"""Extract scene data from Blender depsgraph for ignis-rt."""

import math
import os
import struct
import numpy as np

# Blender Z-up → Vulkan Y-up conversion (applied to positions, normals, transforms)
# Swaps Y↔Z, negates new Z:  x' = x,  y' = z,  z' = -y
def _bake_sky_texture(sky_node, width=256, height=128):
    """Bake a Blender Sky Texture node to an equirectangular image.

    Uses Blender's built-in render to evaluate the sky at every direction.
    Returns a bpy.types.Image or None on failure.
    """
    import bpy

    # Create a temporary image
    img_name = f"__ignis_sky_bake_{id(sky_node)}"
    if img_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[img_name])
    img = bpy.data.images.new(img_name, width, height, alpha=False, float_buffer=True)

    # Evaluate sky at each pixel direction
    pixels = np.zeros(width * height * 4, dtype=np.float32)

    # Get sky type and parameters
    sky_type = getattr(sky_node, 'sky_type', 'NISHITA')
    sun_elevation = getattr(sky_node, 'sun_elevation', 0.5)
    sun_rotation = getattr(sky_node, 'sun_rotation', 0.0)
    sun_intensity = getattr(sky_node, 'sun_intensity', 1.0)
    sun_size = getattr(sky_node, 'sun_size', 0.009512)
    altitude = getattr(sky_node, 'altitude', 0.0)
    air_density = getattr(sky_node, 'air_density', 1.0)
    dust_density = getattr(sky_node, 'dust_density', 1.0)
    ozone_density = getattr(sky_node, 'ozone_density', 1.0)

    # Simple Preetham-like sky model for baking (matches our GPU procedural sky)
    sun_dir = np.array([
        math.cos(sun_elevation) * math.sin(sun_rotation),
        math.sin(sun_elevation),
        math.cos(sun_elevation) * math.cos(sun_rotation)
    ])
    sun_dir /= max(np.linalg.norm(sun_dir), 1e-6)

    for y in range(height):
        theta = math.pi * (y + 0.5) / height  # 0=top, pi=bottom
        for x in range(width):
            phi = 2.0 * math.pi * (x + 0.5) / width  # 0=front, 2pi=around

            # Direction from equirectangular coords
            dx = math.sin(theta) * math.sin(phi)
            dy = math.cos(theta)  # up
            dz = math.sin(theta) * math.cos(phi)
            d = np.array([dx, dy, dz])

            # Sky evaluation (simplified Preetham)
            cos_gamma = max(min(np.dot(d, sun_dir), 1.0), -1.0)
            cos_theta = max(d[1], 0.001)  # elevation angle

            # Rayleigh
            rayleigh_phase = 0.75 * (1.0 + cos_gamma * cos_gamma)
            sun_alt = sun_dir[1]
            sunset_shift = max(0.0, min(1.0, 1.0 - sun_alt * 2.0))
            r_color = np.array([0.30, 0.42, 0.68]) * (1 - sunset_shift * 0.6) + np.array([0.7, 0.35, 0.15]) * (sunset_shift * 0.6)
            opt_depth = 1.0 / (cos_theta + 0.15 * max(93.885 - math.degrees(math.acos(cos_theta)), 0.1)**(-1.253))
            opt_depth = min(opt_depth, 40.0)
            rayleigh = r_color * rayleigh_phase * math.exp(-opt_depth * 0.1)

            # Mie
            g = 0.76
            mie_phase = 1.5 * ((1 - g*g) / (2 + g*g)) * (1 + cos_gamma*cos_gamma) / max((1 + g*g - 2*g*cos_gamma)**1.5, 0.001)
            mie = np.array([1.0, 0.95, 0.9]) * mie_phase * 0.02

            sun_int = max(0.05, min(1.0, sun_alt * 3.0 + 0.3))
            sky = (rayleigh + mie) * sun_int * sun_intensity

            # Sun disk
            sky += max(cos_gamma, 0.0)**512 * 5.0 * sun_intensity
            sky += max(cos_gamma, 0.0)**16 * 0.4 * sun_intensity

            # Below horizon
            if d[1] < 0:
                ground_t = max(0.0, min(1.0, -d[1] * 5.0))
                fog = np.array([0.4, 0.45, 0.5]) * sun_int
                sky = fog * (1 - ground_t) + np.array([0.1, 0.1, 0.1]) * ground_t

            sky = np.maximum(sky, 0.0)
            idx = (y * width + x) * 4
            pixels[idx] = sky[0]
            pixels[idx+1] = sky[1]
            pixels[idx+2] = sky[2]
            pixels[idx+3] = 1.0

    img.pixels.foreach_set(pixels)
    img.pack()
    return img


COORD_CONV = np.array([
    [1,  0,  0,  0],
    [0,  0,  1,  0],
    [0, -1,  0,  0],
    [0,  0,  0,  1],
], dtype=np.float32)


def _convert_positions(arr):
    """Convert Nx3 positions from Blender Z-up to Vulkan Y-up."""
    out = np.empty_like(arr)
    out[:, 0] = arr[:, 0]
    out[:, 1] = arr[:, 2]
    out[:, 2] = -arr[:, 1]
    return out


def _convert_normals(arr):
    """Convert Nx3 normals from Blender Z-up to Vulkan Y-up."""
    return _convert_positions(arr)  # same swizzle for direction vectors


def _matrix_to_3x4_row_major(blender_matrix):
    """Convert a Blender 4x4 Matrix to a 12-float row-major 3x4 array (Vulkan TLAS)."""
    m = np.array(blender_matrix, dtype=np.float32)  # 4x4 column-major from Blender
    # Apply coordinate conversion: COORD_CONV @ M
    m = COORD_CONV @ m
    # Extract top 3 rows (row-major), 4 columns → 12 floats
    return m[:3, :].flatten().tolist()


def _export_particle_hair(eval_obj, particle_system, depsgraph):
    """Convert particle hair strands to cross-shaped quad geometry.

    Each strand becomes 2 perpendicular quads (4 tris) for visibility from all angles.
    Uses co_object() for correct coordinate space (like Cycles' BKE_particle_co_hair).

    Returns mesh dict compatible with unique_meshes format, or None if no hair.
    """
    ps = particle_system
    particles = ps.particles
    if len(particles) == 0:
        return None

    settings = ps.settings

    # Find the ParticleSystem modifier for co_object()
    ps_modifier = None
    for mod in eval_obj.modifiers:
        if mod.type == 'PARTICLE_SYSTEM' and mod.particle_system == ps:
            ps_modifier = mod
            break

    # Hair width: use particle render size, or estimate from strand length
    hair_radius = getattr(settings, 'radius_scale', 0.0)
    if hair_radius <= 0:
        # Estimate from first strand using co_object for correct positions
        if len(particles) > 0 and len(particles[0].hair_keys) >= 2:
            p0_keys = particles[0].hair_keys
            if ps_modifier:
                p0 = np.array(p0_keys[0].co_object(eval_obj, ps_modifier, particles[0]), dtype=np.float32)
                p1 = np.array(p0_keys[-1].co_object(eval_obj, ps_modifier, particles[0]), dtype=np.float32)
            else:
                p0 = np.array(p0_keys[0].co, dtype=np.float32)
                p1 = np.array(p0_keys[-1].co, dtype=np.float32)
            strand_len = np.linalg.norm(p1 - p0)
            hair_radius = max(strand_len * 0.05, 0.005)
        else:
            hair_radius = 0.01

    all_positions = []
    all_normals = []
    all_uvs = []
    all_indices = []
    all_mat_indices = []
    vert_offset = 0

    # Material index from particle settings
    mat_idx = getattr(settings, 'material', 1) - 1  # 1-based in Blender
    if mat_idx < 0:
        mat_idx = 0

    for p_idx, particle in enumerate(particles):
        hair_keys = particle.hair_keys
        n_keys = len(hair_keys)
        if n_keys < 2:
            continue

        # Extract hair key positions in OBJECT LOCAL space
        # co_object() returns the correct position (like Cycles' BKE_particle_co_hair)
        keys = np.empty((n_keys, 3), dtype=np.float32)
        if ps_modifier:
            for ki in range(n_keys):
                keys[ki] = hair_keys[ki].co_object(eval_obj, ps_modifier, particle)
        else:
            raw = np.empty(n_keys * 3, dtype=np.float32)
            hair_keys.foreach_get("co", raw)
            keys = raw.reshape(-1, 3)

        root = keys[0]
        tip = keys[-1]
        strand_dir = tip - root
        strand_len = np.linalg.norm(strand_dir)

        # Log first strand
        if p_idx == 0:
            import os
            try:
                with open(os.path.join(os.path.expanduser("~"), "ignis-rt.log"), "a") as _lf:
                    _lf.write(f"[ignis-hair] co_object strand: root=({root[0]:.4f},{root[1]:.4f},{root[2]:.4f}) "
                              f"tip=({tip[0]:.4f},{tip[1]:.4f},{tip[2]:.4f}) len={strand_len:.4f} radius={hair_radius:.4f}\n")
                    _lf.flush()
            except Exception:
                pass
        if strand_len < 1e-8:
            continue

        # Generate a cross-shaped pair of quads per strand for visibility from all angles.
        # Each quad: 2 triangles, aligned along the strand, rotated 90° from each other.
        # This gives volume without needing camera-facing billboards.
        strand_up = strand_dir / strand_len

        # First perpendicular direction
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(strand_up, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        perp1 = np.cross(strand_up, ref)
        perp1 /= max(np.linalg.norm(perp1), 1e-8)

        # Second perpendicular (90° rotated)
        perp2 = np.cross(strand_up, perp1)
        perp2 /= max(np.linalg.norm(perp2), 1e-8)

        # For each cross direction, create a quad from root to tip
        for perp in [perp1, perp2]:
            v_base = vert_offset
            quad_verts = np.empty((4, 3), dtype=np.float32)
            quad_normals = np.empty((4, 3), dtype=np.float32)
            quad_uvs = np.empty((4, 2), dtype=np.float32)

            # Root: wider
            quad_verts[0] = root - perp * hair_radius
            quad_verts[1] = root + perp * hair_radius
            # Tip: narrower (taper)
            quad_verts[2] = tip - perp * hair_radius * 0.3
            quad_verts[3] = tip + perp * hair_radius * 0.3

            # Normal: perpendicular to quad face
            normal = np.cross(perp, strand_up)
            n_len = np.linalg.norm(normal)
            if n_len > 1e-8:
                normal /= n_len
            quad_normals[:] = normal

            quad_uvs[0] = [0.0, 0.0]
            quad_uvs[1] = [1.0, 0.0]
            quad_uvs[2] = [0.0, 1.0]
            quad_uvs[3] = [1.0, 1.0]

            all_positions.append(quad_verts)
            all_normals.append(quad_normals)
            all_uvs.append(quad_uvs)

            # Two triangles per quad (both sides visible — no backface cull for hair)
            all_indices.extend([v_base, v_base+2, v_base+1,
                                v_base+1, v_base+2, v_base+3])
            all_mat_indices.extend([mat_idx, mat_idx])
            vert_offset += 4

    if not all_positions:
        return None

    positions = np.concatenate(all_positions, axis=0).astype(np.float32)
    normals = np.concatenate(all_normals, axis=0).astype(np.float32)
    uvs = np.concatenate(all_uvs, axis=0).astype(np.float32)
    indices = np.array(all_indices, dtype=np.uint32)
    mat_indices = np.array(all_mat_indices, dtype=np.int32)

    tri_count = len(mat_indices)
    vert_count = len(positions)

    return {
        "name": f"hair_{eval_obj.name}",
        "positions": np.ascontiguousarray(positions.flatten(), dtype=np.float32),
        "normals": np.ascontiguousarray(normals.flatten(), dtype=np.float32),
        "uvs": np.ascontiguousarray(uvs.flatten(), dtype=np.float32),
        "indices": np.ascontiguousarray(indices, dtype=np.uint32),
        "vertex_count": vert_count,
        "index_count": len(indices),
        "tri_count": tri_count,
        "raw_vert_count": vert_count,
        "tri_material_indices": mat_indices,
    }


def export_meshes(depsgraph):
    """Export mesh data with instancing and vertex deduplication.

    Same mesh data shared by multiple instances is exported only once.
    Identical vertices (same pos+normal+UV) are merged via np.unique.

    Returns (unique_meshes, instances) where:
        unique_meshes: dict of mesh_key → {
            "positions": np.ndarray (float32, vertexCount*3),
            "normals":   np.ndarray (float32, vertexCount*3),
            "uvs":       np.ndarray (float32, vertexCount*2),
            "indices":   np.ndarray (uint32,  indexCount),
            "vertex_count": int,
            "index_count": int,
            "tri_count": int,
            "raw_vert_count": int,  (before dedup, for diagnostics)
            "tri_material_indices": np.ndarray (int32, triCount),
        }
        instances: list of {
            "mesh_key": str,
            "transform_3x4": list[float] (12 values, row-major),
            "material_slots": list[Material | None],
        }
    """
    unique_meshes = {}
    instances = []
    skipped_meshes = set()
    # Map object name → mesh key (identity for now, but enables incremental lookup)
    obj_to_mesh_key = {}

    # Build set of objects used as Boolean modifier cutters (should not be rendered)
    boolean_cutters = set()
    for obj in depsgraph.objects:
        if obj.type != 'MESH':
            continue
        for mod in obj.modifiers:
            if mod.type == 'BOOLEAN' and hasattr(mod, 'object') and mod.object:
                boolean_cutters.add(mod.object.name)

    # Build set of objects in collections hidden from viewport.
    # Walk the view layer's layer_collection tree to find excluded/hidden collections.
    hidden_by_collection = set()
    def _walk_layer_collections(lc, parent_hidden=False):
        hidden = parent_hidden or lc.hide_viewport or lc.exclude
        if hidden:
            for obj in lc.collection.objects:
                hidden_by_collection.add(obj.name)
        for child in lc.children:
            _walk_layer_collections(child, hidden)
    try:
        _walk_layer_collections(depsgraph.view_layer.layer_collection)
    except Exception as _e:
        import os
        with open(os.path.join(os.path.expanduser("~"), "ignis-rt.log"), "a") as _lf:
            _lf.write(f"[ignis-export] ERROR walking layer_collections: {_e}\n")

    # Log what we found
    import os
    try:
        with open(os.path.join(os.path.expanduser("~"), "ignis-rt.log"), "a") as _lf:
            _lf.write(f"[ignis-export] Hidden by collection: {len(hidden_by_collection)} objects\n")
            if hidden_by_collection:
                for name in sorted(hidden_by_collection)[:10]:
                    _lf.write(f"  hidden: '{name}'\n")
    except Exception:
        pass

    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.type != 'MESH':
            continue

        # Skip Boolean modifier cutters
        if obj.name in boolean_cutters:
            continue

        # Skip objects in hidden/excluded collections
        if obj.name in hidden_by_collection:
            continue

        # Skip objects hidden at object level
        if not instance.show_self:
            continue
        if obj.hide_viewport:
            continue
        try:
            if obj.hide_get():
                continue
        except RuntimeError:
            pass
        if hasattr(obj, 'visible_camera') and not obj.visible_camera:
            continue

        # Each object gets its own BLAS (safe: modifiers produce unique geometry).
        # Include library path for linked objects to avoid name collisions
        # (e.g., "Cube.026" in classroom vs "Cube.026" in coatStand.blend)
        if obj.library:
            mesh_key = f"{obj.library.filepath}:{obj.name}"
        else:
            mesh_key = obj.name
        # Use same library-qualified key for the mapping (avoids collision in transform sync)
        obj_to_mesh_key[mesh_key] = mesh_key

        if mesh_key in skipped_meshes:
            continue

        if mesh_key not in unique_meshes:
            # Get evaluated mesh (modifiers applied)
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            if mesh is None:
                skipped_meshes.add(mesh_key)
                continue

            mesh.calc_loop_triangles()
            if len(mesh.loop_triangles) == 0:
                eval_obj.to_mesh_clear()
                skipped_meshes.add(mesh_key)
                continue

            tri_count = len(mesh.loop_triangles)
            raw_vert_count = tri_count * 3

            # Loop indices (which mesh.loops form each triangle)
            tri_loops = np.empty(raw_vert_count, dtype=np.int32)
            mesh.loop_triangles.foreach_get("loops", tri_loops)

            # Vertex indices (which mesh.vertices form each triangle)
            tri_verts = np.empty(raw_vert_count, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", tri_verts)

            # --- All vertex positions (local space, Blender Z-up) ---
            all_positions = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", all_positions)
            all_positions = all_positions.reshape(-1, 3)

            # --- Per-corner normals ---
            all_loop_normals = np.empty(len(mesh.loops) * 3, dtype=np.float32)
            try:
                mesh.corner_normals.foreach_get("vector", all_loop_normals)
            except (AttributeError, RuntimeError):
                mesh.calc_normals_split()
                mesh.loops.foreach_get("normal", all_loop_normals)
            all_loop_normals = all_loop_normals.reshape(-1, 3)

            # Unroll per triangle corner (positions stay in Blender local space;
            # TLAS transform handles Z-up → Y-up conversion)
            positions = all_positions[tri_verts]       # (raw_vert_count, 3)
            normals = all_loop_normals[tri_loops]      # (raw_vert_count, 3)

            # UVs
            uvs = np.zeros((raw_vert_count, 2), dtype=np.float32)
            if mesh.uv_layers.active is not None:
                uv_data = mesh.uv_layers.active.data
                all_loop_uvs = np.empty(len(uv_data) * 2, dtype=np.float32)
                uv_data.foreach_get("uv", all_loop_uvs)
                all_loop_uvs = all_loop_uvs.reshape(-1, 2)
                uvs = all_loop_uvs[tri_loops]

            # ---- Vertex deduplication ----
            # Skip dedup for large meshes (>50K tris) — np.unique is O(N log N)
            # and takes seconds on multi-million triangle meshes
            DEDUP_THRESHOLD = 500000
            if tri_count <= DEDUP_THRESHOLD:
                combined = np.ascontiguousarray(
                    np.hstack([positions, normals, uvs]), dtype=np.float32)
                void_dt = np.dtype((np.void, combined.dtype.itemsize * combined.shape[1]))
                _, unique_idx, inverse = np.unique(
                    combined.view(void_dt).ravel(),
                    return_index=True, return_inverse=True)
                dedup_pos = combined[unique_idx, :3]
                dedup_nrm = combined[unique_idx, 3:6]
                dedup_uvs = combined[unique_idx, 6:8]
                dedup_indices = inverse.astype(np.uint32)
                dedup_vert_count = len(unique_idx)
            else:
                # Large mesh: use raw unrolled vertices (no dedup, more VRAM but instant)
                dedup_pos = positions
                dedup_nrm = normals
                dedup_uvs = uvs
                dedup_indices = np.arange(raw_vert_count, dtype=np.uint32)
                dedup_vert_count = raw_vert_count

            # Per-triangle material indices
            tri_mat_indices = np.empty(tri_count, dtype=np.int32)
            mesh.loop_triangles.foreach_get("material_index", tri_mat_indices)

            unique_meshes[mesh_key] = {
                "name": mesh_key,
                "positions": np.ascontiguousarray(dedup_pos.flatten(), dtype=np.float32),
                "normals": np.ascontiguousarray(dedup_nrm.flatten(), dtype=np.float32),
                "uvs": np.ascontiguousarray(dedup_uvs.flatten(), dtype=np.float32),
                "indices": np.ascontiguousarray(dedup_indices, dtype=np.uint32),
                "vertex_count": dedup_vert_count,
                "index_count": len(dedup_indices),
                "tri_count": tri_count,
                "raw_vert_count": raw_vert_count,
                "tri_material_indices": tri_mat_indices,
            }
            eval_obj.to_mesh_clear()

        xform = _matrix_to_3x4_row_major(instance.matrix_world)
        instances.append({
            "mesh_key": mesh_key,
            "transform_3x4": xform,
            "material_slots": [s.material for s in obj.material_slots],
            "is_instance": instance.is_instance,
            "display_type": obj.display_type,
            "hide_render": obj.hide_render,
            "hide_viewport": obj.hide_viewport,
            "visible_camera": getattr(obj, 'visible_camera', '?'),
        })

    # Export particle hair as ribbon meshes
    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.type != 'MESH':
            continue
        if not instance.show_self or obj.name in hidden_by_collection:
            continue
        # Check for particle systems with HAIR type
        eval_obj = obj.evaluated_get(depsgraph)
        for ps_idx, ps in enumerate(eval_obj.particle_systems):
            if ps.settings.type != 'HAIR':
                continue
            if ps.settings.render_type != 'PATH':
                continue

            hair_key = f"{obj.name}__hair_{ps_idx}"
            if obj.library:
                hair_key = f"{obj.library.filepath}:{hair_key}"
            if hair_key in unique_meshes:
                continue

            # Extract hair strands and convert to ribbon mesh
            hair_mesh = _export_particle_hair(eval_obj, ps, depsgraph)
            if hair_mesh is not None:
                unique_meshes[hair_key] = hair_mesh
                obj_to_mesh_key[hair_key] = hair_key
                xform = _matrix_to_3x4_row_major(instance.matrix_world)
                # Use parent object's material slots
                instances.append({
                    "mesh_key": hair_key,
                    "transform_3x4": xform,
                    "material_slots": [s.material for s in obj.material_slots],
                    "is_instance": instance.is_instance,
                    "display_type": obj.display_type,
                    "hide_render": obj.hide_render,
                    "hide_viewport": obj.hide_viewport,
                    "visible_camera": getattr(obj, 'visible_camera', '?'),
                })
                import os
                try:
                    with open(os.path.join(os.path.expanduser("~"), "ignis-rt.log"), "a") as _lf:
                        _lf.write(f"[ignis-export] Hair '{hair_key}': {hair_mesh['tri_count']} tris from {len(ps.particles)} strands\n")
                        _lf.flush()
                except Exception:
                    pass

    # Dump ALL instances to file for ghost mesh diagnosis
    import os
    try:
        _dump_path = os.path.join(os.path.expanduser("~"), "ignis-instances.txt")
        with open(_dump_path, "w") as _df:
            _df.write(f"Total: {len(unique_meshes)} meshes, {len(instances)} instances\n\n")
            for idx, inst in enumerate(instances):
                t = inst["transform_3x4"]
                _df.write(f"[{idx:3d}] mesh='{inst['mesh_key']}' "
                          f"hide_vp={inst.get('hide_viewport',False)} hide_r={inst.get('hide_render',False)} "
                          f"vis_cam={inst.get('visible_camera','?')} "
                          f"pos=({t[3]:.2f}, {t[7]:.2f}, {t[11]:.2f})\n")
    except Exception:
        pass

    # Log instance stats + check for duplicate transforms (ghost mesh diagnosis)
    import os
    _log_path = os.path.join(os.path.expanduser("~"), "ignis-rt.log")
    try:
        from collections import Counter
        key_counts = Counter(i["mesh_key"] for i in instances)
        dupes = {k: v for k, v in key_counts.items() if v > 1}
        if dupes:
            with open(_log_path, "a") as _lf:
                _lf.write(f"[ignis-export] Instanced objects ({len(dupes)} unique, {sum(dupes.values())} instances):\n")
                for k, v in sorted(dupes.items(), key=lambda x: -x[1])[:10]:
                    _lf.write(f"  '{k}' x{v}\n")
                # Log transforms for instanced "Small Windows" or similar
                for k in list(dupes.keys())[:3]:
                    _lf.write(f"  Transforms for '{k}':\n")
                    for inst in instances:
                        if inst["mesh_key"] == k:
                            t = inst["transform_3x4"]
                            _lf.write(f"    pos=({t[3]:.2f}, {t[7]:.2f}, {t[11]:.2f})\n")
                _lf.flush()
    except Exception:
        pass

    return unique_meshes, instances, obj_to_mesh_key, hidden_by_collection


def export_camera(context):
    """Extract camera matrices from the 3D viewport.

    Returns a dict with 16-float lists (column-major) for:
        view, view_inv, proj, proj_inv
    """
    rv3d = context.region_data  # RegionView3D
    region = context.region

    view_matrix = rv3d.view_matrix.copy()        # 4x4
    proj_matrix = rv3d.window_matrix.copy()       # 4x4 perspective

    # Apply coordinate conversion to view matrix
    # In Blender the view matrix transforms world→camera.
    # We need: view_vk = view_blender @ COORD_CONV_INV
    # Since COORD_CONV is orthogonal, its inverse is its transpose.
    import mathutils
    coord_conv_mat = mathutils.Matrix((
        (1,  0,  0,  0),
        (0,  0, -1,  0),
        (0,  1,  0,  0),
        (0,  0,  0,  1),
    ))
    view_matrix = view_matrix @ coord_conv_mat

    # Convert OpenGL projection to D3D/Vulkan Z convention:
    # OpenGL NDC Z is [-1, 1], D3D/Vulkan is [0, 1].
    # NRD expects D3D convention.
    # Y stays positive (D3D convention, shader handles Y-flip).
    # Remap: new clip Z = 0.5 * old clip Z + 0.5 * old clip W
    #   => row2 = 0.5 * row2 + 0.5 * row3
    for col in range(4):
        r2 = proj_matrix[2][col]
        r3 = proj_matrix[3][col]
        proj_matrix[2][col] = 0.5 * r2 + 0.5 * r3

    view_inv = view_matrix.inverted()
    proj_inv = proj_matrix.inverted()

    def flatten(mat):
        """Flatten Blender Matrix to 16 floats (column-major, matching Blender storage)."""
        # Blender stores matrices column-major internally.
        # mat[col][row] — but iterating via list() gives columns.
        result = []
        for col in range(4):
            for row in range(4):
                result.append(mat[row][col])
        return result

    return {
        "view": flatten(view_matrix),
        "view_inv": flatten(view_inv),
        "proj": flatten(proj_matrix),
        "proj_inv": flatten(proj_inv),
        "width": region.width,
        "height": region.height,
    }


def export_world_hdri(depsgraph):
    """Extract HDRI environment texture from Blender's World node tree.

    Returns dict with image data or None if no HDRI found:
        {"name": str, "data": bytes, "width": int, "height": int, "strength": float}
    """
    import bpy

    scene = depsgraph.scene
    world = scene.world
    if not world or not world.use_nodes or not world.node_tree:
        return None

    # Find Background node and its color input
    bg_node = None
    env_tex_node = None
    strength = 1.0

    for node in world.node_tree.nodes:
        if node.type == 'BACKGROUND':
            bg_node = node
            # Get strength
            s_inp = node.inputs.get('Strength')
            if s_inp:
                strength = float(s_inp.default_value)
            # Trace color input to find Environment Texture
            color_inp = node.inputs.get('Color')
            if color_inp and color_inp.is_linked:
                src = color_inp.links[0].from_node
                # Follow through Mapping/Math/etc. to find the texture
                _depth = 0
                while src and _depth < 8:
                    if src.type == 'TEX_ENVIRONMENT':
                        env_tex_node = src
                        break
                    # Follow first linked input
                    found = False
                    for inp in src.inputs:
                        if inp.is_linked:
                            src = inp.links[0].from_node
                            found = True
                            break
                    if not found:
                        break
                    _depth += 1
            break

    if env_tex_node is None or env_tex_node.image is None:
        return None

    image = env_tex_node.image
    w, h = image.size[0], image.size[1]
    if w == 0 or h == 0:
        return None

    # For HDRI (EXR/HDR): extract float pixels, scale to [0,1], encode as PNG manually.
    import io, zlib
    px = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    px_rgb = px.reshape(-1, 4)[:, :3]
    peak = float(np.percentile(px_rgb[px_rgb > 0], 99.5)) if np.any(px_rgb > 0) else 1.0
    peak = max(peak, 1.0)
    px_scaled = np.clip(px / peak, 0.0, 1.0)
    px_u8 = (px_scaled * 255.0 + 0.5).astype(np.uint8).reshape(h, w, 4)
    px_u8 = px_u8[::-1]  # Blender stores bottom-up, PNG is top-down

    # Minimal PNG encoder (RGBA8, no filtering)
    def _write_png_chunk(out, chunk_type, chunk_data):
        out.write(struct.pack('>I', len(chunk_data)))
        out.write(chunk_type)
        out.write(chunk_data)
        out.write(struct.pack('>I', zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF))

    buf = io.BytesIO()
    buf.write(b'\x89PNG\r\n\x1a\n')  # PNG signature
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)  # 8-bit RGBA
    _write_png_chunk(buf, b'IHDR', ihdr)
    # IDAT: raw pixel rows with filter byte 0 (none) per row
    raw_rows = bytearray()
    for y in range(h):
        raw_rows.append(0)  # filter: none
        raw_rows.extend(px_u8[y].tobytes())
    _write_png_chunk(buf, b'IDAT', zlib.compress(bytes(raw_rows), 6))
    _write_png_chunk(buf, b'IEND', b'')
    data = buf.getvalue()

    strength *= peak

    return {
        "name": f"__world_hdri__{image.name}",
        "data": data,
        "width": image.size[0],
        "height": image.size[1],
        "strength": strength,
    }


def export_sun(depsgraph):
    """Find the first SUN light and extract direction/intensity.

    Returns a dict with sun_elevation, sun_azimuth (degrees), sun_intensity.
    Falls back to defaults if no sun light exists.
    """
    defaults = {
        "sun_elevation": 45.0,
        "sun_azimuth": 180.0,
        "sun_intensity": 0.0,
        "sun_color": (1.0, 1.0, 1.0),
    }

    for obj in depsgraph.objects:
        if obj.type != 'LIGHT':
            continue
        light = obj.data
        if light.type != 'SUN':
            continue

        # Sun position direction = +Z axis of the light's world matrix
        # (light travels along -Z local; we want the direction TO the sun)
        mat = obj.matrix_world
        direction = mat.col[2].xyz.normalized()

        # Convert to Vulkan Y-up: Blender Z → Vulkan Y, Blender -Y → Vulkan Z
        dx = direction.x
        dy = direction.z       # Blender Z → Vulkan Y
        dz = -direction.y      # Blender -Y → Vulkan Z

        # Elevation = angle from horizon (XZ plane) toward Y
        elevation = math.degrees(math.asin(max(-1.0, min(1.0, dy))))
        # Azimuth = angle around Y axis from +Z toward +X
        azimuth = math.degrees(math.atan2(dx, dz))
        if azimuth < 0:
            azimuth += 360.0

        final_intensity = light.energy * math.pi

        # Cycles sun: strength is irradiance (W/m²) applied as Li in the
        # rendering equation.  Our Cook-Torrance diffuse = albedo/PI * Li * NdotL,
        # so with Li=1 a white surface gives ~0.25.  Cycles produces the same
        # math but its viewport "feels" brighter because Color Management exposure
        # defaults differ.  Multiply by PI so energy=1 in Blender matches
        # Cycles' perceived brightness (the PI cancels the BRDF's 1/PI).
        return {
            "sun_elevation": elevation,
            "sun_azimuth": azimuth,
            "sun_intensity": light.energy * math.pi,
            "sun_color": (light.color[0], light.color[1], light.color[2]),
        }

    return defaults


def export_lights(depsgraph):
    """Export point/spot/area lights (max 8) for NEE direct sampling.

    Returns a flat list of floats: 16 floats per light.
    [posX, posY, posZ, range, colR, colG, colB, intensity, dirX, dirY, dirZ, sizeX, tanX, tanY, tanZ, sizeY]
    Coordinate conversion: Blender Z-up -> Vulkan Y-up.
    """
    lights = []
    for obj in depsgraph.objects:
        if obj.type != 'LIGHT':
            continue
        light = obj.data
        if light.type == 'SUN':
            continue  # sun is handled separately
        if len(lights) >= 128:  # 8 lights × 16 floats each
            break

        # World position (Blender Z-up -> Vulkan Y-up)
        pos = obj.matrix_world.translation
        vk_x = pos.x
        vk_y = pos.z        # Blender Z -> Vulkan Y
        vk_z = -pos.y       # Blender -Y -> Vulkan Z

        energy = light.energy
        color = light.color
        estimated_range = max(math.sqrt(energy / 0.01), 1.0) if energy > 0 else 10.0
        estimated_range = min(estimated_range, 100.0)

        # Cycles: point light radiance = strength / (4*PI) / r^2
        # Our shader: radiance = color * intensity / r^2
        intensity = energy / math.pi

        # Direction and tangent (for area lights)
        # Area light emits along -Z local axis in Blender
        mat_w = obj.matrix_world
        # Light normal: -Z local → world, then Blender→Vulkan
        bl_normal = -mat_w.col[2].xyz.normalized()
        dir_x = bl_normal.x
        dir_y = bl_normal.z       # Blender Z → Vulkan Y
        dir_z = -bl_normal.y      # Blender -Y → Vulkan Z

        # Light tangent: +X local → world, then Blender→Vulkan
        bl_tangent = mat_w.col[0].xyz.normalized()
        tan_x = bl_tangent.x
        tan_y = bl_tangent.z
        tan_z = -bl_tangent.y

        size_x = 0.0
        size_y = 0.0

        if light.type == 'AREA':
            size_x = light.size if hasattr(light, 'size') else 1.0
            size_y = light.size_y if light.shape in ('RECTANGLE', 'ELLIPSE') else size_x
            # Negative range signals area light to shader
            export_range = -max(size_x, size_y) * 0.5
            # Cycles area light: energy is total power (Watts).
            # Radiance = power * PI / area.  The extra PI compensates the
            # BRDF's 1/PI diffuse term so energy=1 matches Cycles brightness.
            area = size_x * size_y
            intensity = energy * math.pi / area if area > 0 else energy
        else:
            export_range = estimated_range

        print(f"[ignis_rt] export_light: '{obj.name}' type={light.type} "
              f"pos=({vk_x:.2f},{vk_y:.2f},{vk_z:.2f}) "
              f"energy={energy:.1f}W intensity={intensity:.2f} "
              f"range={estimated_range:.1f} "
              f"dir=({dir_x:.2f},{dir_y:.2f},{dir_z:.2f}) "
              f"size=({size_x:.2f},{size_y:.2f}) "
              f"color=({color[0]:.2f},{color[1]:.2f},{color[2]:.2f})")

        lights.extend([
            vk_x, vk_y, vk_z, export_range,
            color[0], color[1], color[2], intensity,
            dir_x, dir_y, dir_z, size_x,
            tan_x, tan_y, tan_z, size_y,
        ])

    return lights


def export_emissive_triangles_fast(depsgraph, unique_meshes, scene_instances, mat_name_to_index):
    """Fast emissive triangle export using already-exported mesh data.

    Avoids re-evaluating meshes from Blender (which hangs on large scenes).
    Uses positions and tri_material_indices from the EXPORT stage.

    Returns a flat list of floats: 16 floats per triangle (max 256 triangles).
    """
    MAX_EMISSIVE_TRIS = 256
    emissive_tris = []

    if not unique_meshes or not scene_instances:
        return []

    # Build material emission info from depsgraph
    mat_emission = {}  # mat_name → (emission_rgb, strength)
    for inst in depsgraph.object_instances:
        obj = inst.object
        if obj.type != 'MESH':
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or mat.name in mat_emission:
                continue
            em_color = (0.0, 0.0, 0.0)
            em_strength = 0.0
            if mat.use_nodes and mat.node_tree:
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        es = node.inputs.get('Emission Strength')
                        em_strength = float(es.default_value) if es else 0.0
                        if em_strength > 0.0:
                            ec = node.inputs.get('Emission Color')
                            if ec and hasattr(ec.default_value, '__len__'):
                                em_color = tuple(float(ec.default_value[i]) for i in range(3))
                        break
                    elif node.type == 'EMISSION':
                        s = node.inputs.get('Strength')
                        c = node.inputs.get('Color')
                        if s: em_strength = float(s.default_value)
                        if c: em_color = tuple(float(c.default_value[i]) for i in range(3))
                        break
            mat_emission[mat.name] = (em_color, em_strength)

    # Process each instance using pre-exported mesh data
    for inst_data in scene_instances:
        if len(emissive_tris) >= MAX_EMISSIVE_TRIS:
            break

        mesh_key = inst_data["mesh_key"]
        if mesh_key not in unique_meshes:
            continue
        mdata = unique_meshes[mesh_key]

        # Check if any material slot has emission
        slots = inst_data.get("material_slots", [])
        slot_emission = []
        for s in slots:
            if s and s.name in mat_emission:
                slot_emission.append(mat_emission[s.name])
            else:
                slot_emission.append(((0.0, 0.0, 0.0), 0.0))
        if not slot_emission:
            slot_emission = [((0.0, 0.0, 0.0), 0.0)]

        # Check if any slot has emission
        if not any(s[1] > 0.0 for s in slot_emission):
            continue

        # Use pre-exported positions (already flattened float32)
        positions = mdata["positions"].reshape(-1, 3).astype(np.float64)
        tri_mat = mdata["tri_material_indices"]
        tri_count = mdata["tri_count"]
        indices = mdata["indices"].reshape(-1, 3) if mdata["index_count"] == tri_count * 3 else None
        if indices is None:
            continue

        # Build transform from instance data
        xf = inst_data["transform_3x4"]
        mat_world = np.array([
            [xf[0], xf[1], xf[2], xf[3]],
            [xf[4], xf[5], xf[6], xf[7]],
            [xf[8], xf[9], xf[10], xf[11]],
            [0.0,   0.0,   0.0,    1.0],
        ], dtype=np.float64)

        # Transform positions to world space (vectorized)
        ones = np.ones((len(positions), 1), dtype=np.float64)
        pos_h = np.hstack([positions, ones])
        world_pos = (mat_world @ pos_h.T).T[:, :3]

        # The transform already converts Z-up → Y-up (done by _matrix_to_3x4_row_major)
        # so no additional swizzle needed

        n_slots = len(slot_emission)
        for ti in range(tri_count):
            if len(emissive_tris) >= MAX_EMISSIVE_TRIS:
                break

            mat_idx = min(int(tri_mat[ti]), n_slots - 1)
            em_color, em_strength = slot_emission[max(mat_idx, 0)]
            if em_strength <= 0.0:
                continue

            vi = indices[ti]
            wp0 = world_pos[vi[0]]
            wp1 = world_pos[vi[1]]
            wp2 = world_pos[vi[2]]

            edge1 = wp1 - wp0
            edge2 = wp2 - wp0
            cross = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(cross)
            if area < 1e-8:
                continue

            emission = (em_color[0] * em_strength,
                        em_color[1] * em_strength,
                        em_color[2] * em_strength)
            luminance = 0.2126 * emission[0] + 0.7152 * emission[1] + 0.0722 * emission[2]
            power = area * luminance

            emissive_tris.append((power, [
                wp0[0], wp0[1], wp0[2], area,
                wp1[0], wp1[1], wp1[2], 0.0,
                wp2[0], wp2[1], wp2[2], 0.0,
                emission[0], emission[1], emission[2], 0.0,
            ]))

    if not emissive_tris:
        return []

    # Sort by power descending, keep top MAX_EMISSIVE_TRIS
    emissive_tris.sort(key=lambda x: x[0], reverse=True)
    emissive_tris = emissive_tris[:MAX_EMISSIVE_TRIS]

    # Build CDF
    total_power = sum(t[0] for t in emissive_tris)
    if total_power <= 0.0:
        return []

    result = []
    cumulative = 0.0
    for power, data in emissive_tris:
        cumulative += power / total_power
        data[7] = cumulative  # CDF
        data[11] = total_power
        result.extend(data)

    return result


def export_emissive_triangles(depsgraph):
    """Export emissive mesh triangles for MIS (Multiple Importance Sampling).

    Iterates all mesh instances, finds triangles whose material has emission > 0,
    computes world-space vertices, builds a CDF weighted by area * luminance.

    Returns a flat list of floats: 16 floats per triangle (max 256 triangles).
    Layout per tri: [v0.xyz, area, v1.xyz, cdf, v2.xyz, totalPower, emission.rgb, 0]
    """
    MAX_EMISSIVE_TRIS = 256
    emissive_tris = []  # list of (power, [16 floats])

    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.type != 'MESH':
            continue

        # Check if any material slot has emission
        has_emission = False
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            if not mat.use_nodes or not mat.node_tree:
                continue
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    es_inp = node.inputs.get('Emission Strength')
                    if es_inp and float(es_inp.default_value) > 0.0:
                        has_emission = True
                        break
                elif node.type == 'EMISSION':
                    strength_inp = node.inputs.get('Strength')
                    if strength_inp and float(strength_inp.default_value) > 0.0:
                        has_emission = True
                        break
            if has_emission:
                break
        if not has_emission:
            continue

        # Get evaluated mesh
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None:
            continue

        mesh.calc_loop_triangles()
        if len(mesh.loop_triangles) == 0:
            eval_obj.to_mesh_clear()
            continue

        # Get world transform (Blender 4x4) and convert to numpy
        mat_world = np.array(instance.matrix_world, dtype=np.float64)

        # Get all vertex positions
        all_positions = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", all_positions)
        all_positions = all_positions.reshape(-1, 3)

        # Build per-slot emission info
        slot_emission = []  # list of (emission_rgb, strength) per slot
        for slot in obj.material_slots:
            mat = slot.material
            em_color = (0.0, 0.0, 0.0)
            em_strength = 0.0
            if mat and mat.use_nodes and mat.node_tree:
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        es_inp = node.inputs.get('Emission Strength')
                        em_strength = float(es_inp.default_value) if es_inp else 0.0
                        if em_strength > 0.0:
                            ec_inp = node.inputs.get('Emission Color')
                            if ec_inp and hasattr(ec_inp.default_value, '__len__'):
                                ec = ec_inp.default_value
                                em_color = (float(ec[0]), float(ec[1]), float(ec[2]))
                            else:
                                em_color = (1.0, 1.0, 1.0)
                        break
                    elif node.type == 'EMISSION':
                        strength_inp = node.inputs.get('Strength')
                        color_inp = node.inputs.get('Color')
                        if strength_inp:
                            em_strength = float(strength_inp.default_value)
                        if color_inp:
                            c = color_inp.default_value
                            em_color = (float(c[0]), float(c[1]), float(c[2]))
                        break
            slot_emission.append((em_color, em_strength))

        if not slot_emission:
            slot_emission = [((0.0, 0.0, 0.0), 0.0)]

        # Vectorized emissive triangle extraction (avoids per-tri Python loop)
        tri_count = len(mesh.loop_triangles)

        # Get per-triangle material indices
        tri_mat_indices = np.empty(tri_count, dtype=np.int32)
        mesh.loop_triangles.foreach_get("material_index", tri_mat_indices)

        # Get per-triangle vertex indices
        tri_verts = np.empty(tri_count * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri_verts)
        tri_verts = tri_verts.reshape(-1, 3)

        # Build emission mask: which triangles have emission > 0
        n_slots = len(slot_emission)
        slot_strengths = np.array([s[1] for s in slot_emission], dtype=np.float64)
        clamped_mat = np.clip(tri_mat_indices, 0, n_slots - 1)
        emissive_mask = slot_strengths[clamped_mat] > 0.0

        emissive_indices = np.where(emissive_mask)[0]
        if len(emissive_indices) == 0:
            eval_obj.to_mesh_clear()
            continue

        # Transform ALL positions to world space at once (vectorized)
        ones = np.ones((len(all_positions), 1), dtype=np.float64)
        pos_h = np.hstack([all_positions, ones])  # (N, 4)
        world_pos = (mat_world @ pos_h.T).T[:, :3]  # (N, 3)

        # Blender Z-up → Vulkan Y-up: swap Y↔Z, negate new Z
        world_pos_vk = world_pos[:, [0, 2, 1]].copy()
        world_pos_vk[:, 2] *= -1.0

        for ti in emissive_indices:
            if len(emissive_tris) >= MAX_EMISSIVE_TRIS:
                break

            mat_idx = clamped_mat[ti]
            em_color, em_strength = slot_emission[mat_idx]

            vi = tri_verts[ti]
            wp0 = world_pos_vk[vi[0]]
            wp1 = world_pos_vk[vi[1]]
            wp2 = world_pos_vk[vi[2]]

            edge1 = wp1 - wp0
            edge2 = wp2 - wp0
            cross = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(cross)
            if area < 1e-8:
                continue

            emission = (em_color[0] * em_strength,
                        em_color[1] * em_strength,
                        em_color[2] * em_strength)

            # Power = area * luminance(emission)
            luminance = 0.2126 * emission[0] + 0.7152 * emission[1] + 0.0722 * emission[2]
            power = area * luminance
            if power < 1e-8:
                continue

            emissive_tris.append((power, [
                wp0[0], wp0[1], wp0[2], area,
                wp1[0], wp1[1], wp1[2], 0.0,  # cdf placeholder
                wp2[0], wp2[1], wp2[2], 0.0,  # totalPower placeholder
                emission[0], emission[1], emission[2], 0.0,
            ]))

        eval_obj.to_mesh_clear()

    if not emissive_tris:
        return []

    # Sort by power descending and cap
    emissive_tris.sort(key=lambda x: x[0], reverse=True)
    emissive_tris = emissive_tris[:MAX_EMISSIVE_TRIS]

    # Build CDF
    total_power = sum(t[0] for t in emissive_tris)
    cumulative = 0.0
    result = []
    for power, data in emissive_tris:
        cumulative += power
        data[7] = cumulative / total_power  # cdf in v1.w
        data[11] = total_power               # totalPower in v2.w
        result.extend(data)

    return result


# ---- GPUMaterial struct layout (140 bytes, scalar) ----
# Must match GPUMaterial in vk_rt_pipeline.h exactly.
# 35 fields x 4 bytes = 140 bytes total.
_GPU_MATERIAL_STRUCT = struct.Struct('<' + 'I' * 5 + 'f' * 3  # tex indices (4) + normalDetail + ks (3)
                                    + 'f' * 4                  # ksSpecularEXP + emissive RGB
                                    + 'f' * 4                  # fresnelC/EXP + detailUVMult + detailNBlend
                                    + 'I' + 'f' + 'I' + 'f'   # flags + alphaRef + shaderType + fresnelMaxLevel
                                    + 'I' * 6                  # multilayer tex indices (6)
                                    + 'f' * 7                  # multilayer mults (7)
                                    + 'f' * 2)                 # sunSpecular + sunSpecularEXP

_NO_TEX = 0xFFFFFFFF


def _pack_gpu_material(
    base_color=(0.8, 0.8, 0.8),
    roughness=0.5,
    metallic=0.0,
    specular_level=0.5,
    emission=(0.0, 0.0, 0.0),
    emission_strength=0.0,
    normal_strength=1.0,
    diffuse_tex=_NO_TEX,
    normal_tex=_NO_TEX,
    orm_tex=_NO_TEX,
    emission_tex=_NO_TEX,
    alpha=1.0,
    ior=1.5,
    transmission=0.0,
    flags=0,
    alpha_ref=0.5,
    transparent_prob=0.0,
    uv_scale_x=1.0,
    uv_scale_y=1.0,
    color_value=1.0,
    color_saturation=1.0,
):
    """Pack one material into 140 bytes matching GPUMaterial."""
    return _GPU_MATERIAL_STRUCT.pack(
        # Texture indices
        diffuse_tex,        # diffuseTexIndex
        normal_tex,         # normalTexIndex
        orm_tex,            # mapsTexIndex (ORM)
        emission_tex,       # detailTexIndex → emission texture
        # normalDetailTexIndex + ksAmbient/ksDiffuse/ksSpecular
        _NO_TEX,
        base_color[0], base_color[1], base_color[2],
        # ksSpecularEXP (= roughness), emissive RGB
        roughness,
        emission[0], emission[1], emission[2],
        # fresnelC (= metallic), fresnelEXP (= specularLevel), detailUVMult (= IOR), detailNormalBlend
        metallic, specular_level, ior, normal_strength,
        # flags, alphaRef, shaderType, fresnelMaxLevel (= emission strength)
        flags,              # flags (bit0=alpha_test, bit1=transmission)
        alpha_ref,          # alphaRef
        100,                # shaderType = SHADER_BLENDER_PBR
        emission_strength,  # fresnelMaxLevel
        # Multilayer tex indices (6x NO_TEX)
        _NO_TEX, _NO_TEX, _NO_TEX, _NO_TEX, _NO_TEX, _NO_TEX,
        # multR=transmission, multG=alpha, multB=transparentProb, rest unused
        transmission, alpha, transparent_prob, uv_scale_x, uv_scale_y, color_value, color_saturation,
        # sunSpecular, sunSpecularEXP
        0.0, 0.0,
    )


def _blackbody_to_rgb(temperature):
    """Convert blackbody temperature (Kelvin) to linear RGB.
    Attempt to use Blender's built-in; fall back to Tanner Helland approximation."""
    try:
        from mathutils import Color
        c = Color()
        c.from_scene_linear_to_srgb = False
        # Blender 4+ has blackbody_to_rgb
        import _cycles
        r, g, b = _cycles.blackbody_to_rgb(temperature)
        return (r, g, b)
    except Exception:
        pass

    # Tanner Helland approximation (sRGB, then linearize)
    t = max(1000.0, min(temperature, 40000.0)) / 100.0
    if t <= 66.0:
        r = 1.0
        g = max(0.0, 0.39008 * math.log(t) - 0.63184)
        b = max(0.0, 0.54321 * math.log(t - 10.0) - 1.19625) if t > 20.0 else 0.0
    else:
        r = max(0.0, 1.292936 * ((t - 60.0) ** -0.1332))
        g = max(0.0, 1.129891 * ((t - 60.0) ** -0.0755))
        b = 1.0
    # Clamp and linearize (approximate sRGB→linear)
    r, g, b = min(r, 1.0), min(g, 1.0), min(b, 1.0)
    r, g, b = r ** 2.2, g ** 2.2, b ** 2.2
    return (r, g, b)


def _resolve_scalar_input(socket, default=0.5, _depth=0):
    """Resolve a scalar socket value, following links recursively."""
    if socket is None or _depth > 8:
        return default
    if not socket.is_linked:
        val = socket.default_value
        return float(val) if not hasattr(val, '__len__') else float(val[0]) if len(val) > 0 else default
    from_node = socket.links[0].from_node

    if from_node.type == 'VALUE':
        out = from_node.outputs.get('Value')
        return float(out.default_value) if out else default

    if from_node.type == 'MATH':
        op = from_node.operation
        a = _resolve_scalar_input(from_node.inputs[0], 0.0, _depth + 1) if len(from_node.inputs) > 0 else 0.0
        b = _resolve_scalar_input(from_node.inputs[1], 0.0, _depth + 1) if len(from_node.inputs) > 1 else 0.0
        import math as _m
        if op == 'ADD': return a + b
        elif op == 'SUBTRACT': return a - b
        elif op == 'MULTIPLY': return a * b
        elif op == 'DIVIDE': return a / b if b != 0 else 0.0
        elif op == 'POWER': return _m.pow(max(a, 0), b) if a >= 0 else 0.0
        elif op == 'MINIMUM': return min(a, b)
        elif op == 'MAXIMUM': return max(a, b)
        elif op == 'ABSOLUTE': return abs(a)
        elif op == 'SQRT': return _m.sqrt(max(a, 0))
        return a

    if from_node.type == 'CLAMP':
        val = _resolve_scalar_input(from_node.inputs.get('Value'), 0.5, _depth + 1)
        mn = _resolve_scalar_input(from_node.inputs.get('Min'), 0.0, _depth + 1)
        mx = _resolve_scalar_input(from_node.inputs.get('Max'), 1.0, _depth + 1)
        return max(mn, min(val, mx))

    if from_node.type == 'MAP_RANGE':
        val = _resolve_scalar_input(from_node.inputs[0], 0.5, _depth + 1)
        from_min = _resolve_scalar_input(from_node.inputs[1], 0.0, _depth + 1)
        from_max = _resolve_scalar_input(from_node.inputs[2], 1.0, _depth + 1)
        to_min = _resolve_scalar_input(from_node.inputs[3], 0.0, _depth + 1)
        to_max = _resolve_scalar_input(from_node.inputs[4], 1.0, _depth + 1)
        if from_max - from_min != 0:
            t = (val - from_min) / (from_max - from_min)
            return to_min + t * (to_max - to_min)
        return to_min

    return default


def _eval_color_ramp(ramp_node, fac):
    """Evaluate a ColorRamp node at a given factor value."""
    elements = ramp_node.color_ramp.elements
    if len(elements) == 0:
        return (0.5, 0.5, 0.5)
    if fac <= elements[0].position:
        c = elements[0].color
        return (c[0], c[1], c[2])
    if fac >= elements[-1].position:
        c = elements[-1].color
        return (c[0], c[1], c[2])
    # Find surrounding stops and interpolate
    for i in range(len(elements) - 1):
        if elements[i].position <= fac <= elements[i + 1].position:
            t = (fac - elements[i].position) / max(elements[i + 1].position - elements[i].position, 1e-6)
            c0, c1 = elements[i].color, elements[i + 1].color
            return (c0[0] + (c1[0] - c0[0]) * t,
                    c0[1] + (c1[1] - c0[1]) * t,
                    c0[2] + (c1[2] - c0[2]) * t)
    c = elements[-1].color
    return (c[0], c[1], c[2])


def _resolve_color_input(socket, default=(0.8, 0.8, 0.8), _depth=0):
    """Resolve a color socket value, following links recursively."""
    if socket is None or _depth > 8:
        return default
    if not socket.is_linked:
        c = socket.default_value
        if hasattr(c, '__len__') and len(c) >= 3:
            return (c[0], c[1], c[2])
        return default
    from_node = socket.links[0].from_node

    # Blackbody → RGB
    if from_node.type == 'BLACKBODY':
        temp_inp = from_node.inputs.get('Temperature')
        temp = float(temp_inp.default_value) if temp_inp else 5000.0
        return _blackbody_to_rgb(temp)

    # RGB constant
    if from_node.type == 'RGB':
        out = from_node.outputs.get('Color')
        if out:
            c = out.default_value
            return (c[0], c[1], c[2])

    # ColorRamp — evaluate the ramp at the input factor
    if from_node.type == 'VALTORGB':
        fac = _resolve_scalar_input(from_node.inputs.get('Fac'), 0.5, _depth + 1)
        return _eval_color_ramp(from_node, fac)

    # MixRGB / Mix — blend two colors
    if from_node.type in ('MIX_RGB', 'MIX'):
        fac = _resolve_scalar_input(from_node.inputs.get('Fac') or from_node.inputs.get('Factor'), 0.5, _depth + 1)
        c1 = _resolve_color_input(from_node.inputs.get('Color1') or from_node.inputs.get('A'), default, _depth + 1)
        c2 = _resolve_color_input(from_node.inputs.get('Color2') or from_node.inputs.get('B'), default, _depth + 1)
        blend_type = getattr(from_node, 'blend_type', 'MIX')
        if blend_type == 'MIX':
            return _lerp_color(c1, c2, fac)
        elif blend_type == 'MULTIPLY':
            return (c1[0] * c2[0], c1[1] * c2[1], c1[2] * c2[2])
        elif blend_type == 'ADD':
            return (c1[0] + c2[0] * fac, c1[1] + c2[1] * fac, c1[2] + c2[2] * fac)
        elif blend_type == 'SUBTRACT':
            return (c1[0] - c2[0] * fac, c1[1] - c2[1] * fac, c1[2] - c2[2] * fac)
        elif blend_type == 'SCREEN':
            return (1-(1-c1[0])*(1-c2[0]*fac), 1-(1-c1[1])*(1-c2[1]*fac), 1-(1-c1[2])*(1-c2[2]*fac))
        elif blend_type == 'OVERLAY':
            def _ov(a, b): return 2*a*b if a < 0.5 else 1-2*(1-a)*(1-b)
            return (_ov(c1[0], c2[0]), _ov(c1[1], c2[1]), _ov(c1[2], c2[2]))
        return _lerp_color(c1, c2, fac)

    # Gamma
    if from_node.type == 'GAMMA':
        c = _resolve_color_input(from_node.inputs.get('Color'), default, _depth + 1)
        gamma = _resolve_scalar_input(from_node.inputs.get('Gamma'), 1.0, _depth + 1)
        if gamma > 0:
            return (pow(max(c[0],0), gamma), pow(max(c[1],0), gamma), pow(max(c[2],0), gamma))
        return c

    # Invert
    if from_node.type == 'INVERT':
        fac = _resolve_scalar_input(from_node.inputs.get('Fac'), 1.0, _depth + 1)
        c = _resolve_color_input(from_node.inputs.get('Color'), default, _depth + 1)
        inv = (1-c[0], 1-c[1], 1-c[2])
        return _lerp_color(c, inv, fac)

    # Bright/Contrast
    if from_node.type == 'BRIGHTCONTRAST':
        c = _resolve_color_input(from_node.inputs.get('Color'), default, _depth + 1)
        bright = _resolve_scalar_input(from_node.inputs.get('Bright'), 0.0, _depth + 1)
        contrast = _resolve_scalar_input(from_node.inputs.get('Contrast'), 0.0, _depth + 1)
        return (max(0, (c[0]-0.5)*max(1+contrast,0)+0.5+bright),
                max(0, (c[1]-0.5)*max(1+contrast,0)+0.5+bright),
                max(0, (c[2]-0.5)*max(1+contrast,0)+0.5+bright))

    # Hue/Saturation/Value
    if from_node.type == 'HUE_SAT':
        c = _resolve_color_input(from_node.inputs.get('Color'), default, _depth + 1)
        sat = _resolve_scalar_input(from_node.inputs.get('Saturation'), 1.0, _depth + 1)
        val = _resolve_scalar_input(from_node.inputs.get('Value'), 1.0, _depth + 1)
        luma = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
        return (max(0, (c[0]-luma)*sat*val + luma),
                max(0, (c[1]-luma)*sat*val + luma),
                max(0, (c[2]-luma)*sat*val + luma))

    # Sky/procedural textures — can't evaluate per-pixel, use subdued default
    # Sky textures produce varied colors; average is much dimmer than pure sky blue
    if from_node.type in ('TEX_SKY', 'TEX_ENVIRONMENT'):
        return (0.03, 0.035, 0.05)  # dim sky average
    if from_node.type in ('TEX_NOISE', 'TEX_VORONOI', 'TEX_MUSGRAVE', 'TEX_CHECKER',
                           'TEX_WAVE', 'TEX_GRADIENT', 'TEX_MAGIC', 'TEX_BRICK'):
        return (0.5, 0.5, 0.5)  # neutral gray for procedurals

    # RGB Curves — passthrough Color input (curve evaluation too complex for CPU)
    if from_node.type == 'CURVE_RGB':
        fac = _resolve_scalar_input(from_node.inputs.get('Fac'), 1.0, _depth + 1)
        c = _resolve_color_input(from_node.inputs.get('Color'), default, _depth + 1)
        return c  # return unmodified (curve not evaluated, but at least not black)

    # Passthrough for other color-producing nodes
    _COLOR_PASSTHROUGH = {'CURVE_VEC', 'SEPRGB', 'SEPARATE_XYZ', 'SEPARATE_COLOR',
                          'COMBRGB', 'COMBINE_COLOR', 'COMBINE_XYZ', 'RGBTOBW',
                          'VECT_MATH', 'NORMAL', 'NORMAL_MAP'}
    if from_node.type in _COLOR_PASSTHROUGH:
        for inp_name in ('Color', 'Color1', 'Vector', 'Image', 'A'):
            inp = from_node.inputs.get(inp_name)
            if inp:
                result = _resolve_color_input(inp, default, _depth + 1)
                if result != default:
                    return result
        return default

    # Follow any other linked node — try Color output
    for out_name in ('Color', 'Result', 'Value', 'Vector'):
        out = from_node.outputs.get(out_name)
        if out and hasattr(out, 'default_value'):
            c = out.default_value
            if hasattr(c, '__len__') and len(c) >= 3:
                return (c[0], c[1], c[2])

    # Unlinked or unknown
    c = socket.default_value
    if hasattr(c, '__len__') and len(c) >= 3:
        return (c[0], c[1], c[2])
    return default


def _get_principled_input(node, input_name, default):
    """Get a Principled BSDF input value (scalar or color)."""
    inp = node.inputs.get(input_name)
    if inp is None:
        return default
    return inp.default_value


def _find_uv_scale(socket, _depth=0):
    """Find Mapping node UV scale in the texture chain. Returns (scaleX, scaleY)."""
    if not socket or not socket.is_linked or _depth > 8:
        return (1.0, 1.0)
    node = socket.links[0].from_node
    if node.type == 'MAPPING':
        scale_inp = node.inputs.get('Scale')
        if scale_inp:
            s = scale_inp.default_value
            return (float(s[0]), float(s[1]))
        return (1.0, 1.0)
    if node.type == 'TEX_IMAGE':
        vec_inp = node.inputs.get('Vector')
        if vec_inp:
            return _find_uv_scale(vec_inp, _depth + 1)
    # Follow any linked input
    for inp in node.inputs:
        if inp.is_linked:
            result = _find_uv_scale(inp, _depth + 1)
            if result != (1.0, 1.0):
                return result
    return (1.0, 1.0)


def _find_image_texture_node(socket, _depth=0):
    """Follow links from a BSDF input to find an Image Texture node.

    Recursively traverses common intermediate nodes (Color Ramp, Mix RGB,
    Gamma, Hue/Sat, Invert, etc.) up to 8 levels deep.
    """
    if not socket or not socket.is_linked or _depth > 8:
        return None
    from_node = socket.links[0].from_node

    if from_node.type == 'TEX_IMAGE':
        return from_node

    # Sky Texture — can't evaluate per-pixel, handled by _resolve_color_input fallback
    if from_node.type == 'TEX_SKY':
        return None

    # Normal Map — check Color input
    if from_node.type == 'NORMAL_MAP':
        return _find_image_texture_node(from_node.inputs.get('Color'), _depth + 1)

    # Mapping node — follow the Vector input (UV transforms don't affect texture identity)
    if from_node.type == 'MAPPING':
        return _find_image_texture_node(from_node.inputs.get('Vector'), _depth + 1)

    # Texture Coordinate — not a texture, stop
    if from_node.type == 'TEX_COORD':
        return None

    # Separate RGB/Color/XYZ — check common input names
    if from_node.type in ('SEPRGB', 'SEPARATE_XYZ', 'SEPARATE_COLOR'):
        for inp_name in ('Image', 'Color', 'Vector'):
            inp = from_node.inputs.get(inp_name)
            result = _find_image_texture_node(inp, _depth + 1)
            if result:
                return result

    # Pass-through nodes: follow the first color/image input
    _PASSTHROUGH_TYPES = {
        'VALTORGB',          # Color Ramp
        'MIX_RGB', 'MIX',   # Mix Color (old and new)
        'HUE_SAT',          # Hue/Saturation
        'GAMMA',            # Gamma
        'BRIGHTCONTRAST',   # Bright/Contrast
        'INVERT',           # Invert
        'CURVE_RGB',        # RGB Curves
        'MAP_RANGE',        # Map Range
        'MATH',             # Math node
        'VECT_MATH',        # Vector Math
        'CLAMP',            # Clamp
        'COMBINE_XYZ',      # Combine XYZ
        'COMBRGB', 'COMBINE_COLOR',  # Combine Color
        'RGBTOBW',          # RGB to BW
        'TEX_NOISE',        # Noise texture (passthrough to find source texture)
        'TEX_VORONOI',      # Voronoi texture
        'TEX_MUSGRAVE',     # Musgrave texture
        'TEX_CHECKER',      # Checker texture
        'TEX_WAVE',         # Wave texture
        'TEX_GRADIENT',     # Gradient texture
    }
    if from_node.type in _PASSTHROUGH_TYPES:
        # Try common color input names
        for inp_name in ('Color', 'Color1', 'Color2', 'Fac', 'Image', 'A', 'B', 'Value'):
            inp = from_node.inputs.get(inp_name)
            result = _find_image_texture_node(inp, _depth + 1)
            if result:
                return result

    # Group nodes — try to follow the first linked input
    if from_node.type == 'GROUP':
        for inp in from_node.inputs:
            result = _find_image_texture_node(inp, _depth + 1)
            if result:
                return result

    return None


_image_bytes_cache = {}  # image.name → bytes (persists across export_materials calls)


def _get_image_bytes(image):
    """Extract file bytes from a Blender image for stb_image decoding.

    Tries packed data first, then file on disk. Returns bytes or None.
    Uses a persistent cache to avoid re-reading on material property tweaks.
    """
    if image.name in _image_bytes_cache:
        return _image_bytes_cache[image.name]

    import os
    import bpy

    data = None

    # Packed in .blend: raw file bytes (PNG/JPG/etc.)
    if image.packed_file:
        raw = bytes(image.packed_file.data)
        # Verify it's a format stb_image can handle (PNG/JPG/BMP headers)
        # PNG: 89 50 4E 47, JPG: FF D8 FF, BMP: 42 4D
        if (raw[:4] == b'\x89PNG' or raw[:3] == b'\xff\xd8\xff' or raw[:2] == b'BM'):
            data = raw
        # else: skip packed data — fall through to pixel extraction

    # File on disk
    if data is None and not (image.size[0] > 0 and image.size[1] > 0 and len(image.pixels) > 0):
        filepath = bpy.path.abspath(image.filepath_from_user())
        if filepath and os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                data = f.read()

    # Pixel extraction fallback: read float pixels from Blender and encode as PNG
    # Handles EXR, 16-bit PNG, generated images, and any format Blender can open
    if data is None and image.size[0] > 0 and image.size[1] > 0:
        import io, zlib, struct as _struct
        w, h = image.size[0], image.size[1]
        px = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(px)
        px_u8 = (np.clip(px, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).reshape(h, w, 4)
        px_u8 = px_u8[::-1]  # flip Y (Blender bottom-up → PNG top-down)
        # Minimal PNG encoder
        def _png_chunk(out, ctype, cdata):
            out.write(_struct.pack('>I', len(cdata)))
            out.write(ctype)
            out.write(cdata)
            out.write(_struct.pack('>I', zlib.crc32(ctype + cdata) & 0xFFFFFFFF))
        buf = io.BytesIO()
        buf.write(b'\x89PNG\r\n\x1a\n')
        _png_chunk(buf, b'IHDR', _struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
        raw_rows = bytearray()
        for y in range(h):
            raw_rows.append(0)
            raw_rows.extend(px_u8[y].tobytes())
        _png_chunk(buf, b'IDAT', zlib.compress(bytes(raw_rows), 6))
        _png_chunk(buf, b'IEND', b'')
        data = buf.getvalue()

    if data is not None:
        _image_bytes_cache[image.name] = data
    return data


def _encode_bmp(rgba_u8, w, h):
    """Encode RGBA8 pixels as a 32-bit BMP (stb_image compatible).

    Uses BITMAPINFOHEADER (40 bytes) with BI_BITFIELDS for BGRA layout.
    stb_image supports this format (v2.28+).
    """
    # Use numpy for fast RGBA→BGRA swizzle
    pixels = np.asarray(rgba_u8, dtype=np.uint8).reshape(h, w, 4)
    bgra = pixels[:, :, [2, 1, 0, 3]]
    # BMP is bottom-up by default — flip rows
    bgra = bgra[::-1]
    pixel_data = bgra.tobytes()

    header_size = 14 + 40  # BITMAPFILEHEADER + BITMAPINFOHEADER
    file_size = header_size + len(pixel_data)

    hdr = bytearray(header_size)
    # BITMAPFILEHEADER (14 bytes)
    hdr[0:2] = b'BM'
    struct.pack_into('<I', hdr, 2, file_size)
    struct.pack_into('<I', hdr, 10, header_size)
    # BITMAPINFOHEADER (40 bytes)
    struct.pack_into('<I', hdr, 14, 40)          # biSize
    struct.pack_into('<i', hdr, 18, w)           # biWidth
    struct.pack_into('<i', hdr, 22, h)           # biHeight (positive = bottom-up)
    struct.pack_into('<H', hdr, 26, 1)           # biPlanes
    struct.pack_into('<H', hdr, 28, 32)          # biBitCount
    struct.pack_into('<I', hdr, 30, 0)           # biCompression = BI_RGB
    struct.pack_into('<I', hdr, 34, len(pixel_data))

    return bytes(hdr) + bytes(pixel_data)


def _find_surface_shader(node_tree):
    """Find the shader node connected to Material Output's Surface input."""
    for node in node_tree.nodes:
        if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
            surface_inp = node.inputs.get('Surface')
            if surface_inp and surface_inp.is_linked:
                return surface_inp.links[0].from_node
    return None


def _extract_normal_texture(node, register_image_fn):
    """Extract normal/bump texture from a BSDF node's Normal input.

    Returns (tex_index, strength, is_bump).
    is_bump=True means it's a height/bump map, False means RGB normal map.
    """
    normal_tex = _NO_TEX
    normal_strength = 1.0
    is_bump = False
    norm_inp = node.inputs.get('Normal') if node else None
    if norm_inp and norm_inp.is_linked:
        from_node = norm_inp.links[0].from_node
        if from_node.type == 'NORMAL_MAP':
            normal_strength = float(from_node.inputs['Strength'].default_value)
            color_inp = from_node.inputs.get('Color')
            if color_inp and color_inp.is_linked:
                nn = color_inp.links[0].from_node
                if nn.type == 'TEX_IMAGE' and nn.image:
                    normal_tex = register_image_fn(nn.image)
        elif from_node.type == 'BUMP':
            strength_inp = from_node.inputs.get('Strength')
            distance_inp = from_node.inputs.get('Distance')
            bump_strength = float(strength_inp.default_value) if strength_inp else 1.0
            bump_distance = float(distance_inp.default_value) if distance_inp else 1.0
            # Cycles formula:
            #   N' = normalize(|det| * N - distance * surfgrad)
            #   N_final = normalize(strength * N' + (1 - strength) * N)
            # We only have one float, so encode both:
            #   magnitude = distance, fractional part encodes strength.
            #   E.g., 0.1 distance + 0.06 strength → encode as distance (shader uses it
            #   as scale) and pass strength * distance as combined effective scale.
            # Simplest correct approach: pass distance as scale, shader handles det.
            normal_strength = bump_distance
            height_inp = from_node.inputs.get('Height')
            tex_node = _find_image_texture_node(height_inp)
            if tex_node and tex_node.image:
                normal_tex = register_image_fn(tex_node.image)
                is_bump = True
                # Pack: integer part = strength * 100, fractional = distance
                # shader extracts: strength = floor(abs * 10) / 10, distance = fract(abs * 10) * 10
                # Actually just encode the two values simply:
                normal_strength = bump_strength + bump_distance * 100.0
    return normal_tex, normal_strength, is_bump


def _extract_shader_props(node, register_image_fn):
    """Extract PBR-approximated properties from a single shader node."""
    props = {
        'base_color': (0.8, 0.8, 0.8),
        'roughness': 0.5,
        'metallic': 0.0,
        'emission': (0.0, 0.0, 0.0),
        'emission_strength': 0.0,
        'diffuse_tex': _NO_TEX,
        'normal_tex': _NO_TEX,
        'normal_strength': 1.0,
        'emission_tex': _NO_TEX,
        'transmission': 0.0,
        'ior': 1.5,
        'transparent_prob': 0.0,
    }
    if node is None:
        return props

    if node.type == 'BSDF_DIFFUSE':
        color_inp = node.inputs.get('Color')
        if color_inp:
            c = color_inp.default_value
            props['base_color'] = (c[0], c[1], c[2])
            tex_node = _find_image_texture_node(color_inp)
            if tex_node and tex_node.image:
                props['diffuse_tex'] = register_image_fn(tex_node.image)
        # Cycles: Diffuse BSDF is always Lambertian (roughness=0) or Oren-Nayar (roughness>0).
        # In PBR terms, a Lambertian surface = fully rough (roughness=1.0).
        props['roughness'] = 1.0
        props['metallic'] = 0.0
    elif node.type == 'BSDF_GLOSSY':
        color_inp = node.inputs.get('Color')
        if color_inp:
            c = color_inp.default_value
            props['base_color'] = (c[0], c[1], c[2])
            tex_node = _find_image_texture_node(color_inp)
            if tex_node and tex_node.image:
                props['diffuse_tex'] = register_image_fn(tex_node.image)
        rough_inp = node.inputs.get('Roughness')
        if rough_inp:
            props['roughness'] = float(rough_inp.default_value)
        props['metallic'] = 1.0
    elif node.type == 'BSDF_GLASS':
        color_inp = node.inputs.get('Color')
        if color_inp:
            c = color_inp.default_value
            props['base_color'] = (c[0], c[1], c[2])
        rough_inp = node.inputs.get('Roughness')
        if rough_inp:
            props['roughness'] = float(rough_inp.default_value)
            # If roughness is connected to a texture, try to extract it as ORM
            if rough_inp.is_linked:
                tex_node = _find_image_texture_node(rough_inp)
                if tex_node and tex_node.image:
                    props['diffuse_tex'] = register_image_fn(tex_node.image)
                    # Use a mid roughness since texture will modulate
                    props['roughness'] = 0.5
        ior_inp = node.inputs.get('IOR')
        if ior_inp:
            props['ior'] = float(ior_inp.default_value)
        props['transmission'] = 1.0
    elif node.type == 'BSDF_TRANSPARENT':
        # Fully transparent — stochastic passthrough like Cycles
        tc = node.inputs.get('Color')
        if tc:
            props['base_color'] = (tc.default_value[0], tc.default_value[1], tc.default_value[2])
        else:
            props['base_color'] = (1.0, 1.0, 1.0)
        props['transparent_prob'] = 1.0
        props['roughness'] = 0.0
    elif node.type == 'EMISSION':
        color_inp = node.inputs.get('Color')
        if color_inp:
            props['emission'] = _resolve_color_input(color_inp, (1.0, 1.0, 1.0))
            props['base_color'] = props['emission']
            tex_node = _find_image_texture_node(color_inp)
            if tex_node and tex_node.image:
                props['emission_tex'] = register_image_fn(tex_node.image)
        strength_inp = node.inputs.get('Strength')
        if strength_inp:
            props['emission_strength'] = float(strength_inp.default_value)
    elif node.type == 'MIX_SHADER':
        # Nested Mix Shader — recurse
        props = _resolve_mix_shader(node, register_image_fn)
    elif node.type == 'ADD_SHADER':
        # Add Shader = both shaders contribute. Cycles samples one stochastically (50/50).
        s1 = node.inputs[0].links[0].from_node if len(node.inputs) > 0 and node.inputs[0].is_linked else None
        s2 = node.inputs[1].links[0].from_node if len(node.inputs) > 1 and node.inputs[1].is_linked else None
        p1 = _extract_shader_props(s1, register_image_fn)
        p2 = _extract_shader_props(s2, register_image_fn)
        # Combine transparent_prob: Add Shader averages both sides' probability
        tp1 = p1.get('transparent_prob', 0.0)
        tp2 = p2.get('transparent_prob', 0.0)
        combined_tp = (tp1 + tp2) * 0.5
        # Use the non-transparent side's visual properties
        if tp1 > tp2:
            props = dict(p2)  # use p2's visual props (the visible BSDF)
        elif tp2 > tp1:
            props = dict(p1)
        else:
            # Both equal (both opaque or both transparent) — blend
            props = {
                'base_color': _lerp_color(p1['base_color'], p2['base_color'], 0.5),
                'roughness': (p1['roughness'] + p2['roughness']) * 0.5,
                'metallic': (p1['metallic'] + p2['metallic']) * 0.5,
                'emission': _lerp_color(p1['emission'], p2['emission'], 0.5),
                'emission_strength': (p1['emission_strength'] + p2['emission_strength']) * 0.5,
                'transmission': max(p1['transmission'], p2['transmission']),
                'ior': p1['ior'] if p1['transmission'] >= p2['transmission'] else p2['ior'],
                'diffuse_tex': p1['diffuse_tex'] if p1['diffuse_tex'] != _NO_TEX else p2['diffuse_tex'],
                'emission_tex': p1['emission_tex'] if p1['emission_tex'] != _NO_TEX else p2['emission_tex'],
                'normal_tex': p1.get('normal_tex', _NO_TEX),
                'normal_strength': p1.get('normal_strength', 1.0),
                'transparent_prob': 0.0,
            }
        props['transparent_prob'] = combined_tp
        if combined_tp > 0.0:
            props['flags'] = props.get('flags', 0) | 2  # shadow rays pass through too

    # Extract normal/bump texture for all BSDF types
    if node.type in ('BSDF_DIFFUSE', 'BSDF_GLOSSY', 'BSDF_GLASS', 'BSDF_PRINCIPLED'):
        ntex, nstr, is_bump = _extract_normal_texture(node, register_image_fn)
        props['normal_tex'] = ntex
        # Negative strength signals bump/height map to the shader
        props['normal_strength'] = -nstr if is_bump else nstr

        # Glass + bump: convert bump intensity to GGX roughness.
        # With low SPP, bump alone can't scatter refraction enough (needs 100+ samples
        # to converge like Cycles). GGX roughness disperses rays analytically in 1 sample.
        if node.type == 'BSDF_GLASS' and is_bump and props['roughness'] < 0.01:
            # nstr is packed as: strength + distance * 100
            # e.g. Strength=0.06, Distance=0.1 → nstr = 10.06
            bump_strength = nstr - int(nstr)  # fractional part = strength
            bump_distance = int(nstr) / 100.0
            # Empirical mapping: roughness that produces similar scatter as
            # Cycles' bump averaging over many samples.
            equiv_roughness = min(bump_strength * bump_distance * 25.0, 0.4)
            equiv_roughness = max(equiv_roughness, 0.05)
            props['roughness'] = equiv_roughness

    return props


def _lerp_color(a, b, t):
    return (a[0] * (1 - t) + b[0] * t,
            a[1] * (1 - t) + b[1] * t,
            a[2] * (1 - t) + b[2] * t)


def _resolve_mix_shader(mix_node, register_image_fn):
    """Resolve a Mix Shader node into blended PBR properties."""
    fac_inp = mix_node.inputs.get('Fac')
    if fac_inp and fac_inp.is_linked:
        # Dynamic factor (Fresnel, Layer Weight, Geometry, etc.)
        # Can't evaluate per-pixel on CPU — use 0.5 as neutral blend
        fac = 0.5
    else:
        fac = float(fac_inp.default_value) if fac_inp else 0.5

    # Get the two input shaders (Shader inputs at index 1 and 2)
    shader1_node = None
    shader2_node = None
    shader1_inp = mix_node.inputs[1] if len(mix_node.inputs) > 1 else None
    shader2_inp = mix_node.inputs[2] if len(mix_node.inputs) > 2 else None
    if shader1_inp and shader1_inp.is_linked:
        shader1_node = shader1_inp.links[0].from_node
    if shader2_inp and shader2_inp.is_linked:
        shader2_node = shader2_inp.links[0].from_node

    p1 = _extract_shader_props(shader1_node, register_image_fn)
    p2 = _extract_shader_props(shader2_node, register_image_fn)

    # ---- Alpha cutout detection (fence, leaf, wireframe materials) ----
    # When Mix Shader Factor is linked to an Image Texture Alpha and one side
    # is Transparent BSDF, this is a texture-driven alpha cutout — NOT stochastic.
    # The texture alpha determines per-pixel visibility (like Cycles).
    s1_type = shader1_node.type if shader1_node else None
    s2_type = shader2_node.type if shader2_node else None
    if fac_inp and fac_inp.is_linked and (s1_type == 'BSDF_TRANSPARENT' or s2_type == 'BSDF_TRANSPARENT'):
        # Trace factor link to find an Image Texture
        fac_source = fac_inp.links[0].from_node
        fac_socket = fac_inp.links[0].from_socket.name
        # Follow through reroute/math nodes
        _depth = 0
        while fac_source and fac_source.type in ('REROUTE', 'MATH', 'MAP_RANGE', 'CLAMP') and _depth < 4:
            for inp in fac_source.inputs:
                if inp.is_linked:
                    fac_source = inp.links[0].from_node
                    fac_socket = inp.links[0].from_socket.name
                    break
            else:
                fac_source = None
            _depth += 1

        if fac_source and fac_source.type == 'TEX_IMAGE' and fac_source.image:
            # Alpha cutout: use the non-transparent shader's props + the texture for alpha test
            if s1_type == 'BSDF_TRANSPARENT':
                result = dict(p2)
            else:
                result = dict(p1)
            # Register the texture so the shader reads both color and alpha
            tex_idx = register_image_fn(fac_source.image)
            if result.get('diffuse_tex', _NO_TEX) == _NO_TEX:
                result['diffuse_tex'] = tex_idx
            result['flags'] = result.get('flags', 0) | 1 | 2  # alpha_test + transmission
            result['alpha'] = 1.0  # texture alpha will be read per-pixel
            result['alpha_ref'] = 0.5
            result['transparent_prob'] = 0.0  # NOT stochastic — texture-driven
            return result

    # Stochastic transparency (like Cycles): compute transparent_prob through the mix.
    # Mix Shader blends transparent_prob by factor, just like any other property.
    tp1 = p1.get('transparent_prob', 0.0)
    tp2 = p2.get('transparent_prob', 0.0)

    # If one side is fully transparent (direct Transparent BSDF), use the other side's
    # visual properties but set transparent_prob from the mix factor.
    s1_type = shader1_node.type if shader1_node else None
    s2_type = shader2_node.type if shader2_node else None
    if s1_type == 'BSDF_TRANSPARENT':
        result = dict(p2)
        result['transparent_prob'] = (1.0 - fac)  # fac=0 → all transparent, fac=1 → all p2
        result['flags'] = result.get('flags', 0) | 2
        return result
    if s2_type == 'BSDF_TRANSPARENT':
        result = dict(p1)
        result['transparent_prob'] = fac  # fac=0 → all p1, fac=1 → all transparent
        result['flags'] = result.get('flags', 0) | 2
        return result

    # General case: blend transparent_prob with mix factor, use non-transparent side's BRDF.
    blended_tp = tp1 * (1.0 - fac) + tp2 * fac

    if blended_tp > 0.0 and (tp1 > 0.0) != (tp2 > 0.0):
        # One side has transparency, the other doesn't — use opaque side's visual props
        if tp1 > 0.0:
            result = dict(p2)
        else:
            result = dict(p1)
        result['transparent_prob'] = blended_tp
        result['flags'] = result.get('flags', 0) | 2
        return result

    # Blend properties by mix factor
    blended = {
        'base_color': _lerp_color(p1['base_color'], p2['base_color'], fac),
        'roughness': p1['roughness'] * (1 - fac) + p2['roughness'] * fac,
        'metallic': p1['metallic'] * (1 - fac) + p2['metallic'] * fac,
        'emission': _lerp_color(p1['emission'], p2['emission'], fac),
        'emission_strength': p1['emission_strength'] * (1 - fac) + p2['emission_strength'] * fac,
        'transmission': p1.get('transmission', 0.0) * (1 - fac) + p2.get('transmission', 0.0) * fac,
        'ior': p1['ior'] * (1 - fac) + p2['ior'] * fac,
        'transparent_prob': blended_tp,
        # Texture: prefer the dominant shader's texture
        'diffuse_tex': p1['diffuse_tex'] if fac <= 0.5 else p2['diffuse_tex'],
        'emission_tex': p1['emission_tex'] if p1['emission_tex'] != _NO_TEX else p2['emission_tex'],
    }
    if blended_tp > 0.0:
        blended['flags'] = p1.get('flags', 0) | p2.get('flags', 0) | 2

    # If dominant texture is missing, fallback to the other
    if blended['diffuse_tex'] == _NO_TEX:
        blended['diffuse_tex'] = p2['diffuse_tex'] if fac <= 0.5 else p1['diffuse_tex']

    return blended


def export_materials(depsgraph, hidden_objects=None, existing_mapping=None):
    """Export Blender Principled BSDF materials as GPUMaterial byte buffer.

    Returns (ctypes byte array, name->global_index dict, textures list).
    textures list = [{"name": str, "data": bytes, "width": int, "height": int}, ...]
    hidden_objects: set of obj.names to skip (from export_meshes)
    existing_mapping: if provided, reuse this mat_name→index mapping to preserve
                      the same order as the initial load (avoids MATIDS mismatch)
    """
    import ctypes
    import bpy

    if hidden_objects is None:
        hidden_objects = set()

    # Collect unique materials from VISIBLE mesh objects only.
    # If existing_mapping is provided, reuse the same index assignment
    # so that per-BLAS material IDs remain valid.
    if existing_mapping:
        mat_name_to_index = dict(existing_mapping)
        mat_list = [None] * len(mat_name_to_index)
    else:
        mat_name_to_index = {}
        mat_list = []

    for inst in depsgraph.object_instances:
        obj = inst.object
        if obj.type != 'MESH':
            continue
        if not inst.show_self:
            continue
        if obj.name in hidden_objects:
            continue
        if obj.hide_viewport:
            continue
        if not obj.material_slots:
            # Mesh with no material slots — register a default "None" material
            if '__ignis_default__' not in mat_name_to_index:
                mat_name_to_index['__ignis_default__'] = len(mat_list)
                mat_list.append(None)
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                # Empty material slot — register default
                if '__ignis_default__' not in mat_name_to_index:
                    mat_name_to_index['__ignis_default__'] = len(mat_list)
                    mat_list.append(None)
                continue
            # Use library-qualified name for materials from linked .blend files
            mat_key = f"{mat.library.filepath}:{mat.name}" if mat.library else mat.name
            if mat_key in mat_name_to_index:
                # Update the material reference (may have changed properties)
                idx = mat_name_to_index[mat_key]
                if idx < len(mat_list):
                    mat_list[idx] = mat
                continue
            mat_name_to_index[mat_key] = len(mat_list)
            mat_list.append(mat)

    # Collect unique textures across all materials
    # image_key → {"index": int, "data": bytes, "name": str, "width": int, "height": int}
    texture_registry = {}
    textures_list = []

    def _register_image(image):
        """Register an image and return its texture index, or _NO_TEX if unavailable."""
        if image is None:
            return _NO_TEX
        key = image.name
        if key in texture_registry:
            return texture_registry[key]["index"]
        data = _get_image_bytes(image)
        if data is None:
            return _NO_TEX
        idx = len(textures_list)
        entry = {
            "index": idx,
            "name": key,
            "data": data,
            "width": image.size[0],
            "height": image.size[1],
        }
        texture_registry[key] = entry
        textures_list.append(entry)
        return idx

    # Always have at least one default material (index 0)
    if not mat_list:
        data = _pack_gpu_material()
        buf = (ctypes.c_uint8 * len(data))(*data)
        return buf, {}, []

    # Dump material order to file for debugging
    import os
    try:
        with open(os.path.join(os.path.expanduser("~"), "ignis-mat-order.txt"), "w") as _mf:
            _mf.write(f"Material buffer order ({len(mat_list)} materials):\n")
            for _mi, _mm in enumerate(mat_list):
                _mn = _mm.name if _mm else "None (default)"
                _lib = f" [{_mm.library.filepath}]" if _mm and _mm.library else ""
                _mf.write(f"  [{_mi:3d}] {_mn}{_lib}\n")
            _mf.write(f"\nmat_name_to_index:\n")
            for _mk, _mv in sorted(mat_name_to_index.items(), key=lambda x: x[1]):
                _mf.write(f"  [{_mv:3d}] {_mk}\n")
    except Exception:
        pass

    # Pack each material (with per-material timing for bottleneck detection)
    import time as _time
    all_bytes = bytearray()
    _mat_total = len(mat_list)
    for _mat_idx, mat in enumerate(mat_list):
        _mat_t0 = _time.perf_counter()
        # Default material for meshes without material assignment
        if mat is None:
            data = _pack_gpu_material(
                base_color=(0.8, 0.8, 0.8), roughness=0.5, metallic=0.0,
            )
            all_bytes.extend(data)
            continue

        base_color = (0.8, 0.8, 0.8)
        roughness = 0.5
        metallic = 0.0
        specular_level = 0.5
        emission = (0.0, 0.0, 0.0)
        emission_strength = 0.0
        normal_strength = 1.0
        diffuse_tex = _NO_TEX
        normal_tex = _NO_TEX
        orm_tex = _NO_TEX
        emission_tex = _NO_TEX
        alpha = 1.0
        ior = 1.5
        transmission = 0.0
        transparent_prob = 0.0
        uv_scale_x = 1.0
        uv_scale_y = 1.0
        color_value = 1.0
        color_saturation = 1.0
        flags = 0
        alpha_ref = 0.5

        if mat.use_nodes and mat.node_tree:
            # First try: find Principled BSDF directly
            principled_node = None
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    principled_node = node
                    break

            # Fallback: trace from Material Output to resolve any shader node
            if principled_node is None:
                surface_node = _find_surface_shader(mat.node_tree)
                # Follow Group nodes and passthrough nodes to find the actual BSDF
                _follow_depth = 0
                while surface_node is not None and _follow_depth < 8:
                    if surface_node.type in ('MIX_SHADER', 'BSDF_DIFFUSE', 'BSDF_GLOSSY',
                                              'BSDF_GLASS', 'BSDF_TRANSPARENT', 'EMISSION',
                                              'BSDF_PRINCIPLED'):
                        break  # found a shader node
                    elif surface_node.type == 'GROUP':
                        # Follow group's internal output → find what's connected
                        inner_tree = surface_node.node_tree
                        if inner_tree:
                            for inode in inner_tree.nodes:
                                if inode.type == 'GROUP_OUTPUT':
                                    inp = inode.inputs.get('Surface') or (inode.inputs[0] if inode.inputs else None)
                                    if inp and inp.is_linked:
                                        surface_node = inp.links[0].from_node
                                        break
                            else:
                                surface_node = None
                        else:
                            surface_node = None
                    elif surface_node.type in ('ADD_SHADER',):
                        # Add Shader — treat like mix at 0.5
                        surface_node.type  # just use first input
                        inp = surface_node.inputs[0] if surface_node.inputs else None
                        if inp and inp.is_linked:
                            surface_node = inp.links[0].from_node
                        else:
                            surface_node = None
                    else:
                        # Unknown node type between Output and BSDF — try first linked input
                        found = False
                        for inp in surface_node.inputs:
                            if inp.is_linked:
                                surface_node = inp.links[0].from_node
                                found = True
                                break
                        if not found:
                            surface_node = None
                    _follow_depth += 1

                if surface_node is not None and surface_node.type == 'BSDF_PRINCIPLED':
                    principled_node = surface_node  # found it through chain!
                elif surface_node is not None and surface_node.type in ('MIX_SHADER', 'BSDF_DIFFUSE', 'BSDF_GLOSSY', 'BSDF_GLASS', 'BSDF_TRANSPARENT', 'EMISSION'):
                    if surface_node.type == 'MIX_SHADER':
                        props = _resolve_mix_shader(surface_node, _register_image)
                    else:
                        props = _extract_shader_props(surface_node, _register_image)
                    base_color = props['base_color']
                    roughness = props['roughness']
                    metallic = props['metallic']
                    emission = props['emission']
                    emission_strength = props['emission_strength']
                    diffuse_tex = props['diffuse_tex']
                    emission_tex = props['emission_tex']
                    transmission = props['transmission']
                    ior = props['ior']
                    if 'alpha' in props:
                        alpha = props['alpha']
                    if 'normal_tex' in props:
                        normal_tex = props['normal_tex']
                    if 'normal_strength' in props:
                        normal_strength = props['normal_strength']
                    if 'flags' in props:
                        flags = props['flags']
                    if 'transparent_prob' in props:
                        transparent_prob = props['transparent_prob']
                    if 'alpha_ref' in props:
                        alpha_ref = props['alpha_ref']
                else:
                    # No Mix Shader — scan individual nodes
                    for node in mat.node_tree.nodes:
                        if node.type == 'EMISSION':
                            color_inp = node.inputs.get('Color')
                            if color_inp:
                                tex_node = _find_image_texture_node(color_inp)
                                if tex_node and tex_node.image:
                                    emission_tex = _register_image(tex_node.image)
                                emission = _resolve_color_input(color_inp, (1.0, 1.0, 1.0))
                            strength_inp = node.inputs.get('Strength')
                            if strength_inp:
                                emission_strength = float(strength_inp.default_value)
                            base_color = emission
                            continue
                        if node.type in ('BSDF_DIFFUSE', 'BSDF_GLOSSY', 'BSDF_GLASS'):
                            color_inp = node.inputs.get('Color')
                            if color_inp:
                                tex_node = _find_image_texture_node(color_inp)
                                if tex_node and tex_node.image:
                                    diffuse_tex = _register_image(tex_node.image)
                                c = color_inp.default_value
                                base_color = (c[0], c[1], c[2])
                            if node.type == 'BSDF_DIFFUSE':
                                # Lambertian = fully rough in PBR terms
                                roughness = 1.0
                            else:
                                rough_inp = node.inputs.get('Roughness')
                                if rough_inp:
                                    roughness = float(rough_inp.default_value)
                            if node.type == 'BSDF_GLOSSY':
                                metallic = 1.0
                            if node.type == 'BSDF_GLASS':
                                transmission = 1.0
                                ior_inp = node.inputs.get('IOR')
                                if ior_inp:
                                    ior = float(ior_inp.default_value)
                            break

            if principled_node is not None:
                node = principled_node
                # Base Color (scalar + texture)
                bc = _get_principled_input(node, 'Base Color', (0.8, 0.8, 0.8, 1.0))
                base_color = (bc[0], bc[1], bc[2])
                bc_node = _find_image_texture_node(node.inputs.get('Base Color'))
                if bc_node and bc_node.image:
                    diffuse_tex = _register_image(bc_node.image)

                # UV scale from Mapping node (applies to all textures)
                uv_scale = _find_uv_scale(node.inputs.get('Base Color'))
                uv_scale_x, uv_scale_y = uv_scale

                # Hue/Saturation/Value node in Base Color chain
                bc_inp = node.inputs.get('Base Color')
                if bc_inp and bc_inp.is_linked:
                    _hsv_node = bc_inp.links[0].from_node
                    _hsv_depth = 0
                    while _hsv_node and _hsv_depth < 8:
                        if _hsv_node.type == 'HUE_SAT':
                            val_inp = _hsv_node.inputs.get('Value')
                            sat_inp = _hsv_node.inputs.get('Saturation')
                            if val_inp:
                                color_value = float(val_inp.default_value)
                            if sat_inp:
                                color_saturation = float(sat_inp.default_value)
                            break
                        # Follow first linked input
                        _found = False
                        for _inp in _hsv_node.inputs:
                            if _inp.is_linked:
                                _hsv_node = _inp.links[0].from_node
                                _found = True
                                break
                        if not _found:
                            break
                        _hsv_depth += 1

                # Roughness
                roughness = float(_get_principled_input(node, 'Roughness', 0.5))

                # Metallic
                metallic = float(_get_principled_input(node, 'Metallic', 0.0))

                # Check for ORM-style texture on Roughness or Metallic inputs
                rough_node = _find_image_texture_node(node.inputs.get('Roughness'))
                metal_node = _find_image_texture_node(node.inputs.get('Metallic'))
                orm_image = None
                if rough_node and rough_node.image:
                    orm_image = rough_node.image
                elif metal_node and metal_node.image:
                    orm_image = metal_node.image
                if orm_image:
                    orm_tex = _register_image(orm_image)

                # Specular IOR Level (Blender 4.0+) or fallback to Specular
                spec_inp = node.inputs.get('Specular IOR Level')
                if spec_inp is None:
                    spec_inp = node.inputs.get('Specular')
                specular_level = float(spec_inp.default_value) if spec_inp else 0.5

                # Emission Color
                ec = _get_principled_input(node, 'Emission Color', (0.0, 0.0, 0.0, 1.0))
                emission = (ec[0], ec[1], ec[2])
                emission_strength = float(_get_principled_input(node, 'Emission Strength', 0.0))

                # Normal/Bump map texture
                ntex, nstr, is_bump = _extract_normal_texture(node, _register_image)
                if ntex != _NO_TEX:
                    normal_tex = ntex
                    normal_strength = -nstr if is_bump else nstr

                # Alpha
                alpha = float(_get_principled_input(node, 'Alpha', 1.0))

                # IOR
                ior = float(_get_principled_input(node, 'IOR', 1.5))

                # Transmission Weight (Blender 4.0+) or Transmission (3.x)
                trans_inp = node.inputs.get('Transmission Weight')
                if trans_inp is None:
                    trans_inp = node.inputs.get('Transmission')
                transmission = float(trans_inp.default_value) if trans_inp else 0.0

                # Emission texture
                ec_node = _find_image_texture_node(node.inputs.get('Emission Color'))
                if ec_node and ec_node.image:
                    emission_tex = _register_image(ec_node.image)

                # Detect alpha testing
                alpha_test = alpha < 1.0
                alpha_inp = node.inputs.get('Alpha')
                if alpha_inp and alpha_inp.is_linked:
                    alpha_test = True
                try:
                    if mat.blend_method in ('CLIP', 'HASHED', 'BLEND'):
                        alpha_test = True
                except AttributeError:
                    pass
                try:
                    alpha_ref = mat.alpha_threshold
                except AttributeError:
                    alpha_ref = 0.5

                flags = 0
                if alpha_test:
                    flags |= 1   # bit0 = alpha_test
                if transmission > 0.0:
                    flags |= 2   # bit1 = transmission

        # Ensure transmission flag is set for all code paths (Mix Shader, individual nodes)
        if transmission > 0.0:
            flags |= 2  # bit1 = transmission

        # Log material to file — dump full node tree for the selected object's material
        import os
        _mat_log_path = os.path.join(os.path.expanduser("~"), "ignis-rt.log")
        try:
            # Get the active object's material name for detailed logging
            _selected_mat = None
            try:
                import bpy
                _sel_obj = bpy.context.active_object
                if _sel_obj and _sel_obj.active_material:
                    _selected_mat = _sel_obj.active_material.name
            except Exception:
                pass

            with open(_mat_log_path, "a") as _mf:
                _mf.write(f"[ignis-mat] '{mat.name}': color=({base_color[0]:.3f},{base_color[1]:.3f},{base_color[2]:.3f}) "
                          f"rough={roughness:.2f} metal={metallic:.2f} trans={transmission:.2f} ior={ior:.2f} "
                          f"emit=({emission[0]:.3f},{emission[1]:.3f},{emission[2]:.3f})*{emission_strength:.2f} "
                          f"diffTex={diffuse_tex} normTex={normal_tex} normStr={normal_strength:.2f} "
                          f"emitTex={emission_tex} flags={flags} tp={transparent_prob:.2f}\n")
                # Dump full node tree only for the selected material
                if mat.use_nodes and mat.node_tree and _selected_mat and mat.name == _selected_mat:
                    _mf.write(f"  [node-tree] '{mat.name}' (selected):\n")
                    for node in mat.node_tree.nodes:
                        _mf.write(f"    node: '{node.name}' type={node.type}\n")
                        for inp in node.inputs:
                            linked = f" <- {inp.links[0].from_node.name}:{inp.links[0].from_socket.name}" if inp.is_linked else ""
                            val = ""
                            if not inp.is_linked:
                                try:
                                    v = inp.default_value
                                    if hasattr(v, '__len__'):
                                        val = f" = ({', '.join(f'{x:.3f}' for x in v)})"
                                    else:
                                        val = f" = {v:.4f}"
                                except:
                                    val = ""
                            _mf.write(f"      in: '{inp.name}'{val}{linked}\n")
                _mf.flush()
        except Exception:
            pass

        all_bytes += _pack_gpu_material(
            base_color=base_color,
            roughness=roughness,
            metallic=metallic,
            specular_level=specular_level,
            emission=emission,
            emission_strength=emission_strength,
            normal_strength=normal_strength,
            diffuse_tex=diffuse_tex,
            normal_tex=normal_tex,
            orm_tex=orm_tex,
            emission_tex=emission_tex,
            alpha=alpha,
            ior=ior,
            transmission=transmission,
            flags=flags,
            alpha_ref=alpha_ref,
            transparent_prob=transparent_prob,
            uv_scale_x=uv_scale_x,
            uv_scale_y=uv_scale_y,
            color_value=color_value,
            color_saturation=color_saturation,
        )

        _mat_dt = _time.perf_counter() - _mat_t0
        if _mat_dt > 0.1:  # Log slow materials (>100ms)
            import os
            _log_path = os.path.join(os.path.expanduser("~"), "ignis-rt.log")
            try:
                with open(_log_path, "a") as _lf:
                    _lf.write(f"[ignis-mat] SLOW: '{mat.name}' took {_mat_dt:.3f}s ({_mat_idx+1}/{_mat_total})\n")
                    _lf.flush()
            except Exception:
                pass

    buf = (ctypes.c_uint8 * len(all_bytes))(*all_bytes)
    return buf, mat_name_to_index, textures_list
