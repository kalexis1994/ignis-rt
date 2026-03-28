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
    """Convert particle hair (parents + children) to ribbon geometry.

    Fully vectorized with numpy — no Python per-strand loops for geometry generation.
    Children are generated by replicating parent strands with random offsets.

    Returns mesh dict compatible with unique_meshes format, or None if no hair.
    """
    ps = particle_system
    particles = ps.particles
    if len(particles) == 0:
        return None

    settings = ps.settings

    ps_modifier = None
    for mod in eval_obj.modifiers:
        if mod.type == 'PARTICLE_SYSTEM' and mod.particle_system == ps:
            ps_modifier = mod
            break

    # Radius settings (matches Cycles: radius = radius_scale * root_radius)
    rad_scale = getattr(settings, 'radius_scale', 0.01)
    rad_root = getattr(settings, 'root_radius', 1.0)
    rad_tip_val = getattr(settings, 'tip_radius', 0.0)
    root_radius = rad_scale * rad_root
    tip_factor = rad_tip_val / max(rad_root, 1e-6) if rad_root > 0 else 0.0
    if root_radius <= 0:
        root_radius = 0.003

    mat_idx = getattr(settings, 'material', 1) - 1
    if mat_idx < 0:
        mat_idx = 0

    # ── Extract parent strand keys ──
    # Find consistent key count from first valid particle
    n_keys = 0
    for particle in particles:
        if len(particle.hair_keys) >= 2:
            n_keys = len(particle.hair_keys)
            break
    if n_keys < 2:
        return None

    # Batch extract parent keys → (n_parents, n_keys, 3)
    parent_list = []
    for particle in particles:
        hk = particle.hair_keys
        if len(hk) != n_keys:
            continue
        keys = np.empty((n_keys, 3), dtype=np.float32)
        if ps_modifier:
            for ki in range(n_keys):
                keys[ki] = hk[ki].co_object(eval_obj, ps_modifier, particle)
        else:
            raw = np.empty(n_keys * 3, dtype=np.float32)
            hk.foreach_get("co", raw)
            keys = raw.reshape(-1, 3)
        parent_list.append(keys)

    if not parent_list:
        return None

    parents = np.array(parent_list, dtype=np.float32)  # (P, K, 3)
    n_parents = parents.shape[0]

    # ── Child generation ──
    child_type = getattr(settings, 'child_type', 'NONE')
    child_nbr = getattr(settings, 'child_nbr',
                        getattr(settings, 'child_percent', 0))

    # Child distribution handled by GPU compute shader (emitter surface + frand table).
    # co_hair() is in an undocumented internal path-cache space that doesn't have
    # a clean transform to co_object space — GPU generation gives correct shapes
    # (clump/kink/roughness match Cycles) with slightly different child placement.
    use_precomputed_children = False
        child_nbr = 0  # GPU does NO child generation — all strands are "parents"

    # ── Extract emitter mesh for GPU child distribution ──
    emitter_verts = np.empty(0, dtype=np.float32)
    emitter_tris = np.empty(0, dtype=np.uint32)
    emitter_cdf = np.empty(0, dtype=np.float32)
    try:
        emesh = eval_obj.to_mesh()
        if emesh and len(emesh.vertices) > 0:
            ev = np.empty(len(emesh.vertices) * 3, dtype=np.float32)
            emesh.vertices.foreach_get('co', ev)
            emitter_verts = ev
            if len(emesh.loop_triangles) == 0:
                emesh.calc_loop_triangles()
            if len(emesh.loop_triangles) > 0:
                et = np.empty(len(emesh.loop_triangles) * 3, dtype=np.int32)
                emesh.loop_triangles.foreach_get('vertices', et)
                emitter_tris = et.astype(np.uint32)
                # Area-weighted CDF for random face selection
                verts_3d = ev.reshape(-1, 3)
                tris_3d = emitter_tris.reshape(-1, 3)
                v0 = verts_3d[tris_3d[:, 0]]
                v1 = verts_3d[tris_3d[:, 1]]
                v2 = verts_3d[tris_3d[:, 2]]
                areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
                total_area = areas.sum()
                if total_area > 0:
                    emitter_cdf = (np.cumsum(areas) / total_area).astype(np.float32)
        eval_obj.to_mesh_clear()
    except Exception:
        pass

    # Compute average parent spacing for GPU noise scaling
    parent_roots = parents[:, 0, :]  # (P, 3)
    avg_spacing = 0.01
    if n_parents > 10:
        rng_s = np.random.RandomState(99)
        si = rng_s.choice(n_parents, min(100, n_parents), replace=False)
        dists = np.linalg.norm(parent_roots[si][:, None, :] - parent_roots[None, :10, :], axis=2)
        np.fill_diagonal(dists[:10, :10], 1e10)
        avg_spacing = float(np.median(dists.min(axis=1)))

    return {
        "name": f"hair_{eval_obj.name}",
        "gpu_hair": True,
        "precomputed_children": use_precomputed_children,
        "parent_keys": np.ascontiguousarray(parents.reshape(-1, 3).flatten(), dtype=np.float32),
        "n_parents": n_parents,
        "n_keys": n_keys,
        "child_nbr": child_nbr if (child_type != 'NONE' and not use_precomputed_children) else 0,
        "root_radius": root_radius,
        "tip_factor": tip_factor,
        "mat_idx": mat_idx,
        "emitter_verts": np.ascontiguousarray(emitter_verts, dtype=np.float32),
        "emitter_tris": np.ascontiguousarray(emitter_tris, dtype=np.uint32),
        "emitter_cdf": np.ascontiguousarray(emitter_cdf, dtype=np.float32),
        "n_emitter_verts": len(emitter_verts) // 3,
        "n_emitter_tris": len(emitter_tris) // 3,
        "avg_spacing": avg_spacing,
        # When using precomputed children, modifiers are already applied by Blender → zero them
        "kink_amplitude": 0.0 if use_precomputed_children else getattr(settings, 'kink_amplitude', 0.0),
        "kink_frequency": getattr(settings, 'kink_frequency', 2.0),
        "clump_factor": 0.0 if use_precomputed_children else getattr(settings, 'clump_factor', 0.0),
        "clump_shape": getattr(settings, 'clump_shape', 0.0),
        "roughness_1": 0.0 if use_precomputed_children else getattr(settings, 'roughness_1', 0.0),
        "roughness_1_size": getattr(settings, 'roughness_1_size', 1.0),
        "roughness_2": 0.0 if use_precomputed_children else getattr(settings, 'roughness_2', 0.0),
        "roughness_2_size": getattr(settings, 'roughness_2_size', 1.0),
        "roughness_endpoint": 0.0 if use_precomputed_children else getattr(settings, 'roughness_endpoint', 0.0),
        "child_mode": 1 if child_type == 'SIMPLE' else 0,
        "kink_shape": getattr(settings, 'kink_shape', 0.0),
        "kink_flat": getattr(settings, 'kink_flat', 0.0),
        "kink_amp_random": 0.0 if use_precomputed_children else getattr(settings, 'kink_amp_random', 0.0),
        "opaque_hair": True,
        "child_length": getattr(settings, 'child_length', 1.0),
        "clump_noise_size": getattr(settings, 'clump_noise_size', 1.0),
        "child_roundness": getattr(settings, 'child_roundness', 0.0),
        "child_size_random": getattr(settings, 'child_size_random', 0.0),
        # When precomputed, ALL strands are "parents" → force useParentParticles
        "use_parent_particles": True if use_precomputed_children else getattr(settings, 'use_parent_particles', False),
        "blender_seed": getattr(settings, 'seed', 0),
    }

    # CPU fallback (kept for reference, not reached)
    if child_type != 'NONE' and child_nbr > 0:
        # Limit children for performance (GPU pipeline will handle full count)
        max_children = min(child_nbr, 5)  # cap at 5 children/parent for CPU
        n_children = n_parents * max_children
        rng = np.random.RandomState(42)

        # Estimate average parent spacing from root positions
        parent_roots = parents[:, 0, :]  # (P, 3)
        if n_parents > 10:
            # Sample 200 random parents, find nearest neighbor distance
            sample_n = min(200, n_parents)
            sample_idx = rng.choice(n_parents, sample_n, replace=False)
            sample_roots = parent_roots[sample_idx]
            # Brute-force nearest neighbor on sample
            diffs = sample_roots[:, None, :] - parent_roots[None, :, :]  # (S, P, 3)
            dists_sq = (diffs * diffs).sum(axis=2)  # (S, P)
            dists_sq[np.arange(sample_n), sample_idx] = 1e30  # exclude self
            avg_spacing = float(np.median(np.sqrt(dists_sq.min(axis=1))))
        else:
            avg_spacing = 0.01

        # For each child: pick a primary parent, offset root, then
        # interpolate with 2 random nearby parents for shape variation
        child_primary = np.arange(n_children, dtype=np.int32) % n_parents

        # Pick 2 additional random parents (close indices = likely nearby on surface)
        spread = max(int(n_parents * 0.02), 3)
        rand_offset1 = rng.randint(-spread, spread + 1, size=n_children)
        rand_offset2 = rng.randint(-spread, spread + 1, size=n_children)
        idx1 = np.clip(child_primary + rand_offset1, 0, n_parents - 1)
        idx2 = np.clip(child_primary + rand_offset2, 0, n_parents - 1)

        # Random barycentric weights
        w = rng.uniform(0.1, 1.0, size=(n_children, 3)).astype(np.float32)
        w /= w.sum(axis=1, keepdims=True)

        # Interpolate keys from 3 parents
        p0 = parents[child_primary]  # (C, K, 3)
        p1 = parents[idx1]
        p2 = parents[idx2]
        children = (p0 * w[:, 0:1, None] +
                    p1 * w[:, 1:2, None] +
                    p2 * w[:, 2:3, None])  # (C, K, 3)

        # Per-key noise increasing toward tip
        t_noise = np.linspace(0, 1, n_keys).reshape(1, n_keys, 1)
        key_noise = rng.normal(0, avg_spacing * 0.05, size=(n_children, n_keys, 3)).astype(np.float32)
        children += key_noise * (0.5 + t_noise * 1.5)

        all_keys = np.concatenate([parents, children], axis=0)
    else:
        all_keys = parents

    S = all_keys.shape[0]  # total strands

    import os
    try:
        with open(os.path.join(os.path.expanduser("~"), "ignis-rt.log"), "a") as _lf:
            _lf.write(f"[ignis-hair] '{eval_obj.name}': {n_parents} parents + "
                      f"{S - n_parents} children = {S} strands, "
                      f"root_r={root_radius:.5f} tip_f={tip_factor:.2f}\n")
    except Exception:
        pass

    # ── Vectorized ribbon geometry for ALL strands at once ──
    K = n_keys

    # Tangents: (S, K, 3) via central differences
    tangents = np.empty_like(all_keys)
    tangents[:, 0] = all_keys[:, 1] - all_keys[:, 0]
    tangents[:, -1] = all_keys[:, -1] - all_keys[:, -2]
    tangents[:, 1:-1] = all_keys[:, 2:] - all_keys[:, :-2]
    t_len = np.linalg.norm(tangents, axis=2, keepdims=True)
    t_len = np.maximum(t_len, 1e-8)
    tangents /= t_len

    # Perpendicular direction: random angle per strand for natural variation
    # (Cycles ribbon mode: each strand faces camera; we approximate with random orientation)
    rng_perp = np.random.RandomState(123)
    angles = rng_perp.uniform(0, np.pi, size=(S, 1, 1)).astype(np.float32)

    # Base perp: cross(tangent, up)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    perp_a = np.cross(tangents, up)  # (S, K, 3)
    p_len = np.linalg.norm(perp_a, axis=2, keepdims=True)
    degenerate = (p_len.squeeze(2) < 0.1)
    alt_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    perp_a[degenerate] = np.cross(tangents, alt_ref)[degenerate]
    p_len = np.linalg.norm(perp_a, axis=2, keepdims=True)
    perp_a /= np.maximum(p_len, 1e-8)

    # Second perp: cross(tangent, perp_a)
    perp_b = np.cross(tangents, perp_a)
    pb_len = np.linalg.norm(perp_b, axis=2, keepdims=True)
    perp_b /= np.maximum(pb_len, 1e-8)

    # Rotate perpendicular by random angle per strand (single ribbon, varied orientation)
    perp = perp_a * np.cos(angles) + perp_b * np.sin(angles)  # (S, K, 3)

    # Radius per key: linear taper root→tip
    t_param = np.linspace(0, 1, K, dtype=np.float32)
    radii = root_radius * (1.0 - t_param + t_param * tip_factor)
    radii = radii.reshape(1, K, 1)

    # Normal: perpendicular to ribbon face
    ribbon_norm = np.cross(perp, tangents)  # (S, K, 3)
    rn_len = np.linalg.norm(ribbon_norm, axis=2, keepdims=True)
    ribbon_norm /= np.maximum(rn_len, 1e-8)

    # Vertices: keys ± perp * radius → 1 ribbon per strand, 2 verts per key
    v_left = all_keys - perp * radii   # (S, K, 3)
    v_right = all_keys + perp * radii

    # Interleave: (S, K*2, 3)
    all_verts = np.empty((S, K * 2, 3), dtype=np.float32)
    all_verts[:, 0::2] = v_left
    all_verts[:, 1::2] = v_right

    all_norms = np.empty((S, K * 2, 3), dtype=np.float32)
    all_norms[:, 0::2] = ribbon_norm
    all_norms[:, 1::2] = ribbon_norm

    # UVs: u=width, v=intercept along strand
    r_uvs = np.empty((K * 2, 2), dtype=np.float32)
    r_uvs[0::2, 0] = 0.0
    r_uvs[1::2, 0] = 1.0
    r_uvs[0::2, 1] = t_param
    r_uvs[1::2, 1] = t_param
    all_uvs_arr = np.tile(r_uvs, (S, 1))  # (S*K*2, 2)

    total_ribbons = S
    verts_per_ribbon = K * 2
    total_verts = total_ribbons * verts_per_ribbon

    positions = all_verts.reshape(-1, 3)
    normals = all_norms.reshape(-1, 3)

    # Indices: triangle strip per ribbon
    # For each ribbon of K*2 verts, (K-1) segments × 2 tris × 3 indices
    segs = K - 1
    tris_per_ribbon = segs * 2
    idx_per_ribbon = tris_per_ribbon * 3
    ribbon_indices = np.empty(idx_per_ribbon, dtype=np.uint32)
    for si in range(segs):
        i0 = si * 2
        base = si * 6
        ribbon_indices[base:base + 6] = [i0, i0 + 2, i0 + 1, i0 + 1, i0 + 2, i0 + 3]
    # Tile for all ribbons with offset
    offsets = np.arange(total_ribbons, dtype=np.uint32) * verts_per_ribbon
    all_indices = (ribbon_indices.reshape(1, -1) + offsets.reshape(-1, 1)).flatten()
    mat_indices = np.full(total_ribbons * tris_per_ribbon, mat_idx, dtype=np.int32)

    return {
        "name": f"hair_{eval_obj.name}",
        "positions": np.ascontiguousarray(positions.flatten(), dtype=np.float32),
        "normals": np.ascontiguousarray(normals.flatten(), dtype=np.float32),
        "uvs": np.ascontiguousarray(all_uvs_arr.flatten(), dtype=np.float32),
        "indices": np.ascontiguousarray(all_indices, dtype=np.uint32),
        "vertex_count": total_verts,
        "index_count": len(all_indices),
        "tri_count": len(mat_indices),
        "raw_vert_count": total_verts,
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

        # Skip objects in hidden/excluded collections — but NOT instances.
        # Collection instances (e.g., column.xxx) have their source collection
        # excluded to hide the original; the instances themselves are visible.
        if obj.name in hidden_by_collection and not instance.is_instance:
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

            # Vertex Colors
            vcols = np.ones((raw_vert_count, 4), dtype=np.float32)  # default white RGBA
            _has_vcols = False
            if mesh.color_attributes and len(mesh.color_attributes) > 0:
                try:
                    color_attr = mesh.color_attributes.active_color
                    if color_attr is None:
                        color_attr = mesh.color_attributes[0]
                    if color_attr.domain == 'CORNER':
                        all_colors = np.empty(len(color_attr.data) * 4, dtype=np.float32)
                        color_attr.data.foreach_get("color", all_colors)
                        all_colors = all_colors.reshape(-1, 4)
                        vcols = all_colors[tri_loops]
                    elif color_attr.domain == 'POINT':
                        all_colors = np.empty(len(color_attr.data) * 4, dtype=np.float32)
                        color_attr.data.foreach_get("color", all_colors)
                        all_colors = all_colors.reshape(-1, 4)
                        # Map point colors to triangle corners via vertex indices
                        tri_verts = mesh.loops  # loop -> vertex mapping
                        loop_to_vert = np.empty(len(tri_verts), dtype=np.int32)
                        tri_verts.foreach_get("vertex_index", loop_to_vert)
                        vcols = all_colors[loop_to_vert[tri_loops]]
                    _has_vcols = True
                    import os as _os_vc2
                    try:
                        avg = vcols.mean(axis=0)
                        with open(_os_vc2.path.join(_os_vc2.path.expanduser("~"), "ignis-rt.log"), "a") as _f:
                            _f.write(f"[ignis-vcol] '{obj.name}' mesh '{mesh.name}': HAS vertex colors — avg=({avg[0]:.3f},{avg[1]:.3f},{avg[2]:.3f},{avg[3]:.3f})\n")
                    except: pass
                except Exception:
                    pass  # Keep default white
            if not _has_vcols:
                import os as _os_vc
                try:
                    with open(_os_vc.path.join(_os_vc.path.expanduser("~"), "ignis-rt.log"), "a") as _f:
                        _f.write(f"[ignis-vcol] '{obj.name}' mesh '{mesh.name}': NO vertex colors — using white default\n")
                except: pass

            # ---- Vertex deduplication ----
            # Skip dedup for large meshes (>50K tris) — np.unique is O(N log N)
            # and takes seconds on multi-million triangle meshes
            DEDUP_THRESHOLD = 500000
            if tri_count <= DEDUP_THRESHOLD:
                combined = np.ascontiguousarray(
                    np.hstack([positions, normals, uvs, vcols]), dtype=np.float32)  # pos(3)+norm(3)+uv(2)+color(4)=12
                void_dt = np.dtype((np.void, combined.dtype.itemsize * combined.shape[1]))
                _, unique_idx, inverse = np.unique(
                    combined.view(void_dt).ravel(),
                    return_index=True, return_inverse=True)
                dedup_pos = combined[unique_idx, :3]
                dedup_nrm = combined[unique_idx, 3:6]
                dedup_uvs = combined[unique_idx, 6:8]
                dedup_vcols = combined[unique_idx, 8:12]
                dedup_indices = inverse.astype(np.uint32)
                dedup_vert_count = len(unique_idx)
            else:
                # Large mesh: use raw unrolled vertices (no dedup, more VRAM but instant)
                dedup_pos = positions
                dedup_nrm = normals
                dedup_uvs = uvs
                dedup_vcols = vcols
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
                "vcols": np.ascontiguousarray(dedup_vcols.flatten(), dtype=np.float32),
                "indices": np.ascontiguousarray(dedup_indices, dtype=np.uint32),
                "vertex_count": dedup_vert_count,
                "index_count": len(dedup_indices),
                "tri_count": tri_count,
                "raw_vert_count": raw_vert_count,
                "tri_material_indices": tri_mat_indices,
            }
            eval_obj.to_mesh_clear()

        xform = _matrix_to_3x4_row_major(instance.matrix_world)
        # Detect collection instance parent for hierarchy caching
        parent_name = None
        parent_matrix_bl = None
        if instance.is_instance and instance.parent:
            parent_name = instance.parent.name
            parent_matrix_bl = np.array(instance.parent.matrix_world, dtype=np.float32)
        instances.append({
            "mesh_key": mesh_key,
            "transform_3x4": xform,
            "material_slots": [s.material for s in obj.material_slots],
            "is_instance": instance.is_instance,
            "display_type": obj.display_type,
            "hide_render": obj.hide_render,
            "hide_viewport": obj.hide_viewport,
            "visible_camera": getattr(obj, 'visible_camera', '?'),
            "parent_name": parent_name,
            "parent_matrix_bl": parent_matrix_bl,
            "child_matrix_bl": np.array(instance.matrix_world, dtype=np.float32) if parent_name else None,
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


def _extract_sun_from_hdri(rgb, w, h):
    """Detect the sun disc in an HDRI and extract direction + intensity.

    Real-time renderers can't importance-sample the tiny bright sun disc
    efficiently, so we extract it as a separate directional light for NEE.

    Args:
        rgb: numpy array (N, 3) of linear HDR pixel values (Blender bottom-up).
        w, h: image dimensions.

    Returns:
        dict with sun_elevation, sun_azimuth, sun_intensity, sun_color
        or None if no bright sun found.
    """
    luminance = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    avg_lum = np.mean(luminance)
    if avg_lum < 1e-6:
        return None

    # Sun threshold: pixels > 500x average luminance are considered sun
    sun_threshold = avg_lum * 500.0
    sun_mask = luminance > sun_threshold
    n_sun = np.sum(sun_mask)
    if n_sun < 4:
        return None  # No significant sun disc found

    # Compute pixel solid angles for equirectangular projection
    # Each row has latitude: theta = (row / h - 0.5) * PI  (Blender: row 0 = bottom)
    row_indices = np.arange(w * h) // w  # row index for each pixel
    theta = (row_indices.astype(np.float64) / h - 0.5) * math.pi  # latitude
    cos_theta = np.cos(theta)
    pixel_solid_angle = (2.0 * math.pi / w) * (math.pi / h) * cos_theta

    # Sun total power (irradiance at perpendicular surface)
    sun_power = np.sum(luminance[sun_mask] * pixel_solid_angle[sun_mask])

    # Sun weighted centroid (luminance-weighted average UV)
    sun_indices = np.where(sun_mask)[0]
    sun_rows = sun_indices // w  # y in Blender coords (0 = bottom)
    sun_cols = sun_indices % w
    sun_weights = luminance[sun_mask]
    total_weight = np.sum(sun_weights)

    avg_col = np.sum(sun_cols * sun_weights) / total_weight
    avg_row = np.sum(sun_rows * sun_weights) / total_weight

    # UV from centroid (Blender bottom-up: row 0 = bottom → v=0)
    u = avg_col / w
    v = avg_row / h

    # Equirectangular UV → 3D direction (Vulkan Y-up, matching shader)
    # Shader: phi = atan(dir.z, dir.x), theta = asin(dir.y)
    #         uv.x = 0.5 + phi/(2*PI), uv.y = 0.5 + theta/PI
    # Inverse:
    phi = (u - 0.5) * 2.0 * math.pi
    lat = (v - 0.5) * math.pi
    dir_x = math.cos(lat) * math.cos(phi)
    dir_y = math.sin(lat)
    dir_z = math.cos(lat) * math.sin(phi)

    # Elevation = angle from horizon (asin of Y component)
    elevation_deg = math.degrees(math.asin(max(-1.0, min(1.0, dir_y))))
    # Azimuth = angle around Y axis from +Z toward +X
    azimuth_deg = math.degrees(math.atan2(dir_x, dir_z))

    # Sun color: luminance-weighted average, normalized to max component = 1
    sun_rgb = np.sum(rgb[sun_mask] * sun_weights[:, np.newaxis], axis=0) / total_weight
    max_comp = max(sun_rgb[0], sun_rgb[1], sun_rgb[2], 1e-6)
    sun_color = (float(sun_rgb[0] / max_comp),
                 float(sun_rgb[1] / max_comp),
                 float(sun_rgb[2] / max_comp))

    # Intensity: match Blender SUN convention (energy in W/m²)
    # export_sun multiplies by PI, so we provide the raw irradiance here
    sun_intensity = float(sun_power)

    print(f"[ignis_rt] HDRI sun extraction: dir=({dir_x:.3f},{dir_y:.3f},{dir_z:.3f}) "
          f"elev={elevation_deg:.1f}° azim={azimuth_deg:.1f}° "
          f"intensity={sun_intensity:.1f} color=({sun_color[0]:.3f},{sun_color[1]:.3f},{sun_color[2]:.3f}) "
          f"sun_pixels={n_sun}")

    return {
        "sun_elevation": elevation_deg,
        "sun_azimuth": azimuth_deg,
        "sun_intensity": sun_intensity,
        "sun_color": sun_color,
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
        # No HDRI texture — but return the Background color + strength
        # so the shader can use it as ambient environment instead of hardcoded gray
        if bg_node:
            color_inp = bg_node.inputs.get('Color')
            if color_inp and not color_inp.is_linked:
                c = color_inp.default_value
                return {
                    "bg_color": (float(c[0]), float(c[1]), float(c[2])),
                    "bg_strength": strength,
                }
        return None

    image = env_tex_node.image
    w, h = image.size[0], image.size[1]
    if w == 0 or h == 0:
        return None

    # Extract float pixels from Blender
    px = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    px_rgba = px.reshape(-1, 4)
    rgb = px_rgba[:, :3].copy()

    # ---- Extract sun from HDRI for directional light NEE ----
    extracted_sun = _extract_sun_from_hdri(rgb, w, h)

    # Upload as float16 RGBA — preserves HDR range without 8-bit clipping.
    # Clamp to prevent fireflies from extreme sun values (67000+) at low SPP.
    # Max ~500 preserves sky/clouds/bright surfaces while the extracted sun NEE
    # handles the actual solar disc separately.
    HDRI_MAX_RADIANCE = 500.0
    px_rgba_clamped = px_rgba.copy()
    px_rgba_clamped[:, :3] = np.minimum(px_rgba_clamped[:, :3], HDRI_MAX_RADIANCE)
    # Flip vertically: Blender stores bottom-up, Vulkan textures are top-down
    px_rgba_flipped = px_rgba_clamped.reshape(h, w, 4)[::-1].copy()
    px_f16 = px_rgba_flipped.astype(np.float16)
    data = px_f16.tobytes()

    return {
        "name": f"__world_hdri__{image.name}",
        "data": data,
        "width": image.size[0],
        "height": image.size[1],
        "strength": strength,
        "dxgi_format": 10,  # DXGI_FORMAT_R16G16B16A16_FLOAT — native HDR
        "extracted_sun": extracted_sun,
    }


def export_sun(depsgraph):
    """Extract sun direction/intensity from the scene.

    Priority:
    1. Blender SUN light (explicit light object)
    2. World Sky Texture (Nishita/Hosek) — uses Cycles' exact sun direction formula
    3. Defaults (no sun)

    Returns a dict with sun_elevation, sun_azimuth (degrees), sun_intensity, sun_color.
    """
    # --- 1. Try explicit SUN light ---
    for obj in depsgraph.objects:
        if obj.type != 'LIGHT':
            continue
        light = obj.data
        if light.type != 'SUN':
            continue

        mat = obj.matrix_world
        direction = mat.col[2].xyz.normalized()

        # Convert to Vulkan Y-up: Blender Z → Vulkan Y, Blender -Y → Vulkan Z
        dx = direction.x
        dy = direction.z
        dz = -direction.y

        elevation = math.degrees(math.asin(max(-1.0, min(1.0, dy))))
        azimuth = math.degrees(math.atan2(dx, dz))
        if azimuth < 0:
            azimuth += 360.0

        # SUN light: use Blender's angle property as sun_size
        sun_angle = getattr(light, 'angle', 0.009512)  # SUN light angular diameter
        return {
            "sun_elevation": elevation,
            "sun_azimuth": azimuth,
            "sun_intensity": light.energy * math.pi,
            "sun_color": (light.color[0], light.color[1], light.color[2]),
            "sun_size": sun_angle,
            "sun_disc_intensity": 1.0,
            "air_density": 1.0,
            "dust_density": 1.0,
            "ozone_density": 1.0,
            "altitude": 0.0,
        }

    # --- 2. Try World Sky Texture (Nishita/Hosek) ---
    scene = depsgraph.scene
    world = scene.world
    if world and world.use_nodes and world.node_tree:
        sky_node = None
        bg_strength = 1.0
        for node in world.node_tree.nodes:
            if node.type == 'TEX_SKY':
                sky_node = node
            elif node.type == 'BACKGROUND':
                s_inp = node.inputs.get('Strength')
                if s_inp:
                    bg_strength = float(s_inp.default_value)

        if sky_node:
            # Cycles formula (from kernel/svm/sky.h + scene/shader_nodes.cpp):
            # sun_dir = (-cos(e)*sin(r), cos(e)*cos(r), sin(e))  [Blender Z-up]
            e = sky_node.sun_elevation  # radians
            r = sky_node.sun_rotation   # radians

            # Blender Z-up direction
            bx = -math.cos(e) * math.sin(r)
            by = math.cos(e) * math.cos(r)
            bz = math.sin(e)

            # Convert to Vulkan Y-up
            dx = bx
            dy = bz        # Blender Z → Vulkan Y
            dz = -by       # Blender -Y → Vulkan Z

            elevation = math.degrees(math.asin(max(-1.0, min(1.0, dy))))
            azimuth = math.degrees(math.atan2(dx, dz))
            if azimuth < 0:
                azimuth += 360.0

            # Read all Sky Texture node properties
            sky_sun_size = getattr(sky_node, 'sun_size', 0.009512)       # angular diameter (rad)
            sky_sun_intensity = getattr(sky_node, 'sun_intensity', 1.0)  # disc brightness multiplier
            sky_air_density = getattr(sky_node, 'air_density', 1.0)
            sky_dust_density = getattr(sky_node, 'dust_density', 1.0)
            sky_ozone_density = getattr(sky_node, 'ozone_density', 1.0)
            sky_altitude = getattr(sky_node, 'altitude', 0.0)           # meters

            # Sun radiance: base × node intensity × background strength × PI (BRDF parity)
            sun_intensity = 5.0 * sky_sun_intensity * bg_strength * math.pi

            return {
                "sun_elevation": elevation,
                "sun_azimuth": azimuth,
                "sun_intensity": sun_intensity,
                "sun_color": (1.0, 0.98, 0.95),  # warm white for Nishita sun
                "from_sky_texture": True,
                "sun_size": sky_sun_size,
                "sun_disc_intensity": sky_sun_intensity,
                "air_density": sky_air_density,
                "dust_density": sky_dust_density,
                "ozone_density": sky_ozone_density,
                "altitude": sky_altitude,
            }

    # --- 3. Defaults (no sun) ---
    return {
        "sun_elevation": 45.0,
        "sun_azimuth": 180.0,
        "sun_intensity": 0.0,
        "sun_color": (1.0, 1.0, 1.0),
        "sun_size": 0.009512,
        "sun_disc_intensity": 1.0,
        "air_density": 1.0,
        "dust_density": 1.0,
        "ozone_density": 1.0,
        "altitude": 0.0,
    }


def _blackbody_to_rgb(temperature_k):
    """Convert Blackbody color temperature (Kelvin) to linear Rec.709 RGB.

    Polynomial coefficients from Blender/Cycles (Apache 2.0 license).
    Source: intern/cycles/kernel/tables.h + svm/math_util.h
    Copyright: 2011-2022 Blender Foundation
    Formula: R,G = a/t + b*t + c;  B = ((a*t + b)*t + c)*t + d
    """
    t = float(temperature_k)

    if t >= 12000.0:
        return (0.8263, 0.9945, 1.0)  # clamped blue
    if t < 800.0:
        return (1.0, 0.0, 0.0)  # very dim red

    # Piecewise table indices (same breakpoints as Cycles)
    _r_table = [
        (1.61919106e+03, -2.05010916e-03, 5.02995757e+00),
        (2.48845471e+03, -1.11330907e-03, 3.22621544e+00),
        (3.34143193e+03, -4.86551192e-04, 1.76486769e+00),
        (4.09461742e+03, -1.27446582e-04, 7.25731635e-01),
        (4.67028036e+03,  2.91258199e-05, 1.26703442e-01),
        (4.59509185e+03,  2.87495649e-05, 1.50345020e-01),
        (3.78717450e+03,  9.35907826e-06, 3.99075871e-01),
    ]
    _g_table = [
        (-4.88999748e+02,  6.04330754e-04, -7.55807526e-02),
        (-7.55994277e+02,  3.16730098e-04,  4.78306139e-01),
        (-1.02363977e+03,  1.20223470e-04,  9.36662319e-01),
        (-1.26571316e+03,  4.87340896e-06,  1.27054498e+00),
        (-1.42529332e+03, -4.01150431e-05,  1.43972784e+00),
        (-1.17554822e+03, -2.16378048e-05,  1.30408023e+00),
        (-5.00799571e+02, -4.59832026e-06,  1.09098763e+00),
    ]
    _b_table = [
        ( 5.96945309e-11, -4.85742887e-08, -9.70622247e-05, -4.07936148e-03),
        ( 2.40430366e-11,  5.55021075e-08, -1.98503712e-04,  2.89312858e-02),
        (-1.40949732e-11,  1.89878968e-07, -3.56632824e-04,  9.10767778e-02),
        (-3.61460868e-11,  2.84822009e-07, -4.93211319e-04,  1.56723440e-01),
        (-1.97075738e-11,  1.75359352e-07, -2.50542825e-04, -2.22783266e-02),
        (-1.61603419e-11,  1.09815345e-07, -1.41329376e-04, -4.42452373e-02),
        (-1.01587682e-11,  6.20469931e-08, -6.59693818e-05, -9.14294750e-03),
    ]
    _breaks = [965.0, 1167.0, 1449.0, 1902.0, 3315.0, 6365.0]

    if t >= _breaks[5]:
        i = 6
    elif t >= _breaks[4]:
        i = 5
    elif t >= _breaks[3]:
        i = 4
    elif t >= _breaks[2]:
        i = 3
    elif t >= _breaks[1]:
        i = 2
    elif t >= _breaks[0]:
        i = 1
    else:
        i = 0

    t_inv = 1.0 / t
    r = _r_table[i]
    g = _g_table[i]
    b = _b_table[i]

    rv = r[0] * t_inv + r[1] * t + r[2]
    gv = g[0] * t_inv + g[1] * t + g[2]
    bv = ((b[0] * t + b[1]) * t + b[2]) * t + b[3]

    return (max(0.0, rv), max(0.0, gv), max(0.0, bv))


def export_lights(depsgraph):
    """Export point/spot/area lights (max 32) for NEE direct sampling.

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
        if len(lights) >= 512:  # 32 lights × 16 floats each
            break

        # World position (Blender Z-up -> Vulkan Y-up)
        pos = obj.matrix_world.translation
        vk_x = pos.x
        vk_y = pos.z        # Blender Z -> Vulkan Y
        vk_z = -pos.y       # Blender -Y -> Vulkan Z

        energy = light.energy
        color = light.color

        # Check for Blackbody node in light's node tree (overrides light.color)
        if hasattr(light, 'use_nodes') and light.use_nodes and light.node_tree:
            for node in light.node_tree.nodes:
                if node.type == 'BLACKBODY':
                    temp_k = node.inputs['Temperature'].default_value
                    color = _blackbody_to_rgb(temp_k)
                    break
        estimated_range = max(math.sqrt(energy / 0.01), 1.0) if energy > 0 else 10.0
        estimated_range = min(estimated_range, 100.0)

        # Point/spot lights emit into the full sphere (4π steradians).
        # Radiant intensity = power / (4π). Matches Cycles: strength / (4*PI) / r^2.
        intensity = energy / (4.0 * math.pi)

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
            # Cycles multiplies light dimensions by object scale (from matrix_world).
            # Extract scale from the transform's basis vectors, like Cycles does:
            # extentu = transform_column(tfm, 0) * sizeu → len_u = length(extentu)
            scale_x = mat_w.col[0].xyz.length
            scale_y = mat_w.col[1].xyz.length
            base_size_x = light.size if hasattr(light, 'size') else 1.0
            base_size_y = light.size_y if light.shape in ('RECTANGLE', 'ELLIPSE') else base_size_x
            size_x = base_size_x * scale_x
            size_y = base_size_y * scale_y
            # Negative range signals area light to shader
            export_range = -max(size_x, size_y) * 0.5
            # Cycles area light: energy is total power (Watts).
            # Radiance = power / (PI * area).  Shader multiplies by areaSize
            # in NEE, so we pass radiance = power / (PI * area).
            area = size_x * size_y
            intensity = energy / (math.pi * area) if area > 0 else energy
        elif light.type == 'SPOT':
            # Spot light: pack cone parameters into size_x (cos_half_angle)
            # and size_y (spot_smooth factor)
            spot_angle = light.spot_size  # full cone angle in radians
            spot_blend = light.spot_blend  # 0 = hard, 1 = full smooth
            cos_half = math.cos(spot_angle * 0.5)
            # Cycles formula: spot_smooth = 1.0 / ((1 - cos_half) * blend)
            if spot_blend > 0.001:
                spot_smooth = 1.0 / ((1.0 - cos_half) * spot_blend)
            else:
                spot_smooth = 1e6  # hard edge
            size_x = cos_half       # packed in size_x
            size_y = -spot_smooth   # NEGATIVE sizeY signals spot light (vs area)
            export_range = -estimated_range
        else:
            export_range = estimated_range

        # Light export logging removed (was too verbose for scenes with many lights)

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

        # Check if any slot has REAL emission (strength > 0 AND non-black color)
        if not any(s[1] > 0.0 and sum(s[0]) > 0.001 for s in slot_emission):
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
            if em_strength <= 0.0 or sum(em_color) < 0.001:
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


# ============================================================================
# Node VM Compiler — translates Blender node trees to bytecode
# ============================================================================

import struct as _struct

def _floatBits(f):
    """Convert float to uint32 (matching GLSL floatBitsToUint)."""
    return _struct.unpack('<I', _struct.pack('<f', float(f)))[0]

# Opcode constants (must match node_vm.glsl)
_OP_NOP            = 0x00
_OP_SAMPLE_TEX     = 0x01
_OP_UV_TRANSFORM   = 0x10
_OP_UV_ROTATE      = 0x11
_OP_MIX            = 0x20
_OP_MIX_REG        = 0x21
_OP_MULTIPLY       = 0x22
_OP_ADD            = 0x23
_OP_SUBTRACT       = 0x24
_OP_SCREEN         = 0x25
_OP_OVERLAY        = 0x26
_OP_INVERT         = 0x28
_OP_GAMMA          = 0x29
_OP_BRIGHT_CONTRAST= 0x2A
_OP_HUE_SAT_VAL    = 0x2B
_OP_COLORRAMP      = 0x30
_OP_RAMP_DATA      = 0x31
_OP_LUMINANCE      = 0x40
_OP_MATH_MUL       = 0x42
_OP_MATH_CLAMP     = 0x47
_OP_MATH_SUB       = 0x49
_OP_MATH_ABS       = 0x4A
_OP_MATH_SQRT      = 0x4B
_OP_MATH_MOD       = 0x4C
_OP_MATH_FLOOR     = 0x4D
_OP_MATH_CEIL      = 0x4E
_OP_MATH_FRACT     = 0x4F
_OP_MATH_SIN       = 0x70
_OP_MATH_COS       = 0x71
_OP_MATH_TAN       = 0x72
_OP_MATH_LESS      = 0x73
_OP_MATH_GREATER   = 0x74
_OP_MATH_ROUND     = 0x75
_OP_MATH_SIGN      = 0x76
_OP_MATH_SMOOTH_MIN = 0x77
_OP_SEPARATE_RGB   = 0x50
_OP_LOAD_CONST     = 0x60
_OP_LOAD_SCALAR    = 0x61
_OP_OUTPUT_COLOR   = 0xF0
_OP_OUTPUT_ROUGH   = 0xF1
_OP_OUTPUT_METAL   = 0xF2
_OP_OUTPUT_EMISSION= 0xF3
_OP_OUTPUT_ALPHA   = 0xF4
_OP_UV_VFLIP       = 0x13
_OP_TEX_CHECKER    = 0x58
_OP_LOAD_WORLD_POS = 0x62
_OP_OUTPUT_UV      = 0xEF
_OP_OUTPUT_IOR     = 0xF6
_OP_OUTPUT_TRANSMISSION = 0xF7
_OP_TEX_NOISE      = 0x80
_OP_TEX_GRADIENT   = 0x81
_OP_TEX_VORONOI    = 0x82
_OP_TEX_WAVE       = 0x83
_OP_RGB_CURVES     = 0x84
_OP_CURVE_DATA     = 0x85
_OP_LOAD_VIEW_DIR  = 0x63
_OP_LAYER_WEIGHT   = 0x64
_OP_FRESNEL_NODE   = 0x65
_OP_VEC_MATH        = 0x86
_OP_MAP_RANGE_FULL  = 0x87
_OP_TEX_WHITE_NOISE = 0x88
_OP_TEX_MAGIC       = 0x91
_OP_TEX_BRICK       = 0x92
_OP_LOAD_NORMAL     = 0x89
_OP_LOAD_INCOMING   = 0x8A
_OP_BACKFACING      = 0x8B
_OP_DARKEN          = 0x27
_OP_LIGHTEN         = 0x2C
_OP_COLOR_DODGE     = 0x2D
_OP_COLOR_BURN      = 0x2E
_OP_SOFT_LIGHT      = 0x2F
_OP_LINEAR_LIGHT    = 0x90
_OP_NOISE_BUMP      = 0x93
_OP_LOAD_VERTEX_COLOR = 0x94
_OP_OBJECT_RANDOM   = 0x95
_OP_OUTPUT_BUMP     = 0xF8


def _make_instr(opcode, dst=0, srcA=0, srcB=0, imm_y=0, imm_z=0, imm_w=0):
    """Pack one VM instruction as a 4-tuple of uint32."""
    x = (opcode & 0xFF) | ((dst & 0x1F) << 8) | ((srcA & 0x1F) << 16) | ((srcB & 0x1F) << 24)
    return (x, imm_y, imm_z, imm_w)


class _NodeVmCompiler:
    """Compiles a Blender Principled BSDF node tree into VM bytecode."""

    def __init__(self, register_image_fn):
        self.instructions = []
        self.register_image = register_image_fn
        self.next_reg = 1  # R0 = UV
        self.node_reg_cache = {}  # (type, name) → register
        self._group_context = None  # Current Group node for cross-boundary compilation

    def _alloc_reg(self):
        r = self.next_reg
        self.next_reg = min(self.next_reg + 1, 31)
        return r

    def _emit(self, opcode, dst=0, srcA=0, srcB=0, imm_y=0, imm_z=0, imm_w=0):
        if len(self.instructions) < 64:
            self.instructions.append(_make_instr(opcode, dst, srcA, srcB, imm_y, imm_z, imm_w))
        return dst

    def _compile_node(self, socket, _depth=0):
        """Compile the node chain feeding into a socket. Returns register index."""
        if socket is None or _depth > 8:
            r = self._alloc_reg()
            self._emit(_OP_LOAD_CONST, r, imm_y=_floatBits(0.8),
                       imm_z=_floatBits(0.8), imm_w=_floatBits(0.8))
            return r

        if not socket.is_linked:
            # Constant value
            r = self._alloc_reg()
            v = socket.default_value
            if hasattr(v, '__len__') and len(v) >= 3:
                self._emit(_OP_LOAD_CONST, r, imm_y=_floatBits(v[0]),
                           imm_z=_floatBits(v[1]), imm_w=_floatBits(v[2]))
            else:
                self._emit(_OP_LOAD_SCALAR, r, imm_y=_floatBits(float(v)))
            return r

        from_node = socket.links[0].from_node
        from_socket = socket.links[0].from_socket
        # Cache key: (type, name, socket_identifier) — unique per output socket.
        # Needed for GROUP_INPUT which has one node but multiple outputs mapping
        # to different Group external inputs.
        _sock_id = getattr(from_socket, 'identifier', from_socket.name)
        node_id = (from_node.type, from_node.name, _sock_id)

        # Check cache — avoid re-compiling the same node
        if node_id in self.node_reg_cache:
            return self.node_reg_cache[node_id]

        dst = self._alloc_reg()

        # ── Image Texture ──
        if from_node.type == 'TEX_IMAGE':
            if from_node.image:
                tex_idx = self.register_image(from_node.image)
                if tex_idx != _NO_TEX:
                    # Check for Mapping node on Vector input
                    uv_reg = 0  # default UV
                    vec_inp = from_node.inputs.get('Vector')
                    if vec_inp and vec_inp.is_linked:
                        uv_reg = self._compile_uv_chain(vec_inp, _depth + 1)
                    # sRGB flag: color textures need gamma decoding, data textures don't
                    cs = getattr(from_node.image, 'colorspace_settings', None)
                    is_srgb = 1 if (cs and cs.name in ('sRGB', 'Filmic sRGB', 'Filmic Log')) else 0
                    # Sample full RGBA into a temp register
                    tex_reg = dst
                    self._emit(_OP_SAMPLE_TEX, tex_reg, srcA=uv_reg, imm_y=is_srgb, imm_z=tex_idx)
                    # If the linked output is "Alpha", extract .a channel
                    if from_socket.name == 'Alpha':
                        alpha_reg = self._alloc_reg()
                        self._emit(_OP_SEPARATE_RGB, alpha_reg, srcA=tex_reg, imm_y=3)  # ch=3 → .a
                        self.node_reg_cache[node_id] = alpha_reg
                        return alpha_reg
                    self.node_reg_cache[node_id] = dst
                    return dst
            # Fallback: gray
            self._emit(_OP_LOAD_CONST, dst, imm_y=_floatBits(0.5),
                       imm_z=_floatBits(0.5), imm_w=_floatBits(0.5))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── ColorRamp ──
        if from_node.type == 'VALTORGB':
            # Compile the factor input
            fac_inp = from_node.inputs.get('Fac')
            fac_reg = self._compile_node(fac_inp, _depth + 1)
            # Need luminance if factor comes from a color
            if fac_inp and fac_inp.is_linked and fac_inp.links[0].from_socket.type in ('RGBA', 'VECTOR'):
                lum_reg = self._alloc_reg()
                self._emit(_OP_LUMINANCE, lum_reg, srcA=fac_reg)
                fac_reg = lum_reg

            ramp = from_node.color_ramp
            elements = sorted(ramp.elements, key=lambda e: e.position)
            stop_count = min(len(elements), 8)

            self._emit(_OP_COLORRAMP, dst, srcA=fac_reg, imm_y=stop_count)

            # Emit ramp data instructions (1 per stop: pos, R, G, B)
            for el in elements[:8]:
                self._emit(_OP_RAMP_DATA, imm_y=_floatBits(el.position),
                           imm_z=_floatBits(el.color[0]),
                           imm_w=_floatBits(el.color[1]))
                # Pack blue into x field of a second approach — actually we have 4 uints per instr
                # Let me redefine: each RAMP_DATA carries pos, R, G, B in y,z,w and x=opcode
                # Wait, x has opcode+dst+srcA+srcB. For data instructions, we can put B in srcB bits
                # Actually, let's use the instruction format properly:
                # x = OP_RAMP_DATA | (0 << 8), y = pos, z = R, w = G
                # But we need B too. Pack R,G,B into y,z,w by redefining the data format.
                # Actually the node_vm.glsl reads: pos from d.x, R from d.y, G from d.z, B from d.w
                # So we need: x=floatBits(pos), y=floatBits(R), z=floatBits(G), w=floatBits(B)
                # But x also has the opcode... Let me fix this.
                pass

            # I need to fix the ramp data format. Let me redesign:
            # The GLSL reads: uvec4 di = nodeVmCode[dataOffset + i]
            # di.x = pos (float as uint), di.y = R, di.z = G, di.w = B
            # But the instruction format packs opcode into di.x!
            # Solution: for RAMP_DATA, use x as raw float (pos), not as opcode+regs
            # The VM skips these via pc advance, so the opcode field doesn't matter.
            self.node_reg_cache[node_id] = dst
            return dst

        # ── RGB Constant ──
        if from_node.type == 'RGB':
            out = from_node.outputs.get('Color')
            if out:
                c = out.default_value
                self._emit(_OP_LOAD_CONST, dst, imm_y=_floatBits(c[0]),
                           imm_z=_floatBits(c[1]), imm_w=_floatBits(c[2]))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Blackbody ──
        if from_node.type == 'BLACKBODY':
            temp_inp = from_node.inputs.get('Temperature')
            temp = float(temp_inp.default_value) if temp_inp else 5000.0
            rgb = _blackbody_to_rgb(temp)
            self._emit(_OP_LOAD_CONST, dst, imm_y=_floatBits(rgb[0]),
                       imm_z=_floatBits(rgb[1]), imm_w=_floatBits(rgb[2]))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── MixRGB / Mix ──
        if from_node.type in ('MIX_RGB', 'MIX'):
            fac_inp = from_node.inputs.get('Fac') or from_node.inputs[0]
            if from_node.type == 'MIX':
                # Blender 4.x MIX node supports FLOAT, VECTOR, RGBA data types
                data_type = getattr(from_node, 'data_type', 'RGBA')
                if data_type == 'FLOAT':
                    # Float MIX: mix(A, B, factor) — scalar inputs
                    float_inputs = [inp for inp in from_node.inputs if inp.type == 'VALUE']
                    # float_inputs[0] = Factor, [1] = A, [2] = B
                    a_inp = float_inputs[1] if len(float_inputs) > 1 else None
                    b_inp = float_inputs[2] if len(float_inputs) > 2 else None
                    a_reg = self._compile_node(a_inp, _depth + 1) if (a_inp and a_inp.is_linked) else self._alloc_reg()
                    if a_inp and not a_inp.is_linked:
                        self._emit(_OP_LOAD_SCALAR, a_reg, imm_y=_floatBits(float(a_inp.default_value)))
                    b_reg = self._compile_node(b_inp, _depth + 1) if (b_inp and b_inp.is_linked) else self._alloc_reg()
                    if b_inp and not b_inp.is_linked:
                        self._emit(_OP_LOAD_SCALAR, b_reg, imm_y=_floatBits(float(b_inp.default_value)))
                    fac_is_linked = fac_inp and fac_inp.is_linked
                    if fac_is_linked:
                        fac_reg = self._compile_node(fac_inp, _depth + 1)
                        self._emit(_OP_MIX_REG, dst, srcA=a_reg, srcB=b_reg, imm_y=fac_reg)
                    else:
                        fac_val = float(fac_inp.default_value) if fac_inp else 0.5
                        self._emit(_OP_MIX, dst, srcA=a_reg, srcB=b_reg, imm_y=_floatBits(fac_val))
                    self.node_reg_cache[node_id] = dst
                    return dst
                rgba_inputs = [inp for inp in from_node.inputs if inp.type == 'RGBA']
                c1_inp = rgba_inputs[0] if len(rgba_inputs) > 0 else None
                c2_inp = rgba_inputs[1] if len(rgba_inputs) > 1 else None
            else:
                c1_inp = from_node.inputs.get('Color1')
                c2_inp = from_node.inputs.get('Color2')

            c1_reg = self._compile_node(c1_inp, _depth + 1)
            c2_reg = self._compile_node(c2_inp, _depth + 1)

            blend_type = getattr(from_node, 'blend_type', 'MIX')

            # Get factor value
            fac_is_linked = fac_inp and fac_inp.is_linked
            fac_val = float(fac_inp.default_value) if (fac_inp and not fac_is_linked) else 0.5

            # Cycles formula: result = mix(A, blend(A,B), factor)
            # With factor=0 → result = A (unchanged) for ALL blend modes.
            # Only compile the blend op if factor > 0.
            if not fac_is_linked and abs(fac_val) < 0.001:
                # Factor is zero — result is just A, skip blend entirely
                self.node_reg_cache[node_id] = c1_reg
                return c1_reg

            if blend_type == 'MIX':
                if fac_is_linked:
                    fac_reg = self._compile_node(fac_inp, _depth + 1)
                    self._emit(_OP_MIX_REG, dst, srcA=c1_reg, srcB=c2_reg, imm_y=fac_reg)
                else:
                    self._emit(_OP_MIX, dst, srcA=c1_reg, srcB=c2_reg, imm_y=_floatBits(fac_val))
            else:
                # Non-MIX blends: compute blend(A,B), then mix(A, blended, fac)
                blended_reg = self._alloc_reg()
                if blend_type == 'MULTIPLY':
                    self._emit(_OP_MULTIPLY, blended_reg, srcA=c1_reg, srcB=c2_reg)
                elif blend_type == 'ADD':
                    self._emit(_OP_ADD, blended_reg, srcA=c1_reg, srcB=c2_reg)
                elif blend_type == 'SUBTRACT':
                    self._emit(_OP_SUBTRACT, blended_reg, srcA=c1_reg, srcB=c2_reg)
                elif blend_type == 'SCREEN':
                    self._emit(_OP_SCREEN, blended_reg, srcA=c1_reg, srcB=c2_reg)
                elif blend_type == 'OVERLAY':
                    self._emit(_OP_OVERLAY, blended_reg, srcA=c1_reg, srcB=c2_reg)
                elif blend_type == 'DARKEN':
                    self._emit(_OP_DARKEN, blended_reg, srcA=c1_reg, srcB=c2_reg)
                elif blend_type == 'LIGHTEN':
                    self._emit(_OP_LIGHTEN, blended_reg, srcA=c1_reg, srcB=c2_reg)
                elif blend_type == 'DODGE':
                    self._emit(_OP_COLOR_DODGE, blended_reg, srcA=c1_reg, srcB=c2_reg)
                elif blend_type == 'BURN':
                    self._emit(_OP_COLOR_BURN, blended_reg, srcA=c1_reg, srcB=c2_reg)
                elif blend_type == 'SOFT_LIGHT':
                    self._emit(_OP_SOFT_LIGHT, blended_reg, srcA=c1_reg, srcB=c2_reg)
                elif blend_type == 'LINEAR_LIGHT':
                    self._emit(_OP_LINEAR_LIGHT, blended_reg, srcA=c1_reg, srcB=c2_reg)
                else:
                    blended_reg = c2_reg  # fallback to B

                # mix(A, blended, factor) — Cycles always applies factor
                if fac_is_linked:
                    fac_reg = self._compile_node(fac_inp, _depth + 1)
                    self._emit(_OP_MIX_REG, dst, srcA=c1_reg, srcB=blended_reg, imm_y=fac_reg)
                else:
                    self._emit(_OP_MIX, dst, srcA=c1_reg, srcB=blended_reg, imm_y=_floatBits(fac_val))

            self.node_reg_cache[node_id] = dst
            return dst

        # ── Invert ──
        if from_node.type == 'INVERT':
            fac = _resolve_scalar_input(from_node.inputs.get('Fac'), 1.0)
            col_reg = self._compile_node(from_node.inputs.get('Color'), _depth + 1)
            self._emit(_OP_INVERT, dst, srcA=col_reg, imm_y=_floatBits(fac))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Gamma ──
        if from_node.type == 'GAMMA':
            col_reg = self._compile_node(from_node.inputs.get('Color'), _depth + 1)
            gamma = _resolve_scalar_input(from_node.inputs.get('Gamma'), 1.0)
            self._emit(_OP_GAMMA, dst, srcA=col_reg, imm_y=_floatBits(gamma))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Bright/Contrast ──
        if from_node.type == 'BRIGHTCONTRAST':
            col_reg = self._compile_node(from_node.inputs.get('Color'), _depth + 1)
            bright = _resolve_scalar_input(from_node.inputs.get('Bright'), 0.0)
            contrast = _resolve_scalar_input(from_node.inputs.get('Contrast'), 0.0)
            self._emit(_OP_BRIGHT_CONTRAST, dst, srcA=col_reg,
                       imm_y=_floatBits(bright), imm_z=_floatBits(contrast))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Hue/Saturation/Value ──
        if from_node.type == 'HUE_SAT':
            col_reg = self._compile_node(from_node.inputs.get('Color'), _depth + 1)
            hue = _resolve_scalar_input(from_node.inputs.get('Hue'), 0.5)
            sat = _resolve_scalar_input(from_node.inputs.get('Saturation'), 1.0)
            val = _resolve_scalar_input(from_node.inputs.get('Value'), 1.0)
            self._emit(_OP_HUE_SAT_VAL, dst, srcA=col_reg,
                       imm_y=_floatBits(hue), imm_z=_floatBits(sat), imm_w=_floatBits(val))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Separate RGB ──
        if from_node.type in ('SEPRGB', 'SEPARATE_COLOR'):
            col_reg = self._compile_node(from_node.inputs[0], _depth + 1)
            # Which output socket was linked? Check the from_socket name
            from_socket = socket.links[0].from_socket
            ch = {'Red': 0, 'Green': 1, 'Blue': 2, 'R': 0, 'G': 1, 'B': 2}.get(from_socket.name, 0)
            self._emit(_OP_SEPARATE_RGB, dst, srcA=col_reg, imm_y=ch)
            self.node_reg_cache[node_id] = dst
            return dst

        # ── RGB to BW ──
        if from_node.type == 'RGBTOBW':
            col_reg = self._compile_node(from_node.inputs.get('Color'), _depth + 1)
            self._emit(_OP_LUMINANCE, dst, srcA=col_reg)
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Value node ──
        if from_node.type == 'VALUE':
            out = from_node.outputs.get('Value')
            val = float(out.default_value) if out else 0.0
            self._emit(_OP_LOAD_SCALAR, dst, imm_y=_floatBits(val))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Math node ──
        if from_node.type == 'MATH':
            a_reg = self._compile_node(from_node.inputs[0], _depth + 1)
            b_reg = self._compile_node(from_node.inputs[1], _depth + 1) if len(from_node.inputs) > 1 else a_reg
            op = from_node.operation
            math_ops = {
                'ADD': 0x41, 'MULTIPLY': 0x42, 'DIVIDE': 0x43,
                'POWER': 0x44, 'MINIMUM': 0x45, 'MAXIMUM': 0x46,
                'SUBTRACT': _OP_MATH_SUB, 'ABSOLUTE': _OP_MATH_ABS,
                'SQRT': _OP_MATH_SQRT, 'MODULO': _OP_MATH_MOD,
                'FLOOR': _OP_MATH_FLOOR, 'CEIL': _OP_MATH_CEIL,
                'FRACT': _OP_MATH_FRACT, 'SINE': _OP_MATH_SIN,
                'COSINE': _OP_MATH_COS, 'TANGENT': _OP_MATH_TAN,
                'LESS_THAN': _OP_MATH_LESS, 'GREATER_THAN': _OP_MATH_GREATER,
                'ROUND': _OP_MATH_ROUND, 'SIGN': _OP_MATH_SIGN,
                'SMOOTH_MIN': _OP_MATH_SMOOTH_MIN,
            }
            opcode = math_ops.get(op, 0x42)  # default to multiply
            self._emit(opcode, dst, srcA=a_reg, srcB=b_reg)
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Vector Math node ──
        if from_node.type == 'VECT_MATH':
            a_reg = self._compile_node(from_node.inputs[0], _depth + 1)
            b_reg = self._compile_node(from_node.inputs[1], _depth + 1) if len(from_node.inputs) > 1 else a_reg
            vec_ops = {
                'ADD': 0, 'SUBTRACT': 1, 'MULTIPLY': 2, 'DIVIDE': 3,
                'CROSS_PRODUCT': 4, 'DOT_PRODUCT': 5, 'LENGTH': 6, 'DISTANCE': 7,
                'NORMALIZE': 8, 'SCALE': 9, 'REFLECT': 10, 'ABSOLUTE': 11,
                'MINIMUM': 12, 'MAXIMUM': 13, 'FLOOR': 14, 'FRACT': 15,
                'MODULO': 16, 'SIGN': 17,
            }
            vop = vec_ops.get(from_node.operation, 0)
            self._emit(_OP_VEC_MATH, dst, srcA=a_reg, srcB=b_reg, imm_y=vop)
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Map Range node (full 5-param) ──
        if from_node.type == 'MAP_RANGE':
            val_reg = self._compile_node(from_node.inputs[0], _depth + 1)
            from_min = _resolve_scalar_input(from_node.inputs.get('From Min') or from_node.inputs[1], 0.0)
            from_max = _resolve_scalar_input(from_node.inputs.get('From Max') or from_node.inputs[2], 1.0)
            to_min = _resolve_scalar_input(from_node.inputs.get('To Min') or from_node.inputs[3], 0.0)
            to_max = _resolve_scalar_input(from_node.inputs.get('To Max') or from_node.inputs[4], 1.0)
            self._emit(_OP_MAP_RANGE_FULL, dst, srcA=val_reg,
                       imm_y=_floatBits(from_min), imm_z=_floatBits(from_max), imm_w=_floatBits(to_min))
            # toMax in next data instruction
            self.instructions.append((_floatBits(to_max), 0, 0, 0))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Clamp node ──
        if from_node.type == 'CLAMP':
            val_inp = from_node.inputs.get('Value') or from_node.inputs[0]
            val_reg = self._compile_node(val_inp, _depth + 1) if val_inp and val_inp.is_linked else self._alloc_reg()
            if not (val_inp and val_inp.is_linked):
                self._emit(_OP_LOAD_SCALAR, val_reg, imm_y=_floatBits(float(val_inp.default_value)))
            mn = _resolve_scalar_input(from_node.inputs.get('Min'), 0.0)
            mx = _resolve_scalar_input(from_node.inputs.get('Max'), 1.0)
            self._emit(_OP_MATH_CLAMP, dst, srcA=val_reg, imm_y=_floatBits(mn), imm_z=_floatBits(mx))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Checker Texture (procedural) ──
        if from_node.type == 'TEX_CHECKER':
            # Get UV input
            vec_inp = from_node.inputs.get('Vector')
            uv_reg = 0  # default UV
            if vec_inp and vec_inp.is_linked:
                uv_reg = self._compile_uv_chain(vec_inp, _depth + 1)

            # Scale
            scale_inp = from_node.inputs.get('Scale')
            scale_val = float(scale_inp.default_value) if scale_inp else 5.0

            # Colors
            c1_inp = from_node.inputs.get('Color1')
            c2_inp = from_node.inputs.get('Color2')
            c1 = (0.8, 0.8, 0.8) if not c1_inp else (c1_inp.default_value[0], c1_inp.default_value[1], c1_inp.default_value[2])
            c2 = (0.2, 0.2, 0.2) if not c2_inp else (c2_inp.default_value[0], c2_inp.default_value[1], c2_inp.default_value[2])

            # Load color1 into a register
            c1_reg = self._alloc_reg()
            self._emit(_OP_LOAD_CONST, c1_reg, imm_y=_floatBits(c1[0]),
                       imm_z=_floatBits(c1[1]), imm_w=_floatBits(c1[2]))

            # OP_TEX_CHECKER: dst=result, srcA=uv, srcB=color1, imm_y=scale, imm_z=c2.r, imm_w=c2.g
            # (c2.b is 0 for now — pack limitation)
            self._emit(_OP_TEX_CHECKER, dst, srcA=uv_reg, srcB=c1_reg,
                       imm_y=_floatBits(scale_val),
                       imm_z=_floatBits(c2[0]), imm_w=_floatBits(c2[1]))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Ambient Occlusion — passthrough Color input (AO handled by GI) ──
        if from_node.type == 'AMBIENT_OCCLUSION':
            color_inp = from_node.inputs.get('Color')
            if color_inp and color_inp.is_linked:
                result = self._compile_node(color_inp, _depth + 1)
                self.node_reg_cache[node_id] = result
                return result
            # Unlinked — use the default color value
            c = color_inp.default_value if color_inp else (0.8, 0.8, 0.8, 1.0)
            self._emit(_OP_LOAD_CONST, dst, imm_y=_floatBits(c[0]),
                       imm_z=_floatBits(c[1]), imm_w=_floatBits(c[2]))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── RGB Curves — passthrough ──
        # Baked LUT produces incorrect blue tint. Needs investigation into
        # how Cycles' BKE_curvemap_evaluateF differs from Python mapping.evaluate().
        if from_node.type == 'CURVE_RGB':
            color_inp = from_node.inputs.get('Color')
            if color_inp and color_inp.is_linked:
                reg = self._compile_node(color_inp, _depth + 1)
                self.node_reg_cache[node_id] = reg
                return reg
            c = color_inp.default_value if color_inp else (0.8, 0.8, 0.8, 1.0)
            self._emit(_OP_LOAD_CONST, dst, imm_y=_floatBits(c[0]),
                       imm_z=_floatBits(c[1]), imm_w=_floatBits(c[2]))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Noise Texture ──
        if from_node.type == 'TEX_NOISE':
            vec_inp = from_node.inputs.get('Vector')
            uv_reg = 0
            if vec_inp and vec_inp.is_linked:
                uv_reg = self._compile_uv_chain(vec_inp, _depth + 1)
            else:
                # Default: Generated coordinates (object-local position)
                uv_reg = self._alloc_reg()
                self._emit(0x66, uv_reg)  # OP_LOAD_LOCAL_POS

            scale_inp = from_node.inputs.get('Scale')
            scale = float(scale_inp.default_value) if scale_inp and not scale_inp.is_linked else 5.0
            detail_inp = from_node.inputs.get('Detail')
            detail = float(detail_inp.default_value) if detail_inp and not detail_inp.is_linked else 2.0
            rough_inp = from_node.inputs.get('Roughness')
            roughness = float(rough_inp.default_value) if rough_inp and not rough_inp.is_linked else 0.5

            self._emit(_OP_TEX_NOISE, dst, srcA=uv_reg,
                       imm_y=_floatBits(scale), imm_z=_floatBits(detail), imm_w=_floatBits(roughness))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Gradient Texture ──
        if from_node.type == 'TEX_GRADIENT':
            vec_inp = from_node.inputs.get('Vector')
            uv_reg = 0
            if vec_inp and vec_inp.is_linked:
                uv_reg = self._compile_uv_chain(vec_inp, _depth + 1)
            else:
                uv_reg = self._alloc_reg()
                self._emit(0x66, uv_reg)  # OP_LOAD_LOCAL_POS (Generated coords)

            grad_type_map = {'LINEAR': 0, 'QUADRATIC': 1, 'EASING': 0, 'DIAGONAL': 0,
                             'RADIAL': 2, 'QUADRATIC_SPHERE': 3, 'SPHERICAL': 3}
            grad_type = grad_type_map.get(getattr(from_node, 'gradient_type', 'LINEAR'), 0)

            self._emit(_OP_TEX_GRADIENT, dst, srcA=uv_reg, imm_y=grad_type)
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Voronoi Texture ──
        if from_node.type == 'TEX_VORONOI':
            vec_inp = from_node.inputs.get('Vector')
            uv_reg = 0
            if vec_inp and vec_inp.is_linked:
                uv_reg = self._compile_uv_chain(vec_inp, _depth + 1)
            else:
                uv_reg = self._alloc_reg()
                self._emit(0x66, uv_reg)  # OP_LOAD_LOCAL_POS (Generated coords)

            scale_inp = from_node.inputs.get('Scale')
            scale = float(scale_inp.default_value) if scale_inp and not scale_inp.is_linked else 5.0

            self._emit(_OP_TEX_VORONOI, dst, srcA=uv_reg, imm_y=_floatBits(scale))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Wave Texture ──
        if from_node.type == 'TEX_WAVE':
            vec_inp = from_node.inputs.get('Vector')
            uv_reg = 0
            if vec_inp and vec_inp.is_linked:
                uv_reg = self._compile_uv_chain(vec_inp, _depth + 1)
            else:
                uv_reg = self._alloc_reg()
                self._emit(0x66, uv_reg)  # OP_LOAD_LOCAL_POS (Generated coords)

            scale_inp = from_node.inputs.get('Scale')
            scale = float(scale_inp.default_value) if scale_inp and not scale_inp.is_linked else 5.0
            dist_inp = from_node.inputs.get('Distortion')
            distortion = float(dist_inp.default_value) if dist_inp and not dist_inp.is_linked else 0.0
            wave_type = 0 if getattr(from_node, 'wave_type', 'SINE') == 'SINE' else 1

            self._emit(_OP_TEX_WAVE, dst, srcA=uv_reg,
                       imm_y=_floatBits(scale), imm_z=_floatBits(distortion), imm_w=wave_type)
            self.node_reg_cache[node_id] = dst
            return dst

        # ── White Noise Texture ──
        if from_node.type == 'TEX_WHITE_NOISE':
            vec_inp = from_node.inputs.get('Vector')
            uv_reg = 0
            if vec_inp and vec_inp.is_linked:
                uv_reg = self._compile_uv_chain(vec_inp, _depth + 1)
            else:
                uv_reg = self._alloc_reg()
                self._emit(0x66, uv_reg)  # OP_LOAD_LOCAL_POS (Generated coords)
            self._emit(_OP_TEX_WHITE_NOISE, dst, srcA=uv_reg)
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Layer Weight ──
        if from_node.type == 'LAYER_WEIGHT':
            blend_inp = from_node.inputs.get('Blend')
            blend = float(blend_inp.default_value) if blend_inp and not blend_inp.is_linked else 0.5
            # Determine which output is connected: Fresnel or Facing
            out_socket_name = socket.links[0].from_socket.name if socket.is_linked else 'Fresnel'
            mode = 0 if out_socket_name == 'Fresnel' else 1

            self._emit(_OP_LAYER_WEIGHT, dst, imm_y=_floatBits(blend), imm_z=mode)
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Fresnel ──
        if from_node.type == 'FRESNEL':
            ior_inp = from_node.inputs.get('IOR')
            ior = float(ior_inp.default_value) if ior_inp and not ior_inp.is_linked else 1.5

            self._emit(_OP_FRESNEL_NODE, dst, imm_y=_floatBits(ior))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Geometry node ──
        if from_node.type == 'NEW_GEOMETRY':
            out_name = socket.links[0].from_socket.name if socket.is_linked else 'Position'
            if out_name == 'Position':
                self._emit(_OP_LOAD_WORLD_POS, dst)
            elif out_name == 'Normal':
                self._emit(_OP_LOAD_NORMAL, dst)
            elif out_name == 'Incoming':
                self._emit(_OP_LOAD_INCOMING, dst)
            elif out_name == 'Backfacing':
                self._emit(_OP_BACKFACING, dst)
            else:
                self._emit(_OP_LOAD_WORLD_POS, dst)  # fallback
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Light Path ──
        if from_node.type == 'LIGHT_PATH':
            # Light Path outputs are all constants in our path tracer:
            # Is Camera Ray = 1.0 (we evaluate materials on primary hits)
            # All other outputs = 0.0 (simplified)
            out_name = socket.links[0].from_socket.name if socket.is_linked else 'Is Camera Ray'
            val = 1.0 if out_name == 'Is Camera Ray' else 0.0
            self._emit(_OP_LOAD_SCALAR, dst, imm_y=_floatBits(val))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Object Info ──
        if from_node.type == 'OBJECT_INFO':
            out_name = socket.links[0].from_socket.name if socket.is_linked else 'Random'
            if out_name == 'Random':
                self._emit(_OP_OBJECT_RANDOM, dst)
            else:
                # Location, Object Index, etc. — not yet supported, return 0
                self._emit(_OP_LOAD_SCALAR, dst, imm_y=_floatBits(0.0))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Magic Texture ──
        if from_node.type == 'TEX_MAGIC':
            vec_inp = from_node.inputs.get('Vector')
            uv_reg = 0
            if vec_inp and vec_inp.is_linked:
                uv_reg = self._compile_uv_chain(vec_inp, _depth + 1)
            else:
                uv_reg = self._alloc_reg()
                self._emit(0x66, uv_reg)  # OP_LOAD_LOCAL_POS (Generated coords)

            scale_inp = from_node.inputs.get('Scale')
            scale = float(scale_inp.default_value) if scale_inp and not scale_inp.is_linked else 5.0
            dist_inp = from_node.inputs.get('Distortion')
            distortion = float(dist_inp.default_value) if dist_inp and not dist_inp.is_linked else 1.0

            self._emit(_OP_TEX_MAGIC, dst, srcA=uv_reg,
                       imm_y=_floatBits(distortion), imm_z=_floatBits(scale))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Brick Texture ──
        if from_node.type == 'TEX_BRICK':
            vec_inp = from_node.inputs.get('Vector')
            uv_reg = 0
            if vec_inp and vec_inp.is_linked:
                uv_reg = self._compile_uv_chain(vec_inp, _depth + 1)
            else:
                uv_reg = self._alloc_reg()
                self._emit(0x66, uv_reg)  # OP_LOAD_LOCAL_POS (Generated coords)

            scale_inp = from_node.inputs.get('Scale')
            scale = float(scale_inp.default_value) if scale_inp and not scale_inp.is_linked else 5.0
            mortar_inp = from_node.inputs.get('Mortar Size')
            mortar = float(mortar_inp.default_value) if mortar_inp and not mortar_inp.is_linked else 0.02

            self._emit(_OP_TEX_BRICK, dst, srcA=uv_reg,
                       imm_y=_floatBits(scale), imm_z=_floatBits(mortar))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Musgrave Texture (deprecated in Blender 4.0, treat as Noise) ──
        if from_node.type == 'TEX_MUSGRAVE':
            vec_inp = from_node.inputs.get('Vector')
            uv_reg = 0
            if vec_inp and vec_inp.is_linked:
                uv_reg = self._compile_uv_chain(vec_inp, _depth + 1)
            else:
                uv_reg = self._alloc_reg()
                self._emit(0x66, uv_reg)  # OP_LOAD_LOCAL_POS (Generated coords)
            scale_inp = from_node.inputs.get('Scale')
            scale = float(scale_inp.default_value) if scale_inp and not scale_inp.is_linked else 5.0
            detail_inp = from_node.inputs.get('Detail')
            detail = float(detail_inp.default_value) if detail_inp and not detail_inp.is_linked else 2.0
            self._emit(0x80, dst, srcA=uv_reg,  # _OP_TEX_NOISE
                       imm_y=_floatBits(scale), imm_z=_floatBits(detail), imm_w=_floatBits(0.5))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Attribute / Vertex Color — load per-vertex color or object color ──
        if from_node.type in ('ATTRIBUTE', 'VERTEX_COLOR'):
            attr_name = getattr(from_node, 'attribute_name', '')
            attr_type = getattr(from_node, 'attribute_type', 'GEOMETRY')
            import os as _os_attr
            try:
                with open(_os_attr.path.join(_os_attr.path.expanduser("~"), "ignis-rt.log"), "a") as _f:
                    _f.write(f"[ignis-attr] '{from_node.name}': type={attr_type} name='{attr_name}'\n")
            except: pass
            self._emit(_OP_LOAD_VERTEX_COLOR, dst)
            self.node_reg_cache[node_id] = dst
            return dst

        # ── Other procedural textures — compile as constant fallback ──
        if from_node.type in ('TEX_SKY', 'TEX_ENVIRONMENT'):
            self._emit(_OP_LOAD_CONST, dst, imm_y=_floatBits(0.5),
                       imm_z=_floatBits(0.5), imm_w=_floatBits(0.5))
            self.node_reg_cache[node_id] = dst
            return dst

        # ── GROUP_INPUT — cross Group boundary to external input ──
        if from_node.type == 'GROUP_INPUT' and self._group_context:
            # Match by socket INDEX (not name — multiple inputs can have same name)
            from_socket = socket.links[0].from_socket
            try:
                sock_idx = list(from_node.outputs).index(from_socket)
            except (ValueError, TypeError):
                sock_idx = -1
            ext_inputs = list(self._group_context.inputs)
            ext_inp = ext_inputs[sock_idx] if 0 <= sock_idx < len(ext_inputs) else None
            if ext_inp and ext_inp.is_linked:
                # Follow the external link (back in main tree)
                old_ctx = self._group_context
                self._group_context = None
                result = self._compile_node(ext_inp, _depth + 1)
                self._group_context = old_ctx
                self.node_reg_cache[node_id] = result
                return result
            elif ext_inp:
                # Constant from Group's external input
                val = ext_inp.default_value
                if hasattr(val, '__len__') and len(val) >= 3:
                    self._emit(_OP_LOAD_CONST, dst, imm_y=_floatBits(val[0]),
                               imm_z=_floatBits(val[1]), imm_w=_floatBits(val[2]))
                else:
                    self._emit(_OP_LOAD_SCALAR, dst, imm_y=_floatBits(float(val)))
                self.node_reg_cache[node_id] = dst
                return dst

        # ── Reroute: transparent passthrough ──
        if from_node.type == 'REROUTE':
            inp = from_node.inputs[0] if from_node.inputs else None
            if inp and inp.is_linked:
                result = self._compile_node(inp, _depth + 1)
                self.node_reg_cache[node_id] = result
                return result

        # ── UV/Vector nodes: delegate to _compile_uv_chain ──
        # These nodes produce 3D positions, not colors. When encountered in
        # _compile_node (e.g., Mapping→TEX_COORD used as color input via MIX),
        # delegate to the UV chain compiler which handles them correctly.
        if from_node.type in ('MAPPING', 'TEX_COORD'):
            result = self._compile_uv_chain(socket, _depth)
            self.node_reg_cache[node_id] = result
            return result

        # ── Fallback: try to follow first linked color input ──
        for inp in from_node.inputs:
            if inp.is_linked and inp.type in ('RGBA', 'VALUE', 'VECTOR', 'CUSTOM'):
                result = self._compile_node(inp, _depth + 1)
                self.node_reg_cache[node_id] = result
                return result

        # Ultimate fallback
        self._emit(_OP_LOAD_CONST, dst, imm_y=_floatBits(0.8),
                   imm_z=_floatBits(0.8), imm_w=_floatBits(0.8))
        self.node_reg_cache[node_id] = dst
        return dst

    def _compile_uv_chain(self, socket, _depth=0):
        """Compile UV manipulation chain (Mapping, Texture Coordinate)."""
        if not socket or not socket.is_linked or _depth > 8:
            return 0  # default UV register

        from_node = socket.links[0].from_node

        if from_node.type == 'MAPPING':
            # Get UV input
            vec_inp = from_node.inputs.get('Vector')
            uv_reg = self._compile_uv_chain(vec_inp, _depth + 1) if vec_inp else 0

            loc = from_node.inputs.get('Location')
            rot = from_node.inputs.get('Rotation')
            scale = from_node.inputs.get('Scale')

            loc_v = loc.default_value if loc else (0, 0, 0)
            rot_v = rot.default_value if rot else (0, 0, 0)
            scale_v = scale.default_value if scale else (1, 1, 1)

            # Check if Location is linked (e.g., Object Info:Location)
            loc_linked = loc and loc.is_linked

            # Detect 3D mapping (non-trivial Z scale/location, or 3D source like Object coords)
            is_3d = (abs(scale_v[2] - 1.0) > 0.001 or abs(loc_v[2]) > 0.001
                     or abs(scale_v[0] - scale_v[1]) < 0.001 and abs(scale_v[0] - scale_v[2]) < 0.001
                     and abs(scale_v[0] - 1.0) > 0.001)

            mapping_type = getattr(from_node, 'vector_type', 'POINT')

            if is_3d:
                # 3D Mapping: result = pos * scale + location (full 3D vector math)
                if mapping_type == 'TEXTURE':
                    s = (1.0/scale_v[0] if abs(scale_v[0])>1e-6 else 1.0,
                         1.0/scale_v[1] if abs(scale_v[1])>1e-6 else 1.0,
                         1.0/scale_v[2] if abs(scale_v[2])>1e-6 else 1.0)
                    l = (-loc_v[0], -loc_v[1], -loc_v[2])
                else:
                    s = (scale_v[0], scale_v[1], scale_v[2])
                    l = (loc_v[0], loc_v[1], loc_v[2])

                has_scale = abs(s[0]-1)>0.001 or abs(s[1]-1)>0.001 or abs(s[2]-1)>0.001
                has_loc = abs(l[0])>0.001 or abs(l[1])>0.001 or abs(l[2])>0.001

                if not has_scale and not has_loc and not loc_linked:
                    return uv_reg

                result_reg = uv_reg
                if has_scale:
                    scale_reg = self._alloc_reg()
                    self._emit(_OP_LOAD_CONST, scale_reg,
                               imm_y=_floatBits(s[0]), imm_z=_floatBits(s[1]), imm_w=_floatBits(s[2]))
                    dst = self._alloc_reg()
                    self._emit(_OP_MULTIPLY, dst, srcA=result_reg, srcB=scale_reg)
                    result_reg = dst
                if has_loc:
                    loc_reg = self._alloc_reg()
                    self._emit(_OP_LOAD_CONST, loc_reg,
                               imm_y=_floatBits(l[0]), imm_z=_floatBits(l[1]), imm_w=_floatBits(l[2]))
                    dst = self._alloc_reg()
                    self._emit(_OP_ADD, dst, srcA=result_reg, srcB=loc_reg)
                    result_reg = dst
                if loc_linked:
                    # Location linked to Object Info:Location or similar
                    loc_reg = self._compile_node(loc, _depth + 1)
                    dst = self._alloc_reg()
                    self._emit(_OP_ADD, dst, srcA=result_reg, srcB=loc_reg)
                    result_reg = dst
                return result_reg

            # 2D UV Mapping (standard UV transforms)
            if mapping_type == 'TEXTURE':
                eff_scale_x = 1.0 / scale_v[0] if abs(scale_v[0]) > 1e-6 else 1.0
                eff_scale_y = 1.0 / scale_v[1] if abs(scale_v[1]) > 1e-6 else 1.0
                eff_loc_x = -loc_v[0]
                eff_loc_y = -loc_v[1]
                eff_rot = -rot_v[2]
            else:
                eff_scale_x = scale_v[0]
                eff_scale_y = scale_v[1]
                eff_loc_x = loc_v[0]
                eff_loc_y = loc_v[1]
                eff_rot = rot_v[2]

            has_scale = abs(eff_scale_x - 1.0) > 0.001 or abs(eff_scale_y - 1.0) > 0.001
            has_loc = abs(eff_loc_x) > 0.001 or abs(eff_loc_y) > 0.001
            has_rot = abs(eff_rot) > 0.001
            has_any = has_scale or has_loc or has_rot or loc_linked

            if not has_any:
                return uv_reg

            dst = self._alloc_reg()

            if has_scale or has_loc or has_rot:
                self._emit(_OP_UV_TRANSFORM, dst, srcA=uv_reg,
                           imm_y=_floatBits(eff_scale_x),
                           imm_z=_floatBits(eff_scale_y),
                           imm_w=_floatBits(eff_loc_x))
                uv_reg = dst

            if has_rot:
                rot_dst = self._alloc_reg()
                self._emit(_OP_UV_ROTATE, rot_dst, srcA=uv_reg,
                           imm_y=_floatBits(eff_rot))
                uv_reg = rot_dst

            if loc_linked:
                loc_reg = self._compile_node(loc, _depth + 1)
                add_dst = self._alloc_reg()
                self._emit(_OP_ADD, add_dst, srcA=uv_reg, srcB=loc_reg)
                return add_dst

            return uv_reg

        if from_node.type == 'TEX_COORD':
            # Check which output is connected (UV, Object, Generated, etc.)
            out_name = socket.links[0].from_socket.name if socket.is_linked else 'UV'
            if out_name == 'Object':
                dst = self._alloc_reg()
                self._emit(0x66, dst)  # OP_LOAD_LOCAL_POS — object-space position
                return dst
            if out_name in ('Generated', 'Camera', 'Window'):
                dst = self._alloc_reg()
                self._emit(_OP_LOAD_WORLD_POS, dst)
                return dst
            return 0  # UV output → default UV register

        # Reroute — follow through
        if from_node.type == 'REROUTE':
            inp = from_node.inputs[0] if from_node.inputs else None
            return self._compile_uv_chain(inp, _depth + 1) if inp else 0

        # Geometry node (Position output) — use world position for texturing
        if from_node.type == 'NEW_GEOMETRY':
            dst = self._alloc_reg()
            self._emit(_OP_LOAD_WORLD_POS, dst)
            return dst

        # Combine XYZ — reconstruct vector from 3 scalar inputs
        # Common pattern: Separate XYZ → Math(mul) → Combine XYZ (scaled world pos)
        if from_node.type in ('COMBXYZ', 'COMBINE_XYZ'):
            x_inp = from_node.inputs.get('X') or from_node.inputs[0]
            y_inp = from_node.inputs.get('Y') or from_node.inputs[1]
            z_inp = from_node.inputs.get('Z') or from_node.inputs[2]

            # Check if all 3 follow the pattern: Math(MULTIPLY, axis, scale)
            # which means it's just worldPos * scale
            scale = [1.0, 1.0, 1.0]
            uses_world_pos = True
            for i, inp in enumerate([x_inp, y_inp, z_inp]):
                if inp and inp.is_linked:
                    math_node = inp.links[0].from_node
                    if math_node.type == 'MATH' and getattr(math_node, 'operation', '') == 'MULTIPLY':
                        # One input should trace back to Geometry.Position (via Separate XYZ)
                        # The other is the scale factor
                        for mi, m_inp in enumerate(math_node.inputs[:2]):
                            if m_inp.is_linked:
                                src = m_inp.links[0].from_node
                                if src.type in ('SEPXYZ', 'SEPARATE_XYZ'):
                                    # This is the position component — check the other input for scale
                                    other = math_node.inputs[1 - mi]
                                    if other.is_linked:
                                        scale_src = other.links[0].from_node
                                        if scale_src.type == 'REROUTE':
                                            # Follow reroute chain
                                            r = scale_src
                                            for _ in range(8):
                                                if r.type == 'REROUTE' and r.inputs[0].is_linked:
                                                    r = r.inputs[0].links[0].from_node
                                                else:
                                                    break
                                            if r.type == 'MATH':
                                                # Math node producing the scale
                                                scale[i] = _resolve_scalar_input(other, 1.0)
                                            elif r.type == 'VALUE':
                                                scale[i] = float(r.outputs[0].default_value)
                                            else:
                                                scale[i] = _resolve_scalar_input(other, 1.0)
                                        else:
                                            scale[i] = _resolve_scalar_input(other, 1.0)
                                    else:
                                        scale[i] = float(other.default_value)
                                    break
                        else:
                            uses_world_pos = False
                    else:
                        uses_world_pos = False
                else:
                    uses_world_pos = False

            if uses_world_pos:
                # Emit: R[dst] = worldPos * scale
                pos_reg = self._alloc_reg()
                self._emit(_OP_LOAD_WORLD_POS, pos_reg)
                scale_reg = self._alloc_reg()
                self._emit(_OP_LOAD_CONST, scale_reg,
                           imm_y=_floatBits(scale[0]),
                           imm_z=_floatBits(scale[1]),
                           imm_w=_floatBits(scale[2]))
                dst = self._alloc_reg()
                self._emit(_OP_MULTIPLY, dst, srcA=pos_reg, srcB=scale_reg)
                return dst

            # Fallback: compile each input separately
            return self._compile_uv_chain(x_inp, _depth + 1) if x_inp else 0

        # Separate XYZ — typically means Geometry.Position was separated
        if from_node.type in ('SEPXYZ', 'SEPARATE_XYZ'):
            vec_inp = from_node.inputs.get('Vector') or from_node.inputs[0]
            return self._compile_uv_chain(vec_inp, _depth + 1) if vec_inp else 0

        # Math node in UV chain — follow first input
        if from_node.type == 'MATH':
            inp = from_node.inputs[0] if from_node.inputs else None
            return self._compile_uv_chain(inp, _depth + 1) if inp else 0

        # GROUP_INPUT — cross Group boundary (use index, not name)
        if from_node.type == 'GROUP_INPUT' and self._group_context:
            from_socket = socket.links[0].from_socket
            try:
                sock_idx = list(from_node.outputs).index(from_socket)
            except (ValueError, TypeError):
                sock_idx = -1
            ext_inputs = list(self._group_context.inputs)
            ext_inp = ext_inputs[sock_idx] if 0 <= sock_idx < len(ext_inputs) else None
            if ext_inp and ext_inp.is_linked:
                old_ctx = self._group_context
                self._group_context = None
                result = self._compile_uv_chain(ext_inp, _depth + 1)
                self._group_context = old_ctx
                return result

        # Fallback: compile as a regular node (handles MIX, Noise, etc. used as Vector)
        return self._compile_node(socket, _depth)

    def _compile_scalar_input(self, socket, default=0.5, _depth=0):
        """Compile a scalar socket (Roughness, Metallic, etc.)."""
        if socket is None or _depth > 8:
            r = self._alloc_reg()
            self._emit(_OP_LOAD_SCALAR, r, imm_y=_floatBits(default))
            return r

        if not socket.is_linked:
            r = self._alloc_reg()
            val = float(socket.default_value) if not hasattr(socket.default_value, '__len__') else default
            self._emit(_OP_LOAD_SCALAR, r, imm_y=_floatBits(val))
            return r

        # Linked — compile the node chain (reuses color compiler)
        return self._compile_node(socket, _depth)

    def _has_nontrivial_mapping(self, tex_node):
        """Check if an Image Texture node has a Mapping with non-default transforms."""
        vec_inp = tex_node.inputs.get('Vector')
        if not vec_inp or not vec_inp.is_linked:
            return False
        node = vec_inp.links[0].from_node
        # Follow reroutes
        while node.type == 'REROUTE' and node.inputs[0].is_linked:
            node = node.inputs[0].links[0].from_node
        if node.type == 'MAPPING':
            scale = node.inputs.get('Scale')
            if scale and (abs(scale.default_value[0] - 1.0) > 0.001 or
                          abs(scale.default_value[1] - 1.0) > 0.001):
                return True
            rot = node.inputs.get('Rotation')
            if rot and (abs(rot.default_value[0]) > 0.001 or
                        abs(rot.default_value[1]) > 0.001 or
                        abs(rot.default_value[2]) > 0.001):
                return True
            loc = node.inputs.get('Location')
            if loc and (abs(loc.default_value[0]) > 0.001 or
                        abs(loc.default_value[1]) > 0.001):
                return True
            # Check if Mapping is fed by Geometry.Position (not UV)
            v_inp = node.inputs.get('Vector')
            if v_inp and v_inp.is_linked:
                src = v_inp.links[0].from_node
                if src.type == 'NEW_GEOMETRY':
                    return True
        # Non-UV source (Geometry, Object, etc.)
        if node.type == 'NEW_GEOMETRY':
            return True
        return False

    def _has_colorramp_in_chain(self, socket, _depth=0):
        """Check if there's a ColorRamp between a socket and an Image Texture."""
        if not socket or not socket.is_linked or _depth > 8:
            return False
        from_node = socket.links[0].from_node
        if from_node.type == 'VALTORGB':
            return True
        # Follow through passthrough nodes
        for inp in from_node.inputs:
            if inp.is_linked and self._has_colorramp_in_chain(inp, _depth + 1):
                return True
        return False

    def compile(self, principled_node, surface_node=None, emission_node=None):
        """Compile shader tree into VM bytecode. VM is the SINGLE AUTHORITY.

        If surface_node is a Mix Shader, compiles BOTH branches and blends
        per-pixel using OP_MIX_REG. Otherwise compiles the single Principled.
        If emission_node is set (from Add Shader), compiles its Color as emission.
        """
        if principled_node is None and surface_node is None:
            return None

        # Mix Shader: compile both branches and blend per-pixel
        if surface_node and surface_node.type == 'MIX_SHADER':
            return self._compile_mix_shader(surface_node)

        if principled_node is None:
            return None
        result = self._compile_principled(principled_node)

        # Add Shader emission: compile the separate Emission node's Color
        if emission_node and emission_node.type == 'EMISSION':
            color_inp = emission_node.inputs.get('Color')
            if color_inp and color_inp.is_linked:
                reg = self._compile_node(color_inp)
                str_inp = emission_node.inputs.get('Strength')
                strength = float(str_inp.default_value) if str_inp else 1.0
                # Scale emission color by strength and store in .a
                if abs(strength - 1.0) > 0.001:
                    str_reg = self._alloc_reg()
                    self._emit(_OP_LOAD_CONST, str_reg,
                               imm_y=_floatBits(strength), imm_z=_floatBits(strength), imm_w=_floatBits(strength))
                    mul_reg = self._alloc_reg()
                    self._emit(_OP_MULTIPLY, mul_reg, srcA=reg, srcB=str_reg)
                    reg = mul_reg
                self._emit(_OP_OUTPUT_EMISSION, srcA=reg)
            else:
                # Constant emission color
                c = color_inp.default_value if color_inp else (1.0, 1.0, 1.0, 1.0)
                str_inp = emission_node.inputs.get('Strength')
                strength = float(str_inp.default_value) if str_inp else 1.0
                reg = self._alloc_reg()
                self._emit(_OP_LOAD_CONST, reg,
                           imm_y=_floatBits(c[0] * strength),
                           imm_z=_floatBits(c[1] * strength),
                           imm_w=_floatBits(c[2] * strength))
                self._emit(_OP_OUTPUT_EMISSION, srcA=reg)
            result = self.instructions if self.instructions else None

        return result

    def _find_principled_in_shader(self, shader_inp):
        """Find Principled BSDF (or internal Mix Shader) from a shader input.

        Returns (principled_node, group_node_or_None, internal_mix_or_None).
        For Groups with internal Mix Shaders, returns the internal Mix Shader
        as the third element for recursive compilation.
        """
        if not shader_inp or not shader_inp.is_linked:
            return None, None, None
        node = shader_inp.links[0].from_node
        if node.type == 'BSDF_PRINCIPLED':
            return node, None, None
        # Diffuse/Glossy BSDFs: treat as pseudo-Principled for VM compilation.
        # Their 'Color' input maps to 'Base Color', 'Roughness' maps directly.
        if node.type in ('BSDF_DIFFUSE', 'BSDF_GLOSSY'):
            return node, None, None
        if node.type == 'GROUP':
            inner_tree = node.node_tree
            if inner_tree:
                # Follow GROUP_OUTPUT to find what the Group actually outputs
                for inode in inner_tree.nodes:
                    if inode.type == 'GROUP_OUTPUT':
                        for inp in inode.inputs:
                            if inp.is_linked:
                                out_node = inp.links[0].from_node
                                if out_node.type == 'BSDF_PRINCIPLED':
                                    return out_node, node, None
                                if out_node.type == 'MIX_SHADER':
                                    # Group outputs a Mix Shader — needs recursive compilation
                                    return None, node, out_node
                # Fallback: find any Principled inside
                for inode in inner_tree.nodes:
                    if inode.type == 'BSDF_PRINCIPLED':
                        return inode, node, None
            return None, None, None
        if node.type == 'MIX_SHADER':
            for si in (1, 2):
                p, g, m = self._find_principled_in_shader(node.inputs[si] if len(node.inputs) > si else None)
                if p or m: return p, g, m
        return None, None, None

    def _compile_principled_color(self, principled):
        """Compile Base Color from a Principled/Diffuse/Glossy BSDF, return register."""
        # Principled uses 'Base Color', Diffuse/Glossy use 'Color'
        bc_inp = principled.inputs.get('Base Color') or principled.inputs.get('Color')
        if bc_inp and bc_inp.is_linked:
            return self._compile_node(bc_inp)
        val = bc_inp.default_value if bc_inp else (0.8, 0.8, 0.8, 1.0)
        r = self._alloc_reg()
        self._emit(_OP_LOAD_CONST, r, imm_y=_floatBits(val[0]),
                   imm_z=_floatBits(val[1]), imm_w=_floatBits(val[2]))
        return r

    def _compile_principled_scalar(self, principled, input_name, default=0.5):
        """Compile scalar input (Roughness, Metallic, etc.), return register."""
        inp = principled.inputs.get(input_name)
        if inp and inp.is_linked:
            return self._compile_node(inp)
        val = float(inp.default_value) if inp else default
        r = self._alloc_reg()
        self._emit(_OP_LOAD_SCALAR, r, imm_y=_floatBits(val))
        return r

    def _compile_with_group_context(self, principled, group_ctx):
        """Compile a Principled BSDF with optional Group context for cross-boundary."""
        old_ctx = self._group_context
        if group_ctx:
            self._group_context = group_ctx
        result = principled
        self._group_context = old_ctx
        return result

    def _compile_principled_color_ctx(self, principled, group_ctx):
        """Compile Base Color with Group context."""
        old_ctx = self._group_context
        if group_ctx: self._group_context = group_ctx
        result = self._compile_principled_color(principled)
        self._group_context = old_ctx
        return result

    def _compile_principled_scalar_ctx(self, principled, input_name, default, group_ctx):
        """Compile scalar input with Group context."""
        old_ctx = self._group_context
        if group_ctx: self._group_context = group_ctx
        result = self._compile_principled_scalar(principled, input_name, default)
        self._group_context = old_ctx
        return result

    def _compile_mix_shader(self, mix_node):
        """Compile Mix Shader: both branches blended per-pixel.
        Supports Group nodes — crosses Group boundary for internal node compilation.
        """
        # Find Principled BSDF in each branch (+ Group context + internal Mix)
        s1_inp = mix_node.inputs[1] if len(mix_node.inputs) > 1 else None
        s2_inp = mix_node.inputs[2] if len(mix_node.inputs) > 2 else None
        p1, g1, m1 = self._find_principled_in_shader(s1_inp)
        p2, g2, m2 = self._find_principled_in_shader(s2_inp)

        # If a branch is an internal Mix Shader inside a Group, compile it recursively
        if m1 and g1:
            old_ctx = self._group_context
            self._group_context = g1
            result = self._compile_mix_shader(m1)
            self._group_context = old_ctx
            return result if not p2 else result  # TODO: blend with other branch
        if m2 and g2:
            # Compile the Group's internal Mix Shader with Group context
            old_ctx = self._group_context
            self._group_context = g2
            # If p1 exists (outer Principled), compile both and blend at outer level
            if p1:
                # Compile outer Principled as one branch
                color_a = self._compile_principled_color_ctx(p1, g1)
                rough_a = self._compile_principled_scalar_ctx(p1, 'Roughness', 0.5, g1)
                metal_a = self._compile_principled_scalar_ctx(p1, 'Metallic', 0.0, g1)
                # Compile inner Mix Shader (recursive — it outputs its own mixed values)
                self._group_context = g2
                inner_result = self._compile_mix_shader(m2)
                self._group_context = old_ctx
                # The inner Mix already emitted OUTPUT_COLOR/ROUGH/METAL.
                # We need to re-blend with the outer Principled at the outer factor.
                # For now, the inner Mix Shader's outputs take priority (it's the complex one)
                return inner_result
            else:
                result = self._compile_mix_shader(m2)
                self._group_context = old_ctx
                return result

        # If one side is Transparent BSDF, use the other side + compile factor as alpha
        # Mix Shader factor=0 → Shader 1, factor=1 → Shader 2
        # So: Transparent on Shader 1 → alpha = factor (factor=1 means opaque)
        #     Transparent on Shader 2 → alpha = 1-factor (factor=0 means opaque)
        s1_node = s1_inp.links[0].from_node if (s1_inp and s1_inp.is_linked) else None
        s2_node = s2_inp.links[0].from_node if (s2_inp and s2_inp.is_linked) else None
        if s1_node and s1_node.type == 'BSDF_TRANSPARENT' and p2:
            old = self._group_context
            if g2: self._group_context = g2
            r = self._compile_principled(p2)
            self._group_context = old
            # Compile Mix Shader factor as alpha (factor=1 → fully Shader 2 → opaque)
            fac_inp = mix_node.inputs.get('Fac') or mix_node.inputs[0]
            if fac_inp and fac_inp.is_linked:
                fac_reg = self._compile_node(fac_inp)
                self._emit(_OP_OUTPUT_ALPHA, srcA=fac_reg)
            else:
                fac_val = float(fac_inp.default_value) if fac_inp else 0.5
                fac_reg = self._alloc_reg()
                self._emit(_OP_LOAD_SCALAR, fac_reg, imm_y=_floatBits(fac_val))
                self._emit(_OP_OUTPUT_ALPHA, srcA=fac_reg)
            return r
        if s2_node and s2_node.type == 'BSDF_TRANSPARENT' and p1:
            old = self._group_context
            if g1: self._group_context = g1
            r = self._compile_principled(p1)
            self._group_context = old
            # Compile Mix Shader factor as inverse alpha (factor=0 → fully Shader 1 → opaque)
            fac_inp = mix_node.inputs.get('Fac') or mix_node.inputs[0]
            if fac_inp and fac_inp.is_linked:
                fac_reg = self._compile_node(fac_inp)
                inv_reg = self._alloc_reg()
                one_reg = self._alloc_reg()
                self._emit(_OP_LOAD_SCALAR, one_reg, imm_y=_floatBits(1.0))
                self._emit(_OP_SUBTRACT, inv_reg, srcA=one_reg, srcB=fac_reg)
                self._emit(_OP_OUTPUT_ALPHA, srcA=inv_reg)
            else:
                fac_val = float(fac_inp.default_value) if fac_inp else 0.5
                fac_reg = self._alloc_reg()
                self._emit(_OP_LOAD_SCALAR, fac_reg, imm_y=_floatBits(1.0 - fac_val))
                self._emit(_OP_OUTPUT_ALPHA, srcA=fac_reg)
            return r

        # If only one side has a Principled, use it with context
        if p1 and not p2:
            old = self._group_context
            if g1: self._group_context = g1
            r = self._compile_principled(p1)
            self._group_context = old
            return r
        if p2 and not p1:
            old = self._group_context
            if g2: self._group_context = g2
            r = self._compile_principled(p2)
            self._group_context = old
            return r
        if not p1 and not p2:
            return None

        # ── Both sides have Principled BSDFs — compile and blend per-pixel ──

        # UV chain from first texture found in either shader
        for p, g in ((p1, g1), (p2, g2)):
            old = self._group_context
            if g: self._group_context = g
            uv_done = False
            for iname in ('Base Color', 'Color', 'Roughness', 'Metallic'):
                inp = p.inputs.get(iname)
                if inp and inp.is_linked:
                    tex = _find_image_texture_node(inp)
                    if tex:
                        vi = tex.inputs.get('Vector')
                        if vi and vi.is_linked:
                            ur = self._compile_uv_chain(vi)
                            if ur != 0:
                                self._emit(_OP_OUTPUT_UV, srcA=ur)
                                uv_done = True
                                break
            self._group_context = old
            if uv_done:
                break

        # Compile factor
        fac_inp = mix_node.inputs.get('Fac') or mix_node.inputs[0]
        if fac_inp and fac_inp.is_linked:
            fac_reg = self._compile_node(fac_inp)
        else:
            fac_val = float(fac_inp.default_value) if fac_inp else 0.5
            fac_reg = self._alloc_reg()
            self._emit(_OP_LOAD_SCALAR, fac_reg, imm_y=_floatBits(fac_val))

        # Compile and blend Base Color (with Group context for each side)
        color_a = self._compile_principled_color_ctx(p1, g1)
        color_b = self._compile_principled_color_ctx(p2, g2)
        color_mix = self._alloc_reg()
        self._emit(_OP_MIX_REG, color_mix, srcA=color_a, srcB=color_b, imm_y=fac_reg & 0x1F)
        self._emit(_OP_OUTPUT_COLOR, srcA=color_mix)

        # Compile and blend Roughness (Diffuse BSDF defaults to 1.0 in PBR)
        _rough_default_a = 1.0 if (p1 and p1.type == 'BSDF_DIFFUSE') else 0.5
        _rough_default_b = 1.0 if (p2 and p2.type == 'BSDF_DIFFUSE') else 0.5
        rough_a = self._compile_principled_scalar_ctx(p1, 'Roughness', _rough_default_a, g1)
        rough_b = self._compile_principled_scalar_ctx(p2, 'Roughness', _rough_default_b, g2)
        rough_mix = self._alloc_reg()
        self._emit(_OP_MIX_REG, rough_mix, srcA=rough_a, srcB=rough_b, imm_y=fac_reg & 0x1F)
        self._emit(_OP_OUTPUT_ROUGH, srcA=rough_mix)

        # Compile and blend Metallic
        # Diffuse BSDF = metallic 0.0, Glossy BSDF = metallic 1.0
        _metal_default_a = 1.0 if (p1 and p1.type == 'BSDF_GLOSSY') else 0.0
        _metal_default_b = 1.0 if (p2 and p2.type == 'BSDF_GLOSSY') else 0.0
        metal_a = self._compile_principled_scalar_ctx(p1, 'Metallic', _metal_default_a, g1)
        metal_b = self._compile_principled_scalar_ctx(p2, 'Metallic', _metal_default_b, g2)
        metal_mix = self._alloc_reg()
        self._emit(_OP_MIX_REG, metal_mix, srcA=metal_a, srcB=metal_b, imm_y=fac_reg & 0x1F)
        self._emit(_OP_OUTPUT_METAL, srcA=metal_mix)

        # Bump: use from whichever shader has it
        for p in (p1, p2):
            norm_inp = p.inputs.get('Normal')
            if norm_inp and norm_inp.is_linked:
                norm_node = norm_inp.links[0].from_node
                if norm_node.type == 'BUMP':
                    height_inp = norm_node.inputs.get('Height')
                    if height_inp and height_inp.is_linked:
                        h_node = height_inp.links[0].from_node
                        if h_node.type == 'TEX_NOISE' and not _find_image_texture_node(height_inp):
                            s_inp = h_node.inputs.get('Scale')
                            scale = float(s_inp.default_value) if s_inp and not s_inp.is_linked else 5.0
                            d_inp = h_node.inputs.get('Detail')
                            detail = float(d_inp.default_value) if d_inp and not d_inp.is_linked else 2.0
                            r_inp = h_node.inputs.get('Roughness')
                            rough = float(r_inp.default_value) if r_inp and not r_inp.is_linked else 0.5
                            str_inp = norm_node.inputs.get('Strength')
                            bstr = float(str_inp.default_value) if str_inp else 1.0
                            dist_inp = norm_node.inputs.get('Distance')
                            bdist = float(dist_inp.default_value) if dist_inp else 1.0
                            pos_reg = self._alloc_reg()
                            v_inp = h_node.inputs.get('Vector')
                            if v_inp and v_inp.is_linked:
                                pos_reg = self._compile_uv_chain(v_inp)
                                if pos_reg == 0:
                                    self._emit(_OP_LOAD_WORLD_POS, pos_reg)
                            else:
                                self._emit(_OP_LOAD_WORLD_POS, pos_reg)
                            grad_reg = self._alloc_reg()
                            self._emit(_OP_NOISE_BUMP, grad_reg, srcA=pos_reg,
                                       imm_y=_floatBits(scale), imm_z=_floatBits(detail), imm_w=_floatBits(rough))
                            self._emit(_OP_OUTPUT_BUMP, srcA=grad_reg,
                                       imm_y=_floatBits(bstr), imm_z=_floatBits(bdist))
                            break

        return self.instructions if self.instructions else None

    def _compile_principled(self, principled_node):
        """Compile a single Principled BSDF's inputs. VM is the SINGLE AUTHORITY."""
        # ── 1. Compile shared UV chain (from first texture with Mapping) ──
        _SCAN_INPUTS = ('Base Color', 'Alpha', 'Roughness', 'Metallic', 'Emission Color')
        for input_name in _SCAN_INPUTS:
            inp = principled_node.inputs.get(input_name)
            if inp and inp.is_linked:
                tex_node = _find_image_texture_node(inp)
                if tex_node:
                    vec_inp = tex_node.inputs.get('Vector')
                    if vec_inp and vec_inp.is_linked:
                        uv_reg = self._compile_uv_chain(vec_inp)
                        if uv_reg != 0:
                            self._emit(_OP_OUTPUT_UV, srcA=uv_reg)
                            break

        # ── 2. Compile Base Color ──
        color_reg = self._compile_principled_color(principled_node)
        self._emit(_OP_OUTPUT_COLOR, srcA=color_reg)

        # ── 3. Compile Roughness + Metallic ──
        for input_name, output_op in [('Roughness', _OP_OUTPUT_ROUGH), ('Metallic', _OP_OUTPUT_METAL)]:
            reg = self._compile_principled_scalar(principled_node, input_name,
                                                   0.5 if input_name == 'Roughness' else 0.0)
            self._emit(output_op, srcA=reg)

        # ── 4. Emission (only if linked) ──
        em_inp = principled_node.inputs.get('Emission Color')
        if em_inp and em_inp.is_linked:
            reg = self._compile_node(em_inp)
            self._emit(_OP_OUTPUT_EMISSION, srcA=reg)

        # ── 5. Transmission Weight (only if linked — per-pixel transparency maps) ──
        trans_inp = principled_node.inputs.get('Transmission Weight') or principled_node.inputs.get('Transmission')
        if trans_inp and trans_inp.is_linked:
            reg = self._compile_node(trans_inp)
            self._emit(_OP_OUTPUT_TRANSMISSION, srcA=reg)

        # ── 6. Alpha (only if linked — per-pixel alpha from texture) ──
        alpha_inp = principled_node.inputs.get('Alpha')
        if alpha_inp and alpha_inp.is_linked:
            reg = self._compile_node(alpha_inp)
            self._emit(_OP_OUTPUT_ALPHA, srcA=reg)

        # ── 7. Compile procedural bump (Bump node with procedural height input) ──
        norm_inp = principled_node.inputs.get('Normal')
        if norm_inp and norm_inp.is_linked:
            norm_node = norm_inp.links[0].from_node
            if norm_node.type == 'BUMP':
                height_inp = norm_node.inputs.get('Height')
                if height_inp and height_inp.is_linked:
                    height_node = height_inp.links[0].from_node
                    # Check if height comes from a procedural texture (not Image Texture)
                    if height_node.type == 'TEX_NOISE' and not _find_image_texture_node(height_inp):
                        # Get noise parameters
                        scale_inp = height_node.inputs.get('Scale')
                        scale = float(scale_inp.default_value) if scale_inp and not scale_inp.is_linked else 5.0
                        detail_inp = height_node.inputs.get('Detail')
                        detail = float(detail_inp.default_value) if detail_inp and not detail_inp.is_linked else 2.0
                        rough_inp = height_node.inputs.get('Roughness')
                        roughness = float(rough_inp.default_value) if rough_inp and not rough_inp.is_linked else 0.5

                        # Get bump parameters
                        strength_inp = norm_node.inputs.get('Strength')
                        bump_strength = float(strength_inp.default_value) if strength_inp else 1.0
                        distance_inp = norm_node.inputs.get('Distance')
                        bump_distance = float(distance_inp.default_value) if distance_inp else 1.0

                        # Get position input (Vector or world pos)
                        vec_inp = height_node.inputs.get('Vector')
                        pos_reg = self._alloc_reg()
                        if vec_inp and vec_inp.is_linked:
                            pos_reg = self._compile_uv_chain(vec_inp)
                            if pos_reg == 0:
                                self._emit(_OP_LOAD_WORLD_POS, pos_reg)
                        else:
                            self._emit(_OP_LOAD_WORLD_POS, pos_reg)

                        # Emit noise bump opcode
                        grad_reg = self._alloc_reg()
                        self._emit(_OP_NOISE_BUMP, grad_reg, srcA=pos_reg,
                                   imm_y=_floatBits(scale), imm_z=_floatBits(detail), imm_w=_floatBits(roughness))
                        self._emit(_OP_OUTPUT_BUMP, srcA=grad_reg,
                                   imm_y=_floatBits(bump_strength), imm_z=_floatBits(bump_distance))

        return self.instructions if self.instructions else None


def _compile_node_vm(principled_node, register_image_fn, surface_node=None, emission_node=None):
    """Compile a shader tree into VM bytecode. VM is the SINGLE AUTHORITY.

    If surface_node is a Mix Shader, compiles BOTH branches and blends per-pixel.
    Otherwise compiles the single Principled BSDF.
    If emission_node is set (from Add Shader), compiles its Color as emission.

    Returns a list of instruction tuples [(x,y,z,w), ...] or None if no VM needed.
    """
    compiler = _NodeVmCompiler(register_image_fn)
    return compiler.compile(principled_node, surface_node=surface_node, emission_node=emission_node)


# Now fix the ColorRamp data format issue.
# The GLSL reads nodeVmCode[offset+i] as: .x=pos, .y=R, .z=G, .w=B
# But our _emit puts opcode in .x. For RAMP_DATA, we override directly.
# Patch: rewrite _NodeVmCompiler's ColorRamp emission to produce correct data.

# Actually let me fix the ColorRamp compilation above. The issue is that
# _emit() packs opcode into .x, but RAMP_DATA needs .x = float(position).
# Solution: bypass _emit() and append raw data tuples.

_orig_compile_node = _NodeVmCompiler._compile_node

def _patched_compile_node(self, socket, _depth=0):
    if socket and socket.is_linked:
        from_node = socket.links[0].from_node
        if from_node.type == 'VALTORGB':
            node_id = (from_node.type, from_node.name)
            if node_id in self.node_reg_cache:
                return self.node_reg_cache[node_id]

            dst = self._alloc_reg()

            # Compile factor input
            fac_inp = from_node.inputs.get('Fac')
            fac_reg = self._compile_node(fac_inp, _depth + 1)
            # Luminance if factor comes from color
            if fac_inp and fac_inp.is_linked and fac_inp.links[0].from_socket.type in ('RGBA', 'VECTOR'):
                lum_reg = self._alloc_reg()
                self._emit(_OP_LUMINANCE, lum_reg, srcA=fac_reg)
                fac_reg = lum_reg

            ramp = from_node.color_ramp
            elements = sorted(ramp.elements, key=lambda e: e.position)
            stop_count = min(len(elements), 8)

            # Emit COLORRAMP opcode
            self._emit(_OP_COLORRAMP, dst, srcA=fac_reg, imm_y=stop_count)

            # Emit raw ramp data (pos, R, G, B per stop) — bypasses opcode packing
            for el in elements[:8]:
                if len(self.instructions) < 64:
                    self.instructions.append((
                        _floatBits(el.position),
                        _floatBits(el.color[0]),
                        _floatBits(el.color[1]),
                        _floatBits(el.color[2]),
                    ))

            self.node_reg_cache[node_id] = dst
            return dst

    return _orig_compile_node(self, socket, _depth)

_NodeVmCompiler._compile_node = _patched_compile_node


# ---- GPUMaterial struct layout (1180 bytes, scalar) ----
# Must match GPUMaterial in vk_rt_pipeline.h exactly.
# 35 base fields (140 bytes) + nodeVmHeader(4) + pad(12) + nodeVmCode(1024) = 1180
_GPU_MATERIAL_BASE = struct.Struct('<' + 'I' * 5 + 'f' * 3  # tex indices (4) + normalDetail + ks (3)
                                   + 'f' * 4                  # ksSpecularEXP + emissive RGB
                                   + 'f' * 4                  # fresnelC/EXP + detailUVMult + detailNBlend
                                   + 'I' + 'f' + 'I' + 'f'   # flags + alphaRef + shaderType + fresnelMaxLevel
                                   + 'I' * 6                  # multilayer tex indices (6)
                                   + 'f' * 7                  # multilayer mults (7)
                                   + 'f' * 2)                 # sunSpecular + sunSpecularEXP
# Node VM: header(1 uint) + pad(3 uints) + code(256 uints) = 260 uints = 1040 bytes
_GPU_MATERIAL_VM = struct.Struct('<' + 'I' * 260)
_GPU_MATERIAL_SIZE = _GPU_MATERIAL_BASE.size + _GPU_MATERIAL_VM.size  # 140 + 1040 = 1180

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
    node_vm_code=None,
    volume_density=0.0,
    volume_anisotropy=0.0,
    volume_noise_scale=0.0,
    volume_noise_detail=0.0,
    volume_noise_brightness=0.0,
    volume_noise_roughness=0.5,
    volume_noise_lacunarity=2.0,
    volume_noise_contrast=0.0,
    volume_mapping_offset=(0.0, 0.0, 0.0),
):
    """Pack one material into 1180 bytes matching GPUMaterial."""
    base = _GPU_MATERIAL_BASE.pack(
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
        # sunSpecular → volume_density, sunSpecularEXP → volume_anisotropy
        volume_density, volume_anisotropy,
    )

    # Node VM bytecode
    vm_uints = [0] * 260  # header(1) + pad(3) + code(256)
    if node_vm_code:
        instr_count = min(len(node_vm_code), 64)
        vm_uints[0] = instr_count  # nodeVmHeader
        # Pack instructions: each is 4 uints (uvec4)
        for i, instr in enumerate(node_vm_code[:64]):
            for j in range(4):
                vm_uints[4 + i * 4 + j] = instr[j] if j < len(instr) else 0

    # Volume noise parameters packed into nodeVmPad[0..2] + nodeVmCode[0..1]
    # Volume-only materials have no VM code, so we reuse those slots.
    if volume_noise_scale > 0.0 or volume_density > 0.0:
        import struct as _struct_vol
        _pf = lambda f: _struct_vol.unpack('I', _struct_vol.pack('f', f))[0]
        # nodeVmPad[0..2]
        vm_uints[1] = _pf(volume_noise_scale)
        vm_uints[2] = _pf(volume_noise_detail)
        vm_uints[3] = _pf(volume_noise_brightness)
        # nodeVmCode[0] = uvec4(roughness, lacunarity, contrast, 0)
        vm_uints[4] = _pf(volume_noise_roughness)
        vm_uints[5] = _pf(volume_noise_lacunarity)
        vm_uints[6] = _pf(volume_noise_contrast)
        vm_uints[7] = 0
        # nodeVmCode[1] = uvec4(mapping.x, mapping.y, mapping.z, 0)
        vm_uints[8] = _pf(volume_mapping_offset[0])
        vm_uints[9] = _pf(volume_mapping_offset[1])
        vm_uints[10] = _pf(volume_mapping_offset[2])
        vm_uints[11] = 0

    return base + _GPU_MATERIAL_VM.pack(*vm_uints)


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

    # Attribute / Vertex Color — return white until vertex color pipeline is implemented
    if from_node.type in ('ATTRIBUTE', 'VERTEX_COLOR'):
        return (1.0, 1.0, 1.0)

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
        fac_inp = from_node.inputs.get('Fac') or from_node.inputs[0]
        fac = _resolve_scalar_input(fac_inp, 0.5, _depth + 1)

        if from_node.type == 'MIX':
            # Blender 4.0+ Mix node: inputs are duplicated for float/vector/RGBA/rotation.
            # Color inputs are type RGBA (indices 6,7 typically). Find them by type.
            rgba_inputs = [inp for inp in from_node.inputs if inp.type == 'RGBA']
            c1_inp = rgba_inputs[0] if len(rgba_inputs) > 0 else None
            c2_inp = rgba_inputs[1] if len(rgba_inputs) > 1 else None
        else:
            # Old MIX_RGB node
            c1_inp = from_node.inputs.get('Color1')
            c2_inp = from_node.inputs.get('Color2')

        c1 = _resolve_color_input(c1_inp, default, _depth + 1)
        c2 = _resolve_color_input(c2_inp, default, _depth + 1)
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

    # Ambient Occlusion — passthrough Color input (AO handled by GI bounces)
    if from_node.type == 'AMBIENT_OCCLUSION':
        color_inp = from_node.inputs.get('Color')
        if color_inp:
            return _resolve_color_input(color_inp, default, _depth + 1)
        return default

    # RGB Curves — passthrough Color input (curve evaluation too complex for CPU)
    if from_node.type == 'CURVE_RGB':
        fac = _resolve_scalar_input(from_node.inputs.get('Fac'), 1.0, _depth + 1)
        c = _resolve_color_input(from_node.inputs.get('Color'), default, _depth + 1)
        return c  # return unmodified (curve not evaluated, but at least not black)

    # Passthrough for other color-producing nodes
    _COLOR_PASSTHROUGH = {'CURVE_VEC', 'SEPRGB', 'SEPARATE_XYZ', 'SEPXYZ', 'SEPARATE_COLOR',
                          'COMBRGB', 'COMBINE_COLOR', 'COMBINE_XYZ', 'COMBXYZ', 'RGBTOBW',
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
    if from_node.type in ('SEPRGB', 'SEPARATE_XYZ', 'SEPXYZ', 'SEPARATE_COLOR'):
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
        'COMBINE_XYZ', 'COMBXYZ',  # Combine XYZ (new + old name)
        'COMBRGB', 'COMBINE_COLOR',  # Combine Color
        'RGBTOBW',          # RGB to BW
        'TEX_NOISE',        # Noise texture (passthrough to find source texture)
        'TEX_VORONOI',      # Voronoi texture
        'TEX_MUSGRAVE',     # Musgrave texture
        'TEX_CHECKER',      # Checker texture
        'TEX_WAVE',         # Wave texture
        'TEX_GRADIENT',     # Gradient texture
        'AMBIENT_OCCLUSION',  # AO — passthrough Color input
        'TEX_WHITE_NOISE',  # White Noise texture
        'NEW_GEOMETRY',     # Geometry node
    }
    if from_node.type in _PASSTHROUGH_TYPES:
        # For Blender 5.x MIX node: search RGBA inputs first (avoid float/vector 'A'/'B')
        if from_node.type == 'MIX':
            for inp in from_node.inputs:
                if inp.type == 'RGBA' and inp.is_linked:
                    result = _find_image_texture_node(inp, _depth + 1)
                    if result:
                        return result
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


def _find_color_ramp_in_chain(socket, _depth=0):
    """Find a ColorRamp node in the chain between a BSDF input and an Image Texture."""
    if not socket or not socket.is_linked or _depth > 8:
        return None
    from_node = socket.links[0].from_node
    if from_node.type == 'VALTORGB':
        # Verify there's an Image Texture feeding into this ColorRamp
        fac_inp = from_node.inputs.get('Fac')
        if fac_inp and _find_image_texture_node(fac_inp, _depth + 1) is not None:
            return from_node
    return None


def _bake_ramp_into_pixels(image, ramp_elements):
    """Read image pixels, apply ColorRamp, return raw RGBA8 bytes (top-down)."""
    import numpy as np
    w, h = image.size[0], image.size[1]
    # Read all pixels as float array [R,G,B,A, R,G,B,A, ...]
    px = np.array(image.pixels[:], dtype=np.float32)
    ch = len(px) // (w * h)
    px = px.reshape(h, w, ch)

    # Compute luminance as ramp factor
    if ch >= 3:
        lum = 0.2126 * px[:, :, 0] + 0.7152 * px[:, :, 1] + 0.0722 * px[:, :, 2]
    else:
        lum = px[:, :, 0]

    # Sort ramp elements by position
    elements = sorted(ramp_elements, key=lambda e: e[0])
    positions = [e[0] for e in elements]
    colors = [np.array(e[1], dtype=np.float32) for e in elements]  # each is [R,G,B,A]

    flat_lum = lum.flatten()
    out = np.zeros((len(flat_lum), 4), dtype=np.float32)

    # Below first stop
    out[flat_lum <= positions[0]] = colors[0]
    # Above last stop
    out[flat_lum >= positions[-1]] = colors[-1]
    # Interpolate between stops
    for i in range(len(positions) - 1):
        p0, p1 = positions[i], positions[i + 1]
        if p1 <= p0:
            continue
        mask = (flat_lum > p0) & (flat_lum < p1)
        if np.any(mask):
            t = ((flat_lum[mask] - p0) / (p1 - p0))[:, None]
            out[mask] = colors[i] * (1.0 - t) + colors[i + 1] * t

    # Preserve original alpha if available
    if ch >= 4:
        out[:, 3] = px[:, :, 3].flatten()

    # Flip both axes: Blender image.pixels[] is bottom-up + left-to-right,
    # but stb_image (normal texture path) delivers top-down. The horizontal
    # flip compensates for UV coordinate convention differences.
    out = out.reshape(h, w, 4)[::-1, ::-1].copy()
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()


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


class _GroupPrincipledProxy:
    """Makes a GROUP node look like a Principled BSDF for material export.

    Maps the GROUP's external inputs to the internal Principled BSDF's input names,
    so code that reads principled_node.inputs.get('Base Color') works transparently.
    """
    def __init__(self, group_node, inner_principled):
        self.type = 'BSDF_PRINCIPLED'
        self.name = f"{group_node.name} (proxy)"
        self._group = group_node
        self._inner = inner_principled

        # Map: inner Principled input name → GROUP external socket
        self._input_map = {}
        for inp in inner_principled.inputs:
            if inp.is_linked:
                from_node = inp.links[0].from_node
                if from_node.type == 'GROUP_INPUT':
                    # The from_socket.name on GROUP_INPUT matches the group's external input name
                    ext_name = inp.links[0].from_socket.name
                    ext_inp = group_node.inputs.get(ext_name)
                    if ext_inp:
                        self._input_map[inp.name] = ext_inp

        self.inputs = _GroupInputsProxy(self._input_map, inner_principled.inputs)


class _GroupInputsProxy:
    """Dict-like proxy for inputs that resolves GROUP connections."""
    def __init__(self, mapped, original):
        self._mapped = mapped  # name → external socket
        self._original = original  # original NodeInputs

    def get(self, name, default=None):
        # If this input is mapped through the group, return the external socket
        if name in self._mapped:
            return self._mapped[name]
        # Otherwise return the internal socket (for unconnected inputs with defaults)
        return self._original.get(name, default)

    def __iter__(self):
        return iter(self._original)

    def __getitem__(self, idx):
        return self._original[idx]


def _find_principled_in_group(group_node):
    """Find a Principled BSDF inside a GROUP node and return a proxy.

    Returns a _GroupPrincipledProxy that maps external group inputs to
    the internal Principled BSDF's inputs, or None if not found.
    """
    inner_tree = group_node.node_tree
    if not inner_tree:
        return None

    # Find Principled BSDF inside the group
    for inode in inner_tree.nodes:
        if inode.type == 'BSDF_PRINCIPLED':
            return _GroupPrincipledProxy(group_node, inode)

    # Follow GROUP_OUTPUT → shader chain inside the group
    for inode in inner_tree.nodes:
        if inode.type == 'GROUP_OUTPUT':
            for inp in inode.inputs:
                if inp.is_linked:
                    src = inp.links[0].from_node
                    if src.type == 'BSDF_PRINCIPLED':
                        return _GroupPrincipledProxy(group_node, src)
    return None


def _find_surface_shader(node_tree):
    """Find the shader node connected to Material Output's Surface input."""
    for node in node_tree.nodes:
        if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
            surface_inp = node.inputs.get('Surface')
            if surface_inp and surface_inp.is_linked:
                return surface_inp.links[0].from_node
    return None


def _find_volume_shader(node_tree):
    """Find the shader node connected to Material Output's Volume input."""
    for node in node_tree.nodes:
        if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
            volume_inp = node.inputs.get('Volume')
            if volume_inp and volume_inp.is_linked:
                return volume_inp.links[0].from_node
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
            # Follow links to resolve actual color (noise chains, Mix nodes, etc.)
            c = _resolve_color_input(color_inp, (0.8, 0.8, 0.8))
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
            c = _resolve_color_input(color_inp, (0.8, 0.8, 0.8))
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
            # If emission resolved to near-black but has a linked texture,
            # use white as emission color so the texture provides the color
            if color_inp.is_linked and sum(props['emission']) < 0.01:
                props['emission'] = (1.0, 1.0, 1.0)
            props['base_color'] = props['emission']
            tex_node = _find_image_texture_node(color_inp)
            if tex_node and tex_node.image:
                props['emission_tex'] = register_image_fn(tex_node.image)
        strength_inp = node.inputs.get('Strength')
        if strength_inp:
            props['emission_strength'] = float(strength_inp.default_value)
    elif node.type == 'BSDF_PRINCIPLED':
        # Extract Principled BSDF properties directly
        color_inp = node.inputs.get('Base Color')
        if color_inp:
            c = color_inp.default_value
            props['base_color'] = (c[0], c[1], c[2])
            tex_node = _find_image_texture_node(color_inp)
            if tex_node and tex_node.image:
                props['diffuse_tex'] = register_image_fn(tex_node.image)
        rough_inp = node.inputs.get('Roughness')
        if rough_inp:
            props['roughness'] = float(rough_inp.default_value)
        metal_inp = node.inputs.get('Metallic')
        if metal_inp:
            props['metallic'] = float(metal_inp.default_value)
        ior_inp = node.inputs.get('IOR')
        if ior_inp:
            props['ior'] = float(ior_inp.default_value)
        trans_inp = node.inputs.get('Transmission Weight') or node.inputs.get('Transmission')
        if trans_inp:
            props['transmission'] = float(trans_inp.default_value)
        ec_inp = node.inputs.get('Emission Color')
        if ec_inp:
            props['emission'] = (ec_inp.default_value[0], ec_inp.default_value[1], ec_inp.default_value[2])
        es_inp = node.inputs.get('Emission Strength')
        if es_inp:
            props['emission_strength'] = float(es_inp.default_value)
        spec_inp = node.inputs.get('Specular IOR Level') or node.inputs.get('Specular')
        if spec_inp:
            props['specular_level'] = float(spec_inp.default_value)
    elif node.type == 'GROUP':
        # Resolve Group node — find Principled BSDF inside
        proxy = _find_principled_in_group(node)
        if proxy:
            props = _extract_shader_props(proxy, register_image_fn)
        else:
            # Try to follow Group's internal output
            inner_tree = node.node_tree
            if inner_tree:
                for inode in inner_tree.nodes:
                    if inode.type == 'GROUP_OUTPUT':
                        inp = inode.inputs[0] if inode.inputs else None
                        if inp and inp.is_linked:
                            props = _extract_shader_props(inp.links[0].from_node, register_image_fn)
                            break
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


def _try_eval_light_path_factor(socket, depth=0):
    """Try to evaluate a linked factor as a compile-time constant when it
    traces back to Light Path outputs (possibly through Math/Clamp/Reroute).
    Returns float value if fully evaluable, or None if dynamic (Fresnel, textures, etc.).

    Light Path outputs are constants in our path tracer:
      Is Camera Ray = 1.0, all others = 0.0
    """
    if depth > 8:
        return None
    if not socket.is_linked:
        try:
            return float(socket.default_value)
        except Exception:
            return None
    node = socket.links[0].from_node
    out_name = socket.links[0].from_socket.name

    if node.type == 'LIGHT_PATH':
        return 1.0 if out_name == 'Is Camera Ray' else 0.0

    if node.type == 'REROUTE':
        inp = node.inputs[0]
        return _try_eval_light_path_factor(inp, depth + 1)

    if node.type == 'CLAMP':
        val = _try_eval_light_path_factor(node.inputs[0], depth + 1)
        if val is None:
            return None
        mn = _try_eval_light_path_factor(node.inputs[1], depth + 1)
        mx = _try_eval_light_path_factor(node.inputs[2], depth + 1)
        if mn is None: mn = 0.0
        if mx is None: mx = 1.0
        return max(mn, min(mx, val))

    if node.type == 'MATH':
        a = _try_eval_light_path_factor(node.inputs[0], depth + 1)
        b = _try_eval_light_path_factor(node.inputs[1], depth + 1) if len(node.inputs) > 1 else None
        if a is None:
            return None
        if b is None and node.operation not in ('ABSOLUTE', 'ROUND', 'FLOOR', 'CEIL',
                                                  'SINE', 'COSINE', 'TANGENT', 'SQRT'):
            b = float(node.inputs[1].default_value) if len(node.inputs) > 1 else 0.0
            if b is None:
                return None
        op = node.operation
        if op == 'ADD': return a + (b or 0)
        if op == 'SUBTRACT': return a - (b or 0)
        if op == 'MULTIPLY': return a * (b or 0)
        if op == 'DIVIDE': return a / b if b and b != 0 else 0.0
        if op == 'MINIMUM': return min(a, b or 0)
        if op == 'MAXIMUM': return max(a, b or 0)
        if op == 'POWER': return a ** (b or 1) if a >= 0 else 0.0
        if op == 'ABSOLUTE': return abs(a)
        if op == 'GREATER_THAN': return 1.0 if a > (b or 0.5) else 0.0
        if op == 'LESS_THAN': return 1.0 if a < (b or 0.5) else 0.0
        return None

    if node.type == 'MAP_RANGE':
        val = _try_eval_light_path_factor(node.inputs[0], depth + 1)
        if val is None:
            return None
        from_min = _try_eval_light_path_factor(node.inputs[1], depth + 1)
        from_max = _try_eval_light_path_factor(node.inputs[2], depth + 1)
        to_min = _try_eval_light_path_factor(node.inputs[3], depth + 1)
        to_max = _try_eval_light_path_factor(node.inputs[4], depth + 1)
        if None in (from_min, from_max, to_min, to_max):
            return None
        rng = from_max - from_min
        if abs(rng) < 1e-6:
            return to_min
        t = (val - from_min) / rng
        t = max(0.0, min(1.0, t))
        return to_min + t * (to_max - to_min)

    # Value node
    if node.type == 'VALUE':
        return float(node.outputs[0].default_value)

    return None  # dynamic node (Fresnel, Layer Weight, Texture, etc.)


def _resolve_mix_shader(mix_node, register_image_fn):
    """Resolve a Mix Shader node into blended PBR properties."""
    fac_inp = mix_node.inputs.get('Fac')
    if fac_inp and fac_inp.is_linked:
        # Try to evaluate as compile-time constant (Light Path patterns)
        evaluated = _try_eval_light_path_factor(fac_inp)
        if evaluated is not None:
            fac = max(0.0, min(1.0, evaluated))
        else:
            # Truly dynamic factor (Fresnel, Layer Weight, Geometry, etc.)
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

    # ---- Factor is 0 or 1: use dominant shader directly (like Cycles) ----
    # When Light Path or constant factor resolves to 0/1, skip all special
    # cases and use the winning shader's properties unchanged.
    if fac <= 0.001:
        return dict(p1)  # 100% Shader 1
    if fac >= 0.999:
        return dict(p2)  # 100% Shader 2

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
        if s2_type == 'BSDF_GLASS':
            # Glass + Transparent: use passthrough for transparency, Glossy for reflection.
            # Don't refract — for architectural glass the entry/exit cancel (flat pane).
            # Set metallic=1 so non-passthrough rays do specular reflection, not diffuse.
            result['transparent_prob'] = (1.0 - fac)
            result['transmission'] = 0.0  # no refraction (passthrough handles transparency)
            result['metallic'] = 1.0      # pure reflection for non-passthrough rays
        else:
            result['transparent_prob'] = (1.0 - fac)
        result['flags'] = result.get('flags', 0) | 2
        return result
    if s2_type == 'BSDF_TRANSPARENT':
        result = dict(p1)
        if s1_type == 'BSDF_GLASS':
            result['transparent_prob'] = fac
            result['transmission'] = 0.0
            result['metallic'] = 1.0
        else:
            result['transparent_prob'] = fac
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
    # Specular level: Diffuse BSDF has NO specular (0), Glossy/Principled have specular (0.5)
    # Blend proportionally so Mix(0.02, Diffuse, Glossy) gives specular_level=0.01 not 0.5
    spec1 = p1.get('specular_level', 0.0 if s1_type == 'BSDF_DIFFUSE' else 0.5)
    spec2 = p2.get('specular_level', 0.0 if s2_type == 'BSDF_DIFFUSE' else 0.5)

    blended = {
        'base_color': _lerp_color(p1['base_color'], p2['base_color'], fac),
        'roughness': p1['roughness'] * (1 - fac) + p2['roughness'] * fac,
        'metallic': p1['metallic'] * (1 - fac) + p2['metallic'] * fac,
        'emission': _lerp_color(p1['emission'], p2['emission'], fac),
        'emission_strength': p1['emission_strength'] * (1 - fac) + p2['emission_strength'] * fac,
        'transmission': p1.get('transmission', 0.0) * (1 - fac) + p2.get('transmission', 0.0) * fac,
        'ior': p1['ior'] * (1 - fac) + p2['ior'] * fac,
        'transparent_prob': blended_tp,
        'specular_level': spec1 * (1 - fac) + spec2 * fac,
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


def extract_texture_bytes(tex_info):
    """Read pixel data for one deferred texture entry. Call once per frame for smooth loading.

    Modifies tex_info in place: fills 'data' with bytes, removes 'image_ref'.
    If 'ramp_elements' is present, bakes the ColorRamp into raw RGBA8 pixels
    and sets dxgi_format=28 to bypass stb_image decoding on the C++ side.
    Returns True if data was extracted, False on failure.
    """
    if tex_info.get("data") is not None:
        return True  # already extracted
    image = tex_info.get("image_ref")
    if image is None:
        return False

    ramp_elements = tex_info.get("ramp_elements")
    if ramp_elements:
        # Bake ColorRamp into raw RGBA8 pixels
        try:
            data = _bake_ramp_into_pixels(image, ramp_elements)
            tex_info["data"] = data
            tex_info["dxgi_format"] = 28  # RGBA8 raw — skip stb_image
        except Exception:
            # Fallback: use original image without ramp
            data = _get_image_bytes(image)
            if data is None:
                return False
            tex_info["data"] = data
    else:
        data = _get_image_bytes(image)
        if data is None:
            return False
        tex_info["data"] = data

    tex_info["width"] = image.size[0]
    tex_info["height"] = image.size[1]
    if "image_ref" in tex_info:
        del tex_info["image_ref"]
    if "ramp_elements" in tex_info:
        del tex_info["ramp_elements"]
    return True


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
        # Don't filter collection instances by hidden_objects — their source
        # collection is excluded but the instances themselves are visible
        if obj.name in hidden_objects and not inst.is_instance:
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

    def _register_image(image, color_ramp_node=None):
        """Register an image and return its texture index, or _NO_TEX if unavailable.

        If color_ramp_node is provided, the ramp will be baked into the texture
        pixels during extraction (uploaded as raw RGBA8, bypassing stb_image).
        """
        if image is None:
            return _NO_TEX
        # Unique key: include ramp id to avoid sharing baked vs raw versions
        ramp_id = f"__ramp_{id(color_ramp_node)}" if color_ramp_node else ""
        key = image.name + ramp_id
        if key in texture_registry:
            return texture_registry[key]["index"]
        if image.size[0] == 0 or image.size[1] == 0:
            return _NO_TEX
        idx = len(textures_list)
        entry = {
            "index": idx,
            "name": image.name + (" [ramp]" if color_ramp_node else ""),
            "data": None,
            "image_ref": image,
            "width": image.size[0],
            "height": image.size[1],
        }
        if color_ramp_node:
            ramp = color_ramp_node.color_ramp
            entry["ramp_elements"] = [
                (e.position, (e.color[0], e.color[1], e.color[2], e.color[3]))
                for e in ramp.elements
            ]
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

        # Volume material properties
        volume_density = 0.0
        volume_anisotropy = 0.0
        volume_noise_scale = 0.0
        volume_noise_detail = 0.0
        volume_noise_brightness = 0.0
        volume_noise_roughness = 0.5
        volume_noise_lacunarity = 2.0
        volume_noise_contrast = 0.0
        volume_mapping_offset = (0.0, 0.0, 0.0)

        if mat.use_nodes and mat.node_tree:
            # Check for Volume Scatter node on Volume input
            volume_node = _find_volume_shader(mat.node_tree)
            if volume_node and volume_node.type in ('VOLUME_SCATTER', 'VOLUME_ABSORPTION', 'PRINCIPLED_VOLUME'):
                flags |= 4  # MAT_FLAG_VOLUME
                # Extract Volume Scatter properties
                color_inp = volume_node.inputs.get('Color')
                if color_inp:
                    if color_inp.is_linked:
                        # Color is linked (noise/texture) — use white, noise modulates density
                        base_color = (1.0, 1.0, 1.0)
                    else:
                        c = color_inp.default_value
                        base_color = (c[0], c[1], c[2])
                density_inp = volume_node.inputs.get('Density')
                if density_inp:
                    volume_density = float(density_inp.default_value)
                aniso_inp = volume_node.inputs.get('Anisotropy')
                if aniso_inp:
                    volume_anisotropy = float(aniso_inp.default_value)

                # Trace Color/Density input chain for noise texture (heterogeneous volume)
                # Extract ALL properties: noise scale/detail/roughness/lacunarity,
                # Bright/Contrast params, Mapping offset
                _vol_color_inp = color_inp
                if _vol_color_inp and _vol_color_inp.is_linked:
                    _trace = _vol_color_inp.links[0].from_node
                    for _vd in range(8):
                        if _trace.type in ('TEX_NOISE', 'TEX_MUSGRAVE'):
                            flags |= 16  # MAT_FLAG_VOLUME_HETERO
                            _s = _trace.inputs.get('Scale')
                            if _s: volume_noise_scale = float(_s.default_value)
                            _d = _trace.inputs.get('Detail')
                            if _d: volume_noise_detail = float(_d.default_value)
                            _r = _trace.inputs.get('Roughness')
                            if _r: volume_noise_roughness = float(_r.default_value)
                            _l = _trace.inputs.get('Lacunarity')
                            if _l: volume_noise_lacunarity = float(_l.default_value)
                            # Follow Vector input for Mapping node
                            _vi = _trace.inputs.get('Vector')
                            if _vi and _vi.is_linked:
                                _map_node = _vi.links[0].from_node
                                if _map_node.type == 'MAPPING':
                                    _loc = _map_node.inputs.get('Location')
                                    if _loc:
                                        volume_mapping_offset = (float(_loc.default_value[0]),
                                                                 float(_loc.default_value[1]),
                                                                 float(_loc.default_value[2]))
                            break
                        elif _trace.type == 'BRIGHTCONTRAST':
                            _b = _trace.inputs.get('Brightness')
                            if _b: volume_noise_brightness = float(_b.default_value)
                            _c = _trace.inputs.get('Contrast')
                            if _c: volume_noise_contrast = float(_c.default_value)
                            _ci = _trace.inputs.get('Color')
                            if _ci and _ci.is_linked:
                                _trace = _ci.links[0].from_node
                            else: break
                        elif _trace.type == 'MAPPING':
                            _loc = _trace.inputs.get('Location')
                            if _loc:
                                volume_mapping_offset = (float(_loc.default_value[0]),
                                                         float(_loc.default_value[1]),
                                                         float(_loc.default_value[2]))
                            _vi = _trace.inputs.get('Vector')
                            if _vi and _vi.is_linked:
                                _trace = _vi.links[0].from_node
                            else: break
                        else:
                            _found = False
                            for _inp in _trace.inputs:
                                if _inp.is_linked:
                                    _trace = _inp.links[0].from_node
                                    _found = True
                                    break
                            if not _found: break

            # ALWAYS resolve from Material Output first (correct shader chain)
            principled_node = None
            _mix_shader_resolved = False  # When True, don't overwrite blended props
            _original_surface_node = None  # Preserved for VM Mix Shader compilation
            _add_shader_emission_node = None  # Emission node from Add Shader (for VM)
            surface_node = _find_surface_shader(mat.node_tree)
            # Volume-only: Surface input empty but Volume input connected
            if surface_node is None and (flags & 4) != 0:
                flags |= 8  # MAT_FLAG_VOLUME_ONLY
            if surface_node and surface_node.type == 'MIX_SHADER':
                _original_surface_node = surface_node
            # Debug: log what Material Output points to
            import os as _os2
            try:
                with open(_os2.path.join(_os2.path.expanduser("~"), "ignis-rt.log"), "a") as _f:
                    _stype = surface_node.type if surface_node else "None"
                    _sname = surface_node.name if surface_node else "N/A"
                    _f.write(f"[ignis-chain] '{mat.name}': MatOutput->Surface = {_stype} ({_sname})\n")
            except: pass

            # Follow chain from Material Output → Surface
            _follow_depth = 0
            while surface_node is not None and _follow_depth < 8:
                if surface_node.type == 'BSDF_PRINCIPLED':
                    principled_node = surface_node
                    break
                elif surface_node.type == 'MIX_SHADER':
                    # Resolve Mix Shader: blend properties from both inputs
                    props = _resolve_mix_shader(surface_node, _register_image)
                    base_color = props['base_color']
                    roughness = props['roughness']
                    metallic = props['metallic']
                    emission = props['emission']
                    emission_strength = props['emission_strength']
                    diffuse_tex = props['diffuse_tex']
                    emission_tex = props.get('emission_tex', _NO_TEX)
                    transmission = props['transmission']
                    ior = props['ior']
                    if 'alpha' in props: alpha = props['alpha']
                    if 'normal_tex' in props: normal_tex = props['normal_tex']
                    if 'normal_strength' in props: normal_strength = props['normal_strength']
                    if 'flags' in props: flags = props['flags']
                    if 'transparent_prob' in props: transparent_prob = props['transparent_prob']
                    if 'alpha_ref' in props: alpha_ref = props['alpha_ref']
                    if 'specular_level' in props: specular_level = props['specular_level']
                    _mix_shader_resolved = True
                    import os
                    try:
                        with open(os.path.join(os.path.expanduser("~"), "ignis-rt.log"), "a") as _f:
                            _f.write(f"[ignis-mix] '{mat.name}': Mix Shader resolved — "
                                     f"color=({base_color[0]:.3f},{base_color[1]:.3f},{base_color[2]:.3f}) "
                                     f"rough={roughness:.2f} metal={metallic:.2f} spec={specular_level:.4f} tp={transparent_prob:.2f}\n")
                    except: pass
                    # Find the BEST Principled BSDF for VM compilation.
                    # Prefer the shader with linked inputs (textures/nodes) over
                    # simple constants, since the VM adds value for per-pixel evaluation.
                    _candidates = []
                    for si in (1, 2):
                        s_inp = surface_node.inputs[si] if len(surface_node.inputs) > si else None
                        if s_inp and s_inp.is_linked:
                            s_node = s_inp.links[0].from_node
                            p_node = None
                            if s_node.type == 'BSDF_PRINCIPLED':
                                p_node = s_node
                            elif s_node.type == 'GROUP':
                                p_node = _find_principled_in_group(s_node)
                            if p_node:
                                # Count linked inputs as complexity score
                                _complexity = sum(1 for inp in p_node.inputs if inp.is_linked)
                                _candidates.append((p_node, _complexity))
                    # Pick the most complex candidate (most linked inputs)
                    if _candidates:
                        _candidates.sort(key=lambda x: x[1], reverse=True)
                        principled_node = _candidates[0][0]
                    break
                elif surface_node.type == 'ADD_SHADER':
                    # Add Shader: find the Principled BSDF and Emission in either input
                    _add_shader_emission_node = None
                    for si in range(len(surface_node.inputs)):
                        inp = surface_node.inputs[si]
                        if inp.is_linked:
                            s_node = inp.links[0].from_node
                            if s_node.type == 'BSDF_PRINCIPLED':
                                principled_node = s_node
                            elif s_node.type == 'EMISSION':
                                _add_shader_emission_node = s_node
                                # Extract emission from Add Shader (legacy)
                                e_props = _extract_shader_props(s_node, _register_image)
                                emission = e_props['emission']
                                emission_strength = e_props['emission_strength']
                                if e_props['diffuse_tex'] != _NO_TEX:
                                    emission_tex = e_props['diffuse_tex']
                    if principled_node:
                        break
                    # Fallback: follow first linked input
                    inp = surface_node.inputs[0] if surface_node.inputs else None
                    surface_node = inp.links[0].from_node if (inp and inp.is_linked) else None
                elif surface_node.type == 'GROUP':
                    proxy = _find_principled_in_group(surface_node)
                    if proxy:
                        principled_node = proxy
                        break
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
                elif surface_node.type in ('BSDF_DIFFUSE', 'BSDF_GLOSSY', 'BSDF_GLASS',
                                           'BSDF_TRANSPARENT', 'EMISSION'):
                    props = _extract_shader_props(surface_node, _register_image)
                    base_color = props['base_color']
                    roughness = props['roughness']
                    metallic = props['metallic']
                    emission = props['emission']
                    emission_strength = props['emission_strength']
                    diffuse_tex = props['diffuse_tex']
                    transmission = props.get('transmission', 0.0)
                    ior = props.get('ior', 1.5)
                    break
                else:
                    # Unknown node — follow first linked input
                    found = False
                    for inp in surface_node.inputs:
                        if inp.is_linked:
                            surface_node = inp.links[0].from_node
                            found = True
                            break
                    if not found:
                        surface_node = None
                _follow_depth += 1

            # Fallback: direct scan if Material Output chain didn't find Principled
            # Skip if Mix Shader already resolved properties correctly
            if principled_node is None and not _mix_shader_resolved:
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        principled_node = node
                        break
                    if node.type == 'GROUP':
                        proxy = _find_principled_in_group(node)
                        if proxy:
                            principled_node = proxy
                            break
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

            if principled_node is not None and not _mix_shader_resolved:
                node = principled_node
                # Base Color (resolve node chain if linked, otherwise use default)
                bc_inp = node.inputs.get('Base Color')
                if bc_inp and bc_inp.is_linked:
                    base_color = _resolve_color_input(bc_inp, (0.8, 0.8, 0.8))
                else:
                    bc = _get_principled_input(node, 'Base Color', (0.8, 0.8, 0.8, 1.0))
                    base_color = (bc[0], bc[1], bc[2])
                bc_node = _find_image_texture_node(node.inputs.get('Base Color'))
                if bc_node and bc_node.image:
                    bc_ramp = _find_color_ramp_in_chain(node.inputs.get('Base Color'))
                    diffuse_tex = _register_image(bc_node.image, bc_ramp)

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
                # Only use as ORM if BOTH come from the SAME texture (packed ORM),
                # or if only one is textured (assume it's a combined map).
                rough_node = _find_image_texture_node(node.inputs.get('Roughness'))
                metal_node = _find_image_texture_node(node.inputs.get('Metallic'))
                orm_image = None
                if rough_node and metal_node and rough_node.image and metal_node.image:
                    if rough_node.image == metal_node.image:
                        # Same texture = packed ORM
                        orm_image = rough_node.image
                    # Different textures = separate maps, VM handles them individually
                elif rough_node and rough_node.image and not (metal_node and metal_node.image):
                    # Only roughness textured, metallic is scalar — use as ORM
                    orm_image = rough_node.image
                elif metal_node and metal_node.image and not (rough_node and rough_node.image):
                    orm_image = metal_node.image
                if orm_image:
                    orm_tex = _register_image(orm_image)

                # Specular IOR Level (Blender 4.0+) or fallback to Specular
                spec_inp = node.inputs.get('Specular IOR Level')
                if spec_inp is None:
                    spec_inp = node.inputs.get('Specular')
                specular_level = float(spec_inp.default_value) if spec_inp else 0.5

                # Emission Color (resolve node chain if linked)
                ec_inp = node.inputs.get('Emission Color')
                if ec_inp and ec_inp.is_linked:
                    emission = _resolve_color_input(ec_inp, (1.0, 1.0, 1.0))
                else:
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

                # Detect alpha testing — only when Alpha is actually < 1 or linked
                alpha_test = alpha < 0.999
                alpha_inp = node.inputs.get('Alpha')
                if alpha_inp and alpha_inp.is_linked:
                    alpha_test = True
                # blend_method only matters if Alpha is variable (linked or < 1)
                if alpha_test:
                    try:
                        if mat.blend_method in ('CLIP', 'HASHED', 'BLEND'):
                            alpha_ref = getattr(mat, 'alpha_threshold', 0.5)
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

        # Compile VM bytecode — VM is the SINGLE AUTHORITY for material evaluation.
        # Legacy scalar path (ksAmbient, ksSpecularEXP, fresnelC) is DEPRECATED.
        # For Mix Shader: pass surface_node so the VM compiles BOTH branches
        # and blends per-pixel. No more scalar blending approximation.
        vm_code = None
        _vm_surface = _original_surface_node if _mix_shader_resolved else None
        _vm_emission = _add_shader_emission_node
        if principled_node is not None or _vm_surface is not None:
            vm_code = _compile_node_vm(principled_node, _register_image, surface_node=_vm_surface, emission_node=_vm_emission)
            if vm_code:
                mat_key_name = mat.name if mat else "?"
                print(f"[ignis-vm] '{mat_key_name}': compiled {len(vm_code)} instructions")
                # Dump VM instructions + Group internals for debugging
                import os as _os3
                try:
                    _opnames = {0x01:'SAMPLE_TEX',0x10:'UV_TRANSFORM',0x11:'UV_ROTATE',
                        0x20:'MIX',0x21:'MIX_REG',0x22:'MULTIPLY',0x2B:'HUE_SAT',
                        0x41:'MATH_ADD',0x42:'MATH_MUL',0x49:'MATH_SUB',
                        0x60:'LOAD_CONST',0x61:'LOAD_SCALAR',
                        0x62:'LOAD_WORLD_POS',0x80:'TEX_NOISE',0x93:'NOISE_BUMP',
                        0x95:'OBJECT_RANDOM',
                        0xEF:'OUTPUT_UV',0xF0:'OUTPUT_COLOR',0xF1:'OUTPUT_ROUGH',
                        0xF2:'OUTPUT_METAL',0xF3:'OUTPUT_EMISSION',0xF4:'OUTPUT_ALPHA',
                        0xF8:'OUTPUT_BUMP'}
                    with open(_os3.path.join(_os3.path.expanduser("~"), "ignis-rt.log"), "a") as _f:
                        _f.write(f"[ignis-vm-dump] '{mat_key_name}' ({len(vm_code)} instrs):\n")
                        for _i, _instr in enumerate(vm_code):
                            _op = _instr[0] & 0xFF
                            _opn = _opnames.get(_op, f'0x{_op:02X}')
                            _dst = (_instr[0] >> 8) & 0x1F
                            _srcA = (_instr[0] >> 16) & 0x1F
                            _f.write(f"  [{_i:2d}] {_opn} dst=R[{_dst}] srcA=R[{_srcA}]\n")
                        # Dump Group internal tree if Mix Shader
                        if _vm_surface and _vm_surface.type == 'MIX_SHADER':
                            for si in (1, 2):
                                sinp = _vm_surface.inputs[si] if len(_vm_surface.inputs) > si else None
                                if sinp and sinp.is_linked:
                                    sn = sinp.links[0].from_node
                                    if sn.type == 'GROUP' and sn.node_tree:
                                        _f.write(f"  [Group '{sn.name}' internal tree]:\n")
                                        for gn in sn.node_tree.nodes:
                                            _f.write(f"    {gn.type}: '{gn.name}'\n")
                                            for gi in gn.inputs:
                                                lnk = f" <- {gi.links[0].from_node.name}:{gi.links[0].from_socket.name}" if gi.is_linked else ""
                                                gv = ""
                                                if not gi.is_linked:
                                                    try:
                                                        v = gi.default_value
                                                        gv = f" = {v:.4f}" if not hasattr(v, '__len__') else f" = ({','.join(f'{x:.3f}' for x in v)})"
                                                    except: pass
                                                _f.write(f"      in: '{gi.name}'{gv}{lnk}\n")
                except: pass
                # Only disable legacy UV scale when VM emits UV transforms
                # (OP_OUTPUT_UV = 0xEF). Otherwise legacy scale is still needed.
                if any((instr[0] & 0xFF) == 0xEF for instr in vm_code):
                    uv_scale_x = 1.0
                    uv_scale_y = 1.0
                # When VM emits OP_OUTPUT_ALPHA (0xF4), it handles alpha per-pixel.
                # Disable stochastic passthrough and enable alpha cutout instead.
                if any((instr[0] & 0xFF) == 0xF4 for instr in vm_code):
                    transparent_prob = 0.0
                    flags |= 1  # enable alpha_test
                    alpha = 1.0  # let VM alpha be sole factor

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
                _vm_tag = f" vm={len(vm_code)}instr" if vm_code else " vm=none"
                _mf.write(f"[ignis-mat] '{mat.name}': color=({base_color[0]:.3f},{base_color[1]:.3f},{base_color[2]:.3f}) "
                          f"rough={roughness:.2f} metal={metallic:.2f} trans={transmission:.2f} ior={ior:.2f} "
                          f"emit=({emission[0]:.3f},{emission[1]:.3f},{emission[2]:.3f})*{emission_strength:.2f} "
                          f"diffTex={diffuse_tex} normTex={normal_tex} normStr={normal_strength:.2f} "
                          f"emitTex={emission_tex} flags={flags} tp={transparent_prob:.2f}"
                          f" uvScale=({uv_scale_x:.2f},{uv_scale_y:.2f}){_vm_tag}\n")
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
            node_vm_code=vm_code,
            volume_density=volume_density,
            volume_anisotropy=volume_anisotropy,
            volume_noise_scale=volume_noise_scale,
            volume_noise_detail=volume_noise_detail,
            volume_noise_brightness=volume_noise_brightness,
            volume_noise_roughness=volume_noise_roughness,
            volume_noise_lacunarity=volume_noise_lacunarity,
            volume_noise_contrast=volume_noise_contrast,
            volume_mapping_offset=volume_mapping_offset,
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
