"""Debug particle systems. Run in Blender Text Editor (Alt+P)."""
import bpy
import numpy as np

print("="*60)
print("PARTICLE SYSTEM DEBUG")
print("="*60)

for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    for ps_idx, ps in enumerate(eval_obj.particle_systems):
        if ps.settings.type != 'HAIR':
            continue
        s = ps.settings
        print(f"\nObject: '{obj.name}' → ParticleSystem[{ps_idx}]: '{ps.name}'")
        print(f"  type={s.type} render_as={s.render_type}")
        print(f"  count (parents)={s.count}")
        print(f"  child_type={s.child_type} child_nbr={s.child_nbr} rendered_child_count={s.rendered_child_count}")
        print(f"  display_step={s.display_step} render_step={s.render_step}")
        print(f"  radius_scale={s.radius_scale}")
        print(f"  root_radius={getattr(s, 'root_radius', '?')}")
        print(f"  tip_radius={getattr(s, 'tip_radius', '?')}")
        print(f"  shape={getattr(s, 'shape', '?')}")
        print(f"  totpart={ps.particles.__len__() if ps.particles else 0}")
        print(f"  child_particles={len(ps.child_particles) if hasattr(ps, 'child_particles') else '?'}")

        # Check first parent strand
        if len(ps.particles) > 0:
            p = ps.particles[0]
            n_keys = len(p.hair_keys)
            print(f"  parent[0] keys={n_keys}")
            if n_keys >= 2:
                # Find modifier
                ps_mod = None
                for mod in eval_obj.modifiers:
                    if mod.type == 'PARTICLE_SYSTEM' and mod.particle_system == ps:
                        ps_mod = mod
                        break
                if ps_mod:
                    root = p.hair_keys[0].co_object(eval_obj, ps_mod, p)
                    tip = p.hair_keys[-1].co_object(eval_obj, ps_mod, p)
                    print(f"    co_object root=({root[0]:.4f},{root[1]:.4f},{root[2]:.4f})")
                    print(f"    co_object tip=({tip[0]:.4f},{tip[1]:.4f},{tip[2]:.4f})")
                root_co = p.hair_keys[0].co
                tip_co = p.hair_keys[-1].co
                print(f"    co (local) root=({root_co[0]:.4f},{root_co[1]:.4f},{root_co[2]:.4f})")
                print(f"    co (local) tip=({tip_co[0]:.4f},{tip_co[1]:.4f},{tip_co[2]:.4f})")

        # Check if we can access child particle data
        if hasattr(ps, 'child_particles') and len(ps.child_particles) > 0:
            print(f"  child_particles[0] attrs: {[a for a in dir(ps.child_particles[0]) if not a.startswith('_')]}")

        # Check new Curves API (Blender 4.0+)
        print(f"\n  Checking Curves API:")
        for child_obj in bpy.data.objects:
            if child_obj.type == 'CURVES' and child_obj.parent == obj:
                curves = child_obj.data
                print(f"    Found CURVES object: '{child_obj.name}'")
                print(f"    curves_num={len(curves.curves)} points_num={len(curves.points)}")
                if len(curves.curves) > 0:
                    c0 = curves.curves[0]
                    print(f"    curve[0]: first_point_index={c0.first_point_index} points_length={c0.points_length}")
                    if c0.points_length > 0:
                        p0 = curves.points[c0.first_point_index]
                        print(f"    point[0]: position=({p0.position[0]:.4f},{p0.position[1]:.4f},{p0.position[2]:.4f}) radius={p0.radius:.4f}")

print(f"\n{'='*60}")
print("Done.")

if False:

MATERIAL_NAME = "Cristal"
mat = bpy.data.materials.get(MATERIAL_NAME)
if not mat:
    print(f"ERROR: '{MATERIAL_NAME}' not found. Available: {[m.name for m in bpy.data.materials]}")
elif mat.use_nodes and mat.node_tree:
    tree = mat.node_tree
    print(f"\n{'='*60}")
    print(f"Material: '{mat.name}'")
    print(f"  blend_method: {getattr(mat, 'blend_method', '?')}")
    print(f"  use_backface_culling: {mat.use_backface_culling}")
    print(f"{'='*60}")
    print(f"\nNodes ({len(tree.nodes)}):")
    for node in tree.nodes:
        print(f"\n  [{node.type}] '{node.name}'")
        for inp in node.inputs:
            if inp.is_linked:
                fn = inp.links[0].from_node
                fs = inp.links[0].from_socket
                print(f"    in: '{inp.name}' <- [{fn.type}] '{fn.name}'.'{fs.name}'")
            else:
                try:
                    v = inp.default_value
                    if hasattr(v, '__len__'):
                        print(f"    in: '{inp.name}' = ({', '.join(f'{x:.4f}' for x in v)})")
                    else:
                        print(f"    in: '{inp.name}' = {v:.4f}")
                except:
                    print(f"    in: '{inp.name}' = ?")
        for out in node.outputs:
            if out.is_linked:
                for lnk in out.links:
                    print(f"    out: '{out.name}' -> [{lnk.to_node.type}] '{lnk.to_node.name}'.'{lnk.to_socket.name}'")
    print(f"{'='*60}")

# ---- Original audit script below (not executed) ----
if False:
 FULLY_SUPPORTED = {
    'BSDF_PRINCIPLED', 'OUTPUT_MATERIAL',
    'TEX_IMAGE', 'TEX_COORD', 'MAPPING',
    'VALTORGB',  # ColorRamp
    'MIX_RGB', 'MIX',  # Mix Color
    'INVERT', 'GAMMA', 'BRIGHTCONTRAST', 'HUE_SAT',
    'NORMAL_MAP', 'BUMP',
    'RGB', 'VALUE',
    'MATH', 'CLAMP', 'MAP_RANGE',
    'SEPRGB', 'SEPARATE_COLOR', 'SEPARATE_XYZ',
    'COMBRGB', 'COMBINE_COLOR', 'COMBINE_XYZ',
    'RGBTOBW',
    'REROUTE', 'FRAME',  # organizational
    'TEX_CHECKER',  # procedural
    'AMBIENT_OCCLUSION',  # passthrough
    'BLACKBODY',
    'GROUP',  # node groups
    'DISPLACEMENT',  # not rendered but supported
    'MIX_SHADER', 'ADD_SHADER',
    'BSDF_DIFFUSE', 'BSDF_GLOSSY', 'BSDF_GLASS', 'BSDF_TRANSPARENT',
    'EMISSION',
}

# Node types with partial support
PARTIAL_SUPPORT = {
    'CURVE_RGB': 'Passthrough only — curves not evaluated',
    'CURVE_VEC': 'Passthrough only — curves not evaluated',
    'TEX_NOISE': 'Constant gray fallback',
    'TEX_VORONOI': 'Constant gray fallback',
    'TEX_MUSGRAVE': 'Constant gray fallback',
    'TEX_WAVE': 'Constant gray fallback',
    'TEX_GRADIENT': 'Constant gray fallback',
    'TEX_MAGIC': 'Constant gray fallback',
    'TEX_BRICK': 'Constant gray fallback',
    'TEX_SKY': 'Constant dim blue fallback',
    'TEX_ENVIRONMENT': 'Handled separately for HDRI',
}

print("=" * 70)
print("MATERIAL AUDIT — All materials in scene")
print("=" * 70)

unsupported_summary = {}
partial_summary = {}

for mat in bpy.data.materials:
    if not mat.node_tree:
        continue

    issues = []
    for node in mat.node_tree.nodes:
        if node.type in FULLY_SUPPORTED:
            continue
        elif node.type in PARTIAL_SUPPORT:
            issues.append(f"  PARTIAL: {node.name} ({node.type}) — {PARTIAL_SUPPORT[node.type]}")
            partial_summary[node.type] = partial_summary.get(node.type, 0) + 1
        else:
            issues.append(f"  UNSUPPORTED: {node.name} ({node.type})")
            unsupported_summary[node.type] = unsupported_summary.get(node.type, 0) + 1

    # Check for specific node configurations
    for node in mat.node_tree.nodes:
        if node.type == 'CURVE_RGB':
            mapping = node.mapping
            mapping.initialize()
            mapping.update()
            # Check if curves are non-trivial (not identity)
            is_identity = True
            for ci, curve in enumerate(mapping.curves):
                for p in curve.points:
                    if abs(p.location.x - p.location.y) > 0.01:
                        is_identity = False
                        break
            if not is_identity:
                issues.append(f"  ACTIVE CURVES: {node.name} — curves modify color (not identity)")
                # Show curve data
                for ci, curve in enumerate(mapping.curves):
                    names = ['Combined', 'Red', 'Green', 'Blue']
                    name = names[ci] if ci < len(names) else f"Ch{ci}"
                    pts = [(round(p.location.x, 3), round(p.location.y, 3)) for p in curve.points]
                    if any(abs(p[0] - p[1]) > 0.01 for p in pts):
                        print_pts = ' '.join(f"({p[0]},{p[1]})" for p in pts)
                        issues.append(f"    {name}: {print_pts}")

        if node.type == 'MAPPING':
            scale = node.inputs.get('Scale')
            rot = node.inputs.get('Rotation')
            if scale and rot:
                has_scale = any(abs(scale.default_value[i] - 1.0) > 0.001 for i in range(3))
                has_rot = abs(rot.default_value[2]) > 0.001
                if has_scale and has_rot:
                    issues.append(f"  KNOWN BUG: {node.name} — Scale+Rotation combined (V-flip issue)")

    if issues:
        print(f"\n{mat.name}:")
        for issue in issues:
            print(issue)

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
if unsupported_summary:
    print("\nUnsupported node types:")
    for ntype, count in sorted(unsupported_summary.items(), key=lambda x: -x[1]):
        print(f"  {ntype}: {count} instances")
if partial_summary:
    print("\nPartially supported node types:")
    for ntype, count in sorted(partial_summary.items(), key=lambda x: -x[1]):
        print(f"  {ntype}: {count} instances — {PARTIAL_SUPPORT[ntype]}")
if not unsupported_summary and not partial_summary:
    print("\nAll nodes fully supported!")
