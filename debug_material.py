"""Full material audit: check ALL materials for unsupported/partially supported nodes."""
import bpy

# Node types we fully support
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
