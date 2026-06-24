"""
Dump a material's FULL shader node graph for diagnosis (surface tree + the colour/value
graph feeding every BSDF input: textures, Mix/Math, Bump, ColorRamp, etc.).

HOW TO RUN (Blender):
  1. Window > Toggle System Console   (so you can see + copy the output, Windows)
  2. Scripting tab > New > paste this whole file (or open it)
  3. Set MATERIAL_NAME below (or "" = active object's active material)
  4. Run (Alt+P, or the ▶ button)
  5. Copy the whole printed block from the System Console and paste it back to Claude.
"""
import bpy

# ── SET YOUR MATERIAL NAME HERE (or "" to use the active object's active material) ──
MATERIAL_NAME = ""


def sval(sock):
    try:
        v = sock.default_value
        if hasattr(v, '__len__'):
            return "(" + ", ".join(f"{x:.4f}" for x in v) + ")"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    except Exception:
        return "<n/a>"


def link_src(sock):
    l = sock.links[0]
    return f"[{l.from_node.type}] '{l.from_node.name}' .{l.from_socket.name}"


def node_props(node, pad):
    """Print node-specific settings that affect the result but aren't sockets."""
    t = node.type
    if t == 'BUMP':
        print(pad + f"  «props» invert={node.invert}")
    elif t in ('MIX_RGB', 'MIX'):
        bt = getattr(node, 'blend_type', None) or getattr(node, 'data_type', '?')
        print(pad + f"  «props» blend_type={bt} clamp={getattr(node,'use_clamp', getattr(node,'clamp_result','?'))}")
    elif t == 'TEX_IMAGE' and node.image:
        print(pad + f"  «props» image='{node.image.name}' colorspace={node.image.colorspace_settings.name} "
                    f"projection={node.projection} extension={node.extension}")
    elif t == 'MATH':
        print(pad + f"  «props» operation={node.operation} clamp={node.use_clamp}")
    elif t == 'VALTORGB':
        els = node.color_ramp.elements
        print(pad + f"  «props» COLORRAMP interp={node.color_ramp.interpolation} stops=" +
              ", ".join(f"{e.position:.3f}:({e.color[0]:.2f},{e.color[1]:.2f},{e.color[2]:.2f})" for e in els))
    elif t == 'CURVE_RGB':
        print(pad + f"  «props» RGB CURVES (mapping)")
    elif t in ('TEX_NOISE', 'TEX_VORONOI', 'TEX_WAVE', 'TEX_MAGIC', 'TEX_GRADIENT', 'TEX_BRICK', 'TEX_CHECKER'):
        extra = []
        for a in ('noise_dimensions', 'voronoi_dimensions', 'feature', 'distance', 'wave_type',
                  'wave_profile', 'gradient_type', 'turbulence_depth'):
            if hasattr(node, a):
                extra.append(f"{a}={getattr(node, a)}")
        if extra:
            print(pad + "  «props» " + " ".join(extra))


def walk_value(node, depth, visited):
    pad = "        " + "  " * depth
    if node is None or depth > 16:
        return
    if node.type == 'REROUTE':
        s = node.inputs[0]
        walk_value(s.links[0].from_node if s.is_linked else None, depth, visited)
        return
    if node.name in visited:
        print(pad + f"[{node.type}] '{node.name}' (shown above)")
        return
    visited.add(node.name)
    print(pad + f"[{node.type}] '{node.name}'")
    node_props(node, pad)
    for inp in node.inputs:
        if inp.is_linked:
            print(pad + f"  {inp.name} <- " + link_src(inp))
            walk_value(inp.links[0].from_node, depth + 1, visited)
        else:
            print(pad + f"  {inp.name} = {sval(inp)}")


def branch(node, depth, visited):
    pad = "    " + "  " * depth
    if depth > 12:
        print(pad + "... (depth cap)"); return
    if node is None:
        print(pad + "<none>"); return
    if node.type == 'REROUTE':
        s = node.inputs[0]
        branch(s.links[0].from_node if s.is_linked else None, depth, visited); return
    print(pad + f"[{node.type}] '{node.name}'")
    if node.type == 'MIX_SHADER':
        fac = node.inputs[0]
        print(pad + "  Fac = " + (("LINKED <- " + link_src(fac)) if fac.is_linked else sval(fac)))
        if fac.is_linked:
            walk_value(fac.links[0].from_node, depth + 1, visited)
        for i, label in ((1, "A (fac=0 side)"), (2, "B (fac=1 side)")):
            s = node.inputs[i]
            print(pad + f"  {label}:")
            branch(s.links[0].from_node if s.is_linked else None, depth + 2, visited)
    elif node.type == 'ADD_SHADER':
        for i, label in ((0, "A"), (1, "B")):
            s = node.inputs[i]
            print(pad + f"  {label}:")
            branch(s.links[0].from_node if s.is_linked else None, depth + 2, visited)
    elif node.type == 'GROUP':
        print(pad + "  (NODE GROUP — real shader is inside; tell Claude the group name)")
    else:
        node_props(node, pad)
        for inp in node.inputs:
            if inp.type == 'SHADER':
                continue
            if inp.is_linked:
                print(pad + f"  {inp.name} <- " + link_src(inp))
                walk_value(inp.links[0].from_node, depth + 2, visited)
            else:
                print(pad + f"  {inp.name} = {sval(inp)}")


def dump(mat):
    print("=" * 72)
    print(f"MATERIAL: '{mat.name}'  use_nodes={mat.use_nodes}")
    if not (mat.use_nodes and mat.node_tree):
        print("  (no node tree)"); print("=" * 72); return
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None) \
        or next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if out is None:
        print("  no Material Output node"); print("=" * 72); return
    surf = out.inputs.get('Surface')
    if not (surf and surf.is_linked):
        print("  Surface not linked"); print("=" * 72); return
    print("  SURFACE TREE (+ colour/value graph):")
    branch(surf.links[0].from_node, 0, set())
    print("=" * 72)


_mat = bpy.data.materials.get(MATERIAL_NAME) if MATERIAL_NAME else None
if _mat is None and not MATERIAL_NAME:
    _o = bpy.context.active_object
    _mat = _o.active_material if (_o and _o.active_material) else None
if _mat is None:
    print(f"Material '{MATERIAL_NAME}' not found. Available materials:")
    for _m in bpy.data.materials:
        print("  -", _m.name)
else:
    dump(_mat)
