"""
Ignis RT — Material dump for diagnostics.

Uso en Blender:
  1. Workspace "Scripting" (o Text Editor).
  2. Pegá / abrí este archivo.
  3. Editá MATERIAL_NAME abajo con el nombre del material a inspeccionar
     (o dejalo en "" para usar el material activo del objeto seleccionado).
  4. Run Script (Alt+P).
  5. Copiá TODO lo que salga en la consola (Window > Toggle System Console) y pasámelo.

No modifica nada — solo lee e imprime.
"""

import bpy

# ── Editá esto ───────────────────────────────────────────────
MATERIAL_NAME = ""   # ej: "Vidrio", "Glass.001". Vacío = material activo del objeto seleccionado.
# ─────────────────────────────────────────────────────────────


def _val(inp):
    """Representación legible del default_value de un socket."""
    try:
        v = inp.default_value
    except Exception:
        return "<sin default>"
    try:
        return tuple(round(float(x), 5) for x in v)  # color/vector
    except TypeError:
        try:
            return round(float(v), 5)
        except Exception:
            return repr(v)


def _socket_src(inp):
    """De dónde viene un input linkeado: 'NodeType.SocketName'."""
    if not inp.is_linked or not inp.links:
        return None
    ln = inp.links[0]
    return f"{ln.from_node.bl_idname}('{ln.from_node.name}').{ln.from_socket.name}"


def dump_material(mat):
    print("=" * 70)
    print(f"MATERIAL: '{mat.name}'")
    print("=" * 70)

    # Settings de material (Eevee/legacy) que delatan intención de transparencia.
    for attr in ("blend_method", "shadow_method", "use_backface_culling",
                 "use_screen_refraction", "refraction_depth", "alpha_threshold",
                 "use_raytrace", "diffuse_color"):
        if hasattr(mat, attr):
            print(f"  mat.{attr:22} = {getattr(mat, attr)}")

    if not mat.use_nodes or mat.node_tree is None:
        print("  (sin nodos — material legacy)")
        return

    nt = mat.node_tree

    # Output activo y su surface.
    out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"
                and n.is_active_output), None)
    if out is None:
        out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    surf = None
    if out and out.inputs.get("Surface") and out.inputs["Surface"].is_linked:
        surf = out.inputs["Surface"].links[0].from_node
    print(f"\n  SURFACE OUTPUT -> {surf.bl_idname if surf else None}"
          f" ('{surf.name}')" if surf else "\n  SURFACE OUTPUT -> None")

    # Resumen de tipos de nodo presentes (rápido para ver Glass/Transparent/Mix/LightPath).
    from collections import Counter
    kinds = Counter(n.bl_idname for n in nt.nodes)
    print("\n  NODOS PRESENTES:")
    for k, c in sorted(kinds.items()):
        print(f"    {c:2} x {k}")

    flag_nodes = [n.bl_idname for n in nt.nodes if n.bl_idname in (
        "ShaderNodeBsdfGlass", "ShaderNodeBsdfTransparent", "ShaderNodeBsdfRefraction",
        "ShaderNodeMixShader", "ShaderNodeAddShader", "ShaderNodeLightPath",
        "ShaderNodeVolumeAbsorption", "ShaderNodeVolumeScatter", "ShaderNodeBsdfPrincipled")]
    print(f"\n  RELEVANTES PARA VIDRIO/TRANSPARENCIA: {flag_nodes}")

    # Detalle completo de cada nodo: inputs (valor o de dónde vienen).
    print("\n  ── DETALLE DE NODOS ──")
    for n in nt.nodes:
        title = f"  [{n.bl_idname}] '{n.name}'"
        extra = []
        for a in ("distribution", "operation", "blend_type", "data_type",
                  "interpolation", "projection", "image"):
            if hasattr(n, a):
                av = getattr(n, a)
                if a == "image":
                    av = (av.name + f" {tuple(av.size)} cs={av.colorspace_settings.name}") if av else None
                extra.append(f"{a}={av}")
        if extra:
            title += "   (" + ", ".join(extra) + ")"
        print(title)
        for inp in n.inputs:
            src = _socket_src(inp)
            if src:
                print(f"        {inp.name:24} <- {src}")
            else:
                print(f"        {inp.name:24} =  {_val(inp)}")

    # Foco en el Principled (lo que el addon lee directo).
    pr = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if pr:
        print("\n  ── PRINCIPLED BSDF (campos que lee el renderer) ──")
        for name in ("Base Color", "Metallic", "Roughness", "IOR", "Alpha",
                     "Transmission Weight", "Transmission", "Specular IOR Level",
                     "Specular", "Coat Weight", "Emission Color", "Emission Strength",
                     "Normal"):
            inp = pr.inputs.get(name)
            if inp is None:
                continue
            src = _socket_src(inp)
            print(f"    {name:22} = {_val(inp)}" + (f"   <- {src}" if src else ""))
    # UV layers de los objetos que usan este material (clave para texturas "mal mapeadas":
    # assets "baked" suelen tener 2 capas UV y la ACTIVA puede no ser la de textura).
    print("\n  ── CAPAS UV de los objetos que usan este material ──")
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if not any(s.material == mat for s in obj.material_slots):
            continue
        me = obj.data
        layers = []
        for uvl in me.uv_layers:
            tag = []
            if uvl.active: tag.append("ACTIVA")
            if getattr(uvl, "active_render", False): tag.append("RENDER")
            layers.append(f"'{uvl.name}'" + ("(" + ",".join(tag) + ")" if tag else ""))
        print(f"    {obj.name}: {len(me.uv_layers)} capas -> {layers}")
    # Fuente UV de cada nodo Image Texture (UVMap node? Mapping? directo?)
    if mat.use_nodes and mat.node_tree:
        print("\n  ── Fuente de coordenadas de cada Image Texture ──")
        for n in mat.node_tree.nodes:
            if n.bl_idname != "ShaderNodeTexImage":
                continue
            vinp = n.inputs.get("Vector")
            if vinp and vinp.is_linked:
                src = vinp.links[0].from_node
                detail = ""
                if src.bl_idname == "ShaderNodeUVMap":
                    detail = f" uv_map='{src.uv_map}'"
                elif src.bl_idname == "ShaderNodeAttribute":
                    detail = f" attr='{getattr(src,'attribute_name','?')}'"
                print(f"    '{n.name}' <- {src.bl_idname}{detail}")
            else:
                print(f"    '{n.name}' <- (UV activa por defecto)")
    print("=" * 70)
    print("FIN — copiá todo lo de arriba.")
    print("=" * 70)


def main():
    mat = None
    if MATERIAL_NAME:
        mat = bpy.data.materials.get(MATERIAL_NAME)
        if mat is None:
            print(f"!! No existe el material '{MATERIAL_NAME}'.")
            print("   Materiales en la escena:")
            for m in bpy.data.materials:
                print(f"     - {m.name}")
            return
    else:
        obj = bpy.context.active_object
        if obj and obj.active_material:
            mat = obj.active_material
            print(f"(MATERIAL_NAME vacío -> uso el activo de '{obj.name}')")
        else:
            print("!! No hay material activo. Seleccioná un objeto o poné MATERIAL_NAME.")
            return
    dump_material(mat)


main()
