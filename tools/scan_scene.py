"""
Ignis RT — Scene scanner for diagnostics.

Vuelca TODOS los materiales de la escena con los props que el addon EXPORTA
(lo que realmente llega al renderer) + la asignacion material<->objeto + flags
de geometria (escala no uniforme, normales). Sirve para detectar problemas
sistematicos de materiales, no de uno solo.

Uso en Blender:
  1. Scripting workspace.
  2. Open/pega este archivo.
  3. Run Script (Alt+P).
  4. Window > Toggle System Console, copia TODO y pasamelo.

Read-only. No modifica nada.
"""

import bpy

# ── intentar importar el modulo de export del addon ──────────────
_se = None
for modname in ("ignis_rt.scene_export", "scene_export"):
    try:
        _se = __import__(modname, fromlist=["_extract_shader_props"])
        break
    except Exception:
        continue
if _se is None:
    # buscar el addon entre los registrados
    import addon_utils
    for mod in addon_utils.modules():
        if "ignis" in mod.__name__.lower():
            try:
                _se = __import__(mod.__name__ + ".scene_export",
                                 fromlist=["_extract_shader_props"])
                break
            except Exception:
                pass

print("=" * 78)
print("IGNIS RT — SCENE SCAN")
print("=" * 78)
print(f"addon scene_export importado: {bool(_se)}")
print(f"Blender: {bpy.app.version_string}   Escena: '{bpy.context.scene.name}'")

# ── stub de registro de imagenes (para llamar al export sin GPU) ──
_img_names = []
def _stub_register(image):
    if image is None:
        return 0xFFFFFFFF
    nm = getattr(image, "name", "?")
    if nm not in _img_names:
        _img_names.append(nm)
    return _img_names.index(nm)


def _surface_node(mat):
    if not mat.use_nodes or mat.node_tree is None:
        return None
    if _se and hasattr(_se, "_find_surface_shader"):
        try:
            return _se._find_surface_shader(mat.node_tree)
        except Exception:
            pass
    for n in mat.node_tree.nodes:
        if n.type == 'OUTPUT_MATERIAL' and n.is_active_output:
            si = n.inputs.get('Surface')
            if si and si.is_linked:
                return si.links[0].from_node
    return None


def _exported_props(mat):
    """Props que el addon exportaria para este material (o None)."""
    if not _se or not hasattr(_se, "_extract_shader_props"):
        return None
    surf = _surface_node(mat)
    if surf is None:
        return None
    try:
        return _se._extract_shader_props(surf, _stub_register)
    except Exception as e:
        return {"_error": repr(e)}


# ── 1. Materiales ────────────────────────────────────────────────
print("\n" + "-" * 78)
print("MATERIALES (props EXPORTADOS por el addon)")
print("-" * 78)

users = {}  # material -> set(objetos)
for obj in bpy.data.objects:
    for slot in getattr(obj, "material_slots", []):
        if slot.material:
            users.setdefault(slot.material.name, set()).add(obj.name)

for mat in bpy.data.materials:
    if mat.users == 0:
        continue
    surf = _surface_node(mat)
    surf_type = surf.bl_idname if surf else "—"
    print(f"\n■ '{mat.name}'   surface={surf_type}   blend={getattr(mat,'blend_method','?')}")
    # tipos de nodo presentes (compacto)
    if mat.use_nodes and mat.node_tree:
        from collections import Counter
        kinds = Counter(n.bl_idname.replace("ShaderNode", "") for n in mat.node_tree.nodes)
        print("    nodos: " + ", ".join(f"{k}x{v}" for k, v in sorted(kinds.items())))
    pr = _exported_props(mat)
    if pr is None:
        print("    (no se pudo extraer props exportados)")
    elif "_error" in pr:
        print(f"    EXPORT ERROR: {pr['_error']}")
    else:
        def g(k, d=None):
            return pr.get(k, d)
        def r(v):
            try: return tuple(round(float(x), 4) for x in v)
            except TypeError: return round(float(v), 4) if v is not None else None
        print(f"    base_color={r(g('base_color'))}  metallic={r(g('metallic'))}  roughness={r(g('roughness'))}")
        print(f"    ior={r(g('ior'))}  transmission={r(g('transmission'))}  transparent_prob={r(g('transparent_prob'))}  spec={r(g('specular_level'))}")
        print(f"    emission={r(g('emission'))} x{r(g('emission_strength'))}")
        print(f"    tex: diffuse={g('diffuse_tex')} normal={g('normal_tex')}(str {r(g('normal_strength'))}) emission={g('emission_tex')}")
        # cualquier clave extra interesante
        extra = {k: g(k) for k in pr.keys() if k not in (
            'base_color','metallic','roughness','emission','emission_strength',
            'diffuse_tex','normal_tex','normal_strength','emission_tex','transmission',
            'ior','transparent_prob','specular_level','hair_shift','hair_radial_roughness','hair_material')}
        if extra:
            print(f"    extra: {extra}")
    print(f"    usado por: {sorted(users.get(mat.name, []))}")

# ── 2. Objetos + slots + geometria ──────────────────────────────
print("\n" + "-" * 78)
print("OBJETOS (slots de material + geometria)")
print("-" * 78)
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH':
        continue
    me = obj.data
    sx, sy, sz = obj.matrix_world.to_scale()
    nonuniform = (abs(sx - sy) > 1e-3 or abs(sy - sz) > 1e-3)
    slots = [s.material.name if s.material else "—" for s in obj.material_slots]
    # auto smooth / custom split normals
    has_split = me.has_custom_normals if hasattr(me, "has_custom_normals") else "?"
    print(f"\n● '{obj.name}'  verts={len(me.vertices)} polys={len(me.polygons)}")
    print(f"    scale=({sx:.3f},{sy:.3f},{sz:.3f})" + ("  <<< NO UNIFORME" if nonuniform else ""))
    print(f"    custom_split_normals={has_split}")
    print(f"    material_slots={slots}")
    # cuantos poligonos por slot (asignacion)
    if len(obj.material_slots) > 1:
        from collections import Counter
        per = Counter(p.material_index for p in me.polygons)
        print(f"    polys por slot index: {dict(sorted(per.items()))}")

print("\n" + "=" * 78)
print(f"Imagenes referenciadas ({len(_img_names)}): {_img_names}")
print("FIN — copia todo.")
print("=" * 78)
