"""Debug one Blender material and dump both node-tree info and Ignis export data.

Usage:
1. Blender Text Editor:
   - Open this file
   - Set MATERIAL_NAME below, or leave it empty to use the active material
   - Run with Alt+P

2. Command line:
   blender --background your_scene.blend --python debug_material.py -- "Material Name"

Output:
- Printed to the Blender console
- Written to ~/ignis-material-<material>.txt
"""

from __future__ import annotations

import difflib
import os
import sys
from importlib import import_module
from io import StringIO
from pathlib import Path

import bpy


MATERIAL_NAME = ""


FULLY_SUPPORTED = {
    "BSDF_PRINCIPLED",
    "BSDF_HAIR_PRINCIPLED",
    "BSDF_HAIR",
    "OUTPUT_MATERIAL",
    "TEX_IMAGE",
    "TEX_COORD",
    "MAPPING",
    "VALTORGB",
    "MIX_RGB",
    "MIX",
    "INVERT",
    "GAMMA",
    "BRIGHTCONTRAST",
    "HUE_SAT",
    "NORMAL_MAP",
    "BUMP",
    "RGB",
    "VALUE",
    "MATH",
    "CLAMP",
    "MAP_RANGE",
    "SEPRGB",
    "SEPARATE_COLOR",
    "SEPARATE_XYZ",
    "COMBRGB",
    "COMBINE_COLOR",
    "COMBINE_XYZ",
    "RGBTOBW",
    "REROUTE",
    "FRAME",
    "TEX_CHECKER",
    "AMBIENT_OCCLUSION",
    "BLACKBODY",
    "GROUP",
    "DISPLACEMENT",
    "MIX_SHADER",
    "ADD_SHADER",
    "BSDF_DIFFUSE",
    "BSDF_GLOSSY",
    "BSDF_GLASS",
    "BSDF_TRANSPARENT",
    "EMISSION",
}


PARTIAL_SUPPORT = {
    "CURVE_RGB": "Passthrough only - curves not evaluated",
    "CURVE_VEC": "Passthrough only - curves not evaluated",
    "TEX_NOISE": "Constant gray fallback or VM-specific path",
    "TEX_VORONOI": "Constant gray fallback",
    "TEX_MUSGRAVE": "Constant gray fallback or volume path",
    "TEX_WAVE": "Constant gray fallback",
    "TEX_GRADIENT": "Constant gray fallback",
    "TEX_MAGIC": "Constant gray fallback",
    "TEX_BRICK": "Constant gray fallback",
    "TEX_SKY": "Handled separately for HDRI",
    "TEX_ENVIRONMENT": "Handled separately for HDRI",
}


GPU_FIELD_NAMES = [
    "diffuseTexIndex",
    "normalTexIndex",
    "mapsTexIndex",
    "detailTexIndex",
    "normalDetailTexIndex",
    "ksAmbient_r",
    "ksAmbient_g",
    "ksAmbient_b",
    "ksSpecularEXP_roughness",
    "emissive_r",
    "emissive_g",
    "emissive_b",
    "fresnelC_metallic",
    "fresnelEXP_specularLevel",
    "detailUVMult_ior",
    "detailNormalBlend_normalStrength",
    "flags",
    "alphaRef",
    "shaderType",
    "fresnelMaxLevel_emissionStrength",
    "mlTex0",
    "mlTex1",
    "mlTex2",
    "mlTex3",
    "mlTex4",
    "mlTex5",
    "multR_transmission",
    "multG_alpha",
    "multB_transparentProb",
    "uv_scale_x",
    "uv_scale_y",
    "color_value",
    "color_saturation",
    "sunSpecular_extra0",
    "sunSpecularEXP_extra1",
]


def _repo_root() -> Path:
    if "__file__" in globals():
        try:
            p = Path(__file__).resolve()
            if p.exists():
                return p.parent
        except Exception:
            pass
    return Path.cwd()


def _candidate_import_roots() -> list[Path]:
    roots: list[Path] = []

    repo = _repo_root()
    roots.append(repo / "blender")

    # Deployed Blender extension path.
    appdata = os.environ.get("APPDATA")
    if appdata:
        roots.append(Path(appdata) / "Blender Foundation" / "Blender" / "5.1" / "extensions" / "user_default")

    # If the deployed addon exists, use its _deploy_root.txt to get back to the source repo.
    deploy_root_txt = (
        Path(appdata) / "Blender Foundation" / "Blender" / "5.1" / "extensions" / "user_default" / "ignis_rt" / "_deploy_root.txt"
        if appdata
        else None
    )
    if deploy_root_txt and deploy_root_txt.exists():
        try:
            deployed_repo = Path(deploy_root_txt.read_text(encoding="utf-8").strip())
            roots.append(deployed_repo / "blender")
        except Exception:
            pass

    # Running from a repo checkout directly.
    for candidate in (
        repo,
        Path.cwd(),
        Path.cwd().parent,
    ):
        roots.append(candidate / "blender")

    out: list[Path] = []
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "ignis_rt" / "scene_export.py").exists():
            out.append(resolved)
    return out


def _load_scene_export():
    try:
        from ignis_rt import scene_export  # type: ignore

        return scene_export
    except Exception:
        errors = []
        for root in _candidate_import_roots():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            try:
                return import_module("ignis_rt.scene_export")
            except Exception as exc:
                errors.append(f"{root}: {exc}")
        raise ModuleNotFoundError(
            "No se pudo importar ignis_rt.scene_export. Rutas probadas:\n  - "
            + "\n  - ".join(errors or ["<ninguna>"])
        )


def _parse_material_name() -> str:
    argv = sys.argv
    if "--" in argv:
        extra = argv[argv.index("--") + 1 :]
        if extra:
            return extra[0]
    if MATERIAL_NAME.strip():
        return MATERIAL_NAME.strip()
    obj = bpy.context.active_object
    if obj and getattr(obj, "active_material", None):
        return obj.active_material.name
    return ""


def _fmt_value(value) -> str:
    try:
        if isinstance(value, str):
            return value
        if hasattr(value, "__len__") and not isinstance(value, (bytes, bytearray)):
            return "(" + ", ".join(f"{float(x):.4f}" for x in value) + ")"
        return f"{float(value):.4f}"
    except Exception:
        return repr(value)


def _mat_key(mat) -> str:
    return f"{mat.library.filepath}:{mat.name}" if mat.library else mat.name


def _resolve_material(material_name: str):
    mat = bpy.data.materials.get(material_name)
    if mat is not None:
        return mat
    for candidate in bpy.data.materials:
        if _mat_key(candidate) == material_name:
            return candidate
    return None


def _safe_node_attr(node, attr_name, fallback="?"):
    try:
        return getattr(node, attr_name)
    except Exception:
        return fallback


def _write_line(buf: StringIO, text: str = "") -> None:
    buf.write(text + "\n")


def _dump_socket(buf: StringIO, prefix: str, sock) -> None:
    if sock.is_linked:
        link = sock.links[0]
        _write_line(
            buf,
            f"{prefix}{sock.name}: <- [{link.from_node.type}] {link.from_node.name}.{link.from_socket.name}",
        )
    else:
        try:
            value = sock.default_value
            _write_line(buf, f"{prefix}{sock.name}: = {_fmt_value(value)}")
        except Exception:
            _write_line(buf, f"{prefix}{sock.name}: = ?")


def _dump_node_tree(buf: StringIO, node_tree, indent: str = "", seen=None) -> None:
    if node_tree is None:
        _write_line(buf, f"{indent}<no node tree>")
        return
    if seen is None:
        seen = set()
    tree_id = id(node_tree)
    if tree_id in seen:
        _write_line(buf, f"{indent}<tree already dumped: {node_tree.name}>")
        return
    seen.add(tree_id)

    _write_line(buf, f"{indent}NodeTree '{node_tree.name}' ({len(node_tree.nodes)} nodes)")
    for node in node_tree.nodes:
        status = "FULL"
        if node.type not in FULLY_SUPPORTED:
            status = "PARTIAL" if node.type in PARTIAL_SUPPORT else "UNSUPPORTED"
        _write_line(
            buf,
            f"{indent}- [{status}] [{node.type}] '{node.name}' label='{_safe_node_attr(node, 'label', '')}' mute={node.mute}",
        )
        for inp in node.inputs:
            _dump_socket(buf, indent + "    in: ", inp)
        for out in node.outputs:
            if out.is_linked:
                for link in out.links:
                    _write_line(
                        buf,
                        f"{indent}    out: {out.name} -> [{link.to_node.type}] {link.to_node.name}.{link.to_socket.name}",
                    )
        if node.type == "TEX_IMAGE" and getattr(node, "image", None):
            img = node.image
            cs = getattr(img, "colorspace_settings", None)
            _write_line(
                buf,
                f"{indent}    image: name='{img.name}' size={tuple(img.size)} source={img.source} "
                f"filepath='{getattr(img, 'filepath', '')}' colorspace='{getattr(cs, 'name', '?')}' "
                f"is_data={getattr(cs, 'is_data', False)} alpha_mode={getattr(img, 'alpha_mode', '?')}",
            )
        if node.type == "GROUP" and node.node_tree:
            _write_line(buf, f"{indent}    [group subtree]")
            _dump_node_tree(buf, node.node_tree, indent + "      ", seen)


def _collect_supported_summary(node_tree, summary=None):
    if summary is None:
        summary = {"full": [], "partial": [], "unsupported": []}
    if node_tree is None:
        return summary
    for node in node_tree.nodes:
        if node.type in FULLY_SUPPORTED:
            summary["full"].append((node.name, node.type))
        elif node.type in PARTIAL_SUPPORT:
            summary["partial"].append((node.name, node.type, PARTIAL_SUPPORT[node.type]))
        else:
            summary["unsupported"].append((node.name, node.type))
        if node.type == "GROUP" and node.node_tree:
            _collect_supported_summary(node.node_tree, summary)
    return summary


def _find_material_users(mat):
    users = []
    for obj in bpy.data.objects:
        for slot_idx, slot in enumerate(obj.material_slots):
            if slot.material == mat:
                users.append((obj.name, slot_idx))
    return users


def _decode_flags(flags: int) -> list[str]:
    names = []
    if flags & 1:
        names.append("ALPHA_TEST")
    if flags & 2:
        names.append("TRANSMISSION")
    if flags & 4:
        names.append("VOLUME")
    if flags & 8:
        names.append("VOLUME_ONLY")
    if flags & 16:
        names.append("VOLUME_HETERO")
    if flags & 32:
        names.append("HAIR")
    if flags & 64:
        names.append("DIFFUSE_SCENE_LINEAR")
    return names


def _unpack_exported_material(scene_export, raw_bytes: bytes):
    base_count = len(GPU_FIELD_NAMES)
    base_values = scene_export._GPU_MATERIAL_BASE.unpack_from(raw_bytes, 0)
    vm_values = scene_export._GPU_MATERIAL_VM.unpack_from(raw_bytes, scene_export._GPU_MATERIAL_BASE.size)
    gpu = dict(zip(GPU_FIELD_NAMES, base_values))
    gpu["nodeVmInstrCount"] = int(vm_values[0])
    gpu["nodeVmPad"] = tuple(int(x) for x in vm_values[1:4])
    gpu["nodeVmCodeWords"] = tuple(int(x) for x in vm_values[4 : 4 + gpu["nodeVmInstrCount"] * 4])
    if len(base_values) != base_count:
        gpu["__warning__"] = f"Expected {base_count} base values, got {len(base_values)}"
    return gpu


def _texture_desc(entry: dict | None) -> str:
    if not entry:
        return "none"
    img = entry.get("image_ref")
    cs = getattr(getattr(img, "colorspace_settings", None), "name", "?") if img else "?"
    is_data = getattr(getattr(img, "colorspace_settings", None), "is_data", False) if img else False
    return (
        f"idx={entry.get('index')} name='{entry.get('name')}' size={entry.get('width')}x{entry.get('height')} "
        f"export_mode={entry.get('export_mode')} scene_linear_export={entry.get('scene_linear_export')} "
        f"colorspace='{cs}' is_data={is_data}"
    )


def _dump_ignis_export(buf: StringIO, mat, scene_export) -> None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    materials_blob, mat_name_to_index, textures_list = scene_export.export_materials(depsgraph)
    key = _mat_key(mat)
    if key not in mat_name_to_index:
        _write_line(buf, f"[Ignis export] material key '{key}' not found in exported mapping")
        matches = difflib.get_close_matches(key, list(mat_name_to_index.keys()), n=8, cutoff=0.2)
        if matches:
            _write_line(buf, f"Closest exported keys: {matches}")
        return

    mat_index = int(mat_name_to_index[key])
    stride = scene_export._GPU_MATERIAL_SIZE
    raw = bytes(materials_blob)[mat_index * stride : (mat_index + 1) * stride]
    gpu = _unpack_exported_material(scene_export, raw)

    _write_line(buf, "Ignis Export")
    _write_line(buf, "-" * 60)
    _write_line(buf, f"material key: {key}")
    _write_line(buf, f"material index: {mat_index}")
    _write_line(buf, f"shaderType: {gpu['shaderType']}")
    _write_line(buf, f"flags: {gpu['flags']} -> {_decode_flags(int(gpu['flags']))}")
    _write_line(
        buf,
        "base_color: "
        f"({_fmt_value([gpu['ksAmbient_r'], gpu['ksAmbient_g'], gpu['ksAmbient_b']])})".replace("((", "(").replace("))", ")"),
    )
    _write_line(buf, f"roughness: {gpu['ksSpecularEXP_roughness']:.6f}")
    _write_line(buf, f"metallic: {gpu['fresnelC_metallic']:.6f}")
    _write_line(buf, f"specular_level: {gpu['fresnelEXP_specularLevel']:.6f}")
    _write_line(buf, f"ior: {gpu['detailUVMult_ior']:.6f}")
    _write_line(buf, f"normal_strength: {gpu['detailNormalBlend_normalStrength']:.6f}")
    _write_line(buf, f"alpha: {gpu['multG_alpha']:.6f}")
    _write_line(buf, f"alpha_ref: {gpu['alphaRef']:.6f}")
    _write_line(buf, f"transmission: {gpu['multR_transmission']:.6f}")
    _write_line(buf, f"transparent_prob: {gpu['multB_transparentProb']:.6f}")
    _write_line(
        buf,
        f"emission: ({gpu['emissive_r']:.6f}, {gpu['emissive_g']:.6f}, {gpu['emissive_b']:.6f}) * {gpu['fresnelMaxLevel_emissionStrength']:.6f}",
    )
    _write_line(buf, f"uv_scale: ({gpu['uv_scale_x']:.6f}, {gpu['uv_scale_y']:.6f})")
    _write_line(buf, f"color_value: {gpu['color_value']:.6f}")
    _write_line(buf, f"color_saturation: {gpu['color_saturation']:.6f}")
    if int(gpu["flags"]) & scene_export._MAT_FLAG_HAIR:
        _write_line(buf, f"hair_shift: {gpu['sunSpecular_extra0']:.6f}")
        _write_line(buf, f"hair_radial_roughness: {gpu['sunSpecularEXP_extra1']:.6f}")
    else:
        _write_line(buf, f"extra0(volume_density or misc): {gpu['sunSpecular_extra0']:.6f}")
        _write_line(buf, f"extra1(volume_anisotropy or misc): {gpu['sunSpecularEXP_extra1']:.6f}")

    tex_map = {
        "diffuseTexIndex": int(gpu["diffuseTexIndex"]),
        "normalTexIndex": int(gpu["normalTexIndex"]),
        "mapsTexIndex": int(gpu["mapsTexIndex"]),
        "detailTexIndex": int(gpu["detailTexIndex"]),
    }
    _write_line(buf, "textures:")
    for label, tex_idx in tex_map.items():
        entry = None if tex_idx == scene_export._NO_TEX else textures_list[tex_idx]
        _write_line(buf, f"  {label}: {_texture_desc(entry)}")

    _write_line(buf, f"node VM instruction count: {gpu['nodeVmInstrCount']}")
    if gpu["nodeVmInstrCount"] > 0:
        preview = list(gpu["nodeVmCodeWords"][: min(16, len(gpu["nodeVmCodeWords"]))])
        _write_line(buf, f"node VM first words: {preview}")


def main():
    scene_export = _load_scene_export()
    mat_name = _parse_material_name()
    buf = StringIO()

    _write_line(buf, "=" * 80)
    _write_line(buf, "IGNIS MATERIAL DEBUG")
    _write_line(buf, "=" * 80)

    if not mat_name:
        _write_line(buf, "ERROR: no material name provided and no active material found.")
        _write_line(buf, "Set MATERIAL_NAME at the top of this script or run with:")
        _write_line(buf, "  blender --background file.blend --python debug_material.py -- \"Material Name\"")
        print(buf.getvalue())
        return

    mat = _resolve_material(mat_name)
    if mat is None:
        _write_line(buf, f"ERROR: material '{mat_name}' not found.")
        names = [m.name for m in bpy.data.materials]
        matches = difflib.get_close_matches(mat_name, names, n=12, cutoff=0.2)
        if matches:
            _write_line(buf, f"Closest matches: {matches}")
        else:
            _write_line(buf, f"Available materials ({len(names)}): {names[:50]}")
        print(buf.getvalue())
        return

    users = _find_material_users(mat)
    _write_line(buf, f"Material: '{mat.name}'")
    _write_line(buf, f"Key: {_mat_key(mat)}")
    _write_line(buf, f"Library: {mat.library.filepath if mat.library else '<local>'}")
    _write_line(buf, f"Use nodes: {mat.use_nodes}")
    _write_line(buf, f"Blend method: {getattr(mat, 'blend_method', '?')}")
    _write_line(buf, f"Shadow method: {getattr(mat, 'shadow_method', '?')}")
    _write_line(buf, f"Alpha threshold: {getattr(mat, 'alpha_threshold', '?')}")
    _write_line(buf, f"Backface culling: {getattr(mat, 'use_backface_culling', '?')}")
    _write_line(buf, f"Users in scene: {len(users)}")
    for obj_name, slot_idx in users[:32]:
        _write_line(buf, f"  - object='{obj_name}' slot={slot_idx}")
    if len(users) > 32:
        _write_line(buf, f"  ... {len(users) - 32} more")

    if mat.use_nodes and mat.node_tree:
        surface = scene_export._find_surface_shader(mat.node_tree)
        volume = scene_export._find_volume_shader(mat.node_tree)
        _write_line(buf, f"Surface shader from active output: {surface.type if surface else 'None'} / {surface.name if surface else 'None'}")
        _write_line(buf, f"Volume shader from active output: {volume.type if volume else 'None'} / {volume.name if volume else 'None'}")

        summary = _collect_supported_summary(mat.node_tree)
        _write_line(buf, f"Support summary: full={len(summary['full'])} partial={len(summary['partial'])} unsupported={len(summary['unsupported'])}")
        if summary["partial"]:
            _write_line(buf, "Partial support nodes:")
            for name, ntype, reason in summary["partial"]:
                _write_line(buf, f"  - [{ntype}] {name}: {reason}")
        if summary["unsupported"]:
            _write_line(buf, "Unsupported nodes:")
            for name, ntype in summary["unsupported"]:
                _write_line(buf, f"  - [{ntype}] {name}")

        _write_line(buf)
        _write_line(buf, "Node Tree")
        _write_line(buf, "-" * 60)
        _dump_node_tree(buf, mat.node_tree)
    else:
        _write_line(buf, "Material has no node tree.")

    _write_line(buf)
    _dump_ignis_export(buf, mat, scene_export)

    output = buf.getvalue()
    print(output)

    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in mat.name)
    out_path = Path(os.path.expanduser("~")) / f"ignis-material-{safe_name}.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
