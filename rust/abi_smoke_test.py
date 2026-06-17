"""ABI smoke test for the Rust ignis_rt.dll (Phase 0 strangler scaffolding).

Replicates exactly what the Blender addon's dll_wrapper.load() does: ctypes.CDLL +
GetProcAddress on every exported symbol, then sets signatures and calls each one.
If every symbol resolves and returns its stub default, the Rust DLL is a valid
drop-in at the C ABI seam and Blender will load the addon without error.

Run:  python rust/abi_smoke_test.py
"""
import ctypes
import os
import sys
from ctypes import (
    c_bool, c_char_p, c_float, c_int, c_uint8, c_uint32, c_void_p, POINTER,
)

DLL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "target", "release", "ignis_rt.dll")

# (name, argtypes, restype, call_args, expected_return)
F32P, U32P, U8P = POINTER(c_float), POINTER(c_uint32), POINTER(c_uint8)
SPECS = [
    ("ignis_set_base_path", [c_char_p], None, [None], None),
    ("ignis_set_log_path", [c_char_p], None, [None], None),
    ("ignis_create", [c_uint32, c_uint32], c_bool, [800, 600], False),
    ("ignis_create_step", [c_uint32, c_uint32], c_char_p, [800, 600], None),
    ("ignis_destroy", [], None, [], None),
    ("ignis_clear_geometry", [], None, [], None),
    ("ignis_upload_mesh", [F32P, c_uint32, U32P, c_uint32], c_int, [None, 0, None, 0], -1),
    ("ignis_refit_blas", [c_int, F32P, c_uint32, U32P, c_uint32], c_bool, [0, None, 0, None, 0], False),
    ("ignis_map_blas_deform_staging", [c_int, c_uint32], c_void_p, [0, 0], None),
    ("ignis_commit_blas_deform", [c_int], c_bool, [0], False),
    ("ignis_upload_mesh_attributes", [c_int, F32P, F32P, c_uint32, F32P], c_bool, [0, None, None, 0, None], False),
    ("ignis_upload_mesh_primitive_materials", [c_int, U32P, c_uint32], c_bool, [0, None, 0], False),
    ("ignis_queue_mesh", [F32P, c_uint32, U32P, c_uint32, F32P, F32P, c_uint32, F32P], c_int,
     [None, 0, None, 0, None, None, 0, None], -1),
    ("ignis_flush_mesh_batch", [], c_int, [], 0),
    ("ignis_free_blas", [c_int], None, [0], None),
    ("ignis_upload_materials", [c_void_p, c_uint32], None, [None, 0], None),
    ("ignis_generate_hair_gpu",
     [F32P, c_uint32, c_uint32, c_uint32, F32P, c_uint32, U32P, c_uint32, F32P,
      c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float,
      c_float, c_float, c_float, c_float, c_float, c_float, c_uint32,
      c_float, c_float, c_float, c_bool, c_float, c_bool, c_bool, c_uint32, F32P, c_uint32],
     c_int,
     [None, 0, 0, 0, None, 0, None, 0, None,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
      0.0, 0.0, 0.0, False, 0.0, False, False, 0, None, 0],
     -1),
    ("ignis_reload_shaders", [], c_bool, [], False),
    ("ignis_build_tlas", [c_void_p, c_uint32], c_bool, [None, 0], False),
    ("ignis_update_instance_transforms", [U32P, F32P, c_uint32], c_bool, [None, None, 0], False),
    ("ignis_set_camera", [F32P, F32P, F32P, F32P, c_uint32], None, [None, None, None, None, 0], None),
    ("ignis_upload_lights", [F32P, c_uint32], None, [None, 0], None),
    ("ignis_upload_emissive_triangles", [F32P, c_uint32], None, [None, 0], None),
    ("ignis_upload_lut", [F32P, c_uint32], c_bool, [None, 0], False),
    ("ignis_render_frame", [], None, [], None),
    ("ignis_readback", [c_void_p, c_uint32], c_bool, [None, 0], False),
    ("ignis_readback_float", [F32P, c_uint32], c_bool, [None, 0], False),
    ("ignis_readback_hdr_float", [F32P, c_uint32], c_bool, [None, 0], False),
    ("ignis_draw_gl", [c_uint32, c_uint32], c_bool, [0, 0], False),
    ("ignis_set_float", [c_char_p, c_float], None, [b"k", 0.0], None),
    ("ignis_set_int", [c_char_p, c_int], None, [b"k", 0], None),
    ("ignis_get_int", [c_char_p], c_int, [b"k"], 0),
    ("ignis_get_float", [c_char_p], c_float, [b"k"], 0.0),
    ("ignis_create_texture_manager", [], c_void_p, [], None),
    ("ignis_destroy_texture_manager", [c_void_p], None, [None], None),
    ("ignis_texture_manager_add", [c_void_p, c_char_p, U8P, c_uint32, c_int, c_int, c_int, c_uint32], c_int,
     [None, None, None, 0, 0, 0, 0, 0], -1),
    ("ignis_texture_manager_upload_all", [c_void_p], c_bool, [None], False),
    ("ignis_texture_manager_upload_one", [c_void_p], c_bool, [None], False),
    ("ignis_texture_manager_pending_count", [c_void_p], c_int, [None], 0),
    ("ignis_update_texture_descriptors", [c_void_p], None, [None], None),
]


def main():
    if not os.path.isfile(DLL):
        print(f"FAIL: DLL not found at {DLL}")
        return 1
    lib = ctypes.CDLL(DLL)
    print(f"Loaded {DLL}\nChecking {len(SPECS)} exported symbols...\n")

    missing, called, mismatched = [], 0, []
    for name, argtypes, restype, args, expected in SPECS:
        try:
            fn = getattr(lib, name)  # triggers GetProcAddress
        except AttributeError:
            missing.append(name)
            continue
        fn.argtypes = argtypes
        fn.restype = restype
        ret = fn(*args)
        called += 1
        if ret != expected:
            mismatched.append(f"{name}: got {ret!r}, expected {expected!r}")

    print(f"  resolved + called : {called}/{len(SPECS)}")
    if missing:
        print(f"  MISSING symbols   : {missing}")
    if mismatched:
        print("  return mismatches :")
        for m in mismatched:
            print(f"    - {m}")

    ok = not missing and not mismatched
    print("\n" + ("PASS — Rust DLL is ABI-compatible with the Blender addon."
                  if ok else "FAIL — see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
