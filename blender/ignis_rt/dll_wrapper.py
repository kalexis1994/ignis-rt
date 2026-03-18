"""ctypes bindings for ignis_rt.dll"""

import ctypes
import os
import sys
from ctypes import (
    c_bool, c_char_p, c_float, c_int, c_uint8, c_uint32, c_void_p,
    POINTER, Structure,
)

import numpy as np


class TLASInstance(Structure):
    """Must match IgnisTLASInstance in ignis_api.cpp (60 bytes)."""
    _fields_ = [
        ("blasIndex", c_int),
        ("transform", c_float * 12),   # 3x4 row-major
        ("customIndex", c_uint32),
        ("mask", c_uint32),
    ]


_lib = None  # type: ctypes.CDLL | None


def _find_dll():
    """Locate ignis_rt.dll in lib/ next to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "lib", "ignis_rt.dll")
    if os.path.isfile(candidate):
        return candidate
    return None


def _np_f32(data):
    """Ensure data is a contiguous float32 numpy array (zero-copy if already)."""
    if isinstance(data, np.ndarray) and data.dtype == np.float32 and data.flags['C_CONTIGUOUS']:
        return data
    return np.ascontiguousarray(data, dtype=np.float32)


def _np_u32(data):
    """Ensure data is a contiguous uint32 numpy array (zero-copy if already)."""
    if isinstance(data, np.ndarray) and data.dtype == np.uint32 and data.flags['C_CONTIGUOUS']:
        return data
    return np.ascontiguousarray(data, dtype=np.uint32)


def load():
    """Load the DLL and set up all function signatures. Returns True on success."""
    global _lib
    if _lib is not None:
        return True

    path = _find_dll()
    if path is None:
        print("[ignis_rt] ERROR: ignis_rt.dll not found in lib/ subfolder")
        return False

    # Add DLL directory so dependent DLLs (vulkan-1.dll etc.) can be found
    dll_dir = os.path.dirname(path)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll_dir)

    try:
        _lib = ctypes.CDLL(path)
    except OSError as e:
        print(f"[ignis_rt] ERROR: Failed to load DLL: {e}")
        return False

    # ------------------------------------------------------------------
    # Pre-init configuration
    # ------------------------------------------------------------------
    _lib.ignis_set_base_path.argtypes = [c_char_p]
    _lib.ignis_set_base_path.restype = None

    _lib.ignis_set_log_path.argtypes = [c_char_p]
    _lib.ignis_set_log_path.restype = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    _lib.ignis_create.argtypes = [c_uint32, c_uint32]
    _lib.ignis_create.restype = c_bool

    _lib.ignis_destroy.argtypes = []
    _lib.ignis_destroy.restype = None

    # ------------------------------------------------------------------
    # Geometry upload
    # ------------------------------------------------------------------
    _lib.ignis_upload_mesh.argtypes = [
        POINTER(c_float), c_uint32,
        POINTER(c_uint32), c_uint32,
    ]
    _lib.ignis_upload_mesh.restype = c_int

    _lib.ignis_upload_mesh_attributes.argtypes = [
        c_int, POINTER(c_float), POINTER(c_float), c_uint32,
    ]
    _lib.ignis_upload_mesh_attributes.restype = c_bool

    _lib.ignis_upload_mesh_primitive_materials.argtypes = [
        c_int, POINTER(c_uint32), c_uint32,
    ]
    _lib.ignis_upload_mesh_primitive_materials.restype = c_bool

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    _lib.ignis_upload_materials.argtypes = [c_void_p, c_uint32]
    _lib.ignis_upload_materials.restype = None

    # ------------------------------------------------------------------
    # Acceleration structures
    # ------------------------------------------------------------------
    _lib.ignis_build_tlas.argtypes = [c_void_p, c_uint32]
    _lib.ignis_build_tlas.restype = c_bool

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    _lib.ignis_set_camera.argtypes = [
        POINTER(c_float), POINTER(c_float),
        POINTER(c_float), POINTER(c_float),
        c_uint32,
    ]
    _lib.ignis_set_camera.restype = None

    # ------------------------------------------------------------------
    # Lights
    # ------------------------------------------------------------------
    _lib.ignis_upload_lights.argtypes = [POINTER(c_float), c_uint32]
    _lib.ignis_upload_lights.restype = None

    _lib.ignis_upload_emissive_triangles.argtypes = [POINTER(c_float), c_uint32]
    _lib.ignis_upload_emissive_triangles.restype = None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    _lib.ignis_render_frame.argtypes = []
    _lib.ignis_render_frame.restype = None

    _lib.ignis_readback.argtypes = [c_void_p, c_uint32]
    _lib.ignis_readback.restype = c_bool

    _lib.ignis_readback_float.argtypes = [POINTER(c_float), c_uint32]
    _lib.ignis_readback_float.restype = c_bool

    _lib.ignis_draw_gl.argtypes = [c_uint32, c_uint32]
    _lib.ignis_draw_gl.restype = c_bool

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    _lib.ignis_set_float.argtypes = [c_char_p, c_float]
    _lib.ignis_set_float.restype = None

    _lib.ignis_set_int.argtypes = [c_char_p, c_int]
    _lib.ignis_set_int.restype = None

    _lib.ignis_get_int.argtypes = [c_char_p]
    _lib.ignis_get_int.restype = c_int

    # ------------------------------------------------------------------
    # Texture management (optional, not used by MVP)
    # ------------------------------------------------------------------
    _lib.ignis_create_texture_manager.argtypes = []
    _lib.ignis_create_texture_manager.restype = c_void_p

    _lib.ignis_destroy_texture_manager.argtypes = [c_void_p]
    _lib.ignis_destroy_texture_manager.restype = None

    _lib.ignis_texture_manager_add.argtypes = [
        c_void_p, c_char_p,
        POINTER(c_uint8), c_uint32,
        c_int, c_int, c_int, c_uint32,
    ]
    _lib.ignis_texture_manager_add.restype = c_int

    _lib.ignis_texture_manager_upload_all.argtypes = [c_void_p]
    _lib.ignis_texture_manager_upload_all.restype = c_bool

    _lib.ignis_update_texture_descriptors.argtypes = [c_void_p]
    _lib.ignis_update_texture_descriptors.restype = None

    return True


# ------------------------------------------------------------------
# Thin Python wrappers
# ------------------------------------------------------------------

def set_base_path(path: str):
    _lib.ignis_set_base_path(path.encode("utf-8"))


def set_log_path(path: str):
    _lib.ignis_set_log_path(path.encode("utf-8"))


def create(width: int, height: int) -> bool:
    return _lib.ignis_create(c_uint32(width), c_uint32(height))


def destroy():
    if _lib is not None:
        _lib.ignis_destroy()


def upload_mesh(vertices, vertex_count: int, indices, index_count: int) -> int:
    """Upload position-only mesh. Returns BLAS handle (>=0) or -1 on error."""
    # Keep numpy arrays alive for the duration of the DLL call
    v = _np_f32(vertices)
    i = _np_u32(indices)
    return _lib.ignis_upload_mesh(
        v.ctypes.data_as(POINTER(c_float)), c_uint32(vertex_count),
        i.ctypes.data_as(POINTER(c_uint32)), c_uint32(index_count))


def upload_mesh_attributes(blas_handle: int, normals, uvs, vertex_count: int) -> bool:
    n = _np_f32(normals)
    u = _np_f32(uvs)
    return _lib.ignis_upload_mesh_attributes(
        c_int(blas_handle),
        n.ctypes.data_as(POINTER(c_float)),
        u.ctypes.data_as(POINTER(c_float)),
        c_uint32(vertex_count))


def build_tlas(instances_array, count: int) -> bool:
    """instances_array must be a ctypes Array of TLASInstance."""
    return _lib.ignis_build_tlas(ctypes.cast(instances_array, c_void_p), c_uint32(count))


def set_camera(view_inv, proj_inv, view, proj, frame_index: int):
    vi = _np_f32(view_inv)
    pi = _np_f32(proj_inv)
    v = _np_f32(view)
    p = _np_f32(proj)
    _lib.ignis_set_camera(
        vi.ctypes.data_as(POINTER(c_float)),
        pi.ctypes.data_as(POINTER(c_float)),
        v.ctypes.data_as(POINTER(c_float)),
        p.ctypes.data_as(POINTER(c_float)),
        c_uint32(frame_index))


def render_frame():
    _lib.ignis_render_frame()


def readback(out_buffer, buffer_size: int) -> bool:
    return _lib.ignis_readback(out_buffer, c_uint32(buffer_size))


def readback_float(out_buffer, pixel_count: int) -> bool:
    """Readback as float32 RGBA (conversion done in C++). out_buffer = ctypes c_float array."""
    return _lib.ignis_readback_float(out_buffer, c_uint32(pixel_count))


def draw_gl(width: int, height: int) -> bool:
    """Draw rendered image directly to current GL context (zero-copy). Returns False if unavailable."""
    return _lib.ignis_draw_gl(c_uint32(width), c_uint32(height))


def upload_materials(data, count: int):
    """Upload material buffer (GPUMaterial array). data = ctypes c_uint8 array."""
    _lib.ignis_upload_materials(ctypes.cast(data, c_void_p), c_uint32(count))


def upload_mesh_primitive_materials(blas_handle: int, material_ids, primitive_count: int) -> bool:
    """Upload per-primitive material IDs for a BLAS."""
    m = _np_u32(material_ids)
    return _lib.ignis_upload_mesh_primitive_materials(
        c_int(blas_handle),
        m.ctypes.data_as(POINTER(c_uint32)),
        c_uint32(primitive_count))


def upload_emissive_triangles(data, count: int):
    """Upload emissive triangle data for MIS. data = flat float array (16 per tri)."""
    if count == 0:
        arr = (c_float * 1)(0.0)
        _lib.ignis_upload_emissive_triangles(arr, c_uint32(0))
        return
    arr = (c_float * len(data))(*data)
    _lib.ignis_upload_emissive_triangles(arr, c_uint32(count))


def upload_lights(light_data_floats, count: int):
    """Upload point/spot/area lights. light_data_floats = flat list of floats (16 per light)."""
    if count == 0:
        arr = (c_float * 1)(0.0)
        _lib.ignis_upload_lights(arr, c_uint32(0))
        return
    arr = (c_float * len(light_data_floats))(*light_data_floats)
    _lib.ignis_upload_lights(arr, c_uint32(count))


def set_float(key: str, value: float):
    _lib.ignis_set_float(key.encode("utf-8"), c_float(value))


def set_int(key: str, value: int):
    _lib.ignis_set_int(key.encode("utf-8"), c_int(value))


def get_int(key: str) -> int:
    if _lib is None:
        return 0
    return _lib.ignis_get_int(key.encode("utf-8"))


def create_texture_manager():
    """Create a texture manager. Returns opaque handle or None on failure."""
    handle = _lib.ignis_create_texture_manager()
    return handle if handle else None


def destroy_texture_manager(handle):
    if handle:
        _lib.ignis_destroy_texture_manager(handle)


def texture_manager_add(handle, name: str, data, data_size: int,
                         width: int, height: int, mip_levels: int,
                         dxgi_format: int) -> int:
    """Add a texture. Returns texture index (>=0) or -1 on error."""
    name_bytes = name.encode("utf-8") if name else b""
    return _lib.ignis_texture_manager_add(
        handle, name_bytes, data, c_uint32(data_size),
        c_int(width), c_int(height), c_int(mip_levels), c_uint32(dxgi_format))


def texture_manager_upload_all(handle) -> bool:
    return _lib.ignis_texture_manager_upload_all(handle)


def update_texture_descriptors(handle):
    _lib.ignis_update_texture_descriptors(handle)
