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

    _lib.ignis_create_step.argtypes = [c_uint32, c_uint32]
    _lib.ignis_create_step.restype = c_char_p

    _lib.ignis_destroy.argtypes = []
    _lib.ignis_destroy.restype = None

    # ------------------------------------------------------------------
    # Geometry upload
    # ------------------------------------------------------------------
    _lib.ignis_clear_geometry.argtypes = []
    _lib.ignis_clear_geometry.restype = None

    _lib.ignis_upload_mesh.argtypes = [
        POINTER(c_float), c_uint32,
        POINTER(c_uint32), c_uint32,
    ]
    _lib.ignis_upload_mesh.restype = c_int

    _lib.ignis_refit_blas.argtypes = [
        c_int, POINTER(c_float), c_uint32, POINTER(c_uint32), c_uint32,
    ]
    _lib.ignis_refit_blas.restype = c_bool

    _lib.ignis_map_blas_deform_staging.argtypes = [c_int, c_uint32]
    _lib.ignis_map_blas_deform_staging.restype = c_void_p
    _lib.ignis_commit_blas_deform.argtypes = [c_int]
    _lib.ignis_commit_blas_deform.restype = c_bool

    _lib.ignis_upload_mesh_attributes.argtypes = [
        c_int, POINTER(c_float), POINTER(c_float), c_uint32, POINTER(c_float),
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
    # GPU hair generation (Blender-exact with emitter mesh)
    _lib.ignis_generate_hair_gpu.argtypes = [
        POINTER(c_float), c_uint32, c_uint32, c_uint32,       # parentKeys, nParents, keysPerStrand, childrenPerParent
        POINTER(c_float), c_uint32,                            # emitterVerts, nEmitterVerts
        POINTER(c_uint32), c_uint32,                           # emitterTris, nEmitterTris
        POINTER(c_float),                                      # emitterCDF
        c_float, c_float, c_float, c_float, c_float, c_float, # rootRadius, tipFactor, clumpNoiseSize, childRoundness, childLength, avgSpacing
        c_float, c_float,                                      # kinkAmplitude, kinkFrequency
        c_float, c_float, c_float, c_float, c_float, c_float, # clump, rough
        c_uint32,                                              # childMode
        c_float, c_float, c_float,                             # kinkShape, kinkFlat, kinkAmpRandom
        c_bool,                                                # opaqueHair
        c_float,                                               # childSizeRandom
        c_bool,                                                # useParentParticles
        c_bool,                                                # precomputedStrands
        c_uint32,                                              # blenderSeed
        POINTER(c_float),                                      # frandTable
        c_uint32,                                              # frandCount
    ]
    _lib.ignis_generate_hair_gpu.restype = c_int

    _lib.ignis_reload_shaders.argtypes = []
    _lib.ignis_reload_shaders.restype = c_bool

    # ------------------------------------------------------------------
    _lib.ignis_build_tlas.argtypes = [c_void_p, c_uint32]
    _lib.ignis_build_tlas.restype = c_bool
    _lib.ignis_update_instance_transforms.argtypes = [
        ctypes.POINTER(c_uint32),  # indices
        ctypes.POINTER(c_float),   # transforms (12 floats per instance)
        c_uint32,                  # count
    ]
    _lib.ignis_update_instance_transforms.restype = c_bool

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

    _lib.ignis_upload_lut.argtypes = [POINTER(c_float), c_uint32]
    _lib.ignis_upload_lut.restype = c_bool

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    _lib.ignis_render_frame.argtypes = []
    _lib.ignis_render_frame.restype = None

    _lib.ignis_readback.argtypes = [c_void_p, c_uint32]
    _lib.ignis_readback.restype = c_bool

    _lib.ignis_readback_float.argtypes = [POINTER(c_float), c_uint32]
    _lib.ignis_readback_float.restype = c_bool

    _lib.ignis_readback_hdr_float.argtypes = [POINTER(c_float), c_uint32]
    _lib.ignis_readback_hdr_float.restype = c_bool

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

    _lib.ignis_get_float.argtypes = [c_char_p]
    _lib.ignis_get_float.restype = c_float

    _lib.ignis_get_string.argtypes = [c_char_p]
    _lib.ignis_get_string.restype = c_char_p

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

    _lib.ignis_texture_manager_upload_one.argtypes = [c_void_p]
    _lib.ignis_texture_manager_upload_one.restype = c_bool

    _lib.ignis_texture_manager_pending_count.argtypes = [c_void_p]
    _lib.ignis_texture_manager_pending_count.restype = c_int

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


def create_step(width: int, height: int):
    """Execute one init step. Returns step name (str) or None when done."""
    result = _lib.ignis_create_step(c_uint32(width), c_uint32(height))
    return result.decode('utf-8') if result else None


def destroy():
    if _lib is not None:
        _lib.ignis_destroy()


def clear_geometry():
    """Destroy all BLAS (call before full scene reload)."""
    _lib.ignis_clear_geometry()


def upload_mesh(vertices, vertex_count: int, indices, index_count: int) -> int:
    """Upload position-only mesh. Returns BLAS handle (>=0) or -1 on error."""
    # Keep numpy arrays alive for the duration of the DLL call
    v = _np_f32(vertices)
    i = _np_u32(indices)
    return _lib.ignis_upload_mesh(
        v.ctypes.data_as(POINTER(c_float)), c_uint32(vertex_count),
        i.ctypes.data_as(POINTER(c_uint32)), c_uint32(index_count))


def upload_mesh_attributes(blas_handle: int, normals, uvs, vertex_count: int, colors=None) -> bool:
    n = _np_f32(normals)
    u = _np_f32(uvs)
    c_ptr = None
    if colors is not None:
        c = _np_f32(colors)
        c_ptr = c.ctypes.data_as(POINTER(c_float))
    return _lib.ignis_upload_mesh_attributes(
        c_int(blas_handle),
        n.ctypes.data_as(POINTER(c_float)),
        u.ctypes.data_as(POINTER(c_float)),
        c_uint32(vertex_count),
        c_ptr)


def map_blas_deform_staging(blas_index: int, vertex_count: int):
    """Map persistent staging buffer for zero-copy deformation writes.
    Layout: [positions N*3*f32] [normals N*3*f32] [UVs N*2*f32]
    Returns raw pointer (int) or 0/None on failure."""
    return _lib.ignis_map_blas_deform_staging(c_int(blas_index), c_uint32(vertex_count))


def commit_blas_deform(blas_index: int) -> bool:
    """DMA staging → GPU buffers + BLAS rebuild. No memcpy — data already in staging."""
    return _lib.ignis_commit_blas_deform(c_int(blas_index))


def generate_hair_gpu(parent_keys, n_parents: int, keys_per_strand: int,
                      children_per_parent: int,
                      emitter_verts=None, n_emitter_verts: int = 0,
                      emitter_tris=None, n_emitter_tris: int = 0,
                      emitter_cdf=None,
                      root_radius: float = 0.003, tip_factor: float = 0.0,
                      clump_noise_size: float = 1.0, child_roundness: float = 0.0,
                      child_length: float = 1.0,
                      avg_spacing: float = 0.01,
                      kink_amplitude: float = 0.0, kink_frequency: float = 2.0,
                      clump_factor: float = 0.0, clump_shape: float = 0.0,
                      rough1: float = 0.0, rough1_size: float = 1.0,
                      rough2: float = 0.0, rough_end: float = 0.0,
                      child_mode: int = 0,
                      kink_shape: float = 0.0, kink_flat: float = 0.0,
                      kink_amp_random: float = 0.0,
                      opaque_hair: bool = False,
                      child_size_random: float = 0.0,
                      use_parent_particles: bool = False,
                      precomputed_strands: bool = False,
                      blender_seed: int = 0,
                      frand_table=None) -> int:
    """Generate hair children + ribbon geometry entirely on GPU. Returns BLAS index."""
    import numpy as np
    ft = _np_f32(frand_table) if frand_table is not None and len(frand_table) > 0 else None
    k = _np_f32(parent_keys)
    ev = _np_f32(emitter_verts) if emitter_verts is not None and len(emitter_verts) > 0 else np.zeros(3, dtype=np.float32)
    et = _np_u32(emitter_tris) if emitter_tris is not None and len(emitter_tris) > 0 else np.zeros(3, dtype=np.uint32)
    ec = _np_f32(emitter_cdf) if emitter_cdf is not None and len(emitter_cdf) > 0 else np.zeros(1, dtype=np.float32)
    return _lib.ignis_generate_hair_gpu(
        k.ctypes.data_as(POINTER(c_float)),
        c_uint32(n_parents), c_uint32(keys_per_strand), c_uint32(children_per_parent),
        ev.ctypes.data_as(POINTER(c_float)), c_uint32(n_emitter_verts),
        et.ctypes.data_as(POINTER(c_uint32)), c_uint32(n_emitter_tris),
        ec.ctypes.data_as(POINTER(c_float)),
        c_float(root_radius), c_float(tip_factor),
        c_float(clump_noise_size), c_float(child_roundness), c_float(child_length), c_float(avg_spacing),
        c_float(kink_amplitude), c_float(kink_frequency),
        c_float(clump_factor), c_float(clump_shape),
        c_float(rough1), c_float(rough1_size), c_float(rough2), c_float(rough_end),
        c_uint32(child_mode),
        c_float(kink_shape), c_float(kink_flat), c_float(kink_amp_random),
        c_bool(opaque_hair),
        c_float(child_size_random),
        c_bool(use_parent_particles),
        c_bool(precomputed_strands),
        c_uint32(blender_seed),
        ft.ctypes.data_as(POINTER(c_float)) if ft is not None else None,
        c_uint32(len(ft) if ft is not None else 0))


def reload_shaders() -> bool:
    """Hot-reload shaders: recompile from disk + recreate RT pipeline. Keeps geometry/TLAS."""
    return _lib.ignis_reload_shaders()


def refit_blas(blas_handle: int, vertices, vertex_count: int, indices, index_count: int) -> bool:
    """Rebuild BLAS with new vertex positions (same topology required)."""
    v = _np_f32(vertices)
    i = _np_u32(indices)
    return _lib.ignis_refit_blas(
        c_int(blas_handle),
        v.ctypes.data_as(POINTER(c_float)), c_uint32(vertex_count),
        i.ctypes.data_as(POINTER(c_uint32)), c_uint32(index_count))


def build_tlas(instances_array, count: int) -> bool:
    """instances_array must be a ctypes Array of TLASInstance."""
    return _lib.ignis_build_tlas(ctypes.cast(instances_array, c_void_p), c_uint32(count))


def update_instance_transforms(indices_np, transforms_np, count: int) -> bool:
    """Patch specific TLAS instance transforms in-place (GPU refit, no full rebuild).
    indices_np: np.array of uint32 TLAS indices
    transforms_np: np.array of float32, 12 floats per instance (3x4 row-major)
    """
    import numpy as np
    idx = np.ascontiguousarray(indices_np, dtype=np.uint32)
    xfm = np.ascontiguousarray(transforms_np, dtype=np.float32)
    return _lib.ignis_update_instance_transforms(
        idx.ctypes.data_as(ctypes.POINTER(c_uint32)),
        xfm.ctypes.data_as(ctypes.POINTER(c_float)),
        c_uint32(count))


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


def readback_hdr_float(out_buffer, pixel_count: int) -> bool:
    """Read back scene-linear HDR as float32 RGBA from the pre-tonemap buffer."""
    return _lib.ignis_readback_hdr_float(out_buffer, c_uint32(pixel_count))


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


def upload_lut(rgb_floats, lut_size: int):
    """Upload tonemap 3D LUT in-memory. rgb_floats = flat float array (3 per entry, lut_size^3 entries)."""
    arr = (c_float * len(rgb_floats))(*rgb_floats)
    return _lib.ignis_upload_lut(arr, c_uint32(lut_size))


def set_float(key: str, value: float):
    _lib.ignis_set_float(key.encode("utf-8"), c_float(value))


def set_sun_params(elevation, azimuth, intensity, cr, cg, cb, sun_size,
                   disc_intensity, air_density, dust_density, ozone_density, altitude):
    """Batch set all sun parameters in 1 DLL call (instead of 12 separate set_float calls)."""
    # Pack into array and send as single call
    _lib.ignis_set_float(b"sun_elevation", c_float(elevation))
    _lib.ignis_set_float(b"sun_azimuth", c_float(azimuth))
    _lib.ignis_set_float(b"sun_intensity", c_float(intensity))
    _lib.ignis_set_float(b"sun_color_r", c_float(cr))
    _lib.ignis_set_float(b"sun_color_g", c_float(cg))
    _lib.ignis_set_float(b"sun_color_b", c_float(cb))
    _lib.ignis_set_float(b"sun_size", c_float(sun_size))
    _lib.ignis_set_float(b"sun_disc_intensity", c_float(disc_intensity))
    _lib.ignis_set_float(b"air_density", c_float(air_density))
    _lib.ignis_set_float(b"dust_density", c_float(dust_density))
    _lib.ignis_set_float(b"ozone_density", c_float(ozone_density))
    _lib.ignis_set_float(b"altitude", c_float(altitude))


def set_int(key: str, value: int):
    _lib.ignis_set_int(key.encode("utf-8"), c_int(value))


def get_int(key: str) -> int:
    if _lib is None:
        return 0
    return _lib.ignis_get_int(key.encode("utf-8"))


def get_float(key: str) -> float:
    if _lib is None:
        return 0.0
    return _lib.ignis_get_float(key.encode("utf-8"))


def get_string(key: str) -> str:
    if _lib is None:
        return ""
    result = _lib.ignis_get_string(key.encode("utf-8"))
    return result.decode("utf-8") if result else ""


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


def texture_manager_upload_one(handle) -> bool:
    """Upload one pending texture to GPU. Returns True if uploaded, False if none pending."""
    return _lib.ignis_texture_manager_upload_one(handle)


def texture_manager_pending_count(handle) -> int:
    """Return number of textures still waiting for GPU upload."""
    return _lib.ignis_texture_manager_pending_count(handle)


def update_texture_descriptors(handle):
    _lib.ignis_update_texture_descriptors(handle)
