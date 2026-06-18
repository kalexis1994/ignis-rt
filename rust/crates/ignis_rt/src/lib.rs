//! Ignis RT — Rust FFI surface (strangler migration of the C++ Vulkan renderer).
//!
//! Each function mirrors a symbol the Blender addon's `dll_wrapper.py` binds via ctypes;
//! the C++ `ignis_rt.dll` is the reference. Functions are ported one at a time and
//! validated A/B on the RTX box. Unported symbols remain stubs returning safe defaults.
//!
//! ABI notes (x86_64-pc-windows-msvc): `extern "C"` matches ctypes.CDLL; Rust `bool` is
//! 1 byte (== C `_Bool`); `#[no_mangle]` + `cdylib` exports each symbol (no .def file).

#![allow(clippy::missing_safety_doc)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

mod config;
mod gpu;
mod log;
mod ngx;
mod renderer;

/// Borrow a C string as a Rust &str (lossy). Returns "" on null.
unsafe fn cstr<'a>(p: *const c_char) -> std::borrow::Cow<'a, str> {
    if p.is_null() {
        std::borrow::Cow::Borrowed("")
    } else {
        CStr::from_ptr(p).to_string_lossy()
    }
}

// ============================================================================
// Configuration (call before ignis_create)
// ============================================================================

#[no_mangle]
pub extern "C" fn ignis_set_base_path(path: *const c_char) {
    let p = unsafe { cstr(path) };
    config::set_base_path(&p);
    // Mirror the log into the addon folder (reachable over the deploy SMB share).
    log::set_mirror(&format!("{p}/ignis-rs.log"));
}

#[no_mangle]
pub extern "C" fn ignis_set_log_path(path: *const c_char) {
    log::set_path(&unsafe { cstr(path) });
}

// ============================================================================
// Lifecycle
// ============================================================================

#[no_mangle]
pub extern "C" fn ignis_create(width: u32, height: u32) -> bool {
    renderer::create(width, height)
}

/// Phased init: returns the current step name, or NULL when done. We do init in one
/// shot inside ignis_create, so report completion immediately (NULL = finished).
#[no_mangle]
pub extern "C" fn ignis_create_step(width: u32, height: u32) -> *const c_char {
    if !renderer::is_created() {
        renderer::create(width, height);
    }
    ptr::null()
}

#[no_mangle]
pub extern "C" fn ignis_destroy() {
    renderer::destroy();
}

// ============================================================================
// Geometry upload
// ============================================================================

#[no_mangle]
pub extern "C" fn ignis_clear_geometry() {
    renderer::clear_geometry();
}

#[no_mangle]
pub extern "C" fn ignis_upload_mesh(
    vertices: *const f32,
    vertex_count: u32,
    indices: *const u32,
    index_count: u32,
) -> c_int {
    if vertices.is_null() || indices.is_null() || vertex_count == 0 || index_count == 0 {
        return -1;
    }
    let positions = unsafe { std::slice::from_raw_parts(vertices, vertex_count as usize * 3) };
    let idx = unsafe { std::slice::from_raw_parts(indices, index_count as usize) };
    let handle = renderer::queue_mesh(positions, vertex_count, idx, &[], &[]);
    renderer::flush_mesh_batch(); // immediate build
    handle
}

#[no_mangle]
pub extern "C" fn ignis_refit_blas(
    _blas_handle: c_int,
    _vertices: *const f32,
    _vertex_count: u32,
    _indices: *const u32,
    _index_count: u32,
) -> bool {
    false
}

#[no_mangle]
pub extern "C" fn ignis_map_blas_deform_staging(
    _blas_index: c_int,
    _vertex_count: u32,
) -> *mut c_void {
    ptr::null_mut()
}

#[no_mangle]
pub extern "C" fn ignis_commit_blas_deform(_blas_handle: c_int) -> bool {
    false
}

#[no_mangle]
pub extern "C" fn ignis_upload_mesh_attributes(
    blas_handle: c_int,
    normals: *const f32,
    uvs: *const f32,
    vertex_count: u32,
    _colors: *const f32,
) -> bool {
    // The addon's incremental upload path: ignis_upload_mesh built positions-only, this attaches
    // the per-vertex normals + UVs so the mesh gets smooth shading and correct texturing.
    if blas_handle < 0 || vertex_count == 0 {
        return false;
    }
    let normals: &[f32] = if !normals.is_null() {
        unsafe { std::slice::from_raw_parts(normals, vertex_count as usize * 3) }
    } else {
        &[]
    };
    let uvs: &[f32] = if !uvs.is_null() {
        unsafe { std::slice::from_raw_parts(uvs, vertex_count as usize * 2) }
    } else {
        &[]
    };
    renderer::upload_mesh_attributes(blas_handle, normals, uvs);
    true
}

#[no_mangle]
pub extern "C" fn ignis_upload_mesh_primitive_materials(
    blas_handle: c_int,
    material_ids: *const u32,
    primitive_count: u32,
) -> bool {
    if material_ids.is_null() || primitive_count == 0 {
        return false;
    }
    let ids = unsafe { std::slice::from_raw_parts(material_ids, primitive_count as usize) };
    renderer::upload_mesh_primitive_materials(blas_handle, ids);
    true
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn ignis_queue_mesh(
    vertices: *const f32,
    vertex_count: u32,
    indices: *const u32,
    index_count: u32,
    normals: *const f32,
    uvs: *const f32,
    attr_vertex_count: u32,
    _colors: *const f32,
) -> c_int {
    if vertices.is_null() || indices.is_null() || vertex_count == 0 || index_count == 0 {
        return -1;
    }
    let positions = unsafe { std::slice::from_raw_parts(vertices, vertex_count as usize * 3) };
    let idx = unsafe { std::slice::from_raw_parts(indices, index_count as usize) };
    let normals: &[f32] = if !normals.is_null() && attr_vertex_count > 0 {
        unsafe { std::slice::from_raw_parts(normals, attr_vertex_count as usize * 3) }
    } else {
        &[]
    };
    let uvs: &[f32] = if !uvs.is_null() && attr_vertex_count > 0 {
        unsafe { std::slice::from_raw_parts(uvs, attr_vertex_count as usize * 2) }
    } else {
        &[]
    };
    renderer::queue_mesh(positions, vertex_count, idx, normals, uvs)
}

#[no_mangle]
pub extern "C" fn ignis_flush_mesh_batch() -> c_int {
    renderer::flush_mesh_batch()
}

#[no_mangle]
pub extern "C" fn ignis_free_blas(_blas_handle: c_int) {}

// ============================================================================
// Materials
// ============================================================================

#[no_mangle]
pub extern "C" fn ignis_upload_materials(data: *const c_void, count: u32) {
    renderer::upload_materials(data as *const u8, count);
}

// ============================================================================
// GPU hair generation
// ============================================================================

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn ignis_generate_hair_gpu(
    _parent_keys: *const f32,
    _n_parents: u32,
    _keys_per_strand: u32,
    _children_per_parent: u32,
    _emitter_verts: *const f32,
    _n_emitter_verts: u32,
    _emitter_tris: *const u32,
    _n_emitter_tris: u32,
    _emitter_cdf: *const f32,
    _root_radius: f32,
    _tip_factor: f32,
    _clump_noise_size: f32,
    _child_roundness: f32,
    _child_length: f32,
    _avg_spacing: f32,
    _kink_amplitude: f32,
    _kink_frequency: f32,
    _clump_factor: f32,
    _clump_shape: f32,
    _rough1: f32,
    _rough1_size: f32,
    _rough2: f32,
    _rough_end: f32,
    _child_mode: u32,
    _kink_shape: f32,
    _kink_flat: f32,
    _kink_amp_random: f32,
    _opaque_hair: bool,
    _child_size_random: f32,
    _use_parent_particles: bool,
    _precomputed_strands: bool,
    _blender_seed: u32,
    _frand_table: *const f32,
    _frand_count: u32,
) -> c_int {
    -1
}

// ============================================================================
// Shaders
// ============================================================================

#[no_mangle]
pub extern "C" fn ignis_reload_shaders() -> bool {
    false
}

// ============================================================================
// Acceleration structures
// ============================================================================

#[no_mangle]
pub extern "C" fn ignis_build_tlas(instances: *const c_void, count: u32) -> bool {
    renderer::build_tlas(instances as *const u8, count)
}

#[no_mangle]
pub extern "C" fn ignis_update_instance_transforms(
    _indices: *const u32,
    _transforms: *const f32,
    _count: u32,
) -> bool {
    false
}

// ============================================================================
// Camera
// ============================================================================

#[no_mangle]
pub extern "C" fn ignis_set_camera(
    view_inverse: *const f32,
    proj_inverse: *const f32,
    _view: *const f32,
    _proj: *const f32,
    _frame_index: u32,
) {
    if view_inverse.is_null() || proj_inverse.is_null() {
        return;
    }
    let vi = unsafe { std::slice::from_raw_parts(view_inverse, 16) };
    let pi = unsafe { std::slice::from_raw_parts(proj_inverse, 16) };
    renderer::set_camera(vi, pi);
}

// ============================================================================
// Lights / emissive
// ============================================================================

#[no_mangle]
pub extern "C" fn ignis_upload_lights(light_data: *const f32, light_count: u32) {
    renderer::upload_lights(light_data, light_count);
}

#[no_mangle]
pub extern "C" fn ignis_upload_emissive_triangles(data: *const f32, triangle_count: u32) {
    renderer::upload_emissive_triangles(data, triangle_count);
}

/// Accept the baked tonemap LUT (size^3 RGB triples) and stash it for later GPU upload.
#[no_mangle]
pub extern "C" fn ignis_upload_lut(rgb_data: *const f32, lut_size: u32) -> bool {
    if rgb_data.is_null() || lut_size < 2 {
        return false;
    }
    let n = (lut_size as usize).pow(3) * 3;
    let data = unsafe { std::slice::from_raw_parts(rgb_data, n) };
    renderer::upload_lut(data, lut_size);
    true
}

// ============================================================================
// Rendering
// ============================================================================

#[no_mangle]
pub extern "C" fn ignis_render_frame() {
    renderer::render_frame();
}

#[no_mangle]
pub extern "C" fn ignis_readback(_out_pixels: *mut c_void, _buffer_size: u32) -> bool {
    false
}

/// Copy the latest GPU-rendered frame (offscreen image -> host buffer) into the addon's
/// RGBA32F output. render_frame must have run first to populate the readback buffer.
#[no_mangle]
pub extern "C" fn ignis_readback_float(out_pixels: *mut f32, pixel_count: u32) -> bool {
    if out_pixels.is_null() || pixel_count == 0 {
        return false;
    }
    renderer::readback(out_pixels, pixel_count)
}

#[no_mangle]
pub extern "C" fn ignis_readback_hdr_float(_out_pixels: *mut f32, _pixel_count: u32) -> bool {
    false
}

#[no_mangle]
pub extern "C" fn ignis_draw_gl(_viewport_width: u32, _viewport_height: u32) -> bool {
    false
}

// ============================================================================
// Configuration (key/value)
// ============================================================================

#[no_mangle]
pub extern "C" fn ignis_set_float(key: *const c_char, value: f32) {
    config::set_float(&unsafe { cstr(key) }, value);
}

#[no_mangle]
pub extern "C" fn ignis_set_int(key: *const c_char, value: c_int) {
    config::set_int(&unsafe { cstr(key) }, value);
}

#[no_mangle]
pub extern "C" fn ignis_get_int(key: *const c_char) -> c_int {
    config::get_int(&unsafe { cstr(key) })
}

#[no_mangle]
pub extern "C" fn ignis_get_float(key: *const c_char) -> f32 {
    config::get_float(&unsafe { cstr(key) })
}

// ============================================================================
// Texture management
// ============================================================================

// Texture state lives in the renderer; the addon's opaque mgr handle is a non-null marker.
#[no_mangle]
pub extern "C" fn ignis_create_texture_manager() -> *mut c_void {
    renderer::texture_manager_reset();
    1 as *mut c_void
}

#[no_mangle]
pub extern "C" fn ignis_destroy_texture_manager(_mgr: *mut c_void) {
    renderer::texture_manager_reset();
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn ignis_texture_manager_add(
    _mgr: *mut c_void,
    _name: *const c_char,
    data: *const u8,
    data_size: u32,
    width: c_int,
    height: c_int,
    _mip_levels: c_int,
    dxgi_format: u32,
) -> c_int {
    if data.is_null() || data_size == 0 {
        return -1;
    }
    let bytes = unsafe { std::slice::from_raw_parts(data, data_size as usize) };
    renderer::texture_add(bytes, width, height, dxgi_format)
}

#[no_mangle]
pub extern "C" fn ignis_texture_manager_upload_all(_mgr: *mut c_void) -> bool {
    renderer::texture_upload_all();
    true
}

#[no_mangle]
pub extern "C" fn ignis_texture_manager_upload_one(_mgr: *mut c_void) -> bool {
    renderer::texture_upload_one()
}

#[no_mangle]
pub extern "C" fn ignis_texture_manager_pending_count(_mgr: *mut c_void) -> c_int {
    renderer::texture_pending_count()
}

#[no_mangle]
pub extern "C" fn ignis_update_texture_descriptors(_mgr: *mut c_void) {
    renderer::update_texture_descriptors();
}
