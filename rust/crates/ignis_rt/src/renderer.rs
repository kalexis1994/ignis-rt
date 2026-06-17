//! Vulkan renderer core (Rust port of the C++ acpt::vk::Renderer).
//!
//! M1: headless device bring-up. M2: offscreen target + readback. M3: compute pipeline +
//! camera (sky from per-pixel rays). M4a: build BLAS per mesh + a TLAS over instances (the
//! shader still renders sky; ray tracing against the TLAS lands in M4b). AS building is only
//! exercised on a real RT GPU — on a non-RT device flush/build_tlas no-op gracefully.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::Mutex;

use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::MemoryLocation;

use crate::gpu::{self, GpuBuffer};
use crate::log::log;

type AccelExt = ash::khr::acceleration_structure::Device;

struct SyncRenderer(Renderer);
unsafe impl Send for SyncRenderer {}

static RENDERER: Mutex<Option<SyncRenderer>> = Mutex::new(None);

const RB_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
const MAX_TEXTURES: u32 = 1024; // bindless texture array capacity

struct Texture {
    image: vk::Image,
    alloc: Option<Allocation>,
    view: vk::ImageView,
}

struct PendingTex {
    data: Vec<u8>,
    width: u32,
    height: u32,
    format: vk::Format,
}

/// DXGI format code (from the addon) -> Vulkan. The addon only emits uncompressed
/// R8G8B8A8 (28) and R16G16B16A16_FLOAT (10); default the rest to RGBA8.
fn dxgi_to_vk(dxgi: u32) -> (vk::Format, u32) {
    match dxgi {
        10 => (vk::Format::R16G16B16A16_SFLOAT, 8),
        87 => (vk::Format::B8G8R8A8_UNORM, 4),
        _ => (vk::Format::R8G8B8A8_UNORM, 4), // 28 and fallback
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CameraPush {
    inv_view_proj: [f32; 16],
    cam_pos: [f32; 4],
    dims: [u32; 2],
    has_tlas: u32,
    num_lights: u32,
    sun_dir: [f32; 4], // xyz = world direction, w = intensity
    sun_col: [f32; 4], // rgb = color
} // 128 bytes (== guaranteed push-constant minimum)

/// TLAS instance as the addon sends it (matches IgnisTLASInstance / TLASInstance, 60 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct TlasInstanceIn {
    blas_index: i32,
    transform: [f32; 12], // 3x4 row-major
    custom_index: u32,
    mask: u32,
}

struct QueuedMesh {
    positions: Vec<f32>, // 3 per vertex
    indices: Vec<u32>,
    normals: Vec<f32>, // 3 per vertex (empty if none)
    uvs: Vec<f32>,     // 2 per vertex (empty if none)
    vertex_count: u32,
}

struct Blas {
    accel: vk::AccelerationStructureKHR,
    buf: GpuBuffer,
    vbuf: GpuBuffer,
    ibuf: GpuBuffer,
    nbuf: Option<GpuBuffer>, // per-vertex normals (for smooth shading)
    ubuf: Option<GpuBuffer>, // per-vertex UVs (for texturing)
    matbuf: Option<GpuBuffer>, // per-triangle material id
    address: vk::DeviceAddress,
}

struct Tlas {
    accel: vk::AccelerationStructureKHR,
    buf: GpuBuffer,
    instbuf: GpuBuffer,
}

pub struct Renderer {
    // Geometry / acceleration structures.
    accel_ext: Option<AccelExt>,
    scratch_align: u64,
    queued: Vec<QueuedMesh>,
    blas_list: Vec<Option<Blas>>,
    tlas: Option<Tlas>,
    tlas_instance_data: Vec<u8>, // last instances (the addon may rebuild every frame)
    geom_table: Option<GpuBuffer>, // per-BLAS [vtx, idx, nrm, mat, uv] addrs, bound at binding 2
    mat_buffer: Option<GpuBuffer>, // material {albedo, texIndices}, bound at binding 3
    pending_tex: Vec<PendingTex>,  // textures staged on CPU, not yet uploaded
    textures: Vec<Texture>,        // uploaded textures, bound bindless at binding 4
    tex_sampler: vk::Sampler,      // shared sampler for all textures
    light_buffer: Option<GpuBuffer>, // scene point/area lights, bound at binding 5
    light_count: u32,
    light_data: Vec<f32>, // last uploaded light floats (the addon re-sends every frame)

    // OCIO view-transform 3D LUT (Blender's AgX/Filmic/etc.), bound at binding 7.
    lut_image: vk::Image, // null until a LUT is uploaded
    lut_alloc: Option<Allocation>,
    lut_view: vk::ImageView,
    lut_sampler: vk::Sampler,
    has_lut: bool,

    // World/background color (the "sky"), bound at binding 8.
    world_buffer: Option<GpuBuffer>,
    world_data: [f32; 8], // [bg_color.rgb, 0, hdri_index, hdri_strength, 0, 0]

    // Compute pipeline.
    offscreen_view: vk::ImageView,
    ds_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    desc_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    inv_view_proj: [f32; 16],
    cam_pos: [f32; 4],

    // Temporal accumulation (path tracing): a persistent radiance accumulator + frame counter
    // that resets when the camera or scene changes.
    accum_image: vk::Image,
    accum_alloc: Option<Allocation>,
    accum_view: vk::ImageView,
    accum_frame: u32,
    prev_view: [f32; 16], // previous viewInverse (no projection jitter) for move detection

    // Offscreen target + readback.
    offscreen_image: vk::Image,
    offscreen_alloc: Option<Allocation>,
    readback_buffer: vk::Buffer,
    readback_alloc: Option<Allocation>,
    readback_ptr: usize,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,

    // Core (destroyed last).
    allocator: Option<Allocator>,
    command_pool: vk::CommandPool,
    device: ash::Device,
    instance: ash::Instance,
    _entry: ash::Entry,

    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    rt_supported: bool,
    rtx_series: u32,
    width: u32,
    height: u32,
}

fn enforce_min(width: u32, height: u32) -> (u32, u32) {
    const MIN_WIDTH: u32 = 1920;
    if width < MIN_WIDTH && width > 0 {
        let aspect = height as f32 / width as f32;
        let h = (MIN_WIDTH as f32 * aspect) as u32;
        (MIN_WIDTH, (h + 1) & !1)
    } else {
        (width, height)
    }
}

fn detect_rtx_series(name: &str) -> u32 {
    let upper = name.to_uppercase();
    if let Some(idx) = upper.find("RTX") {
        for c in upper[idx + 3..].chars() {
            if c == ' ' {
                continue;
            }
            return c.to_digit(10).map(|d| d * 1000).unwrap_or(0);
        }
    }
    0
}

fn align_up(addr: u64, align: u64) -> u64 {
    if align <= 1 {
        addr
    } else {
        (addr + align - 1) & !(align - 1)
    }
}

fn submit_oneshot(
    device: &ash::Device,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    record: impl FnOnce(&ash::Device, vk::CommandBuffer),
) {
    unsafe {
        let _ = device.reset_fences(&[fence]);
        let _ = device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        );
        record(device, cmd);
        let _ = device.end_command_buffer(cmd);
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        let _ = device.queue_submit(queue, &[submit], fence);
        let _ = device.wait_for_fences(&[fence], true, u64::MAX);
    }
}

// ===================== Public module API (called from lib.rs) =====================

pub fn create(width: u32, height: u32) -> bool {
    let (w, h) = enforce_min(width, height);
    log(&format!("ignis_create({w}, {h})"));
    let mut guard = RENDERER.lock().unwrap();
    if guard.is_some() {
        log("renderer already created");
        return true;
    }
    match build(w, h) {
        Ok(mut r) => {
            crate::config::set_int("render_width", r.width as i32);
            crate::config::set_int("render_height", r.height as i32);
            // Upload the OCIO LUT the addon baked before create (if any).
            if let Some((size, data)) = crate::config::get_lut() {
                r.set_lut(&data, size);
            }
            *guard = Some(SyncRenderer(r));
            log("ignis_create OK");
            true
        }
        Err(e) => {
            log(&format!("ignis_create FAILED: {e}"));
            false
        }
    }
}

pub fn destroy() {
    if RENDERER.lock().unwrap().take().is_some() {
        log("ignis_destroy: renderer torn down");
    }
}

pub fn is_created() -> bool {
    RENDERER.lock().unwrap().is_some()
}

pub fn set_camera(view_inverse: &[f32], proj_inverse: &[f32]) {
    if view_inverse.len() < 16 || proj_inverse.len() < 16 {
        return;
    }
    if let Some(r) = RENDERER.lock().unwrap().as_mut() {
        let vi = glam::Mat4::from_cols_slice(&view_inverse[..16]);
        let pi = glam::Mat4::from_cols_slice(&proj_inverse[..16]);
        // inverse(proj*view) = viewInverse * projInverse
        // Reset accumulation only on real camera movement — compare the view-inverse
        // (camera world transform), NOT invViewProj, which the addon jitters every frame
        // for sub-pixel sampling. Letting the jitter through actually supersamples (AA).
        let view = view_inverse[..16].to_vec();
        if view
            .iter()
            .zip(r.0.prev_view.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6)
        {
            r.0.accum_frame = 0;
            r.0.prev_view.copy_from_slice(&view);
        }
        r.0.inv_view_proj = (vi * pi).to_cols_array();
        r.0.cam_pos = vi.w_axis.to_array();
    }
}

pub fn queue_mesh(
    positions: &[f32],
    vertex_count: u32,
    indices: &[u32],
    normals: &[f32],
    uvs: &[f32],
) -> i32 {
    match RENDERER.lock().unwrap().as_mut() {
        Some(r) => r.0.queue_mesh(positions, vertex_count, indices, normals, uvs),
        None => -1,
    }
}

pub fn flush_mesh_batch() -> i32 {
    match RENDERER.lock().unwrap().as_mut() {
        Some(r) => r.0.flush_mesh_batch(),
        None => 0,
    }
}

pub fn build_tlas(instances: *const u8, count: u32) -> bool {
    match RENDERER.lock().unwrap().as_mut() {
        Some(r) => r.0.build_tlas(instances, count),
        None => false,
    }
}

pub fn clear_geometry() {
    if let Some(r) = RENDERER.lock().unwrap().as_mut() {
        r.0.clear_geometry();
    }
}

/// Material albedos parsed from the addon's GPUMaterial buffer (base_color at byte 20 of
/// each 2204-byte material). Replaces the per-instance hash with the real surface color.
pub fn upload_materials(data: *const u8, count: u32) {
    if data.is_null() || count == 0 {
        return;
    }
    // Upload the full 2204-byte GPUMaterial structs (incl. the Node VM bytecode) verbatim;
    // the shader reads the fields + runs the VM (scalar layout matches the C++ struct).
    const STRIDE: usize = 2204;
    let bytes = unsafe { std::slice::from_raw_parts(data, count as usize * STRIDE) };
    if let Some(r) = RENDERER.lock().unwrap().as_mut() {
        r.0.set_materials(bytes);
    }
}

pub fn upload_mesh_primitive_materials(handle: i32, ids: &[u32]) {
    if let Some(r) = RENDERER.lock().unwrap().as_mut() {
        r.0.set_blas_materials(handle, ids);
    }
}

pub fn upload_mesh_attributes(handle: i32, normals: &[f32], uvs: &[f32]) {
    if let Some(r) = RENDERER.lock().unwrap().as_mut() {
        r.0.set_blas_attributes(handle, normals, uvs);
    }
}

/// OCIO view-transform 3D LUT (size^3 RGB). Stored, and uploaded to the GPU now if the
/// renderer exists (else create() uploads it from config).
pub fn upload_lut(data: &[f32], size: u32) {
    crate::config::store_lut(size, data.to_vec());
    if let Some(r) = RENDERER.lock().unwrap().as_mut() {
        r.0.set_lut(data, size);
    }
}

/// Scene point/spot/area lights (16 floats each) for direct lighting (NEE).
pub fn upload_lights(data: *const f32, count: u32) {
    let floats: &[f32] = if data.is_null() || count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data, count as usize * 16) }
    };
    if let Some(r) = RENDERER.lock().unwrap().as_mut() {
        r.0.set_lights(floats, count);
    }
}

// --- Texture manager (state lives in the renderer; the addon's mgr handle is ignored) ---

pub fn texture_manager_reset() {
    if let Some(r) = RENDERER.lock().unwrap().as_mut() {
        r.0.clear_textures();
    }
}

pub fn texture_add(data: &[u8], width: i32, height: i32, dxgi: u32) -> i32 {
    if width <= 0 || height <= 0 {
        return -1;
    }
    match RENDERER.lock().unwrap().as_mut() {
        Some(r) => r.0.texture_add(data.to_vec(), width as u32, height as u32, dxgi),
        None => -1,
    }
}

pub fn texture_upload_all() {
    if let Some(r) = RENDERER.lock().unwrap().as_mut() {
        r.0.upload_textures();
    }
}

pub fn texture_upload_one() -> bool {
    match RENDERER.lock().unwrap().as_mut() {
        Some(r) => r.0.upload_one_texture(),
        None => false,
    }
}

pub fn texture_pending_count() -> i32 {
    match RENDERER.lock().unwrap().as_ref() {
        Some(r) => r.0.pending_tex.len() as i32,
        None => 0,
    }
}

pub fn update_texture_descriptors() {
    if let Some(r) = RENDERER.lock().unwrap().as_mut() {
        r.0.update_texture_descriptors();
    }
}

pub fn render_frame() {
    if let Some(r) = RENDERER.lock().unwrap().as_mut() {
        r.0.render();
    }
}

pub fn readback(out: *mut f32, pixel_count: u32) -> bool {
    match RENDERER.lock().unwrap().as_ref() {
        Some(r) => {
            r.0.copy_to(out, pixel_count as usize);
            true
        }
        None => false,
    }
}

// ===================== Build / device setup =====================

fn build(width: u32, height: u32) -> Result<Renderer, String> {
    let entry = unsafe { ash::Entry::load() }
        .map_err(|e| format!("Entry::load (vulkan-1.dll missing?): {e}"))?;

    let app = vk::ApplicationInfo::default()
        .application_name(c"Ignis RT")
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(c"Ignis")
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_2);
    let inst_ci = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = unsafe { entry.create_instance(&inst_ci, None) }
        .map_err(|e| format!("create_instance: {e}"))?;

    let pds = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| format!("enumerate_physical_devices: {e}"))?;
    if pds.is_empty() {
        unsafe { instance.destroy_instance(None) };
        return Err("no Vulkan physical devices found".into());
    }
    let pd = pds
        .iter()
        .copied()
        .find(|&p| {
            unsafe { instance.get_physical_device_properties(p) }.device_type
                == vk::PhysicalDeviceType::DISCRETE_GPU
        })
        .unwrap_or(pds[0]);

    let props = unsafe { instance.get_physical_device_properties(pd) };
    let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    log(&format!("GPU: {name}"));

    let dev_exts = unsafe { instance.enumerate_device_extension_properties(pd) }
        .map_err(|e| format!("enumerate_device_extension_properties: {e}"))?;
    let has = |n: &CStr| {
        dev_exts
            .iter()
            .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == n)
    };
    let rt_supported = has(ash::khr::acceleration_structure::NAME)
        && has(ash::khr::ray_query::NAME)
        && has(ash::khr::ray_tracing_pipeline::NAME)
        && has(ash::khr::buffer_device_address::NAME)
        && has(ash::khr::deferred_host_operations::NAME);
    let rtx_series = detect_rtx_series(&name);
    log(&format!(
        "ray tracing supported: {rt_supported}, RTX series: {rtx_series}"
    ));

    // Scratch alignment for AS builds (only meaningful with RT).
    let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut as_props);
    unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
    let scratch_align = as_props
        .min_acceleration_structure_scratch_offset_alignment
        .max(1) as u64;

    let qfs = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    let queue_family = qfs
        .iter()
        .position(|q| {
            q.queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        })
        .ok_or("no graphics+compute queue family")? as u32;

    let mut dev_ext_ptrs: Vec<*const c_char> = Vec::new();
    if rt_supported {
        for n in [
            ash::khr::acceleration_structure::NAME,
            ash::khr::ray_query::NAME,
            ash::khr::ray_tracing_pipeline::NAME,
            ash::khr::buffer_device_address::NAME,
            ash::khr::deferred_host_operations::NAME,
            ash::ext::descriptor_indexing::NAME,
            ash::khr::spirv_1_4::NAME,
            ash::khr::shader_float_controls::NAME,
        ] {
            if has(n) {
                dev_ext_ptrs.push(n.as_ptr());
            }
        }
    }

    let supported = unsafe { instance.get_physical_device_features(pd) };
    let mut base = vk::PhysicalDeviceFeatures::default();
    base.sampler_anisotropy = supported.sampler_anisotropy;
    base.shader_int64 = supported.shader_int64;
    base.shader_storage_image_write_without_format =
        supported.shader_storage_image_write_without_format;

    let mut f_rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
    let mut f_rp =
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
    let mut f_as =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
    let mut f_di = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
        .descriptor_binding_partially_bound(true)
        .runtime_descriptor_array(true)
        .shader_sampled_image_array_non_uniform_indexing(true);
    let mut f_bda =
        vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
    let mut f_ai =
        vk::PhysicalDeviceShaderAtomicInt64Features::default().shader_buffer_int64_atomics(true);
    // Scalar block layout: lets the GLSL material struct match the C++ 2204-byte layout exactly
    // (needed for the Node VM bytecode at offset 156, which std430 would misalign).
    let mut f_scalar =
        vk::PhysicalDeviceScalarBlockLayoutFeatures::default().scalar_block_layout(true);

    let prio = [1.0f32];
    let q_ci = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&prio)];

    let mut dci = vk::DeviceCreateInfo::default()
        .queue_create_infos(&q_ci)
        .enabled_extension_names(&dev_ext_ptrs)
        .enabled_features(&base);
    if rt_supported {
        dci = dci
            .push_next(&mut f_rq)
            .push_next(&mut f_rp)
            .push_next(&mut f_as)
            .push_next(&mut f_di)
            .push_next(&mut f_bda)
            .push_next(&mut f_ai)
            .push_next(&mut f_scalar);
    }

    let device = unsafe { instance.create_device(pd, &dci, None) }
        .map_err(|e| format!("create_device: {e}"))?;
    let queue = unsafe { device.get_device_queue(queue_family, 0) };
    let accel_ext = if rt_supported {
        Some(AccelExt::new(&instance, &device))
    } else {
        None
    };

    let cp_ci = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family);
    let command_pool = unsafe { device.create_command_pool(&cp_ci, None) }
        .map_err(|e| format!("create_command_pool: {e}"))?;

    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device: pd,
        debug_settings: Default::default(),
        buffer_device_address: rt_supported,
        allocation_sizes: Default::default(),
    })
    .map_err(|e| format!("allocator init: {e}"))?;

    // Offscreen image.
    let img_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(RB_FORMAT)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let offscreen_image = unsafe { device.create_image(&img_ci, None) }
        .map_err(|e| format!("create_image: {e}"))?;
    let img_req = unsafe { device.get_image_memory_requirements(offscreen_image) };
    let offscreen_alloc = allocator
        .allocate(&AllocationCreateDesc {
            name: "offscreen",
            requirements: img_req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("offscreen alloc: {e}"))?;
    unsafe {
        device
            .bind_image_memory(offscreen_image, offscreen_alloc.memory(), offscreen_alloc.offset())
            .map_err(|e| format!("bind image: {e}"))?;
    }
    let offscreen_view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(offscreen_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(RB_FORMAT)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
            None,
        )
    }
    .map_err(|e| format!("create_image_view: {e}"))?;

    // Accumulation image (RGBA32F storage, persistent across frames for path-trace averaging).
    let accum_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(RB_FORMAT)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let accum_image = unsafe { device.create_image(&accum_ci, None) }
        .map_err(|e| format!("create accum image: {e}"))?;
    let accum_req = unsafe { device.get_image_memory_requirements(accum_image) };
    let accum_alloc = allocator
        .allocate(&AllocationCreateDesc {
            name: "accum",
            requirements: accum_req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("accum alloc: {e}"))?;
    unsafe {
        device
            .bind_image_memory(accum_image, accum_alloc.memory(), accum_alloc.offset())
            .map_err(|e| format!("bind accum: {e}"))?;
    }
    let accum_view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(accum_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(RB_FORMAT)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
            None,
        )
    }
    .map_err(|e| format!("accum view: {e}"))?;

    // Readback buffer.
    let rb_bytes = (width as u64) * (height as u64) * 16;
    let buf_ci = vk::BufferCreateInfo::default()
        .size(rb_bytes)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let readback_buffer = unsafe { device.create_buffer(&buf_ci, None) }
        .map_err(|e| format!("create_buffer: {e}"))?;
    let buf_req = unsafe { device.get_buffer_memory_requirements(readback_buffer) };
    let readback_alloc = allocator
        .allocate(&AllocationCreateDesc {
            name: "readback",
            requirements: buf_req,
            location: MemoryLocation::GpuToCpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("readback alloc: {e}"))?;
    unsafe {
        device
            .bind_buffer_memory(readback_buffer, readback_alloc.memory(), readback_alloc.offset())
            .map_err(|e| format!("bind buffer: {e}"))?;
    }
    let readback_ptr = readback_alloc
        .mapped_ptr()
        .ok_or("readback buffer not host-mapped")?
        .as_ptr() as usize;

    // World buffer (binding 8, RT only): vec4 background color + vec4 HDRI params
    // (x = environment texture index or -1, y = strength).
    let world_buffer = GpuBuffer::new(
        &device, &mut allocator, 32,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "world",
    )?;
    world_buffer.write_bytes(gpu::as_bytes(&[0.0f32, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0]));

    // Compute pipeline.
    // Binding 0: offscreen storage image (always). Binding 1: TLAS — only on RT devices
    // (partially bound: written after the scene loads, read only when hasTlas == 1). A
    // non-RT device can't have an acceleration-structure descriptor or ray query, so it
    // falls back to the sky shader and a single-binding layout.
    let mut bindings = vec![vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    let mut binding_flags = vec![vk::DescriptorBindingFlags::empty()];
    let mut pool_sizes = vec![vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)];
    if rt_supported {
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        binding_flags.push(vk::DescriptorBindingFlags::PARTIALLY_BOUND);
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
        );
        // Binding 2: geometry table. Binding 3: materials. (storage buffers)
        for binding in [2u32, 3u32] {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
            binding_flags.push(vk::DescriptorBindingFlags::PARTIALLY_BOUND);
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1),
            );
        }
        // Binding 4: bindless texture array.
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_TEXTURES)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        binding_flags.push(vk::DescriptorBindingFlags::PARTIALLY_BOUND);
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_TEXTURES),
        );
        // Binding 5: scene lights (storage buffer).
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        binding_flags.push(vk::DescriptorBindingFlags::PARTIALLY_BOUND);
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
        );
        // Binding 6: accumulation image (storage image).
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        binding_flags.push(vk::DescriptorBindingFlags::empty());
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
        );
        // Binding 7: OCIO 3D LUT (combined image sampler, uploaded later).
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        binding_flags.push(vk::DescriptorBindingFlags::PARTIALLY_BOUND);
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1),
        );
        // Binding 8: world/background color (storage buffer).
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(8)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        binding_flags.push(vk::DescriptorBindingFlags::empty());
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
        );
    }
    let mut flags_info =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
    let ds_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings)
                .push_next(&mut flags_info),
            None,
        )
    }
    .map_err(|e| format!("descriptor set layout: {e}"))?;

    let desc_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )
    }
    .map_err(|e| format!("descriptor pool: {e}"))?;

    let desc_set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(&[ds_layout]),
        )
    }
    .map_err(|e| format!("allocate descriptor set: {e}"))?[0];

    let img_info = [vk::DescriptorImageInfo::default()
        .image_view(offscreen_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let accum_info = [vk::DescriptorImageInfo::default()
        .image_view(accum_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let mut writes = vec![vk::WriteDescriptorSet::default()
        .dst_set(desc_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(&img_info)];
    let world_info = [vk::DescriptorBufferInfo::default()
        .buffer(world_buffer.buffer)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    if rt_supported {
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&accum_info),
        );
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(8)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&world_info),
        );
    }
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    let push_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<CameraPush>() as u32)];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[ds_layout])
                .push_constant_ranges(&push_range),
            None,
        )
    }
    .map_err(|e| format!("pipeline layout: {e}"))?;

    // Ray-query shader on RT devices, sky fallback otherwise (both embedded at build time).
    let spv_bytes: &[u8] = if rt_supported {
        &include_bytes!(concat!(env!("OUT_DIR"), "/trace.comp.spv"))[..]
    } else {
        &include_bytes!(concat!(env!("OUT_DIR"), "/sky.comp.spv"))[..]
    };
    let spv = ash::util::read_spv(&mut std::io::Cursor::new(spv_bytes))
        .map_err(|e| format!("read shader spv: {e}"))?;
    let shader_module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
    }
    .map_err(|e| format!("shader module: {e}"))?;

    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(c"main");
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout)],
            None,
        )
    }
    .map_err(|(_, e)| format!("compute pipeline: {e}"))?[0];
    unsafe { device.destroy_shader_module(shader_module, None) };

    let cmd = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .map_err(|e| format!("alloc cmd buffer: {e}"))?[0];
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .map_err(|e| format!("create_fence: {e}"))?;

    let tex_sampler = unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .max_lod(vk::LOD_CLAMP_NONE),
            None,
        )
    }
    .map_err(|e| format!("create_sampler: {e}"))?;

    // Clamp sampler for the OCIO LUT (no wrapping at the edges of the cube).
    let lut_sampler = unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            None,
        )
    }
    .map_err(|e| format!("create_lut_sampler: {e}"))?;

    // Put the accumulation image in GENERAL once (contents undefined; frame 0 overwrites them).
    submit_oneshot(&device, queue, cmd, fence, |d, c| unsafe {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(accum_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        d.cmd_pipeline_barrier(
            c,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    });

    log("Vulkan device + compute pipeline + offscreen target ready");

    Ok(Renderer {
        accel_ext,
        scratch_align,
        queued: Vec::new(),
        blas_list: Vec::new(),
        tlas: None,
        tlas_instance_data: Vec::new(),
        geom_table: None,
        mat_buffer: None,
        pending_tex: Vec::new(),
        textures: Vec::new(),
        tex_sampler,
        light_buffer: None,
        light_count: 0,
        light_data: Vec::new(),
        lut_image: vk::Image::null(),
        lut_alloc: None,
        lut_view: vk::ImageView::null(),
        lut_sampler,
        has_lut: false,
        world_buffer: Some(world_buffer),
        world_data: [-1.0; 8], // impossible -> first render writes the real value
        offscreen_view,
        ds_layout,
        desc_pool,
        desc_set,
        pipeline_layout,
        pipeline,
        inv_view_proj: [0.0; 16],
        cam_pos: [0.0; 4],
        accum_image,
        accum_alloc: Some(accum_alloc),
        accum_view,
        accum_frame: 0,
        prev_view: [0.0; 16],
        offscreen_image,
        offscreen_alloc: Some(offscreen_alloc),
        readback_buffer,
        readback_alloc: Some(readback_alloc),
        readback_ptr,
        cmd,
        fence,
        allocator: Some(allocator),
        command_pool,
        device,
        instance,
        _entry: entry,
        physical_device: pd,
        queue,
        queue_family,
        rt_supported,
        rtx_series,
        width,
        height,
    })
}

impl Renderer {
    fn queue_mesh(
        &mut self,
        positions: &[f32],
        vertex_count: u32,
        indices: &[u32],
        normals: &[f32],
        uvs: &[f32],
    ) -> i32 {
        let handle = (self.blas_list.len() + self.queued.len()) as i32;
        self.queued.push(QueuedMesh {
            positions: positions.to_vec(),
            indices: indices.to_vec(),
            normals: normals.to_vec(),
            uvs: uvs.to_vec(),
            vertex_count,
        });
        handle
    }

    fn flush_mesh_batch(&mut self) -> i32 {
        if self.accel_ext.is_none() {
            log("flush_mesh_batch: RT not supported, skipping BLAS build");
            self.queued.clear();
            return 0;
        }
        let device = self.device.clone();
        let accel = self.accel_ext.clone().unwrap();
        let (queue, cmd, fence, scratch_align) =
            (self.queue, self.cmd, self.fence, self.scratch_align);
        let queued = std::mem::take(&mut self.queued);
        let alloc = self.allocator.as_mut().unwrap();
        let mut built = 0;
        for mesh in queued {
            match build_blas(&device, alloc, &accel, queue, cmd, fence, scratch_align, &mesh) {
                Ok(b) => {
                    log(&format!(
                        "BLAS built: {} verts, {} tris -> {:#x}",
                        mesh.vertex_count,
                        mesh.indices.len() / 3,
                        b.address
                    ));
                    self.blas_list.push(Some(b));
                    built += 1;
                }
                Err(e) => {
                    log(&format!("BLAS build FAILED: {e}"));
                    self.blas_list.push(None);
                }
            }
        }
        log(&format!("flush_mesh_batch: {built} BLAS built"));
        self.update_geom_table();
        built
    }

    /// Rebuild the per-instance geometry table (vertex/index buffer device addresses, indexed
    /// by BLAS handle = TLAS instanceCustomIndex) and point descriptor binding 2 at it.
    fn update_geom_table(&mut self) {
        let device = self.device.clone();
        let addr = |b: &Option<GpuBuffer>| b.as_ref().map(|x| x.device_address(&device)).unwrap_or(0);
        let mut descs: Vec<[u64; 6]> = Vec::with_capacity(self.blas_list.len());
        for slot in &self.blas_list {
            match slot {
                Some(b) => descs.push([
                    b.vbuf.device_address(&device),
                    b.ibuf.device_address(&device),
                    addr(&b.nbuf),
                    addr(&b.matbuf),
                    addr(&b.ubuf),
                    0,
                ]),
                None => descs.push([0; 6]),
            }
        }
        if descs.is_empty() {
            return;
        }
        let alloc = self.allocator.as_mut().unwrap();
        if let Some(old) = self.geom_table.take() {
            old.destroy(&device, alloc);
        }
        let buf = match GpuBuffer::new(
            &device,
            alloc,
            (descs.len() * 48) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
            "geom_table",
        ) {
            Ok(b) => b,
            Err(e) => {
                log(&format!("geom_table alloc FAILED: {e}"));
                return;
            }
        };
        buf.write_bytes(gpu::as_bytes(&descs));

        let info = [vk::DescriptorBufferInfo::default()
            .buffer(buf.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.desc_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info);
        unsafe { device.update_descriptor_sets(&[write], &[]) };
        self.geom_table = Some(buf);
        log(&format!("geom_table: {} entries", descs.len()));
    }

    fn set_materials(&mut self, bytes: &[u8]) {
        if bytes.is_empty() || self.accel_ext.is_none() {
            return;
        }
        let device = self.device.clone();
        let alloc = self.allocator.as_mut().unwrap();
        if let Some(old) = self.mat_buffer.take() {
            old.destroy(&device, alloc);
        }
        let buf = match GpuBuffer::new(
            &device,
            alloc,
            bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
            "materials",
        ) {
            Ok(b) => b,
            Err(e) => {
                log(&format!("materials alloc FAILED: {e}"));
                return;
            }
        };
        buf.write_bytes(bytes);
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(buf.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.desc_set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info);
        unsafe { device.update_descriptor_sets(&[write], &[]) };
        self.mat_buffer = Some(buf);
        self.accum_frame = 0;
        log(&format!("materials: {} full (2204b each)", bytes.len() / 2204));
    }

    fn set_blas_materials(&mut self, handle: i32, ids: &[u32]) {
        if handle < 0 || ids.is_empty() {
            return;
        }
        let device = self.device.clone();
        let idx = handle as usize;
        let alloc = self.allocator.as_mut().unwrap();
        let buf = match GpuBuffer::new(
            &device,
            alloc,
            (ids.len() * 4) as u64,
            GEOM_REF_USAGE,
            MemoryLocation::CpuToGpu,
            "blas_matids",
        ) {
            Ok(b) => b,
            Err(e) => {
                log(&format!("matids alloc FAILED: {e}"));
                return;
            }
        };
        buf.write_bytes(gpu::as_bytes(ids));
        if let Some(Some(b)) = self.blas_list.get_mut(idx) {
            if let Some(old) = b.matbuf.take() {
                old.destroy(&device, alloc);
            }
            b.matbuf = Some(buf);
        } else {
            buf.destroy(&device, alloc);
        }
    }

    /// Attach per-vertex normals + UVs to an already-built BLAS (the addon's incremental upload
    /// path: ignis_upload_mesh builds positions-only, then ignis_upload_mesh_attributes adds
    /// these). Without this the mesh has no smooth normals (flat/faceted) and no UVs (everything
    /// samples (0,0) -> badly mapped textures). Rebuilds the geom table so the new addresses bind.
    fn set_blas_attributes(&mut self, handle: i32, normals: &[f32], uvs: &[f32]) {
        if handle < 0 || self.accel_ext.is_none() {
            return;
        }
        let idx = handle as usize;
        if !matches!(self.blas_list.get(idx), Some(Some(_))) {
            return;
        }
        let device = self.device.clone();
        let alloc = self.allocator.as_mut().unwrap();
        let mut mk = |src: &[f32], name: &str| -> Option<GpuBuffer> {
            if src.is_empty() {
                return None;
            }
            match GpuBuffer::new(
                &device, alloc, (src.len() * 4) as u64,
                GEOM_REF_USAGE, MemoryLocation::CpuToGpu, name,
            ) {
                Ok(b) => { b.write_bytes(gpu::as_bytes(src)); Some(b) }
                Err(e) => { log(&format!("blas attr alloc FAILED: {e}")); None }
            }
        };
        let nbuf = mk(normals, "blas_normals");
        let ubuf = mk(uvs, "blas_uvs");
        if let Some(Some(b)) = self.blas_list.get_mut(idx) {
            if nbuf.is_some() {
                if let Some(old) = b.nbuf.take() { old.destroy(&device, alloc); }
                b.nbuf = nbuf;
            }
            if ubuf.is_some() {
                if let Some(old) = b.ubuf.take() { old.destroy(&device, alloc); }
                b.ubuf = ubuf;
            }
        }
        self.update_geom_table();
        self.accum_frame = 0;
    }

    fn set_lights(&mut self, floats: &[f32], count: u32) {
        // The addon re-sends lights every frame; skip work + accumulation reset if unchanged.
        if count == self.light_count && floats == self.light_data.as_slice() {
            return;
        }
        self.light_count = count;
        self.light_data = floats.to_vec();
        if count == 0 || self.accel_ext.is_none() {
            return;
        }
        let device = self.device.clone();
        let alloc = self.allocator.as_mut().unwrap();
        if let Some(old) = self.light_buffer.take() {
            old.destroy(&device, alloc);
        }
        let buf = match GpuBuffer::new(
            &device,
            alloc,
            (count as usize * 64) as u64, // 16 floats per light
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
            "lights",
        ) {
            Ok(b) => b,
            Err(e) => {
                log(&format!("lights alloc FAILED: {e}"));
                return;
            }
        };
        buf.write_bytes(gpu::as_bytes(floats));
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(buf.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.desc_set)
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info);
        unsafe { device.update_descriptor_sets(&[write], &[]) };
        self.light_buffer = Some(buf);
        self.accum_frame = 0;
        log(&format!("lights: {count} uploaded"));
    }

    /// Upload the OCIO view-transform LUT as a 3D texture (RGB -> RGBA) bound at binding 7.
    fn set_lut(&mut self, data: &[f32], size: u32) {
        if self.accel_ext.is_none() || size < 2 {
            return;
        }
        let n = (size as usize).pow(3);
        if data.len() < n * 3 {
            log("lut: data too small");
            return;
        }
        let mut rgba = vec![0f32; n * 4];
        for i in 0..n {
            rgba[i * 4] = data[i * 3];
            rgba[i * 4 + 1] = data[i * 3 + 1];
            rgba[i * 4 + 2] = data[i * 3 + 2];
            rgba[i * 4 + 3] = 1.0;
        }

        let device = self.device.clone();
        let (queue, cmd, fence) = (self.queue, self.cmd, self.fence);
        let alloc = self.allocator.as_mut().unwrap();

        if self.has_lut {
            unsafe {
                device.destroy_image_view(self.lut_view, None);
                device.destroy_image(self.lut_image, None);
            }
            if let Some(a) = self.lut_alloc.take() {
                let _ = alloc.free(a);
            }
            self.has_lut = false;
        }

        let img_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(RB_FORMAT)
            .extent(vk::Extent3D { width: size, height: size, depth: size })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = match unsafe { device.create_image(&img_ci, None) } {
            Ok(i) => i,
            Err(e) => { log(&format!("lut image: {e}")); return; }
        };
        let req = unsafe { device.get_image_memory_requirements(image) };
        let ialloc = match alloc.allocate(&AllocationCreateDesc {
            name: "lut",
            requirements: req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => { unsafe { device.destroy_image(image, None) }; log(&format!("lut alloc: {e}")); return; }
        };
        unsafe { let _ = device.bind_image_memory(image, ialloc.memory(), ialloc.offset()); }

        let staging = match GpuBuffer::new(
            &device, alloc, (rgba.len() * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "lut_staging",
        ) {
            Ok(b) => b,
            Err(e) => { log(&format!("lut staging: {e}")); return; }
        };
        staging.write_bytes(gpu::as_bytes(&rgba));

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        submit_oneshot(&device, queue, cmd, fence, |d, c| unsafe {
            let to_dst = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED).new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image).subresource_range(range);
            d.cmd_pipeline_barrier(c, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_extent(vk::Extent3D { width: size, height: size, depth: size });
            d.cmd_copy_buffer_to_image(c, staging.buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
            let to_read = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image).subresource_range(range);
            d.cmd_pipeline_barrier(c, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &[], &[], &[to_read]);
        });
        staging.destroy(&device, alloc);

        let view = match unsafe {
            device.create_image_view(&vk::ImageViewCreateInfo::default()
                .image(image).view_type(vk::ImageViewType::TYPE_3D).format(RB_FORMAT).subresource_range(range), None)
        } {
            Ok(v) => v,
            Err(e) => { log(&format!("lut view: {e}")); return; }
        };
        let info = [vk::DescriptorImageInfo::default()
            .image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).sampler(self.lut_sampler)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.desc_set).dst_binding(7).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&info);
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        self.lut_image = image;
        self.lut_alloc = Some(ialloc);
        self.lut_view = view;
        self.has_lut = true;
        log(&format!("lut: {size}^3 uploaded"));
    }

    fn texture_add(&mut self, data: Vec<u8>, width: u32, height: u32, dxgi: u32) -> i32 {
        let (format, _bpp) = dxgi_to_vk(dxgi);
        let idx = (self.pending_tex.len() + self.textures.len()) as i32;
        self.pending_tex.push(PendingTex { data, width, height, format });
        idx
    }

    /// Upload one pending texture (front of the queue) to the GPU. Returns false when empty.
    fn upload_one_texture(&mut self) -> bool {
        if self.pending_tex.is_empty() {
            return false;
        }
        if self.accel_ext.is_none() {
            self.pending_tex.clear();
            return false;
        }
        let pt = self.pending_tex.remove(0);
        let device = self.device.clone();
        let (queue, cmd, fence) = (self.queue, self.cmd, self.fence);
        let alloc = self.allocator.as_mut().unwrap();
        let tex = upload_texture(&device, alloc, queue, cmd, fence, &pt).or_else(|e| {
            log(&format!("texture {}x{} upload FAILED: {e} — using 1x1 white", pt.width, pt.height));
            let white = PendingTex {
                data: vec![255u8; 4],
                width: 1,
                height: 1,
                format: vk::Format::R8G8B8A8_UNORM,
            };
            upload_texture(&device, alloc, queue, cmd, fence, &white)
        });
        match tex {
            Ok(t) => {
                self.textures.push(t);
                true
            }
            Err(e) => {
                log(&format!("texture fallback also failed: {e}"));
                false
            }
        }
    }

    fn upload_textures(&mut self) {
        let mut n = 0;
        while self.upload_one_texture() {
            n += 1;
        }
        if n > 0 {
            log(&format!("textures: {n} uploaded, {} total", self.textures.len()));
        }
    }

    fn update_texture_descriptors(&mut self) {
        if self.textures.is_empty() {
            return;
        }
        let device = self.device.clone();
        let infos: Vec<vk::DescriptorImageInfo> = self
            .textures
            .iter()
            .map(|t| {
                vk::DescriptorImageInfo::default()
                    .image_view(t.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(self.tex_sampler)
            })
            .collect();
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.desc_set)
            .dst_binding(4)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&infos);
        unsafe { device.update_descriptor_sets(&[write], &[]) };
        log(&format!("texture descriptors: {} bound", infos.len()));
    }

    fn clear_textures(&mut self) {
        let device = self.device.clone();
        if let Some(alloc) = self.allocator.as_mut() {
            for t in self.textures.drain(..) {
                unsafe {
                    device.destroy_image_view(t.view, None);
                    device.destroy_image(t.image, None);
                }
                if let Some(a) = t.alloc {
                    let _ = alloc.free(a);
                }
            }
        }
        self.pending_tex.clear();
    }

    fn build_tlas(&mut self, instances: *const u8, count: u32) -> bool {
        let accel = match self.accel_ext.clone() {
            Some(a) => a,
            None => {
                log("build_tlas: RT not supported, skipping");
                return false;
            }
        };
        if instances.is_null() || count == 0 {
            return false;
        }
        // The addon may rebuild the TLAS every frame; skip if the instances are unchanged
        // (avoids a per-frame accumulation reset that prevents convergence).
        let in_bytes = unsafe { std::slice::from_raw_parts(instances, count as usize * 60) };
        if in_bytes == self.tlas_instance_data.as_slice() {
            return true;
        }
        self.tlas_instance_data = in_bytes.to_vec();

        // Refresh the geometry table: per-triangle material ids arrive after flush.
        self.update_geom_table();
        let in_slice =
            unsafe { std::slice::from_raw_parts(instances as *const TlasInstanceIn, count as usize) };

        // Convert to VkAccelerationStructureInstanceKHR.
        let mut vk_instances: Vec<vk::AccelerationStructureInstanceKHR> = Vec::with_capacity(count as usize);
        let mut missing = 0;
        for inst in in_slice {
            let addr = self
                .blas_list
                .get(inst.blas_index as usize)
                .and_then(|o| o.as_ref())
                .map(|b| b.address)
                .unwrap_or(0);
            if addr == 0 {
                missing += 1;
            }
            vk_instances.push(vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: inst.transform },
                instance_custom_index_and_mask: vk::Packed24_8::new(
                    inst.custom_index & 0x00FF_FFFF,
                    (inst.mask & 0xFF) as u8,
                ),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: addr,
                },
            });
        }
        if missing > 0 {
            log(&format!("build_tlas: {missing}/{count} instances had no BLAS"));
        }

        let device = self.device.clone();
        let (queue, cmd, fence, scratch_align) =
            (self.queue, self.cmd, self.fence, self.scratch_align);
        // Replace any prior TLAS.
        if let (Some(old), Some(alloc)) = (self.tlas.take(), self.allocator.as_mut()) {
            unsafe { accel.destroy_acceleration_structure(old.accel, None) };
            old.buf.destroy(&device, alloc);
            old.instbuf.destroy(&device, alloc);
        }
        let alloc = self.allocator.as_mut().unwrap();
        match build_tlas_inner(
            &device, alloc, &accel, queue, cmd, fence, scratch_align, &vk_instances,
        ) {
            Ok(t) => {
                // Point the compute shader's binding 1 at the new TLAS.
                let handles = [t.accel];
                let mut as_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                    .acceleration_structures(&handles);
                let mut write = vk::WriteDescriptorSet::default()
                    .dst_set(self.desc_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .push_next(&mut as_write);
                write.descriptor_count = 1;
                unsafe { self.device.update_descriptor_sets(&[write], &[]) };

                self.tlas = Some(t);
                self.accum_frame = 0; // scene changed -> restart accumulation
                log(&format!("TLAS built: {count} instances"));
                true
            }
            Err(e) => {
                log(&format!("TLAS build FAILED: {e}"));
                false
            }
        }
    }

    fn clear_geometry(&mut self) {
        let device = self.device.clone();
        let accel = self.accel_ext.clone();
        if let (Some(accel), Some(alloc)) = (accel.as_ref(), self.allocator.as_mut()) {
            if let Some(t) = self.tlas.take() {
                unsafe { accel.destroy_acceleration_structure(t.accel, None) };
                t.buf.destroy(&device, alloc);
                t.instbuf.destroy(&device, alloc);
            }
            if let Some(g) = self.geom_table.take() {
                g.destroy(&device, alloc);
            }
            if let Some(m) = self.mat_buffer.take() {
                m.destroy(&device, alloc);
            }
            for slot in self.blas_list.drain(..) {
                if let Some(b) = slot {
                    unsafe { accel.destroy_acceleration_structure(b.accel, None) };
                    b.buf.destroy(&device, alloc);
                    b.vbuf.destroy(&device, alloc);
                    b.ibuf.destroy(&device, alloc);
                    if let Some(n) = b.nbuf {
                        n.destroy(&device, alloc);
                    }
                    if let Some(u) = b.ubuf {
                        u.destroy(&device, alloc);
                    }
                    if let Some(m) = b.matbuf {
                        m.destroy(&device, alloc);
                    }
                }
            }
        }
        self.queued.clear();
        log("clear_geometry: all BLAS/TLAS released");
    }

    fn render(&mut self) {
        // World buffer: background color (color*strength*0.15) + HDRI params, set by the addon.
        // HDRI is uploaded into the bindless texture array; we just need its index + strength.
        let wd = [
            crate::config::get_float("world_bg_r"),
            crate::config::get_float("world_bg_g"),
            crate::config::get_float("world_bg_b"),
            0.0,
            crate::config::get_int("hdri_tex_index") as f32, // -1 = no HDRI
            crate::config::get_float("hdri_strength"),
            0.0,
            0.0,
        ];
        if wd != self.world_data {
            self.world_data = wd;
            if let Some(b) = &self.world_buffer {
                b.write_bytes(gpu::as_bytes(&wd));
            }
            self.accum_frame = 0; // world changed -> restart accumulation
        }

        let d = &self.device;
        // Sun from scene config (azimuth/elevation -> direction, Y-up, matching the C++).
        let az = crate::config::get_float("sun_azimuth").to_radians();
        let el = crate::config::get_float("sun_elevation").to_radians();
        let mut intensity = crate::config::get_float("sun_intensity");
        if intensity <= 0.0 {
            intensity = 1.0;
        }
        let mut sc = [
            crate::config::get_float("sun_color_r"),
            crate::config::get_float("sun_color_g"),
            crate::config::get_float("sun_color_b"),
        ];
        if sc[0] + sc[1] + sc[2] <= 0.0 {
            sc = [1.0, 1.0, 1.0];
        }
        // Sun angular radius (radians) for soft shadows — packed into the unused cam_pos.w.
        let sun_size = crate::config::get_float("sun_size");
        let mut cam = self.cam_pos;
        cam[3] = if sun_size > 0.0 { sun_size } else { 0.009 };
        let push = CameraPush {
            inv_view_proj: self.inv_view_proj,
            cam_pos: cam,
            dims: [self.width, self.height],
            has_tlas: (self.tlas.is_some() as u32) | (if self.has_lut { 2 } else { 0 }),
            num_lights: self.light_count,
            sun_dir: [az.sin() * el.cos(), el.sin(), az.cos() * el.cos(), intensity],
            sun_col: [sc[0], sc[1], sc[2], self.accum_frame as f32],
        };
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        unsafe {
            let _ = d.reset_fences(&[self.fence]);
            let _ = d.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            );

            let to_general = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.offscreen_image)
                .subresource_range(range);
            // Accumulation image stays in GENERAL across frames; this barrier makes the
            // previous frame's writes available to this frame's read-modify-write.
            let accum_bar = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.accum_image)
                .subresource_range(range);
            d.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_general, accum_bar],
            );

            d.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.desc_set],
                &[],
            );
            let bytes = std::slice::from_raw_parts(
                &push as *const CameraPush as *const u8,
                std::mem::size_of::<CameraPush>(),
            );
            d.cmd_push_constants(
                self.cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
            d.cmd_dispatch(self.cmd, self.width.div_ceil(8), self.height.div_ceil(8), 1);

            let to_src = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.offscreen_image)
                .subresource_range(range);
            d.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_src],
            );

            let region = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                });
            d.cmd_copy_image_to_buffer(
                self.cmd,
                self.offscreen_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback_buffer,
                &[region],
            );

            let _ = d.end_command_buffer(self.cmd);
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd));
            let _ = d.queue_submit(self.queue, &[submit], self.fence);
            let _ = d.wait_for_fences(&[self.fence], true, u64::MAX);
        }
        self.accum_frame = self.accum_frame.wrapping_add(1);
    }

    fn copy_to(&self, out: *mut f32, pixel_count: usize) {
        if out.is_null() {
            return;
        }
        let avail = (self.width as usize) * (self.height as usize) * 4;
        let n = (pixel_count * 4).min(avail);
        unsafe { std::ptr::copy_nonoverlapping(self.readback_ptr as *const f32, out, n) };
    }
}

// ===================== Acceleration structure builders =====================

const AS_INPUT_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR.as_raw()
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
);
const AS_STORAGE_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR.as_raw()
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
);
const SCRATCH_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
);
// Buffers read in-shader via buffer_reference (normals): storage + device address.
const GEOM_REF_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
);

#[allow(clippy::too_many_arguments)]
fn build_blas(
    device: &ash::Device,
    allocator: &mut Allocator,
    accel: &AccelExt,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    scratch_align: u64,
    mesh: &QueuedMesh,
) -> Result<Blas, String> {
    let vbuf = GpuBuffer::new(
        device,
        allocator,
        (mesh.positions.len() * 4) as u64,
        AS_INPUT_USAGE,
        MemoryLocation::CpuToGpu,
        "blas_verts",
    )?;
    vbuf.write_bytes(gpu::as_bytes(&mesh.positions));
    let ibuf = GpuBuffer::new(
        device,
        allocator,
        (mesh.indices.len() * 4) as u64,
        AS_INPUT_USAGE,
        MemoryLocation::CpuToGpu,
        "blas_idx",
    )?;
    ibuf.write_bytes(gpu::as_bytes(&mesh.indices));

    let tri_count = (mesh.indices.len() / 3) as u32;
    let vaddr = vbuf.device_address(device);
    let iaddr = ibuf.device_address(device);

    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vaddr })
        .vertex_stride(12)
        .max_vertex(mesh.vertex_count.saturating_sub(1))
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR { device_address: iaddr });
    let geos = [vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        .flags(vk::GeometryFlagsKHR::OPAQUE)];

    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geos);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        accel.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[tri_count],
            &mut sizes,
        );
    }

    let buf = GpuBuffer::new(
        device,
        allocator,
        sizes.acceleration_structure_size,
        AS_STORAGE_USAGE,
        MemoryLocation::GpuOnly,
        "blas",
    )?;
    let accel_struct = unsafe {
        accel.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(buf.buffer)
                .size(sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
            None,
        )
    }
    .map_err(|e| format!("create BLAS: {e}"))?;

    let scratch = GpuBuffer::new(
        device,
        allocator,
        sizes.build_scratch_size + scratch_align,
        SCRATCH_USAGE,
        MemoryLocation::GpuOnly,
        "blas_scratch",
    )?;
    let scratch_addr = align_up(scratch.device_address(device), scratch_align);

    let build_info = build_info
        .dst_acceleration_structure(accel_struct)
        .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_addr });
    let range = vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(tri_count);

    submit_oneshot(device, queue, cmd, fence, |_, c| unsafe {
        accel.cmd_build_acceleration_structures(
            c,
            std::slice::from_ref(&build_info),
            &[std::slice::from_ref(&range)],
        );
    });

    let address = unsafe {
        accel.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                .acceleration_structure(accel_struct),
        )
    };
    scratch.destroy(device, allocator);

    // Per-vertex normals + UVs (read in-shader via buffer_reference).
    let mut mk = |src: &[f32], name: &str| -> Result<Option<GpuBuffer>, String> {
        if src.is_empty() {
            return Ok(None);
        }
        let b = GpuBuffer::new(
            device,
            allocator,
            (src.len() * 4) as u64,
            GEOM_REF_USAGE,
            MemoryLocation::CpuToGpu,
            name,
        )?;
        b.write_bytes(gpu::as_bytes(src));
        Ok(Some(b))
    };
    let nbuf = mk(&mesh.normals, "blas_normals")?;
    let ubuf = mk(&mesh.uvs, "blas_uvs")?;

    Ok(Blas { accel: accel_struct, buf, vbuf, ibuf, nbuf, ubuf, matbuf: None, address })
}

#[allow(clippy::too_many_arguments)]
fn build_tlas_inner(
    device: &ash::Device,
    allocator: &mut Allocator,
    accel: &AccelExt,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    scratch_align: u64,
    instances: &[vk::AccelerationStructureInstanceKHR],
) -> Result<Tlas, String> {
    let inst_bytes = std::mem::size_of_val(instances) as u64;
    let instbuf = GpuBuffer::new(
        device,
        allocator,
        inst_bytes,
        AS_INPUT_USAGE,
        MemoryLocation::CpuToGpu,
        "tlas_instances",
    )?;
    instbuf.write_bytes(gpu::as_bytes(instances));
    let inst_addr = instbuf.device_address(device);

    let geos = [vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR { device_address: inst_addr }),
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE)];

    let count = instances.len() as u32;
    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geos);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        accel.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[count],
            &mut sizes,
        );
    }

    let buf = GpuBuffer::new(
        device,
        allocator,
        sizes.acceleration_structure_size,
        AS_STORAGE_USAGE,
        MemoryLocation::GpuOnly,
        "tlas",
    )?;
    let accel_struct = unsafe {
        accel.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(buf.buffer)
                .size(sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
            None,
        )
    }
    .map_err(|e| format!("create TLAS: {e}"))?;

    let scratch = GpuBuffer::new(
        device,
        allocator,
        sizes.build_scratch_size + scratch_align,
        SCRATCH_USAGE,
        MemoryLocation::GpuOnly,
        "tlas_scratch",
    )?;
    let scratch_addr = align_up(scratch.device_address(device), scratch_align);

    let build_info = build_info
        .dst_acceleration_structure(accel_struct)
        .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_addr });
    let range = vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count);

    submit_oneshot(device, queue, cmd, fence, |_, c| unsafe {
        accel.cmd_build_acceleration_structures(
            c,
            std::slice::from_ref(&build_info),
            &[std::slice::from_ref(&range)],
        );
    });

    scratch.destroy(device, allocator);
    Ok(Tlas { accel: accel_struct, buf, instbuf })
}

fn upload_texture(
    device: &ash::Device,
    allocator: &mut Allocator,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    pt: &PendingTex,
) -> Result<Texture, String> {
    let img_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(pt.format)
        .extent(vk::Extent3D { width: pt.width, height: pt.height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let image = unsafe { device.create_image(&img_ci, None) }.map_err(|e| format!("tex image: {e}"))?;
    let req = unsafe { device.get_image_memory_requirements(image) };
    let alloc = allocator
        .allocate(&AllocationCreateDesc {
            name: "texture",
            requirements: req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("tex alloc: {e}"))?;
    unsafe { device.bind_image_memory(image, alloc.memory(), alloc.offset()) }
        .map_err(|e| format!("tex bind: {e}"))?;

    let staging = GpuBuffer::new(
        device,
        allocator,
        pt.data.len().max(1) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
        "tex_staging",
    )?;
    staging.write_bytes(&pt.data);

    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);
    submit_oneshot(device, queue, cmd, fence, |d, c| unsafe {
        let to_dst = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        d.cmd_pipeline_barrier(
            c,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_dst],
        );
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D { width: pt.width, height: pt.height, depth: 1 });
        d.cmd_copy_buffer_to_image(
            c,
            staging.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
        let to_read = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        d.cmd_pipeline_barrier(
            c,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_read],
        );
    });
    staging.destroy(device, allocator);

    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(pt.format)
                .subresource_range(range),
            None,
        )
    }
    .map_err(|e| format!("tex view: {e}"))?;

    Ok(Texture { image, alloc: Some(alloc), view })
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let device = self.device.clone();
        unsafe { let _ = device.device_wait_idle(); }

        // Geometry first (needs accel_ext + allocator).
        let accel = self.accel_ext.clone();
        if let (Some(accel), Some(alloc)) = (accel.as_ref(), self.allocator.as_mut()) {
            if let Some(t) = self.tlas.take() {
                unsafe { accel.destroy_acceleration_structure(t.accel, None) };
                t.buf.destroy(&device, alloc);
                t.instbuf.destroy(&device, alloc);
            }
            if let Some(g) = self.geom_table.take() {
                g.destroy(&device, alloc);
            }
            if let Some(m) = self.mat_buffer.take() {
                m.destroy(&device, alloc);
            }
            for slot in self.blas_list.drain(..) {
                if let Some(b) = slot {
                    unsafe { accel.destroy_acceleration_structure(b.accel, None) };
                    b.buf.destroy(&device, alloc);
                    b.vbuf.destroy(&device, alloc);
                    b.ibuf.destroy(&device, alloc);
                    if let Some(n) = b.nbuf {
                        n.destroy(&device, alloc);
                    }
                    if let Some(u) = b.ubuf {
                        u.destroy(&device, alloc);
                    }
                    if let Some(m) = b.matbuf {
                        m.destroy(&device, alloc);
                    }
                }
            }
        }

        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.desc_pool, None);
            device.destroy_descriptor_set_layout(self.ds_layout, None);
            device.destroy_image_view(self.offscreen_view, None);
            device.destroy_image_view(self.accum_view, None);
            device.destroy_fence(self.fence, None);
            device.destroy_image(self.offscreen_image, None);
            device.destroy_image(self.accum_image, None);
            device.destroy_buffer(self.readback_buffer, None);
            if let Some(alloc) = &mut self.allocator {
                for t in self.textures.drain(..) {
                    device.destroy_image_view(t.view, None);
                    device.destroy_image(t.image, None);
                    if let Some(a) = t.alloc {
                        let _ = alloc.free(a);
                    }
                }
                if let Some(a) = self.accum_alloc.take() {
                    let _ = alloc.free(a);
                }
                if let Some(a) = self.offscreen_alloc.take() {
                    let _ = alloc.free(a);
                }
                if let Some(a) = self.readback_alloc.take() {
                    let _ = alloc.free(a);
                }
                if let Some(b) = self.light_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if let Some(b) = self.world_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if self.has_lut {
                    device.destroy_image_view(self.lut_view, None);
                    device.destroy_image(self.lut_image, None);
                    if let Some(a) = self.lut_alloc.take() {
                        let _ = alloc.free(a);
                    }
                }
            }
            device.destroy_sampler(self.tex_sampler, None);
            device.destroy_sampler(self.lut_sampler, None);
            self.allocator = None;
            device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = (
            self.desc_set,
            self.readback_ptr,
            self.physical_device,
            self.queue_family,
            self.rt_supported,
            self.rtx_series,
        );
    }
}
