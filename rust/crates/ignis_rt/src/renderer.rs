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

/// Decode a texture payload to GPU-ready pixels. Material textures come from the addon as
/// ENCODED file bytes (PNG/JPEG/BMP, dxgi 0); decode them to RGBA8 like the C++ stb_image path.
/// Raw payloads (float16 dxgi 10, raw RGBA8 dxgi 28) pass through unchanged.
fn decode_texture(data: Vec<u8>, width: u32, height: u32, dxgi: u32) -> (Vec<u8>, u32, u32, vk::Format) {
    let encoded = data.len() >= 4
        && (&data[..4] == b"\x89PNG"               // PNG
            || data[..3] == [0xFF, 0xD8, 0xFF]      // JPEG
            || &data[..2] == b"BM");                // BMP
    if encoded {
        match image::load_from_memory(&data) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (w, h) = (rgba.width(), rgba.height());
                return (rgba.into_raw(), w, h, vk::Format::R8G8B8A8_UNORM);
            }
            Err(e) => log(&format!("image decode failed ({} bytes): {e} — using raw", data.len())),
        }
    }
    let (format, _bpp) = dxgi_to_vk(dxgi);
    (data, width, height, format)
}

/// DLSS render (trace) resolution for a display size + quality mode (addon enum 1..6). Matches the
/// C++ fallback ratios; DLAA (6) and unknown = native (render == display, no upscaling). Rounded up
/// to even and clamped to the display size.
fn dlss_render_res(display_w: u32, display_h: u32, quality: i32) -> (u32, u32) {
    let ratio: f32 = match quality {
        1 => 3.0, // Ultra Performance
        2 => 2.0, // Performance
        3 => 1.7, // Balanced
        4 => 1.5, // Quality
        5 => 1.3, // Ultra Quality
        _ => 1.0, // DLAA / native
    };
    let scale = |d: u32| -> u32 { (((d as f32 / ratio) as u32 + 1) & !1u32).clamp(2, d) };
    (scale(display_w), scale(display_h))
}

/// NGX PerfQuality value for the addon quality enum (NGX: MaxPerf=0, Balanced=1, MaxQuality=2,
/// UltraPerformance=3, UltraQuality=4, DLAA=5).
fn dlss_perf_quality(quality: i32) -> i32 {
    match quality {
        1 => 3, // Ultra Performance
        2 => 0, // Performance (MaxPerf)
        3 => 1, // Balanced
        4 => 2, // Quality (MaxQuality)
        5 => 4, // Ultra Quality
        _ => 5, // DLAA
    }
}

/// Create a GPU-only 2D storage image + view (UNDEFINED layout). Used for the DLSS guide buffers
/// (depth / normal+roughness / albedo / motion vectors) written by the path tracer.
fn create_storage_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    width: u32,
    height: u32,
    format: vk::Format,
    name: &'static str,
) -> Result<(vk::Image, Allocation, vk::ImageView), String> {
    let ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let image = unsafe { device.create_image(&ci, None) }.map_err(|e| format!("{name} image: {e}"))?;
    let req = unsafe { device.get_image_memory_requirements(image) };
    let alloc = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements: req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("{name} alloc: {e}"))?;
    unsafe { device.bind_image_memory(image, alloc.memory(), alloc.offset()) }
        .map_err(|e| format!("{name} bind: {e}"))?;
    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
            None,
        )
    }
    .map_err(|e| format!("{name} view: {e}"))?;
    Ok((image, alloc, view))
}

/// Hybrid-raster G-buffer + graphics pipeline (primary visibility). The raster pass writes
/// (instanceId, primId) + depth; the wavefront ray-gen reads the hit instead of casting the primary
/// ray. Allocated at display res; a trace-res viewport renders into the top-left so a DLSS quality
/// change (which moves trace res) needs no reallocation. See the wavefront plan in memory.
struct RasterRes {
    hit: (vk::Image, Allocation, vk::ImageView),     // R32G32_UINT: (instanceId, primId)
    pos: (vk::Image, Allocation, vk::ImageView),     // R32G32B32A32_SFLOAT: world position of the hit
    depth: (vk::Image, Allocation, vk::ImageView),   // D32 depth
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// Per-instance push constants for the raster vertex+fragment shaders (matches raster.vert / .frag).
#[repr(C)]
#[derive(Clone, Copy)]
struct RasterPush {
    mvp: [f32; 16],   // viewProj * objectToWorld
    o2w: [f32; 12],   // objectToWorld rows (row-major 3x4), for the world-position G-buffer
    instance_id: u32, // stored in the hit G-buffer
}

/// Per-instance raster record (binding 23), indexed by draw-order instanceId. Lets the wavefront
/// resolve the raster hit's geometry (customIndex) + transform (objectToWorld). Matches the GLSL
/// RasterInst in wf_pathstate.glsl: 3 vec4 (o2w rows) + customIndex + padding = 64 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct RasterInst {
    o2w: [f32; 12],   // row-major 3x4 object-to-world (3 vec4 rows in std430)
    custom_index: u32, // geometry-table index (== BLAS slot)
    _pad: [u32; 3],
}

fn create_shader_module(device: &ash::Device, spv_bytes: &[u8]) -> Result<vk::ShaderModule, String> {
    let spv = ash::util::read_spv(&mut std::io::Cursor::new(spv_bytes)).map_err(|e| format!("read spv: {e}"))?;
    unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None) }
        .map_err(|e| format!("shader module: {e}"))
}

/// A TLAS instance transform (row-major 3x4 object-to-world) as a column-major glam Mat4 with an
/// implicit (0,0,0,1) bottom row — for the hybrid-raster per-instance MVP.
fn mat4_from_3x4(t: &[f32; 12]) -> glam::Mat4 {
    glam::Mat4::from_cols_array(&[
        t[0], t[4], t[8], 0.0,
        t[1], t[5], t[9], 0.0,
        t[2], t[6], t[10], 0.0,
        t[3], t[7], t[11], 1.0,
    ])
}

fn create_attachment_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    name: &'static str,
) -> Result<(vk::Image, Allocation, vk::ImageView), String> {
    let ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let image = unsafe { device.create_image(&ci, None) }.map_err(|e| format!("{name} image: {e}"))?;
    let req = unsafe { device.get_image_memory_requirements(image) };
    let alloc = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements: req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("{name} alloc: {e}"))?;
    unsafe { device.bind_image_memory(image, alloc.memory(), alloc.offset()) }
        .map_err(|e| format!("{name} bind: {e}"))?;
    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(aspect)
                        .level_count(1)
                        .layer_count(1),
                ),
            None,
        )
    }
    .map_err(|e| format!("{name} view: {e}"))?;
    Ok((image, alloc, view))
}

fn create_raster_resources(
    device: &ash::Device,
    allocator: &mut Allocator,
    width: u32,
    height: u32,
) -> Result<RasterRes, String> {
    let hit = create_attachment_image(
        device, allocator, width, height, vk::Format::R32G32_UINT,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
        vk::ImageAspectFlags::COLOR, "raster_hit",
    )?;
    let pos = create_attachment_image(
        device, allocator, width, height, vk::Format::R32G32B32A32_SFLOAT,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
        vk::ImageAspectFlags::COLOR, "raster_pos",
    )?;
    let depth = create_attachment_image(
        device, allocator, width, height, vk::Format::D32_SFLOAT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::ImageAspectFlags::DEPTH, "raster_depth",
    )?;

    // Render pass: a uint color attachment (hit) + depth. Both cleared each frame; the hit attachment
    // ends in GENERAL so the wavefront can read it as a storage image without a layout transition.
    let attachments = [
        vk::AttachmentDescription::default()
            .format(vk::Format::R32G32_UINT) // 0: hit (instanceId, primId)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::GENERAL),
        vk::AttachmentDescription::default()
            .format(vk::Format::R32G32B32A32_SFLOAT) // 1: world position
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::GENERAL),
        vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT) // 2: depth
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    let color_ref = [
        vk::AttachmentReference::default().attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        vk::AttachmentReference::default().attachment(1).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
    ];
    let depth_ref = vk::AttachmentReference::default()
        .attachment(2)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let subpass = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_ref)
        .depth_stencil_attachment(&depth_ref)];
    let render_pass = unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default().attachments(&attachments).subpasses(&subpass),
            None,
        )
    }
    .map_err(|e| format!("raster render pass: {e}"))?;

    let fb_views = [hit.2, pos.2, depth.2];
    let framebuffer = unsafe {
        device.create_framebuffer(
            &vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&fb_views)
                .width(width)
                .height(height)
                .layers(1),
            None,
        )
    }
    .map_err(|e| format!("raster framebuffer: {e}"))?;

    // Pipeline layout: only the per-instance push (mvp + instanceId), no descriptor sets.
    let pc_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<RasterPush>() as u32)];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&pc_range),
            None,
        )
    }
    .map_err(|e| format!("raster pipeline layout: {e}"))?;

    let vmod = create_shader_module(device, &include_bytes!(concat!(env!("OUT_DIR"), "/raster.vert.spv"))[..])?;
    let fmod = create_shader_module(device, &include_bytes!(concat!(env!("OUT_DIR"), "/raster.frag.spv"))[..])?;
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vmod)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fmod)
            .name(c"main"),
    ];
    // Vertex input: position only (the BLAS vertex buffer is 3 floats/vertex).
    let vtx_binding = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(12)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let vtx_attr = [vk::VertexInputAttributeDescription::default()
        .location(0)
        .binding(0)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(0)];
    let vtx_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vtx_binding)
        .vertex_attribute_descriptions(&vtx_attr);
    let input_asm = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let raster_state = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE) // matches the RT TRIANGLE_FACING_CULL_DISABLE
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let ms_state = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);
    let blend_attach = [
        vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA),
        vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA),
    ];
    let blend_state = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attach);
    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dyn_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
    let pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vtx_input)
        .input_assembly_state(&input_asm)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster_state)
        .multisample_state(&ms_state)
        .depth_stencil_state(&depth_state)
        .color_blend_state(&blend_state)
        .dynamic_state(&dyn_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
    }
    .map_err(|(_, e)| format!("raster pipeline: {e}"))?[0];
    unsafe {
        device.destroy_shader_module(vmod, None);
        device.destroy_shader_module(fmod, None);
    }

    Ok(RasterRes { hit, pos, depth, render_pass, framebuffer, pipeline_layout, pipeline })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CameraPush {
    inv_view_proj: [f32; 16],
    cam_pos: [f32; 4],
    dims: [u32; 2],
    has_tlas: u32,   // bit0 = hasTlas, bit1 = hasLut, bit2 = write DLSS guide buffers
    num_lights: u32,
    sun_dir: [f32; 4], // xyz = world direction, w = intensity
    sun_col: [f32; 4], // rgb = color
    prev_view_proj: [f32; 16], // forward view-proj of the previous frame (DLSS motion vectors)
    jitter: [f32; 2],  // DLSS-RR sub-pixel offset from pixel center [-0.5,0.5]; 0 when RR inactive
    emissive_count: u32, // emissive triangle count for NEE (0 = none)
    wf_in: u32,          // wavefront compaction: read side (0/1) of the ping-pong live-path queue
    wf_out: u32,         // wavefront compaction: write side (0/1)
    max_bounces: u32,    // path-tracing bounce budget (Max Bounces selector)
    spp: u32,            // samples per pixel per frame (Samples/Pixel selector; wavefront loops this)
    sample_idx: u32,     // current sample 0..spp-1, set per round in the wavefront dispatch loop
} // 224 bytes (push-constant max is 256 on the target GPUs)

// SHARC (Spatial Hash Radiance Cache) world-space cache size, in voxels/slots. Power of two. The basic
// phase uses 1M (~40 MiB: keys = cap*8, data = cap*32). NVIDIA's baseline is 2^22; bump once it works.
const SHARC_CAPACITY: u32 = 1 << 20;

/// Upper bound on TLAS instances for the previous-transform buffer (object motion vectors). The
/// shader only reads indices below the live instance count, so this is a hard allocation cap.
const MAX_INSTANCES: u32 = 65536;

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
    index_count: u32, // for the hybrid-raster drawIndexed (3 * triangle count)
}

struct Tlas {
    accel: vk::AccelerationStructureKHR,
    buf: GpuBuffer,
    instbuf: GpuBuffer,
}

/// DLSS Ray Reconstruction guide buffers — written by trace.comp at the primary hit, read by NGX.
struct GuideBuffers {
    depth: (vk::Image, Allocation, vk::ImageView),        // R32F linear view-space depth (binding 9)
    normal_rough: (vk::Image, Allocation, vk::ImageView), // RGBA16F rgb=normal, a=roughness (binding 10)
    albedo: (vk::Image, Allocation, vk::ImageView),       // RGBA16F diffuse albedo (binding 11)
    motion: (vk::Image, Allocation, vk::ImageView),       // RGBA16F pixel-space motion vectors (binding 12)
    noisy: (vk::Image, Allocation, vk::ImageView),        // RGBA16F noisy 1-spp linear color (binding 13)
    clean: (vk::Image, Allocation, vk::ImageView),        // RGBA16F NGX denoised linear color (binding 14)
    spec_albedo: (vk::Image, Allocation, vk::ImageView),  // RGBA16F specular albedo / EnvBRDF (binding 15)
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
    emissive_buffer: Option<GpuBuffer>, // emissive triangles (NEE area lights), bound at binding 16
    emissive_count: u32,
    emissive_data: Vec<f32>, // last uploaded emissive-tri floats (16 per tri; dedup like lights)

    // OCIO view-transform 3D LUT (Blender's AgX/Filmic/etc.), bound at binding 7.
    lut_image: vk::Image, // null until a LUT is uploaded
    lut_alloc: Option<Allocation>,
    lut_view: vk::ImageView,
    lut_sampler: vk::Sampler,
    has_lut: bool,

    // World/background color (the "sky"), bound at binding 8.
    world_buffer: Option<GpuBuffer>,
    world_data: [f32; 8], // [bg_color.rgb, 0, hdri_index, hdri_strength, 0, 0]

    // Object motion vectors (binding 18): previous-frame instance transforms so moving objects
    // get correct motion vectors. instance_transforms = current (parsed in build_tlas);
    // prev_instance_transforms = as of the last render. Each is a flat [f32;12] (3x4 row-major) per
    // instance, indexed by gl_InstanceID.
    prev_xform_buffer: Option<GpuBuffer>,
    instance_transforms: Vec<[f32; 12]>,
    prev_instance_transforms: Vec<[f32; 12]>,
    instance_blas: Vec<u32>, // BLAS index per instance, for the hybrid-raster per-instance draws

    // Wavefront path tracer (binding 19 + compute pipelines). Off by default; the megakernel
    // (trace.rgen) renders unless use_wavefront is set. See [[wavefront-plan]].
    wf_pathstate_buffer: Option<GpuBuffer>,
    wf_queue_buffer: Option<GpuBuffer>,
    wf_ctrl_buffer: Option<GpuBuffer>,
    sharc_keys_buffer: Option<GpuBuffer>, // SHARC hash keys (uint64), binding 25
    sharc_data_buffer: Option<GpuBuffer>, // SHARC accum + resolved radiance, binding 26
    wf_gen_pipeline: Option<vk::Pipeline>,
    wf_extend_pipeline: Option<vk::Pipeline>,
    wf_resolve_pipeline: Option<vk::Pipeline>,
    wf_compact_pipeline: Option<vk::Pipeline>,
    // Hybrid rasterization (primary-visibility G-buffer). Off by default; feeds wf_gen when enabled.
    raster: Option<RasterRes>,
    raster_debug_pipeline: Option<vk::Pipeline>, // R1 visualization (hashColor of the hit G-buffer)
    // Per-instance (objectToWorld + customIndex) for the raster hit, indexed by draw-order instanceId.
    // Host-visible; rewritten when the instance transforms change. Binding 23.
    raster_inst_buffer: Option<GpuBuffer>,

    // Compute pipeline.
    offscreen_view: vk::ImageView,
    ds_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    desc_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline, // trace ray-generation pipeline (RT pipeline; ray query inline)
    tonemap_pipeline: Option<vk::Pipeline>, // DLSS-RR clean->display tonemap (compute, shares layout)
    rt_pipeline_ext: Option<ash::khr::ray_tracing_pipeline::Device>, // vkCmdTraceRaysKHR etc.
    sbt_buffer: Option<GpuBuffer>, // shader binding table (one raygen record)
    sbt_region: vk::StridedDeviceAddressRegionKHR, // raygen region for traceRays
    env_buffer: Option<GpuBuffer>, // HDRI luminance distribution (binding 16->17), env importance sampling
    env_pipeline: Option<vk::Pipeline>, // env_cdf.comp prepass (compute, shares pipeline_layout)
    env_dirty: bool, // rebuild the env distribution before the next trace (world/HDRI changed)
    inv_view_proj: [f32; 16],
    cam_pos: [f32; 4],

    // Temporal accumulation (path tracing): a persistent radiance accumulator + frame counter
    // that resets when the camera or scene changes.
    accum_image: vk::Image,
    accum_alloc: Option<Allocation>,
    accum_view: vk::ImageView,
    accum_frame: u32,
    prev_view: [f32; 16], // previous viewInverse (no projection jitter) for move detection

    // DLSS guide buffers (bindings 9-12), written by the path tracer at the primary hit.
    guide: Option<GuideBuffers>,
    prev_view_proj: [f32; 16], // forward view-proj of the previous frame (motion vectors)
    rr: Option<crate::ngx::RrFeature>, // DLSS Ray Reconstruction feature (None if unsupported)
    fg: Option<crate::ngx::FgFeature>, // DLSS Frame Generation feature (None if unsupported/off)
    fg_interp: Option<(vk::Image, Allocation, vk::ImageView)>, // DLSS-G interpolated output
    fg_real: Option<(vk::Image, Allocation, vk::ImageView)>,   // DLSS-G real-frame passthrough output
    fg_present: bool, // frame-gen cadence: true = present the held real frame (skip the path trace)

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
    ser: bool, // Shader Execution Reordering active (sets hasTlas bit4)
    ft_ema: f32, // GPU frame time EMA (ms), for perf measurement
    ft_count: u32,
    last_render: Option<std::time::Instant>, // for the real frame delta fed to DLSS-RR
    pending_reset: bool, // a hard cut (object/camera teleport) -> reset DLSS history next frame
    width: u32,        // display (output) resolution — readback + DLSS-RR output
    height: u32,
    trace_width: u32,  // path-tracer / guide resolution — DLSS-RR upscales this to width/height
    trace_height: u32, // (== width/height for DLAA / no DLSS)
    dlss_quality: i32, // quality mode the DLSS feature + guides were last built for (live re-init)
}

/// Halton low-discrepancy sequence value (radical inverse) — used for DLSS-RR sub-pixel jitter.
fn halton(mut i: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
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
    if let Some(mut r) = RENDERER.lock().unwrap().take() {
        use ash::vk::Handle;
        unsafe { let _ = r.0.device.device_wait_idle(); }
        if let Some(rr) = r.0.rr.take() {
            crate::ngx::release_rr(rr); // release the RR feature before NGX shutdown / device destroy
        }
        if let Some(fg) = r.0.fg.take() {
            crate::ngx::release_fg(fg);
        }
        crate::ngx::shutdown(r.0.device.handle().as_raw());
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
        // Compare the view-inverse (camera world transform), NOT invViewProj, which the addon
        // jitters every frame for sub-pixel sampling.
        let view = view_inverse[..16].to_vec();
        let moved = view
            .iter()
            .zip(r.0.prev_view.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        if moved {
            r.0.prev_view.copy_from_slice(&view);
            // With DLSS-RR driving the image, camera motion is handled by the motion vectors + the
            // denoiser's internal disocclusion (RTXDI just increments frameIndex — it never resets
            // accumulation on camera movement). Resetting here would (a) discard RR history AND
            // (b) force prev_view_proj == current, zeroing the motion vectors, so the frame visibly
            // rebuilds while moving. Genuine discontinuities (geometry/material/world changes) still
            // reset elsewhere. Only the non-RR accumulation path needs the per-move reset.
            let rr_active = r.0.rr.is_some() && crate::config::get_int("dlss_rr_enabled") != 0;
            if !rr_active {
                r.0.accum_frame = 0;
            }
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

pub fn update_instance_transforms(indices: &[u32], transforms: &[f32]) -> bool {
    match RENDERER.lock().unwrap().as_mut() {
        Some(r) => r.0.update_instance_transforms(indices, transforms),
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

/// Emissive triangles for NEE area lighting — 16 floats each, importance-sampling CDF already baked
/// by the addon: [v0.xyz, area, v1.xyz, cdf, v2.xyz, totalPower, emission.rgb, pad].
pub fn upload_emissive_triangles(data: *const f32, count: u32) {
    let floats: &[f32] = if data.is_null() || count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data, count as usize * 16) }
    };
    if let Some(r) = RENDERER.lock().unwrap().as_mut() {
        r.0.set_emissive_triangles(floats, count);
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
    // DLSS render (trace) resolution: lower than the display size for the upscaling quality modes,
    // equal to it for DLAA / no DLSS. The guide buffers + path tracer run at this size; DLSS-RR
    // upscales to the display (width/height). Read from config at create — changing quality at
    // runtime needs a renderer recreate (the addon restarts the engine on a quality change).
    let dlss_rr = crate::config::get_int("dlss_rr_enabled") != 0;
    let quality = crate::config::get_int("dlss_quality");
    let (trace_width, trace_height) = if dlss_rr && (1..=5).contains(&quality) {
        dlss_render_res(width, height, quality)
    } else {
        (width, height)
    };
    log(&format!(
        "ignis build: display {width}x{height}, trace {trace_width}x{trace_height} (dlss_rr={dlss_rr}, q={quality})"
    ));
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
            // NGX/DLSS required device extensions (CUDA interop) — without these, NGX feature
            // creation fails with NotInitialized. Match vk_context.cpp.
            c"VK_KHR_external_memory",
            c"VK_KHR_external_memory_win32",
            c"VK_KHR_push_descriptor",
            c"VK_NVX_binary_import",
            c"VK_NVX_image_view_handle",
            // Shader Execution Reordering (perf): required by trace.rgen's reorderThreadNV.
            c"VK_NV_ray_tracing_invocation_reorder",
        ] {
            if has(n) {
                dev_ext_ptrs.push(n.as_ptr());
            }
        }
    }
    let has_ser = rt_supported && has(c"VK_NV_ray_tracing_invocation_reorder");

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
    let mut f_ser = vk::PhysicalDeviceRayTracingInvocationReorderFeaturesNV::default()
        .ray_tracing_invocation_reorder(true);

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
        if has_ser {
            dci = dci.push_next(&mut f_ser);
        }
    }

    let device = unsafe { instance.create_device(pd, &dci, None) }
        .map_err(|e| format!("create_device: {e}"))?;
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    // DLSS / NGX init (Stage 0): verify the runtime chain on the RTX. Harmless on non-RTX
    // (returns a failure result, logged). Uses raw Vulkan handles. base_path holds the log dir.
    {
        use ash::vk::Handle;
        let base = crate::config::base_path();
        let gipa = entry.static_fn().get_instance_proc_addr as *const std::ffi::c_void;
        let gdpa = instance.fp_v1_0().get_device_proc_addr as *const std::ffi::c_void;
        crate::ngx::init(
            instance.handle().as_raw(),
            pd.as_raw(),
            device.handle().as_raw(),
            gipa,
            gdpa,
            if base.is_empty() { "." } else { &base },
        );
    }
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

    // Accumulation image (RGBA32F storage, persistent across frames for path-trace averaging). Only
    // the non-RR path uses it (and there trace == display), so it stays at display resolution and is
    // untouched by a DLSS quality change.
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

    // DLSS guide buffers (RT devices only — DLSS-RR needs ray tracing). Written by the path tracer.
    let guide = if rt_supported {
        let f16 = vk::Format::R16G16B16A16_SFLOAT;
        Some(GuideBuffers {
            // Guides are DLSS-RR *inputs* → trace resolution. clean is the DLSS *output* → display.
            depth: create_storage_image(&device, &mut allocator, trace_width, trace_height, vk::Format::R32_SFLOAT, "g_depth")?,
            normal_rough: create_storage_image(&device, &mut allocator, trace_width, trace_height, f16, "g_normal")?,
            albedo: create_storage_image(&device, &mut allocator, trace_width, trace_height, f16, "g_albedo")?,
            motion: create_storage_image(&device, &mut allocator, trace_width, trace_height, f16, "g_motion")?,
            noisy: create_storage_image(&device, &mut allocator, trace_width, trace_height, f16, "g_noisy")?,
            clean: create_storage_image(&device, &mut allocator, width, height, f16, "g_clean")?,
            spec_albedo: create_storage_image(&device, &mut allocator, trace_width, trace_height, f16, "g_specalb")?,
        })
    } else {
        None
    };

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

    // Previous-frame instance transforms (binding 18, object motion vectors): 3 vec4 (48 bytes) per
    // TLAS instance, host-visible so it can be rewritten each frame. Sized for a generous instance
    // cap; the shader only reads indices < instance count, which never exceeds it.
    let prev_xform_buffer = GpuBuffer::new(
        &device, &mut allocator, (MAX_INSTANCES as u64) * 48,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "prev_xforms",
    )?;

    // Wavefront PathState buffer (binding 19): one path in flight per pixel, 112 bytes (7 vec4) — the
    // serialized PathCtx + rng + spreadAngle + flags (s0..s4) + the injected raster primary hit
    // (s5..s6). Only used when use_wavefront is on; the megakernel ignores it. Sized at display res.
    let wf_pathstate_buffer = if rt_supported {
        Some(GpuBuffer::new(
            &device, &mut allocator, (width as u64) * (height as u64) * 112,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "wf_pathstate",
        )?)
    } else {
        None
    };
    // Wavefront compaction (binding 20): ping-pong live-path queues — 2 halves of N u32 each holding
    // the pixel indices still alive. The extend stage reads one half and atomic-appends survivors to
    // the other. (binding 21) control buffer: [count0, count1, argsX, argsY, argsZ] — the per-round
    // live counts + the VkDispatchIndirectCommand the next extend dispatches over (offset 8).
    let wf_queue_buffer = if rt_supported {
        Some(GpuBuffer::new(
            &device, &mut allocator, (width as u64) * (height as u64) * 4 * 2,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "wf_queue",
        )?)
    } else {
        None
    };
    let wf_ctrl_buffer = if rt_supported {
        Some(GpuBuffer::new(
            &device, &mut allocator, 32,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly, "wf_ctrl",
        )?)
    } else {
        None
    };
    // SHARC SSBOs (bindings 25 = uint64 keys, 26 = [accum: cap*4 uint][resolved: cap*4 uint]). GpuOnly,
    // zero-filled at scene init; TRANSFER_DST for the fill. Allocated even when SHARC is off (the config
    // pushes capacity 0 to disable in-shader) so the descriptor set stays static.
    let sharc_usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let sharc_keys_buffer = if rt_supported {
        Some(GpuBuffer::new(&device, &mut allocator, SHARC_CAPACITY as u64 * 8,
            sharc_usage, MemoryLocation::GpuOnly, "sharc_keys")?)
    } else {
        None
    };
    let sharc_data_buffer = if rt_supported {
        Some(GpuBuffer::new(&device, &mut allocator, SHARC_CAPACITY as u64 * 32,
            sharc_usage, MemoryLocation::GpuOnly, "sharc_data")?)
    } else {
        None
    };

    // Compute pipeline.
    // Binding 0: offscreen storage image (always). Binding 1: TLAS — only on RT devices
    // (partially bound: written after the scene loads, read only when hasTlas == 1). A
    // non-RT device can't have an acceleration-structure descriptor or ray query, so it
    // falls back to the sky shader and a single-binding layout.
    let mut bindings = vec![vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE)];
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
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE),
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
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE),
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
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE),
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
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE),
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
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE),
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
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE),
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
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE),
        );
        binding_flags.push(vk::DescriptorBindingFlags::empty());
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
        );
        // Bindings 9-12: DLSS guide buffers (depth, normal+rough, albedo, motion); 13 = noisy color
        // (trace writes, NGX reads); 14 = clean color (NGX writes, tonemap reads); 15 = specular
        // albedo (trace writes, NGX reads). All storage images.
        for b in 9u32..=15u32 {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE),
            );
            binding_flags.push(vk::DescriptorBindingFlags::empty());
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1),
            );
        }
        // Binding 16: emissive triangles (NEE area lights) — storage buffer, written when present.
        // Binding 17: environment (HDRI) luminance distribution — built by env_cdf.comp, read by trace.
        // Binding 18: previous-frame instance transforms (object motion vectors), written per frame.
        // Binding 19: wavefront PathState buffer (gen writes, extend/shade consumes; megakernel ignores).
        // Binding 20: wavefront live-path queue (ping-pong). Binding 21: wavefront compaction control.
        for b in 16u32..=21u32 {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE),
            );
            binding_flags.push(vk::DescriptorBindingFlags::PARTIALLY_BOUND);
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1),
            );
        }
        // Binding 22: hybrid-raster hit G-buffer (instanceId, primId). The raster pass writes it as a
        // color attachment; wf_gen / the debug viz read it as a storage image.
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(22)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        binding_flags.push(vk::DescriptorBindingFlags::PARTIALLY_BOUND);
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
        );
        // Binding 23: per-instance raster data (objectToWorld + customIndex) for the raster hit.
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(23)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE),
        );
        binding_flags.push(vk::DescriptorBindingFlags::PARTIALLY_BOUND);
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
        );
        // Binding 24: hybrid-raster world-position G-buffer (raster writes it, wf_gen reads it).
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(24)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        binding_flags.push(vk::DescriptorBindingFlags::PARTIALLY_BOUND);
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
        );
        // Bindings 25-26: SHARC hash keys (uint64) + data (accum/resolved). Both PT paths use them.
        for b in [25u32, 26u32] {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE),
            );
            binding_flags.push(vk::DescriptorBindingFlags::PARTIALLY_BOUND);
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1),
            );
        }
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
    let prev_xform_info = [vk::DescriptorBufferInfo::default()
        .buffer(prev_xform_buffer.buffer)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let wf_pathstate_info = wf_pathstate_buffer.as_ref().map(|b| {
        [vk::DescriptorBufferInfo::default().buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE)]
    });
    let wf_queue_info = wf_queue_buffer.as_ref().map(|b| {
        [vk::DescriptorBufferInfo::default().buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE)]
    });
    let wf_ctrl_info = wf_ctrl_buffer.as_ref().map(|b| {
        [vk::DescriptorBufferInfo::default().buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE)]
    });
    let sharc_keys_info = sharc_keys_buffer.as_ref().map(|b| {
        [vk::DescriptorBufferInfo::default().buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE)]
    });
    let sharc_data_info = sharc_data_buffer.as_ref().map(|b| {
        [vk::DescriptorBufferInfo::default().buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE)]
    });
    let gi = |v: vk::ImageView| [vk::DescriptorImageInfo::default()
        .image_view(v)
        .image_layout(vk::ImageLayout::GENERAL)];
    let guide_infos: [[vk::DescriptorImageInfo; 1]; 4] = match guide.as_ref() {
        Some(g) => [gi(g.depth.2), gi(g.normal_rough.2), gi(g.albedo.2), gi(g.motion.2)],
        None => Default::default(),
    };
    let noisy_info = guide.as_ref().map(|g| gi(g.noisy.2)).unwrap_or_default();
    let clean_info = guide.as_ref().map(|g| gi(g.clean.2)).unwrap_or_default();
    let specalb_info = guide.as_ref().map(|g| gi(g.spec_albedo.2)).unwrap_or_default();
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
        for (i, info) in guide_infos.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(9 + i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(info),
            );
        }
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(13)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&noisy_info),
        );
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(14)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&clean_info),
        );
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(15)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&specalb_info),
        );
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(18)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&prev_xform_info),
        );
        if let Some(info) = wf_pathstate_info.as_ref() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(19)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info),
            );
        }
        if let Some(info) = wf_queue_info.as_ref() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(20)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info),
            );
        }
        if let Some(info) = wf_ctrl_info.as_ref() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(21)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info),
            );
        }
        if let Some(info) = sharc_keys_info.as_ref() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(25)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info),
            );
        }
        if let Some(info) = sharc_data_info.as_ref() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(26)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info),
            );
        }
    }
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    let push_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::COMPUTE)
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

    // RT devices: the path tracer is a ray-generation pipeline (trace.rgen, ray query inline) so it
    // can use Shader Execution Reordering. Non-RT devices keep the sky.comp compute fallback.
    let (pipeline, rt_pipeline_ext, sbt_buffer, sbt_region) = if rt_supported {
        let rt_ext = ash::khr::ray_tracing_pipeline::Device::new(&instance, &device);
        // SER variant (trace_ser.spv) only where VK_NV_ray_tracing_invocation_reorder is available
        // (Ada+); the plain trace.rgen.spv carries no SER capability and runs on any RT GPU.
        let rgen_spv: &[u8] = if has_ser {
            &include_bytes!(concat!(env!("OUT_DIR"), "/trace_ser.spv"))[..]
        } else {
            &include_bytes!(concat!(env!("OUT_DIR"), "/trace.rgen.spv"))[..]
        };
        let spv = ash::util::read_spv(&mut std::io::Cursor::new(rgen_spv))
            .map_err(|e| format!("read rgen spv: {e}"))?;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
        }
        .map_err(|e| format!("rgen module: {e}"))?;
        let stages = [vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .module(module)
            .name(c"main")];
        let groups = [vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)];
        let pipe = unsafe {
            rt_ext.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[vk::RayTracingPipelineCreateInfoKHR::default()
                    .stages(&stages)
                    .groups(&groups)
                    .max_pipeline_ray_recursion_depth(1)
                    .layout(pipeline_layout)],
                None,
            )
        }
        .map_err(|(_, e)| format!("rt pipeline: {e}"))?[0];
        unsafe { device.destroy_shader_module(module, None) };

        // Shader binding table: one raygen record. Aligned to shaderGroupBaseAlignment.
        let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut rt_props);
        unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
        let handle_size = rt_props.shader_group_handle_size as usize;
        let handles = unsafe { rt_ext.get_ray_tracing_shader_group_handles(pipe, 0, 1, handle_size) }
            .map_err(|e| format!("sbt handles: {e}"))?;
        let stride = align_up(handle_size as u64, rt_props.shader_group_base_alignment as u64);
        let sbt = GpuBuffer::new(
            &device,
            &mut allocator,
            stride,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
            "sbt",
        )
        .map_err(|e| format!("sbt buffer: {e}"))?;
        sbt.write_bytes(&handles[..handle_size]);
        let region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt.device_address(&device))
            .stride(stride)
            .size(stride);
        (pipe, Some(rt_ext), Some(sbt), region)
    } else {
        let spv = ash::util::read_spv(&mut std::io::Cursor::new(
            &include_bytes!(concat!(env!("OUT_DIR"), "/sky.comp.spv"))[..],
        ))
        .map_err(|e| format!("read sky spv: {e}"))?;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
        }
        .map_err(|e| format!("sky module: {e}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipe = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)],
                None,
            )
        }
        .map_err(|(_, e)| format!("sky pipeline: {e}"))?[0];
        unsafe { device.destroy_shader_module(module, None) };
        (pipe, None, None, vk::StridedDeviceAddressRegionKHR::default())
    };

    // DLSS-RR final tonemap pass — reads NGX's clean output, writes the display offscreen. Shares
    // the trace pipeline layout (same descriptor set + push constants). RT devices only.
    let tonemap_pipeline = if rt_supported {
        let tspv_bytes: &[u8] = &include_bytes!(concat!(env!("OUT_DIR"), "/tonemap.comp.spv"))[..];
        let tspv = ash::util::read_spv(&mut std::io::Cursor::new(tspv_bytes))
            .map_err(|e| format!("read tonemap spv: {e}"))?;
        let tmod = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&tspv), None)
        }
        .map_err(|e| format!("tonemap module: {e}"))?;
        let tstage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(tmod)
            .name(c"main");
        let tp = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(tstage)
                    .layout(pipeline_layout)],
                None,
            )
        }
        .map_err(|(_, e)| format!("tonemap pipeline: {e}"))?[0];
        unsafe { device.destroy_shader_module(tmod, None) };
        Some(tp)
    } else {
        None
    };

    // Wavefront path tracer pipelines (compute, share the pipeline layout). Phase 0: gen + extend/shade.
    let (wf_gen_pipeline, wf_extend_pipeline, wf_resolve_pipeline, wf_compact_pipeline, raster_debug_pipeline) = if rt_supported {
        let mk = |spv_bytes: &[u8], name: &str| -> Result<vk::Pipeline, String> {
            let spv = ash::util::read_spv(&mut std::io::Cursor::new(spv_bytes))
                .map_err(|e| format!("read {name} spv: {e}"))?;
            let m = unsafe {
                device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
            }
            .map_err(|e| format!("{name} module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(m)
                .name(c"main");
            let p = unsafe {
                device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)],
                    None,
                )
            }
            .map_err(|(_, e)| format!("{name} pipeline: {e}"))?[0];
            unsafe { device.destroy_shader_module(m, None) };
            Ok(p)
        };
        let g = mk(&include_bytes!(concat!(env!("OUT_DIR"), "/wf_gen.comp.spv"))[..], "wf_gen")?;
        let e = mk(&include_bytes!(concat!(env!("OUT_DIR"), "/wf_extend_shade.comp.spv"))[..], "wf_extend")?;
        let r = mk(&include_bytes!(concat!(env!("OUT_DIR"), "/wf_resolve.comp.spv"))[..], "wf_resolve")?;
        let c = mk(&include_bytes!(concat!(env!("OUT_DIR"), "/wf_compact.comp.spv"))[..], "wf_compact")?;
        let dbg = mk(&include_bytes!(concat!(env!("OUT_DIR"), "/raster_debug.comp.spv"))[..], "raster_debug")?;
        (Some(g), Some(e), Some(r), Some(c), Some(dbg))
    } else {
        (None, None, None, None, None)
    };

    // Hybrid-raster G-buffer + graphics pipeline (primary visibility). Allocated at display res; the
    // raster + wavefront render into a trace-res viewport. Only on RT devices (it feeds the wavefront).
    let raster = if rt_supported {
        Some(create_raster_resources(&device, &mut allocator, width, height)?)
    } else {
        None
    };
    // Bind the raster hit G-buffer at binding 22 (created after the main descriptor writes above).
    if let Some(r) = raster.as_ref() {
        let hit_info = [vk::DescriptorImageInfo::default().image_view(r.hit.2).image_layout(vk::ImageLayout::GENERAL)];
        let pos_info = [vk::DescriptorImageInfo::default().image_view(r.pos.2).image_layout(vk::ImageLayout::GENERAL)];
        let w = [
            vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(22)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&hit_info),
            vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(24)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&pos_info),
        ];
        unsafe { device.update_descriptor_sets(&w, &[]) };
    }
    // Per-instance raster data (binding 23): objectToWorld (3 vec4) + customIndex, indexed by the
    // draw-order instanceId. Host-visible; rewritten when the instance transforms change.
    let raster_inst_buffer = if rt_supported {
        Some(GpuBuffer::new(
            &device, &mut allocator, (MAX_INSTANCES as u64) * 64,
            vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "raster_inst",
        )?)
    } else {
        None
    };
    if let Some(b) = raster_inst_buffer.as_ref() {
        let info = [vk::DescriptorBufferInfo::default().buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE)];
        let w = [vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(23)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info)];
        unsafe { device.update_descriptor_sets(&w, &[]) };
    }

    // Environment (HDRI) importance-sampling distribution: a storage buffer (binding 17) built by the
    // env_cdf.comp prepass (compute, shares the pipeline layout). Sized for the 256x128 Distribution2D.
    let (env_buffer, env_pipeline) = if rt_supported {
        const ENV_W: u64 = 256;
        const ENV_H: u64 = 128;
        let env_floats = 1 + ENV_H + ENV_H * ENV_W + ENV_H * ENV_W;
        let buf = GpuBuffer::new(
            &device,
            &mut allocator,
            env_floats * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "env_dist",
        )
        .map_err(|e| format!("env buffer: {e}"))?;
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(buf.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(17)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info);
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        let espv = ash::util::read_spv(&mut std::io::Cursor::new(
            &include_bytes!(concat!(env!("OUT_DIR"), "/env_cdf.comp.spv"))[..],
        ))
        .map_err(|e| format!("read env_cdf spv: {e}"))?;
        let emod = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&espv), None)
        }
        .map_err(|e| format!("env_cdf module: {e}"))?;
        let estage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(emod)
            .name(c"main");
        let ep = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().stage(estage).layout(pipeline_layout)],
                None,
            )
        }
        .map_err(|(_, e)| format!("env_cdf pipeline: {e}"))?[0];
        unsafe { device.destroy_shader_module(emod, None) };
        (Some(buf), Some(ep))
    } else {
        (None, None)
    };

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

    // Create the DLSS Ray Reconstruction feature (RT devices only). NGX records initialization
    // into a one-shot command buffer that we submit + wait on here. Falls back to None on failure.
    let (rr, fg) = if rt_supported {
        use ash::vk::Handle;
        unsafe {
            let _ = device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            );
            let feat = crate::ngx::create_rr(
                device.handle().as_raw(),
                cmd.as_raw(),
                trace_width,
                trace_height,
                width,
                height,
                dlss_perf_quality(quality),
            );
            // DLSS Frame Generation feature (if the GPU supports it) — recorded into the same one-shot
            // init command buffer. The backbuffer is the display-res offscreen (RB_FORMAT).
            let fg_feat = if crate::ngx::dlssg_max_frames() > 0 {
                crate::ngx::create_fg(device.handle().as_raw(), cmd.as_raw(), width, height, RB_FORMAT.as_raw())
            } else {
                None
            };
            let _ = device.end_command_buffer(cmd);
            let _ = device.reset_fences(&[fence]);
            let _ = device.queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))],
                fence,
            );
            let _ = device.wait_for_fences(&[fence], true, u64::MAX);
            (feat, fg_feat)
        }
    } else {
        (None, None)
    };

    // DLSS-G output images (interpolated frame + real passthrough), display res, RB_FORMAT. Only when
    // the feature exists. STORAGE (NGX writes them) + TRANSFER_SRC (read back to present).
    let (fg_interp, fg_real) = if fg.is_some() {
        let u = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC;
        (
            Some(create_attachment_image(&device, &mut allocator, width, height, RB_FORMAT, u, vk::ImageAspectFlags::COLOR, "fg_interp")?),
            Some(create_attachment_image(&device, &mut allocator, width, height, RB_FORMAT, u, vk::ImageAspectFlags::COLOR, "fg_real")?),
        )
    } else {
        (None, None)
    };

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
        emissive_buffer: None,
        emissive_count: 0,
        emissive_data: Vec::new(),
        lut_image: vk::Image::null(),
        lut_alloc: None,
        lut_view: vk::ImageView::null(),
        lut_sampler,
        has_lut: false,
        world_buffer: Some(world_buffer),
        world_data: [-1.0; 8], // impossible -> first render writes the real value
        prev_xform_buffer: Some(prev_xform_buffer),
        instance_transforms: Vec::new(),
        prev_instance_transforms: Vec::new(),
        instance_blas: Vec::new(),
        wf_pathstate_buffer,
        wf_queue_buffer,
        wf_ctrl_buffer,
        sharc_keys_buffer,
        sharc_data_buffer,
        wf_gen_pipeline,
        wf_extend_pipeline,
        wf_resolve_pipeline,
        wf_compact_pipeline,
        raster,
        raster_debug_pipeline,
        raster_inst_buffer,
        offscreen_view,
        ds_layout,
        desc_pool,
        desc_set,
        pipeline_layout,
        pipeline,
        tonemap_pipeline,
        rt_pipeline_ext,
        sbt_buffer,
        sbt_region,
        env_buffer,
        env_pipeline,
        env_dirty: true, // build the env distribution before the first frame
        inv_view_proj: [0.0; 16],
        cam_pos: [0.0; 4],
        accum_image,
        accum_alloc: Some(accum_alloc),
        accum_view,
        accum_frame: 0,
        prev_view: [0.0; 16],
        guide,
        prev_view_proj: [0.0; 16],
        rr,
        fg,
        fg_interp,
        fg_real,
        fg_present: false,
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
        // Toggle to A/B the Shader Execution Reordering perf (no shader recompile — gates the bit).
        ser: has_ser && SER_ENABLED,
        ft_ema: 0.0,
        ft_count: 0,
        last_render: None,
        pending_reset: false,
        width,
        height,
        trace_width,
        trace_height,
        dlss_quality: quality,
    })
}

/// Master switch for Shader Execution Reordering — flip to false to measure the baseline.
const SER_ENABLED: bool = true;
/// HDRI environment importance sampling (cleaner HDRI lighting). Cheap in practice (limited to the
/// first two bounces); kept as a toggle for a future viewport-vs-final-render distinction.
const ENV_IS_ENABLED: bool = true;

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

    /// Restart the accumulator on an incremental edit (light/material/mesh/world change) — but only
    /// outside DLSS-RR. In RR mode the denoiser absorbs the change via motion vectors + history
    /// clamping, so a reset would just flash a denoiser restart (the noise/blotches seen when moving
    /// or toggling lights). This mirrors shipping path tracers (e.g. Cyberpunk RT Overdrive), which
    /// never hard-reset on edits — ReSTIR + the denoiser re-converge continuously.
    fn reset_accum_for_edit(&mut self) {
        let rr_active = self.rr.is_some() && crate::config::get_int("dlss_rr_enabled") != 0;
        if !rr_active {
            self.accum_frame = 0;
        }
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
        self.reset_accum_for_edit();
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
        self.reset_accum_for_edit();
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
        self.reset_accum_for_edit();
        log(&format!("lights: {count} uploaded"));
    }

    fn set_emissive_triangles(&mut self, floats: &[f32], count: u32) {
        // Re-sent every frame; skip work + accumulation reset if unchanged.
        if count == self.emissive_count && floats == self.emissive_data.as_slice() {
            return;
        }
        self.emissive_count = count;
        self.emissive_data = floats.to_vec();
        if count == 0 || self.accel_ext.is_none() {
            return;
        }
        let device = self.device.clone();
        let alloc = self.allocator.as_mut().unwrap();
        if let Some(old) = self.emissive_buffer.take() {
            old.destroy(&device, alloc);
        }
        let buf = match GpuBuffer::new(
            &device,
            alloc,
            (count as usize * 64) as u64, // 16 floats per triangle
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
            "emissive_tris",
        ) {
            Ok(b) => b,
            Err(e) => {
                log(&format!("emissive alloc FAILED: {e}"));
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
            .dst_binding(16)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info);
        unsafe { device.update_descriptor_sets(&[write], &[]) };
        self.emissive_buffer = Some(buf);
        self.reset_accum_for_edit();
        log(&format!("emissive triangles: {count} uploaded"));
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
        // Material textures arrive as ENCODED file bytes (PNG/JPEG/BMP) with dxgi 0 — the C++
        // decodes them with stb_image. Decode here too; otherwise the compressed bytes get
        // uploaded as raw RGBA8 and render as garbage (a dotted/noise pattern).
        let (data, width, height, format) = decode_texture(data, width, height, dxgi);
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
        if self.accel_ext.is_none() {
            log("build_tlas: RT not supported, skipping");
            return false;
        }
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
        // In DLSS-RR mode let the denoiser + motion vectors absorb the change (don't reset the
        // accumulator) — otherwise undo/redo and live edits flash a denoiser restart. Non-RR keeps
        // the per-change reset its accumulation needs.
        let rr_active = self.rr.is_some() && crate::config::get_int("dlss_rr_enabled") != 0;
        self.rebuild_tlas(!rr_active)
    }

    /// Rebuild the TLAS from the stored instance bytes (tlas_instance_data). Shared by the full
    /// build_tlas and the live transform sync (update_instance_transforms). reset_accum restarts the
    /// accumulator; the live sync skips it in DLSS-RR mode, where the motion vectors carry the object
    /// motion so the denoiser need not reset. Also refreshes the motion-vector transform snapshot.
    fn rebuild_tlas(&mut self, reset_accum: bool) -> bool {
        let accel = match self.accel_ext.clone() {
            Some(a) => a,
            None => return false,
        };
        let count = (self.tlas_instance_data.len() / 60) as u32;
        if count == 0 {
            return false;
        }
        let in_slice = unsafe {
            std::slice::from_raw_parts(
                self.tlas_instance_data.as_ptr() as *const TlasInstanceIn,
                count as usize,
            )
        };

        // Snapshot the current instance transforms (object motion vectors). gl_InstanceID in the
        // shader is this array order, so the previous-transform buffer is indexed the same way.
        self.instance_transforms = in_slice.iter().map(|i| i.transform).collect();
        self.instance_blas = in_slice.iter().map(|i| i.blas_index as u32).collect();
        // Per-instance raster records (binding 23): the wavefront resolves the raster hit's geometry +
        // transform from these. custom_index == BLAS slot (descs[] is indexed by BLAS slot).
        if let Some(b) = self.raster_inst_buffer.as_ref() {
            let recs: Vec<RasterInst> = in_slice
                .iter()
                .map(|i| RasterInst {
                    o2w: i.transform,
                    custom_index: i.custom_index,
                    _pad: [0; 3],
                })
                .collect();
            b.write_bytes(crate::gpu::as_bytes(&recs));
        }

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
                if reset_accum {
                    self.accum_frame = 0; // scene/transform changed -> restart accumulation
                }
                true
            }
            Err(e) => {
                log(&format!("TLAS build FAILED: {e}"));
                false
            }
        }
    }

    /// Live transform sync (the fast path the addon calls while dragging an object): patch the given
    /// instances' transforms in the stored instance bytes and rebuild the TLAS, without re-collecting
    /// geometry or materials. This was a no-op stub before — moving an object only updated on the next
    /// full build_tlas (i.e. after deselecting), so the render appeared frozen mid-drag.
    fn update_instance_transforms(&mut self, indices: &[u32], transforms: &[f32]) -> bool {
        if self.tlas_instance_data.is_empty() {
            return false;
        }
        let n_inst = self.tlas_instance_data.len() / 60;
        let mut hard_cut = false;
        for (k, &idx) in indices.iter().enumerate() {
            let idx = idx as usize;
            if idx >= n_inst || (k + 1) * 12 > transforms.len() {
                continue;
            }
            let nt = &transforms[k * 12..k * 12 + 12];
            // Teleport detection: the translation jump (row-major 3x4 -> indices 3/7/11) relative to
            // the camera distance ~= its screen-space size. A fast drag stays small; a Ctrl+Z snap is
            // a large instant jump whose stale DLSS history would ghost -> flag a hard cut (reset).
            if let Some(old) = self.instance_transforms.get(idx) {
                let (dx, dy, dz) = (nt[3] - old[3], nt[7] - old[7], nt[11] - old[11]);
                let disp = (dx * dx + dy * dy + dz * dz).sqrt();
                let (cx, cy, cz) = (self.cam_pos[0], self.cam_pos[1], self.cam_pos[2]);
                let dist = ((nt[3] - cx).powi(2) + (nt[7] - cy).powi(2) + (nt[11] - cz).powi(2))
                    .sqrt()
                    .max(0.1);
                if disp / dist > 0.25 {
                    hard_cut = true;
                }
            }
            // The transform is a 48-byte [f32;12] at offset 4 within the 60-byte TlasInstanceIn.
            let off = idx * 60 + 4;
            self.tlas_instance_data[off..off + 48].copy_from_slice(gpu::as_bytes(nt));
        }
        if hard_cut {
            self.pending_reset = true; // reset DLSS history next frame to drop the teleported ghost
        }
        // In DLSS-RR mode the motion vectors carry the object motion, so don't reset accumulation
        // (that would discard the denoiser history and zero the very motion vectors we just wrote).
        let rr_active = self.rr.is_some() && crate::config::get_int("dlss_rr_enabled") != 0;
        self.rebuild_tlas(!rr_active)
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

    /// Re-initialise the DLSS pipeline for a new quality / RR-enabled state WITHOUT reloading geometry:
    /// re-allocate the guide buffers at the new trace resolution, rebind them, and recreate the NGX
    /// feature. Triggered live from render() when the addon changes the DLSS settings.
    fn reinit_dlss(&mut self, quality: i32) {
        self.dlss_quality = quality;
        if self.rr.is_none() || self.guide.is_none() {
            return;
        }
        let rr_enabled = crate::config::get_int("dlss_rr_enabled") != 0;
        let (tw, th) = if rr_enabled && (1..=5).contains(&quality) {
            dlss_render_res(self.width, self.height, quality)
        } else {
            (self.width, self.height)
        };
        let device = self.device.clone();
        let (dw, dh) = (self.width, self.height);
        let desc_set = self.desc_set;
        let (cmd, queue, fence) = (self.cmd, self.queue, self.fence);
        let perf_q = dlss_perf_quality(quality);
        unsafe {
            let _ = device.device_wait_idle();
        }

        // Re-allocate the guide buffers at the new trace resolution (inputs); clean stays display res.
        let new_guide = {
            let alloc = match self.allocator.as_mut() {
                Some(a) => a,
                None => return,
            };
            if let Some(old) = self.guide.take() {
                for (img, a, view) in [
                    old.depth, old.normal_rough, old.albedo, old.motion, old.noisy, old.clean, old.spec_albedo,
                ] {
                    unsafe {
                        device.destroy_image_view(view, None);
                        device.destroy_image(img, None);
                    }
                    let _ = alloc.free(a);
                }
            }
            let f16 = vk::Format::R16G16B16A16_SFLOAT;
            (|| -> Result<GuideBuffers, String> {
                Ok(GuideBuffers {
                    depth: create_storage_image(&device, alloc, tw, th, vk::Format::R32_SFLOAT, "g_depth")?,
                    normal_rough: create_storage_image(&device, alloc, tw, th, f16, "g_normal")?,
                    albedo: create_storage_image(&device, alloc, tw, th, f16, "g_albedo")?,
                    motion: create_storage_image(&device, alloc, tw, th, f16, "g_motion")?,
                    noisy: create_storage_image(&device, alloc, tw, th, f16, "g_noisy")?,
                    clean: create_storage_image(&device, alloc, dw, dh, f16, "g_clean")?,
                    spec_albedo: create_storage_image(&device, alloc, tw, th, f16, "g_specalb")?,
                })
            })()
        };
        let g = match new_guide {
            Ok(g) => g,
            Err(e) => {
                log(&format!("DLSS re-init: guide realloc failed: {e}"));
                return;
            }
        };

        // Rebind bindings 9..=15 (depth, normal, albedo, motion, noisy, clean, spec).
        let gi = |v: vk::ImageView| {
            [vk::DescriptorImageInfo::default()
                .image_view(v)
                .image_layout(vk::ImageLayout::GENERAL)]
        };
        let infos = [
            gi(g.depth.2), gi(g.normal_rough.2), gi(g.albedo.2), gi(g.motion.2),
            gi(g.noisy.2), gi(g.clean.2), gi(g.spec_albedo.2),
        ];
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(9 + i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(info)
            })
            .collect();
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
        self.guide = Some(g);

        // Recreate the NGX feature at the new render/display resolution.
        if let Some(old) = self.rr.take() {
            crate::ngx::release_rr(old);
        }
        {
            use ash::vk::Handle;
            unsafe {
                let _ = device.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                );
                let feat = crate::ngx::create_rr(
                    device.handle().as_raw(),
                    cmd.as_raw(),
                    tw, th, dw, dh, perf_q,
                );
                let _ = device.end_command_buffer(cmd);
                let _ = device.reset_fences(&[fence]);
                let _ = device.queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))],
                    fence,
                );
                let _ = device.wait_for_fences(&[fence], true, u64::MAX);
                self.rr = feat;
            }
        }
        self.trace_width = tw;
        self.trace_height = th;
        self.accum_frame = 0;
        log(&format!("DLSS re-init: trace {tw}x{th} -> display {dw}x{dh} (q={quality})"));
    }

    fn render(&mut self) {
        // Live DLSS quality / RR-enable change: re-init the DLSS pipeline (re-allocate guides +
        // recreate the NGX feature at the new trace resolution) without reloading geometry.
        if self.rr.is_some() {
            let cfg_q = crate::config::get_int("dlss_quality");
            let cfg_rr = crate::config::get_int("dlss_rr_enabled") != 0;
            let (want_tw, want_th) = if cfg_rr && (1..=5).contains(&cfg_q) {
                dlss_render_res(self.width, self.height, cfg_q)
            } else {
                (self.width, self.height)
            };
            if (want_tw, want_th) != (self.trace_width, self.trace_height) {
                self.reinit_dlss(cfg_q);
            }
        }
        // Real frame-to-frame delta for DLSS-RR's temporal feedback. A constant value (we used 16.6)
        // mistunes the denoiser when the actual frame time is much larger -> contributes to boiling.
        let now = std::time::Instant::now();
        let frame_delta_ms = self
            .last_render
            .map(|t| (now.duration_since(t).as_secs_f32() * 1000.0).clamp(1.0, 200.0))
            .unwrap_or(16.6);
        self.last_render = Some(now);
        // DLSS history reset: on a fresh accumulation OR a detected hard cut (object/camera teleport).
        // Smooth motion is carried by the motion vectors; a sudden jump would otherwise leave a ghost.
        let dlss_reset = self.accum_frame <= 1 || self.pending_reset;
        self.pending_reset = false;
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
            self.reset_accum_for_edit(); // world/HDRI changed (no reset in RR — denoiser adapts)
            self.env_dirty = true; // rebuild the HDRI importance-sampling distribution
        }

        // Object motion vectors: upload each current instance's PREVIOUS-frame transform to binding 18,
        // then snapshot the current set for next frame. New instances default to their current transform
        // (zero motion). bit6 of has_tlas gates the shader on a populated buffer.
        let n = self.instance_transforms.len();
        let have_motion = self.tlas.is_some() && n > 0 && n <= MAX_INSTANCES as usize;
        if have_motion {
            // If the instance count changed since last frame the order likely shifted (object added or
            // removed), so the stored previous transforms no longer line up with gl_InstanceID. Fall
            // back to zero object motion this frame, otherwise meshes get reprojected with another
            // object's motion and briefly smear/vanish until the snapshot catches up next frame.
            let order_stable = n == self.prev_instance_transforms.len();
            let mut data: Vec<f32> = Vec::with_capacity(n * 12);
            for i in 0..n {
                let t = if order_stable {
                    self.prev_instance_transforms[i]
                } else {
                    self.instance_transforms[i]
                };
                data.extend_from_slice(&t);
            }
            if let Some(b) = &self.prev_xform_buffer {
                b.write_bytes(gpu::as_bytes(&data));
            }
            self.prev_instance_transforms = self.instance_transforms.clone();
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
        // Forward view-proj this frame (= inverse of inv_view_proj) for DLSS motion vectors.
        let cur_vp = glam::Mat4::from_cols_array(&self.inv_view_proj).inverse().to_cols_array();
        let prev_vp = if self.accum_frame == 0 { cur_vp } else { self.prev_view_proj };
        let write_guide = self.guide.is_some();
        // DLSS Ray Reconstruction runs when the feature created, the tonemap pass exists, and the
        // addon enabled it. When active the path tracer emits 1-spp noisy linear color (no accum),
        // NGX denoises, and a separate pass tonemaps the clean output.
        // Wavefront path tracer (Phase 1c): gen -> extend×MAX_BOUNCES (one real bounce per dispatch)
        // -> resolve. Replaces the megakernel trace and writes the offscreen directly, bypassing DLSS
        // for now. Megakernel stays the default. See [[wavefront-plan]].
        let use_wavefront = self.wf_gen_pipeline.is_some()
            && self.wf_extend_pipeline.is_some()
            && self.wf_resolve_pipeline.is_some()
            && self.wf_compact_pipeline.is_some()
            && crate::config::get_int("use_wavefront") != 0;
        // Hybrid raster: rasterize primary visibility into the G-buffer so wf_gen skips the primary ray
        // query. It feeds the wavefront, so it only does anything when the wavefront is also on.
        let use_raster = self.raster.is_some()
            && crate::config::get_int("hybrid_rasterization") != 0;
        let raster_active = use_raster && use_wavefront;
        // The wavefront feeds DLSS the same way the megakernel does (gNoisy + guides at trace res),
        // so RR runs for both. When RR is off, the path tracer writes the offscreen directly.
        let use_rr = self.rr.is_some()
            && self.tonemap_pipeline.is_some()
            && write_guide
            && crate::config::get_int("dlss_rr_enabled") != 0;
        crate::config::set_int("dlss_rr_active", use_rr as i32);
        // Frame Generation cadence: active when a feature + output images exist, the selector is on,
        // DLSS-RR provides the guides, and we're at native res (DLAA) so the guides match the
        // display-res backbuffer. Alternates render (full trace + interpolate) / present (held real).
        let fg_active = self.fg.is_some()
            && self.fg_interp.is_some()
            && use_rr
            && crate::config::get_int("fg_frames") > 0
            && self.trace_width == self.width
            && self.trace_height == self.height;
        if !fg_active {
            self.fg_present = false;
        }
        if crate::config::get_int("fg_frames") > 0 && self.accum_frame % 120 == 0 {
            log(&format!(
                "fg: active={fg_active} (feature={}, interp={}, rr={use_rr}, dlaa={}, frames={})",
                self.fg.is_some(), self.fg_interp.is_some(), self.trace_width == self.width,
                crate::config::get_int("fg_frames"),
            ));
        }
        // Deterministic Halton(2,3) sub-pixel jitter, scaled by camera_jitter_scale (default 0.75 —
        // full 1.0 leaves visible shimmer). The shader samples at pixel-center + (jx, jy) in y-down
        // screen space; NGX is told (jx, -jy) — the projection-space jitter, since the shader's
        // screen-UV -> NDC step flips Y (1 - uv.y*2). Matches the C++ jitterData = (jx, -jy).
        // Hybrid raster rasterizes at pixel centres (no jitter yet), so feed DLSS zero jitter when it's
        // active — otherwise the guide motion vectors (which use pc.jitter) wouldn't match the rays.
        let (jitter, ngx_jitter) = if use_rr && !raster_active {
            let scale = {
                let s = crate::config::get_float("camera_jitter_scale");
                if s <= 0.0 { 0.75 } else { s.min(1.0) }
            };
            let idx = self.accum_frame % 256 + 1;
            let jx = (halton(idx, 2) - 0.5) * scale;
            let jy = (halton(idx, 3) - 0.5) * scale;
            ([jx, jy], [jx, -jy])
        } else {
            ([0.0, 0.0], [0.0, 0.0])
        };
        // Max Bounces selector (Blender pushes "max_bounces", range 2-8). 0/unset -> the historical 8.
        let max_bounces = match crate::config::get_int("max_bounces") {
            v if v <= 0 => 8u32,
            v => (v as u32).clamp(1, 16),
        };
        // Samples/Pixel selector (Blender pushes "spp", range 1-10). 0/unset -> 1. Wavefront only.
        let spp = match crate::config::get_int("spp") {
            v if v <= 0 => 1u32,
            v => (v as u32).clamp(1, 16),
        };
        let push = CameraPush {
            inv_view_proj: self.inv_view_proj,
            cam_pos: cam,
            dims: [self.trace_width, self.trace_height], // path tracer runs at trace (render) res
            has_tlas: (self.tlas.is_some() as u32)
                | (if self.has_lut { 2 } else { 0 })
                | (if write_guide { 4 } else { 0 }) // bit2 = emit DLSS guide buffers
                | (if use_rr { 8 } else { 0 })      // bit3 = DLSS-RR mode (noisy linear out)
                | (if self.ser { 16 } else { 0 })   // bit4 = Shader Execution Reordering
                | (if ENV_IS_ENABLED { 32 } else { 0 }) // bit5 = HDRI environment importance sampling
                | (if have_motion { 64 } else { 0 }) // bit6 = object motion vectors (prev transforms)
                | (if raster_active { 128 } else { 0 }), // bit7 = hybrid-raster primary hit available
            num_lights: self.light_count,
            sun_dir: [az.sin() * el.cos(), el.sin(), az.cos() * el.cos(), intensity],
            sun_col: [sc[0], sc[1], sc[2], self.accum_frame as f32],
            prev_view_proj: prev_vp,
            jitter,
            emissive_count: self.emissive_count,
            wf_in: 0,  // set per round in the wavefront dispatch loop
            wf_out: 0,
            max_bounces,
            spp,
            sample_idx: 0, // set per sample in the wavefront dispatch loop
        };
        self.prev_view_proj = cur_vp;
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        // The path trace writes its images from the ray-generation stage (RT pipeline) or the
        // compute stage (sky fallback); the NGX eval + tonemap are compute. Bracket-barriers that
        // touch the trace's outputs use this combined mask so no dependency is missed either way.
        // The wavefront writes its noisy colour + guides from compute; the megakernel from ray-gen.
        let trace_stage = if use_wavefront {
            vk::PipelineStageFlags::COMPUTE_SHADER
        } else if self.rt_pipeline_ext.is_some() {
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
        } else {
            vk::PipelineStageFlags::COMPUTE_SHADER
        };
        let trace_or_compute = trace_stage | vk::PipelineStageFlags::COMPUTE_SHADER;

        unsafe {
            let _ = d.reset_fences(&[self.fence]);
            let _ = d.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            );

            // ── Frame-gen PRESENT phase: show DLSS-FG's retained real frame (out_real) ──
            // The render phase's eval copied the rendered frame into out_real (still in GENERAL); just
            // read it back without re-rendering. Per the SDK: present the generated frame, then this
            // retained real frame. This is the in-between that doubles the presented frame rate.
            if fg_active && self.fg_present {
                self.fg_present = false;
                let real_img = self.fg_real.as_ref().unwrap().0;
                let to_src = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(real_img)
                    .subresource_range(range);
                d.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
                let region = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                    .image_extent(vk::Extent3D { width: self.width, height: self.height, depth: 1 });
                d.cmd_copy_image_to_buffer(self.cmd, real_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, self.readback_buffer, &[region]);
                let _ = d.end_command_buffer(self.cmd);
                let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd));
                let _ = d.queue_submit(self.queue, &[submit], self.fence);
                let _ = d.wait_for_fences(&[self.fence], true, u64::MAX);
                return;
            }

            // Rebuild the HDRI importance-sampling distribution when the world/HDRI changed (rare).
            // One workgroup; barrier so the trace sees the written buffer.
            if self.env_dirty {
                if let Some(ep) = self.env_pipeline {
                    d.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, ep);
                    d.cmd_bind_descriptor_sets(
                        self.cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipeline_layout,
                        0,
                        &[self.desc_set],
                        &[],
                    );
                    d.cmd_dispatch(self.cmd, 1, 1, 1);
                    let mem = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ);
                    d.cmd_pipeline_barrier(
                        self.cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        trace_stage,
                        vk::DependencyFlags::empty(),
                        &[mem],
                        &[],
                        &[],
                    );
                }
                self.env_dirty = false;
            }

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
            // Guide buffers: discard last frame and make them writable (regenerated every frame).
            let mut barriers = vec![to_general, accum_bar];
            if let Some(g) = &self.guide {
                for img in [g.depth.0, g.normal_rough.0, g.albedo.0, g.motion.0, g.noisy.0, g.clean.0, g.spec_albedo.0] {
                    barriers.push(
                        vk::ImageMemoryBarrier::default()
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::GENERAL)
                            .src_access_mask(vk::AccessFlags::empty())
                            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(img)
                            .subresource_range(range),
                    );
                }
            }
            d.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                trace_or_compute,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );

            let bytes = std::slice::from_raw_parts(
                &push as *const CameraPush as *const u8,
                std::mem::size_of::<CameraPush>(),
            );
            // Path trace: hybrid-raster debug (R1) when use_raster; else the wavefront (writes the
            // offscreen) when enabled; else the megakernel ray-gen (RT) or the sky compute fallback.
            if use_wavefront {
                // Hybrid raster (R2): rasterize primary visibility into the G-buffer at trace res, so
                // wf_gen resolves the primary hit without a ray query. Runs only when both toggles on.
                if raster_active {
                let r = self.raster.as_ref().unwrap();
                let (tw, th) = (self.trace_width, self.trace_height);
                let extent = vk::Extent2D { width: tw, height: th };
                let clear = [
                    vk::ClearValue { color: vk::ClearColorValue { uint32: [0xFFFF_FFFF, 0, 0, 0] } }, // hit (miss)
                    vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] } },   // world pos
                    vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
                ];
                d.cmd_begin_render_pass(
                    self.cmd,
                    &vk::RenderPassBeginInfo::default()
                        .render_pass(r.render_pass)
                        .framebuffer(r.framebuffer)
                        .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                        .clear_values(&clear),
                    vk::SubpassContents::INLINE,
                );
                d.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::GRAPHICS, r.pipeline);
                d.cmd_set_viewport(self.cmd, 0, &[vk::Viewport {
                    // Negative height flips Y so the raster matches the path tracer's ndc_y (+1 = top).
                    x: 0.0, y: th as f32, width: tw as f32, height: -(th as f32), min_depth: 0.0, max_depth: 1.0,
                }]);
                d.cmd_set_scissor(self.cmd, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent }]);
                let view_proj = glam::Mat4::from_cols_array(&cur_vp);
                let rlayout = r.pipeline_layout;
                for i in 0..self.instance_transforms.len() {
                    let bi = self.instance_blas.get(i).copied().unwrap_or(u32::MAX) as usize;
                    let blas = match self.blas_list.get(bi).and_then(|o| o.as_ref()) {
                        Some(b) if b.index_count > 0 => b,
                        _ => continue,
                    };
                    let rpush = RasterPush {
                        mvp: (view_proj * mat4_from_3x4(&self.instance_transforms[i])).to_cols_array(),
                        o2w: self.instance_transforms[i],
                        instance_id: i as u32,
                    };
                    let rbytes = std::slice::from_raw_parts(
                        &rpush as *const RasterPush as *const u8, std::mem::size_of::<RasterPush>());
                    d.cmd_push_constants(self.cmd, rlayout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, rbytes);
                    d.cmd_bind_vertex_buffers(self.cmd, 0, &[blas.vbuf.buffer], &[0]);
                    d.cmd_bind_index_buffer(self.cmd, blas.ibuf.buffer, 0, vk::IndexType::UINT32);
                    d.cmd_draw_indexed(self.cmd, blas.index_count, 1, 0, 0, 0);
                }
                d.cmd_end_render_pass(self.cmd);
                // Raster color writes (now in GENERAL) -> compute read of the hit G-buffer.
                let mb = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                d.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                }
                // The wavefront runs at trace (render) resolution = push.dims. In RR mode it emits
                // gNoisy + guides there for DLSS to upscale; with RR off, trace res == display res and
                // resolve writes the offscreen directly. wf_gen reads the raster G-buffer when present.
                let wf_push = push;
                let groups = (self.trace_width * self.trace_height).div_ceil(64);
                d.cmd_bind_descriptor_sets(
                    self.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.desc_set], &[],
                );
                // Phase 2 compaction: a ping-pong live-path queue (binding 20) + control buffer (21).
                // Each extend round reads the live pixel indices, traces one bounce, and atomic-appends
                // survivors to the other half; resolve reads every path's final radiance. One broad
                // barrier covers the compute stages + the fillBuffer count resets (over-sync but correct).
                let n = self.trace_width * self.trace_height;
                let ctrl = self.wf_ctrl_buffer.as_ref().unwrap().buffer;
                let cmd = self.cmd;
                let layout = self.pipeline_layout;
                let wf_mb = vk::MemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
                            | vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::INDIRECT_COMMAND_READ,
                    );
                let wf_stage = vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::DRAW_INDIRECT;
                let do_barrier = || unsafe {
                    d.cmd_pipeline_barrier(cmd, wf_stage, wf_stage, vk::DependencyFlags::empty(), &[wf_mb], &[], &[]);
                };
                let push_wf = |wf_in: u32, wf_out: u32, sample_idx: u32| unsafe {
                    let mut p = wf_push;
                    p.wf_in = wf_in;
                    p.wf_out = wf_out;
                    p.sample_idx = sample_idx;
                    let bytes = std::slice::from_raw_parts(
                        &p as *const CameraPush as *const u8,
                        std::mem::size_of::<CameraPush>(),
                    );
                    d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
                };
                // Samples/Pixel: run the whole wavefront `spp` times this frame. Each sample carries a
                // distinct sample_idx; the gen walks a fresh point of the per-pixel sampling sequence
                // (global sample = frame*spp + sample_idx) and the resolve accumulates them — RR averages
                // into gNoisy, progressive sums into the accumulator. spp == 1 is the realtime default.
                for s in 0u32..spp {
                    do_barrier(); // isolate this sample's PathState/accum from the previous sample
                    // Control init: count0 = N (every pixel live in round 0), count1 = 0, and the round-0
                    // indirect args = ceil(N / 64) workgroups (offset 8) + argsY = argsZ = 1. Later rounds
                    // get their args from the compaction pass.
                    d.cmd_fill_buffer(cmd, ctrl, 0, 4, n);
                    d.cmd_fill_buffer(cmd, ctrl, 4, 4, 0);
                    d.cmd_fill_buffer(cmd, ctrl, 8, 4, n.div_ceil(64));
                    d.cmd_fill_buffer(cmd, ctrl, 12, 8, 1);
                    // Stage 1: ray generation -> PathState + identity live queue (half 0).
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.wf_gen_pipeline.unwrap());
                    push_wf(0, 0, s);
                    d.cmd_dispatch(cmd, groups, 1, 1);
                    // Stage 2: extend, ONE real bounce per dispatch, over the LIVE paths only — a
                    // dispatch-indirect over the compacted count, so dead paths never get threads. Looped
                    // max_bounces times; the shader's c.b >= maxBounces check terminates paths in lockstep.
                    for r in 0u32..max_bounces {
                        let (wi, wo) = (r & 1, (r + 1) & 1);
                        do_barrier();
                        d.cmd_fill_buffer(cmd, ctrl, (wo as u64) * 4, 4, 0); // reset the write side's live count
                        do_barrier();
                        push_wf(wi, wo, s);
                        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.wf_extend_pipeline.unwrap());
                        d.cmd_dispatch_indirect(cmd, ctrl, 8); // ceil(live / 64) workgroups
                        do_barrier();
                        // Compaction: turn the survivor count into the next round's indirect args (1 thread).
                        push_wf(wi, wo, s);
                        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.wf_compact_pipeline.unwrap());
                        d.cmd_dispatch(cmd, 1, 1, 1);
                    }
                    // Stage 3: resolve -> firefly clamp + accumulate (RR -> gNoisy; progressive -> accum + tonemap).
                    do_barrier();
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.wf_resolve_pipeline.unwrap());
                    push_wf(0, 0, s);
                    d.cmd_dispatch(cmd, groups, 1, 1);
                }
            } else if let Some(rt) = self.rt_pipeline_ext.clone() {
                d.cmd_bind_pipeline(
                    self.cmd,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.pipeline,
                );
                d.cmd_bind_descriptor_sets(
                    self.cmd,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.pipeline_layout,
                    0,
                    &[self.desc_set],
                    &[],
                );
                d.cmd_push_constants(
                    self.cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::RAYGEN_KHR,
                    0,
                    bytes,
                );
                let empty = vk::StridedDeviceAddressRegionKHR::default();
                rt.cmd_trace_rays(
                    self.cmd,
                    &self.sbt_region,
                    &empty,
                    &empty,
                    &empty,
                    self.trace_width,
                    self.trace_height,
                    1,
                );
            } else {
                d.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
                d.cmd_bind_descriptor_sets(
                    self.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[self.desc_set],
                    &[],
                );
                d.cmd_push_constants(
                    self.cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes,
                );
                d.cmd_dispatch(self.cmd, self.trace_width.div_ceil(8), self.trace_height.div_ceil(8), 1);
            }

            // DLSS Ray Reconstruction: denoise the noisy 1-spp color (with the guide buffers) into
            // the clean image, then tonemap that to the display offscreen. NGX records its own
            // compute work into self.cmd; all its resources stay in GENERAL, so plain memory
            // barriers (not layout transitions) bracket it.
            if use_rr {
                use ash::vk::Handle;
                let g = self.guide.as_ref().unwrap();
                let rr = self.rr.as_ref().unwrap();
                let wr_to_rd = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                // Trace's noisy-color + guide writes (ray-gen stage) -> NGX reads (compute).
                d.cmd_pipeline_barrier(
                    self.cmd,
                    trace_stage,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[wr_to_rd],
                    &[],
                    &[],
                );
                let f16 = vk::Format::R16G16B16A16_SFLOAT.as_raw();
                let r32 = vk::Format::R32_SFLOAT.as_raw();
                let img = |t: &(vk::Image, Allocation, vk::ImageView), fmt: i32| crate::ngx::RrImage {
                    view: t.2.as_raw(),
                    image: t.0.as_raw(),
                    format: fmt,
                };
                crate::ngx::evaluate_rr(
                    self.cmd.as_raw(),
                    rr,
                    img(&g.noisy, f16),
                    img(&g.clean, f16),
                    img(&g.depth, r32),
                    img(&g.motion, f16),
                    img(&g.normal_rough, f16),
                    img(&g.albedo, f16),
                    img(&g.spec_albedo, f16),
                    self.trace_width,  // inputs at trace (render) resolution
                    self.trace_height,
                    self.width,        // output (clean) at display resolution
                    self.height,
                    ngx_jitter[0],
                    ngx_jitter[1],
                    dlss_reset,             // fresh accumulation or a detected hard cut (teleport)
                    frame_delta_ms,         // real frame delta (ms) for DLSS-RR temporal feedback
                );
                // NGX's clean output -> tonemap reads.
                d.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[wr_to_rd],
                    &[],
                    &[],
                );
                d.cmd_bind_pipeline(
                    self.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.tonemap_pipeline.unwrap(),
                );
                d.cmd_bind_descriptor_sets(
                    self.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[self.desc_set],
                    &[],
                );
                // The tonemap runs at DISPLAY resolution (it reads the upscaled clean image), so the
                // shared push's dims — set to the trace resolution for the path tracer — would clip
                // the output to the top-left trace-res region. Override them with the display size.
                let mut tm_push = push;
                tm_push.dims = [self.width, self.height];
                let tm_bytes = std::slice::from_raw_parts(
                    &tm_push as *const CameraPush as *const u8,
                    std::mem::size_of::<CameraPush>(),
                );
                d.cmd_push_constants(
                    self.cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    tm_bytes,
                );
                d.cmd_dispatch(self.cmd, self.width.div_ceil(8), self.height.div_ceil(8), 1);
            }

            // ── Frame-gen RENDER phase: interpolate between the held previous frame and this one ──
            // DLSS-G keeps the previous backbuffer internally; feeding the (final, tonemapped) offscreen
            // produces the in-between frame in fg_interp, which we present this frame. The NEXT frame is
            // the present phase (the held real offscreen). That alternation is what doubles the FPS.
            let mut rb_img = self.offscreen_image;
            if fg_active {
                use ash::vk::Handle;
                let interp = self.fg_interp.as_ref().unwrap();
                let real = self.fg_real.as_ref().unwrap();
                let g = self.guide.as_ref().unwrap();
                // Make the tonemap's offscreen write + the guides visible to NGX (compute), and put the
                // output images into GENERAL (NGX writes them as storage; their prior contents are
                // discarded each frame, so UNDEFINED -> GENERAL).
                let mb = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                let to_gen = |img: vk::Image| vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(img)
                    .subresource_range(range);
                let img_bars = [to_gen(interp.0), to_gen(real.0)];
                d.cmd_pipeline_barrier(self.cmd, trace_or_compute, vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(), &[mb], &[], &img_bars);
                // ClipToPrevClip: current clip -> world (invViewProj) -> previous clip (prevViewProj).
                // Use the REAL previous-frame view-proj (the local `prev_vp`), NOT self.prev_view_proj,
                // which render() already advanced to the current frame above — that would make
                // clipToPrevClip the identity and collapse the interpolation. clipToPrevClip = current
                // clip -> world -> previous clip; prevClipToClip is its inverse. glam's column-major +
                // pre-multiply layout matches NGX's row-major + post-multiply -> no transpose.
                let cur_vp_m = glam::Mat4::from_cols_array(&cur_vp);
                let prev_vp_m = glam::Mat4::from_cols_array(&prev_vp);
                let clip_to_prev_m = prev_vp_m * cur_vp_m.inverse();
                let clip_to_prev = clip_to_prev_m.to_cols_array();
                let prev_clip = clip_to_prev_m.inverse().to_cols_array();
                let ri = |image: u64, view: u64, fmt: i32| crate::ngx::RrImage { view, image, format: fmt };
                let rb = RB_FORMAT.as_raw();
                let f16 = vk::Format::R16G16B16A16_SFLOAT.as_raw();
                let r32 = vk::Format::R32_SFLOAT.as_raw();
                crate::ngx::evaluate_fg(
                    self.cmd.as_raw(),
                    self.fg.as_ref().unwrap(),
                    ri(self.offscreen_image.as_raw(), self.offscreen_view.as_raw(), rb),
                    ri(g.motion.0.as_raw(), g.motion.2.as_raw(), f16),
                    ri(g.depth.0.as_raw(), g.depth.2.as_raw(), r32),
                    ri(interp.0.as_raw(), interp.2.as_raw(), rb),
                    ri(real.0.as_raw(), real.2.as_raw(), rb),
                    self.width,
                    self.height,
                    crate::config::get_int("fg_frames") as u32,
                    clip_to_prev,
                    prev_clip,
                    0.1,
                    10000.0,
                    self.accum_frame == 0,
                );
                // TEMP: present out_real (DLSS-FG's clean passthrough, which works) on both phases —
                // no flicker — until the interpolation is fixed. out_interp comes out black because our
                // depth guide is LINEAR view-space (for DLSS-RR) but DLSS-FG needs HARDWARE/non-linear
                // depth; with bad depth it can't interpolate and leaves out_interp untouched. Fix =
                // feed DLSS-FG a separate hardware-depth buffer (clip.z/clip.w). See frame-gen-plan.
                rb_img = real.0;
                self.fg_present = true; // next phase: present the held real (out_real)
            }

            // For the fg readback, DLSS-FG writes rb_img (out_interp) via its own internal mix of
            // compute + transfer + OFA stages, so a SHADER_WRITE/compute source mask would race the
            // copy and read undefined (black). Wait on ALL_COMMANDS / MEMORY_WRITE in that case.
            let (rb_src_stage, rb_src_access) = if fg_active {
                (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_WRITE)
            } else {
                (trace_or_compute, vk::AccessFlags::SHADER_WRITE)
            };
            let to_src = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(rb_src_access)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(rb_img)
                .subresource_range(range);
            d.cmd_pipeline_barrier(
                self.cmd,
                rb_src_stage,
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
                rb_img,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback_buffer,
                &[region],
            );

            let _ = d.end_command_buffer(self.cmd);
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd));
            // GPU frame time: render() submits then blocks on the fence, so wall-clock around the
            // wait ~= GPU work. EMA-smoothed; logged periodically to compare SER on vs off.
            let t0 = std::time::Instant::now();
            let _ = d.queue_submit(self.queue, &[submit], self.fence);
            let _ = d.wait_for_fences(&[self.fence], true, u64::MAX);
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            self.ft_ema = if self.ft_ema <= 0.0 { ms } else { self.ft_ema * 0.9 + ms * 0.1 };
            self.ft_count = self.ft_count.wrapping_add(1);
            if self.ft_count % 60 == 0 {
                log(&format!(
                    "frame time: {:.2} ms ({:.0} fps), SER {}",
                    self.ft_ema,
                    1000.0 / self.ft_ema.max(1e-3),
                    if self.ser { "on" } else { "off" }
                ));
            }
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

    Ok(Blas { accel: accel_struct, buf, vbuf, ibuf, nbuf, ubuf, matbuf: None, address,
        index_count: mesh.indices.len() as u32 })
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
            if let Some(tp) = self.tonemap_pipeline.take() {
                device.destroy_pipeline(tp, None);
            }
            if let Some(ep) = self.env_pipeline.take() {
                device.destroy_pipeline(ep, None);
            }
            if let Some(p) = self.wf_gen_pipeline.take() {
                device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.wf_extend_pipeline.take() {
                device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.wf_resolve_pipeline.take() {
                device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.wf_compact_pipeline.take() {
                device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.raster_debug_pipeline.take() {
                device.destroy_pipeline(p, None);
            }
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
                if let Some(g) = self.guide.take() {
                    for (image, galloc, view) in
                        [g.depth, g.normal_rough, g.albedo, g.motion, g.noisy, g.clean, g.spec_albedo]
                    {
                        device.destroy_image_view(view, None);
                        device.destroy_image(image, None);
                        let _ = alloc.free(galloc);
                    }
                }
                if let Some(r) = self.raster.take() {
                    device.destroy_pipeline(r.pipeline, None);
                    device.destroy_pipeline_layout(r.pipeline_layout, None);
                    device.destroy_framebuffer(r.framebuffer, None);
                    device.destroy_render_pass(r.render_pass, None);
                    for (image, ralloc, view) in [r.hit, r.pos, r.depth] {
                        device.destroy_image_view(view, None);
                        device.destroy_image(image, None);
                        let _ = alloc.free(ralloc);
                    }
                }
                for slot in [self.fg_interp.take(), self.fg_real.take()] {
                    if let Some((image, falloc, view)) = slot {
                        device.destroy_image_view(view, None);
                        device.destroy_image(image, None);
                        let _ = alloc.free(falloc);
                    }
                }
                if let Some(b) = self.light_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if let Some(b) = self.emissive_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if let Some(b) = self.sbt_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if let Some(b) = self.env_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if let Some(b) = self.world_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if let Some(b) = self.prev_xform_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if let Some(b) = self.wf_pathstate_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if let Some(b) = self.raster_inst_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if let Some(b) = self.wf_queue_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if let Some(b) = self.wf_ctrl_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if let Some(b) = self.sharc_keys_buffer.take() {
                    b.destroy(&device, alloc);
                }
                if let Some(b) = self.sharc_data_buffer.take() {
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
