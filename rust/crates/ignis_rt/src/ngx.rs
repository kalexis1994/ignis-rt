//! NVIDIA NGX (DLSS) FFI — pure-Rust bindings to nvsdk_ngx_d.lib for DLSS Ray Reconstruction.
//!
//! Stage 0: just initialize NGX after device creation and log whether it (and DLSS-RR) is
//! available on the GPU. The heavy machinery (RR feature create + per-frame evaluate with the
//! guide buffers) lands in later stages. The lib links a static loader that pulls in
//! nvngx_dlss*.dll at runtime — those DLLs ship next to ignis_rt.dll on the RTX box.
//!
//! Compiled only when build.rs found the NGX SDK (`have_ngx` cfg). When absent, `init()` is a
//! no-op stub so non-RTX / SDK-less builds still work.

use crate::log::log;

#[cfg(have_ngx)]
mod ffi {
    use std::os::raw::{c_char, c_void};

    // NVSDK_NGX_Result: Success = 0x1, failures are 0xBAD0000X. Treated as i32.
    pub const NGX_SUCCESS: i32 = 0x1;
    // NVSDK_NGX_Version_API (nvsdk_ngx_defs.h: 0x0000015 = 1.5.0).
    pub const NGX_VERSION_API: i32 = 0x0000015;

    pub type VkHandle = *const c_void; // dispatchable Vulkan handle (instance/physdev/device)

    pub const NGX_FEATURE_RAY_RECONSTRUCTION: i32 = 13;
    // Create params.
    pub const PERF_QUALITY_DLAA: i32 = 5;
    pub const DLSS_FLAG_IS_HDR: i32 = 1 << 0;
    pub const DLSS_FLAG_MV_LOWRES: i32 = 1 << 1;
    pub const DLSS_FLAG_AUTO_EXPOSURE: i32 = 1 << 6;
    pub const DENOISE_MODE_DL_UNIFIED: i32 = 1;
    pub const ROUGHNESS_MODE_PACKED: u32 = 1;

    // NVSDK_NGX_Resource_VK — union(ImageViewInfo_VK, 48 bytes) + Type + ReadWrite. We only ever
    // wrap image views, so the union is modeled as ImageViewInfo_VK inline.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct ResourceVk {
        pub image_view: u64, // VkImageView
        pub image: u64,      // VkImage
        pub subresource: ash::vk::ImageSubresourceRange, // 20 bytes
        pub format: i32,     // VkFormat
        pub width: u32,
        pub height: u32,
        pub res_type: i32,   // 0 = IMAGEVIEW
        pub read_write: u8,  // bool
    }

    extern "C" {
        // The exported C symbol (9 args). gipa/gdpa/feature_info may be null — NGX then uses the
        // already-loaded Vulkan loader. Matches the non-NGX_SNIPPET_BUILD declaration.
        pub fn NVSDK_NGX_VULKAN_Init(
            app_id: u64,
            app_data_path: *const u16, // wchar_t* (UTF-16 on Windows)
            instance: VkHandle,
            phys_device: VkHandle,
            device: VkHandle,
            gipa: *const c_void,          // PFN_vkGetInstanceProcAddr
            gdpa: *const c_void,          // PFN_vkGetDeviceProcAddr
            feature_info: *const c_void,  // NVSDK_NGX_FeatureCommonInfo*
            version: i32,
        ) -> i32;

        pub fn NVSDK_NGX_VULKAN_Shutdown1(device: VkHandle) -> i32;

        pub fn NVSDK_NGX_VULKAN_AllocateParameters(out_params: *mut *mut c_void) -> i32;
        pub fn NVSDK_NGX_VULKAN_GetCapabilityParameters(out_params: *mut *mut c_void) -> i32;
        pub fn NVSDK_NGX_VULKAN_DestroyParameters(params: *mut c_void) -> i32;
        pub fn NVSDK_NGX_Parameter_GetI(params: *mut c_void, name: *const c_char, out: *mut i32) -> i32;
        pub fn NVSDK_NGX_Parameter_SetUI(params: *mut c_void, name: *const c_char, value: u32);
        pub fn NVSDK_NGX_Parameter_SetI(params: *mut c_void, name: *const c_char, value: i32);
        pub fn NVSDK_NGX_Parameter_SetF(params: *mut c_void, name: *const c_char, value: f32);
        pub fn NVSDK_NGX_Parameter_SetVoidPointer(params: *mut c_void, name: *const c_char, value: *mut c_void);
        pub fn NVSDK_NGX_VULKAN_CreateFeature1(
            device: VkHandle, cmd: VkHandle, feature_id: i32,
            params: *mut c_void, out_handle: *mut *mut c_void,
        ) -> i32;
        pub fn NVSDK_NGX_VULKAN_EvaluateFeature_C(
            cmd: VkHandle, handle: *mut c_void, params: *mut c_void, callback: *const c_void,
        ) -> i32;
        pub fn NVSDK_NGX_VULKAN_ReleaseFeature(handle: *mut c_void) -> i32;
    }
}

/// Directory containing THIS dll (ignis_rt.dll) — that's where the nvngx_dlss*.dll snippets ship,
/// and where NGX must look to find them. Uses GetModuleHandleEx(FROM_ADDRESS) on an in-module
/// symbol so we get our own module regardless of the loading executable.
#[cfg(have_ngx)]
fn dll_dir() -> Option<String> {
    use std::os::raw::c_void;
    extern "system" {
        fn GetModuleHandleExW(flags: u32, addr: *const u16, out: *mut *mut c_void) -> i32;
        fn GetModuleFileNameW(module: *mut c_void, filename: *mut u16, size: u32) -> u32;
    }
    static ANCHOR: u8 = 0; // a symbol guaranteed to live in this DLL
    const FROM_ADDRESS: u32 = 0x4;
    const UNCHANGED_REFCOUNT: u32 = 0x2;
    unsafe {
        let mut hmod: *mut c_void = std::ptr::null_mut();
        if GetModuleHandleExW(FROM_ADDRESS | UNCHANGED_REFCOUNT, &ANCHOR as *const u8 as *const u16, &mut hmod) == 0 {
            return None;
        }
        let mut buf = [0u16; 520];
        let n = GetModuleFileNameW(hmod, buf.as_mut_ptr(), buf.len() as u32);
        if n == 0 {
            return None;
        }
        let path = String::from_utf16_lossy(&buf[..n as usize]);
        std::path::Path::new(&path).parent().map(|p| p.to_string_lossy().into_owned())
    }
}

/// Initialize NGX for the given Vulkan handles (raw `as_raw()` pointers). Logs the outcome.
/// Returns true if NGX initialized successfully (RTX + driver + runtime DLLs present).
#[cfg(have_ngx)]
pub fn init(
    instance: u64,
    phys_device: u64,
    device: u64,
    gipa: *const std::os::raw::c_void, // vkGetInstanceProcAddr
    gdpa: *const std::os::raw::c_void, // vkGetDeviceProcAddr
    _app_data_path: &str,
) -> bool {
    use ffi::*;
    // NGX searches the application data path for the feature snippets (nvngx_dlssd.dll). Point it
    // at this DLL's own directory (where the snippets are deployed), not the caller's path.
    let app_data = dll_dir().unwrap_or_else(|| ".".to_string());
    log(&format!("NGX: app data path = {app_data}"));
    let mut path16: Vec<u16> = app_data.encode_utf16().collect();
    path16.push(0);
    // Application id used by the C++ build (arbitrary but stable).
    const APP_ID: u64 = 0x1337BEEF;
    let res = unsafe {
        NVSDK_NGX_VULKAN_Init(
            APP_ID,
            path16.as_ptr(),
            instance as VkHandle,
            phys_device as VkHandle,
            device as VkHandle,
            gipa, // real proc-addrs: NGX needs these to set up feature creation
            gdpa,
            std::ptr::null(), // feature_info
            NGX_VERSION_API,
        )
    };
    if res != NGX_SUCCESS {
        log(&format!("NGX: init FAILED (result {res:#x}) — DLSS unavailable on this device"));
        return false;
    }
    // GetCapabilityParameters initializes the NGX feature subsystem (loads the snippet, queries
    // the driver) — required before CreateFeature, otherwise it returns NotInitialized.
    let mut caps: *mut std::os::raw::c_void = std::ptr::null_mut();
    let cres = unsafe { NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut caps) };
    if cres == NGX_SUCCESS && !caps.is_null() {
        let name = std::ffi::CString::new("SuperSampling.Available").unwrap();
        let mut avail: i32 = 0;
        unsafe { NVSDK_NGX_Parameter_GetI(caps, name.as_ptr(), &mut avail) };
        log(&format!("NGX: init OK; DLSS available = {avail}"));
    } else {
        log(&format!("NGX: init OK but GetCapabilityParameters failed (result {cres:#x})"));
    }
    true
}

/// A created DLSS Ray Reconstruction feature + its NGX parameter map.
#[cfg(have_ngx)]
pub struct RrFeature {
    pub handle: *mut std::os::raw::c_void,
    pub params: *mut std::os::raw::c_void,
}
#[cfg(have_ngx)]
unsafe impl Send for RrFeature {} // raw NGX pointers; only touched from the render thread

/// Create the DLSS-RR feature in DLAA mode (render == output resolution). `cmd` must be a
/// command buffer in the recording state (NGX records initialization into it); the caller
/// submits + waits. Returns None on failure (logged).
#[cfg(have_ngx)]
pub fn create_rr(device: u64, cmd: u64, width: u32, height: u32) -> Option<RrFeature> {
    use ffi::*;
    use std::ffi::CString;
    use std::os::raw::c_void;
    let mut params: *mut c_void = std::ptr::null_mut();
    if unsafe { NVSDK_NGX_VULKAN_AllocateParameters(&mut params) } != NGX_SUCCESS || params.is_null() {
        log("NGX: AllocateParameters failed");
        return None;
    }
    let set_ui = |n: &str, v: u32| { let c = CString::new(n).unwrap(); unsafe { NVSDK_NGX_Parameter_SetUI(params, c.as_ptr(), v) }; };
    let set_i  = |n: &str, v: i32| { let c = CString::new(n).unwrap(); unsafe { NVSDK_NGX_Parameter_SetI(params, c.as_ptr(), v) }; };
    set_ui("CreationNodeMask", 1);
    set_ui("VisibilityNodeMask", 1);
    set_ui("Width", width);
    set_ui("Height", height);
    set_ui("OutWidth", width);   // DLAA: render resolution == output resolution
    set_ui("OutHeight", height);
    set_i("PerfQualityValue", PERF_QUALITY_DLAA);
    set_i("DLSS.Feature.Create.Flags", DLSS_FLAG_IS_HDR | DLSS_FLAG_MV_LOWRES | DLSS_FLAG_AUTO_EXPOSURE);
    set_i("DLSS.Denoise.Mode", DENOISE_MODE_DL_UNIFIED);
    set_ui("DLSS.Roughness.Mode", ROUGHNESS_MODE_PACKED);
    set_ui("DLSS.Use.HW.Depth", 0);
    let mut handle: *mut c_void = std::ptr::null_mut();
    let res = unsafe {
        NVSDK_NGX_VULKAN_CreateFeature1(
            device as VkHandle, cmd as VkHandle, NGX_FEATURE_RAY_RECONSTRUCTION, params, &mut handle,
        )
    };
    if res == NGX_SUCCESS && !handle.is_null() {
        log(&format!("NGX: DLSS-RR feature created OK ({width}x{height}, DLAA)"));
        Some(RrFeature { handle, params })
    } else {
        log(&format!("NGX: CreateFeature RR FAILED (result {res:#x})"));
        unsafe { NVSDK_NGX_VULKAN_DestroyParameters(params) };
        None
    }
}

/// One input/output image for the RR evaluate (raw `as_raw()` view+image handles + VkFormat).
#[cfg(have_ngx)]
#[derive(Clone, Copy)]
pub struct RrImage {
    pub view: u64,
    pub image: u64,
    pub format: i32,
}

/// Run DLSS Ray Reconstruction for one frame. Wraps each image in an NVSDK_NGX_Resource_VK, sets
/// the essential eval parameters (the helper sets many more, but they are optional/null), and
/// dispatches EvaluateFeature_C on `cmd`. `width`/`height` are the render (== output, DLAA) size.
#[cfg(have_ngx)]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_rr(
    cmd: u64,
    rr: &RrFeature,
    color: RrImage,    // noisy 1-spp linear color (input)
    output: RrImage,   // clean denoised color (output, read-write)
    depth: RrImage,
    motion: RrImage,
    normal_rough: RrImage, // normals (rgb) + roughness packed in .a
    albedo: RrImage,       // diffuse albedo
    spec_albedo: RrImage,  // specular albedo (EnvBRDFApprox)
    width: u32,
    height: u32,
    jitter_x: f32,
    jitter_y: f32,
    reset: bool,
    frame_delta_ms: f32,
) -> bool {
    use ffi::*;
    use std::ffi::CString;
    use std::os::raw::c_void;

    let range = ash::vk::ImageSubresourceRange {
        aspect_mask: ash::vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let mk = |img: RrImage, rw: bool| ResourceVk {
        image_view: img.view,
        image: img.image,
        subresource: range,
        format: img.format,
        width,
        height,
        res_type: 0, // IMAGEVIEW
        read_write: rw as u8,
    };
    // These must outlive the EvaluateFeature_C call (referenced by pointer).
    let mut r_color = mk(color, false);
    let mut r_output = mk(output, true);
    let mut r_depth = mk(depth, false);
    let mut r_motion = mk(motion, false);
    let mut r_normal = mk(normal_rough, false);
    let mut r_albedo = mk(albedo, false);
    let mut r_specalb = mk(spec_albedo, false);

    let p = rr.params;
    let set_ptr = |n: &str, r: *mut ResourceVk| { let c = CString::new(n).unwrap(); unsafe { NVSDK_NGX_Parameter_SetVoidPointer(p, c.as_ptr(), r as *mut c_void) }; };
    let set_f = |n: &str, v: f32| { let c = CString::new(n).unwrap(); unsafe { NVSDK_NGX_Parameter_SetF(p, c.as_ptr(), v) }; };
    let set_i = |n: &str, v: i32| { let c = CString::new(n).unwrap(); unsafe { NVSDK_NGX_Parameter_SetI(p, c.as_ptr(), v) }; };
    let set_ui = |n: &str, v: u32| { let c = CString::new(n).unwrap(); unsafe { NVSDK_NGX_Parameter_SetUI(p, c.as_ptr(), v) }; };

    set_ptr("Color", &mut r_color);
    set_ptr("Output", &mut r_output);
    set_ptr("Depth", &mut r_depth);
    set_ptr("MotionVectors", &mut r_motion);
    set_ptr("GBuffer.Normals", &mut r_normal);
    set_ptr("GBuffer.Roughness", &mut r_normal);      // roughness packed in normals.a
    set_ptr("DLSS.Input.DiffuseAlbedo", &mut r_albedo);
    set_ptr("DLSS.Input.SpecularAlbedo", &mut r_specalb); // EnvBRDFApprox specular albedo
    set_f("Jitter.Offset.X", jitter_x);
    set_f("Jitter.Offset.Y", jitter_y);
    set_i("Reset", reset as i32);
    set_f("MV.Scale.X", 1.0);
    set_f("MV.Scale.Y", 1.0);
    set_ui("DLSS.Render.Subrect.Dimensions.Width", width);
    set_ui("DLSS.Render.Subrect.Dimensions.Height", height);
    set_f("FrameTimeDeltaInMsec", frame_delta_ms);

    let res = unsafe { NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd as VkHandle, rr.handle, p, std::ptr::null()) };
    if res != NGX_SUCCESS {
        log(&format!("NGX: EvaluateFeature RR failed (result {res:#x})"));
        return false;
    }
    true
}

#[cfg(have_ngx)]
pub fn release_rr(rr: RrFeature) {
    unsafe {
        ffi::NVSDK_NGX_VULKAN_ReleaseFeature(rr.handle);
        ffi::NVSDK_NGX_VULKAN_DestroyParameters(rr.params);
    }
}

#[cfg(have_ngx)]
pub fn shutdown(device: u64) {
    unsafe { ffi::NVSDK_NGX_VULKAN_Shutdown1(device as ffi::VkHandle) };
}

// ── Stubs when the NGX SDK wasn't available at build time ──
#[cfg(not(have_ngx))]
pub fn init(
    _instance: u64,
    _phys_device: u64,
    _device: u64,
    _gipa: *const std::os::raw::c_void,
    _gdpa: *const std::os::raw::c_void,
    _app_data_path: &str,
) -> bool {
    log("NGX: built without the NGX SDK — DLSS disabled");
    false
}

#[cfg(not(have_ngx))]
pub fn shutdown(_device: u64) {}

#[cfg(not(have_ngx))]
pub struct RrFeature;
#[cfg(not(have_ngx))]
pub fn create_rr(_device: u64, _cmd: u64, _w: u32, _h: u32) -> Option<RrFeature> { None }
#[cfg(not(have_ngx))]
pub fn release_rr(_rr: RrFeature) {}

#[cfg(not(have_ngx))]
#[derive(Clone, Copy)]
pub struct RrImage { pub view: u64, pub image: u64, pub format: i32 }
#[cfg(not(have_ngx))]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_rr(
    _cmd: u64, _rr: &RrFeature,
    _color: RrImage, _output: RrImage, _depth: RrImage, _motion: RrImage,
    _normal_rough: RrImage, _albedo: RrImage, _spec_albedo: RrImage,
    _w: u32, _h: u32, _jx: f32, _jy: f32, _reset: bool, _delta: f32,
) -> bool { false }
