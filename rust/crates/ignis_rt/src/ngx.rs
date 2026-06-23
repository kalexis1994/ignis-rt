//! NVIDIA NGX (DLSS) FFI — pure-Rust bindings to nvsdk_ngx_d.lib for DLSS Ray Reconstruction.
//!
//! Stage 0: just initialize NGX after device creation and log whether it (and DLSS-RR) is
//! available on the GPU. The heavy machinery (RR feature create + per-frame evaluate with the
//! guide buffers) lands in later stages. The lib links a static loader that pulls in the DLSS
//! snippets at runtime (nvngx_dlss*.dll on Windows, libnvidia-ngx-dlss*.so on Linux), shipped next
//! to the renderer's own shared library.
//!
//! Compiled only when build.rs found the NGX SDK (`have_ngx` cfg). When absent, `init()` is a
//! no-op stub so non-RTX / SDK-less builds still work.

use crate::log::log;
use std::sync::atomic::{AtomicU32, Ordering};

/// Max number of GENERATED frames DLSS-G supports on this GPU (0 = unavailable, 1 = 2x / Ada,
/// 3 = up to 4x / Blackwell). Detected at NGX init; the addon reads it to build the selector.
static DLSSG_MAX_FRAMES: AtomicU32 = AtomicU32::new(0);

/// The detected DLSS-G max generated-frame count (see DLSSG_MAX_FRAMES). 0 if unavailable/SDK-less.
pub fn dlssg_max_frames() -> u32 {
    DLSSG_MAX_FRAMES.load(Ordering::Relaxed)
}

#[cfg(have_ngx)]
mod ffi {
    use std::os::raw::{c_char, c_void};

    // NVSDK_NGX_Result: Success = 0x1, failures are 0xBAD0000X. Treated as i32.
    pub const NGX_SUCCESS: i32 = 0x1;
    // NVSDK_NGX_Version_API (nvsdk_ngx_defs.h: 0x0000015 = 1.5.0).
    pub const NGX_VERSION_API: i32 = 0x0000015;

    pub type VkHandle = *const c_void; // dispatchable Vulkan handle (instance/physdev/device)
    // wchar_t is 2 bytes (UTF-16) on Windows but 4 bytes (UTF-32) on Linux. NGX's app-data-path
    // argument is wchar_t*, so its element width must follow the platform.
    #[cfg(windows)] pub type WChar = u16;
    #[cfg(not(windows))] pub type WChar = u32;

    // NVSDK_NGX_Logging_Level
    pub const NGX_LOGGING_LEVEL_ON: i32 = 1;
    pub const NGX_LOGGING_LEVEL_VERBOSE: i32 = 2;
    pub type AppLogCallback = extern "C" fn(message: *const c_char, level: i32, feature: i32);
    // NVSDK_NGX_PathListInfo — extra directories NGX searches for the feature snippet (.so/.dll),
    // besides the default (the executable's folder).
    #[repr(C)]
    pub struct PathListInfo {
        pub path: *const *const WChar, // wchar_t const* const*
        pub length: u32,
    }
    // NVSDK_NGX_LoggingInfo — app logging callback + minimum level.
    #[repr(C)]
    pub struct LoggingInfo {
        pub callback: Option<AppLogCallback>,
        pub min_level: i32,
        pub disable_other_sinks: u8, // bool
    }
    // NVSDK_NGX_FeatureCommonInfo — passed to Init so NGX can find the snippet via PathListInfo
    // and pipe its diagnostics through our callback.
    #[repr(C)]
    pub struct FeatureCommonInfo {
        pub path_list: PathListInfo,
        pub internal_data: *mut c_void,
        pub logging: LoggingInfo,
    }

    pub const NGX_FEATURE_RAY_RECONSTRUCTION: i32 = 13;
    pub const NGX_FEATURE_FRAME_GENERATION: i32 = 11; // DLSS-G (NVSDK_NGX_Feature_FrameGeneration)
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
            app_data_path: *const WChar, // wchar_t* (UTF-16 Windows / UTF-32 Linux)
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

/// Encode a filesystem path as a platform `wchar_t` string (NUL-terminated): UTF-16 on Windows,
/// UTF-32 on Linux. NGX's app-data-path argument is `wchar_t*`, whose element width is platform-set.
#[cfg(have_ngx)]
fn encode_wide(s: &str) -> Vec<ffi::WChar> {
    #[cfg(windows)]
    { s.encode_utf16().chain(std::iter::once(0u16)).collect() }
    #[cfg(not(windows))]
    { s.chars().map(|c| c as u32).chain(std::iter::once(0u32)).collect() }
}

/// Directory containing THIS module's shared library — that's where the DLSS snippets ship
/// (nvngx_dlss*.dll on Windows, libnvidia-ngx-dlss*.so on Linux) and where NGX must look to find
/// them. Resolved from an in-module symbol so we get our own library regardless of the host process.
#[cfg(all(have_ngx, windows))]
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

/// Linux: `dladdr` on an in-module symbol yields the path of the .so that contains it.
#[cfg(all(have_ngx, unix))]
fn dll_dir() -> Option<String> {
    use std::os::raw::{c_char, c_int, c_void};
    #[repr(C)]
    struct DlInfo {
        dli_fname: *const c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const c_char,
        dli_saddr: *mut c_void,
    }
    extern "C" {
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
    }
    static ANCHOR: u8 = 0; // a symbol guaranteed to live in this shared object
    unsafe {
        let mut info: DlInfo = std::mem::zeroed();
        if dladdr(&ANCHOR as *const u8 as *const c_void, &mut info) == 0 || info.dli_fname.is_null() {
            return None;
        }
        let path = std::ffi::CStr::from_ptr(info.dli_fname).to_string_lossy().into_owned();
        std::path::Path::new(&path).parent().map(|p| p.to_string_lossy().into_owned())
    }
}

/// NGX's internal logging callback — forwards NGX's own diagnostics into our log so a failure to
/// locate/validate the DLSS snippet is visible. Called from arbitrary threads; must be thread-safe.
#[cfg(have_ngx)]
extern "C" fn ngx_log_callback(message: *const std::os::raw::c_char, _level: i32, _feature: i32) {
    if message.is_null() { return; }
    let s = unsafe { std::ffi::CStr::from_ptr(message) }.to_string_lossy();
    log(&format!("NGX[core]: {}", s.trim_end()));
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
    let path_w = encode_wide(&app_data);
    // NGX looks for the DLSS snippet next to the *executable* (Blender) or in the paths listed in
    // FeatureCommonInfo.PathListInfo — NOT in the app-data-path (that's only for logs/temp). Point
    // PathListInfo at our own lib dir (where the snippets are deployed) and install the log callback.
    let path_list: [*const WChar; 1] = [path_w.as_ptr()];
    let common = FeatureCommonInfo {
        path_list: PathListInfo { path: path_list.as_ptr(), length: 1 },
        internal_data: std::ptr::null_mut(),
        logging: LoggingInfo {
            callback: Some(ngx_log_callback),
            min_level: NGX_LOGGING_LEVEL_VERBOSE,
            disable_other_sinks: 0,
        },
    };
    // Application id used by the C++ build (arbitrary but stable).
    const APP_ID: u64 = 0x1337BEEF;
    let res = unsafe {
        NVSDK_NGX_VULKAN_Init(
            APP_ID,
            path_w.as_ptr(),
            instance as VkHandle,
            phys_device as VkHandle,
            device as VkHandle,
            gipa, // real proc-addrs: NGX needs these to set up feature creation
            gdpa,
            &common as *const FeatureCommonInfo as *const std::os::raw::c_void, // feature_info (snippet path + logging)
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
        let get_i = |n: &str| -> i32 {
            let c = std::ffi::CString::new(n).unwrap();
            let mut v: i32 = 0;
            unsafe { NVSDK_NGX_Parameter_GetI(caps, c.as_ptr(), &mut v) };
            v
        };
        let dlss = get_i("SuperSampling.Available");
        let fg = get_i("FrameGeneration.Available"); // DLSS-G — Ada (40-series) + recent driver
        // Max GENERATED frames: 1 = 2x (Ada), 3 = up to 4x (Blackwell MFG). 0 if DLSS-G unavailable.
        // The param is unset (0) on GPUs without multiframe, but single-frame gen (1) still works.
        let mfc_max = get_i("DLSSG.MultiFrameCountMax");
        let max_gen: u32 = if fg != 0 { mfc_max.max(1) as u32 } else { 0 };
        DLSSG_MAX_FRAMES.store(max_gen, Ordering::Relaxed);
        log(&format!("NGX: init OK; DLSS available = {dlss}, DLSS-G available = {fg}, max generated frames = {max_gen}"));
        if fg == 0 {
            // Why DLSS-G is unavailable: feature init result + whether a newer driver is needed.
            let needs_drv = get_i("FrameGeneration.NeedsUpdatedDriver");
            log(&format!(
                "NGX: DLSS-G unavailable — initResult={:#x}, needsUpdatedDriver={needs_drv}, minDriver={}.{}",
                get_i("FrameGeneration.FeatureInitResult"),
                get_i("FrameGeneration.MinDriverVersionMajor"),
                get_i("FrameGeneration.MinDriverVersionMinor"),
            ));
        }
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
pub fn create_rr(
    device: u64,
    cmd: u64,
    render_w: u32,
    render_h: u32,
    display_w: u32,
    display_h: u32,
    perf_quality: i32,
) -> Option<RrFeature> {
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
    set_ui("Width", render_w);   // render (trace) resolution — DLSS upscales from this
    set_ui("Height", render_h);
    set_ui("OutWidth", display_w); // output (display) resolution
    set_ui("OutHeight", display_h);
    set_i("PerfQualityValue", perf_quality);
    // RR does NOT support exposure / auto-exposure (RR Integration Guide §3.7) — must NOT set the
    // AUTO_EXPOSURE flag. Pre.Exposure / Exposure.Scale stay pinned to 1.0 (set at eval) to avoid the
    // NGX divide-by-zero, which is the real reason exposure is neutralised here.
    set_i("DLSS.Feature.Create.Flags", DLSS_FLAG_IS_HDR | DLSS_FLAG_MV_LOWRES);
    set_i("DLSS.Denoise.Mode", DENOISE_MODE_DL_UNIFIED);
    set_ui("DLSS.Roughness.Mode", ROUGHNESS_MODE_PACKED);
    set_ui("DLSS.Use.HW.Depth", 0);
    // RR model preset: E (5) = latest transformer model. Without an explicit preset NGX falls back
    // to a *different* default RR model that boils/shimmers noticeably on a static camera — this is
    // why our static image wavered where the C++/RTXPT reference stays calm. D (4) for perf modes.
    // Must be set before CreateFeature. Matches the C++ create exactly.
    set_ui("RayReconstruction.Hint.Render.Preset.DLAA", 5);
    set_ui("RayReconstruction.Hint.Render.Preset.Quality", 5);
    set_ui("RayReconstruction.Hint.Render.Preset.Balanced", 5);
    set_ui("RayReconstruction.Hint.Render.Preset.Performance", 4);
    set_ui("RayReconstruction.Hint.Render.Preset.UltraPerformance", 4);
    set_ui("RayReconstruction.Hint.Render.Preset.UltraQuality", 5);
    set_i("DLSS.Enable.Output.Subrects", 0);
    let mut handle: *mut c_void = std::ptr::null_mut();
    let res = unsafe {
        NVSDK_NGX_VULKAN_CreateFeature1(
            device as VkHandle, cmd as VkHandle, NGX_FEATURE_RAY_RECONSTRUCTION, params, &mut handle,
        )
    };
    if res == NGX_SUCCESS && !handle.is_null() {
        log(&format!("NGX: DLSS-RR feature created OK (render {render_w}x{render_h} -> display {display_w}x{display_h}, q={perf_quality})"));
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
    render_w: u32,  // trace resolution — inputs (color/depth/motion/normals/albedo)
    render_h: u32,
    display_w: u32, // output resolution — the upscaled clean image
    display_h: u32,
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
    let mk = |img: RrImage, rw: bool, w: u32, h: u32| ResourceVk {
        image_view: img.view,
        image: img.image,
        subresource: range,
        format: img.format,
        width: w,
        height: h,
        res_type: 0, // IMAGEVIEW
        read_write: rw as u8,
    };
    // These must outlive the EvaluateFeature_C call (referenced by pointer). Inputs are at the trace
    // resolution; the output (clean) is the full display resolution that DLSS upscales into.
    let mut r_color = mk(color, false, render_w, render_h);
    let mut r_output = mk(output, true, display_w, display_h);
    let mut r_depth = mk(depth, false, render_w, render_h);
    let mut r_motion = mk(motion, false, render_w, render_h);
    let mut r_normal = mk(normal_rough, false, render_w, render_h);
    let mut r_albedo = mk(albedo, false, render_w, render_h);
    let mut r_specalb = mk(spec_albedo, false, render_w, render_h);

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
    set_ui("DLSS.Render.Subrect.Dimensions.Width", render_w);
    set_ui("DLSS.Render.Subrect.Dimensions.Height", render_h);
    set_f("FrameTimeDeltaInMsec", frame_delta_ms);
    // Pre-exposure / exposure scale MUST be set: NGX reads an unset param as 0, and DLSS then tries
    // to un-apply a pre-exposure of 0 (divide by zero) -> normalization blows up / temporal boiling.
    // The helper guards 0 -> 1.0 for exactly this reason; we set params by hand, so do it ourselves.
    set_f("DLSS.Pre.Exposure", 1.0);
    set_f("DLSS.Exposure.Scale", 1.0);

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

/// A created DLSS Frame Generation (DLSS-G) feature + its NGX parameter map.
#[cfg(have_ngx)]
pub struct FgFeature {
    pub handle: *mut std::os::raw::c_void,
    pub params: *mut std::os::raw::c_void,
}
#[cfg(have_ngx)]
unsafe impl Send for FgFeature {}

/// Create the DLSS-G feature at width x height (the presented / backbuffer resolution). `cmd` must be
/// in the recording state; the caller submits + waits. `backbuffer_format` is the color VkFormat.
#[cfg(have_ngx)]
pub fn create_fg(device: u64, cmd: u64, width: u32, height: u32, backbuffer_format: i32) -> Option<FgFeature> {
    use ffi::*;
    use std::ffi::CString;
    use std::os::raw::c_void;
    let mut params: *mut c_void = std::ptr::null_mut();
    if unsafe { NVSDK_NGX_VULKAN_AllocateParameters(&mut params) } != NGX_SUCCESS || params.is_null() {
        log("NGX: AllocateParameters (DLSS-G) failed");
        return None;
    }
    let set_ui = |n: &str, v: u32| { let c = CString::new(n).unwrap(); unsafe { NVSDK_NGX_Parameter_SetUI(params, c.as_ptr(), v) }; };
    set_ui("CreationNodeMask", 1);
    set_ui("VisibilityNodeMask", 1);
    set_ui("Width", width);
    set_ui("Height", height);
    set_ui("DLSSG.BackbufferFormat", backbuffer_format as u32);
    let mut handle: *mut c_void = std::ptr::null_mut();
    let res = unsafe {
        NVSDK_NGX_VULKAN_CreateFeature1(device as VkHandle, cmd as VkHandle, NGX_FEATURE_FRAME_GENERATION, params, &mut handle)
    };
    if res == NGX_SUCCESS && !handle.is_null() {
        log(&format!("NGX: DLSS-G feature created OK ({width}x{height})"));
        Some(FgFeature { handle, params })
    } else {
        log(&format!("NGX: CreateFeature DLSS-G FAILED (result {res:#x})"));
        unsafe { NVSDK_NGX_VULKAN_DestroyParameters(params) };
        None
    }
}

/// Run DLSS Frame Generation for one frame: interpolate between the previously-fed frame and the
/// current `color`, writing the generated frame to `out_interp` (and `color` through to `out_real`).
/// `clip_to_prev_clip` reprojects current clip space to the previous frame's (prevViewProj * invVP).
#[cfg(have_ngx)]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_fg(
    cmd: u64,
    fg: &FgFeature,
    color: RrImage,      // current frame (final tonemapped color = backbuffer)
    mvec: RrImage,
    depth: RrImage,
    out_interp: RrImage, // generated interpolated frame (output)
    out_real: RrImage,   // real frame passthrough (output)
    width: u32,
    height: u32,
    multi_frame_count: u32,
    clip_to_prev_clip: [f32; 16], // row-major (NGX convention), current clip -> previous clip
    prev_clip_to_clip: [f32; 16], // row-major, previous clip -> current clip (the inverse)
    cam_near: f32,
    cam_far: f32,
    reset: bool,
) -> bool {
    use ffi::*;
    use std::ffi::CString;
    use std::os::raw::c_void;
    let range = ash::vk::ImageSubresourceRange {
        aspect_mask: ash::vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1,
    };
    let mk = |img: RrImage, rw: bool| ResourceVk {
        image_view: img.view, image: img.image, subresource: range, format: img.format, width, height, res_type: 0, read_write: rw as u8,
    };
    // Must outlive EvaluateFeature_C (referenced by pointer).
    let mut r_color = mk(color, false);
    let mut r_mvec = mk(mvec, false);
    let mut r_depth = mk(depth, false);
    let mut r_interp = mk(out_interp, true);
    let mut r_real = mk(out_real, true);
    let mut m_clip_prev = clip_to_prev_clip;
    let mut m_prev_clip = prev_clip_to_clip;

    let p = fg.params;
    let set_ptr = |n: &str, r: *mut c_void| { let c = CString::new(n).unwrap(); unsafe { NVSDK_NGX_Parameter_SetVoidPointer(p, c.as_ptr(), r) }; };
    let set_f = |n: &str, v: f32| { let c = CString::new(n).unwrap(); unsafe { NVSDK_NGX_Parameter_SetF(p, c.as_ptr(), v) }; };
    let set_ui = |n: &str, v: u32| { let c = CString::new(n).unwrap(); unsafe { NVSDK_NGX_Parameter_SetUI(p, c.as_ptr(), v) }; };

    set_ptr("DLSSG.Backbuffer", &mut r_color as *mut _ as *mut c_void);
    set_ptr("DLSSG.MVecs", &mut r_mvec as *mut _ as *mut c_void);
    set_ptr("DLSSG.Depth", &mut r_depth as *mut _ as *mut c_void);
    set_ptr("DLSSG.OutputInterpolated", &mut r_interp as *mut _ as *mut c_void);
    set_ptr("DLSSG.OutputReal", &mut r_real as *mut _ as *mut c_void);
    set_ptr("DLSSG.ClipToPrevClip", m_clip_prev.as_mut_ptr() as *mut c_void);
    set_ptr("DLSSG.PrevClipToClip", m_prev_clip.as_mut_ptr() as *mut c_void);
    set_ui("DLSSG.MultiFrameCount", multi_frame_count.max(1));
    set_ui("DLSSG.MultiFrameIndex", 1);
    set_f("DLSSG.CameraNear", cam_near);
    set_f("DLSSG.CameraFar", cam_far);
    set_f("DLSSG.MvecScaleX", 1.0);
    set_f("DLSSG.MvecScaleY", 1.0);
    set_ui("DLSSG.DepthInverted", 0);
    set_ui("DLSSG.ColorBuffersHDR", 0); // the backbuffer is tonemapped display (LDR), not full HDR
    set_ui("DLSSG.CameraMotionIncluded", 1); // our motion vectors include camera motion
    set_ui("DLSSG.Reset", reset as u32);

    let res = unsafe { NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd as VkHandle, fg.handle, p, std::ptr::null()) };
    if res != NGX_SUCCESS {
        log(&format!("NGX: EvaluateFeature DLSS-G failed (result {res:#x})"));
        return false;
    }
    true
}

#[cfg(have_ngx)]
pub fn release_fg(fg: FgFeature) {
    unsafe {
        ffi::NVSDK_NGX_VULKAN_ReleaseFeature(fg.handle);
        ffi::NVSDK_NGX_VULKAN_DestroyParameters(fg.params);
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
pub fn create_rr(_device: u64, _cmd: u64, _rw: u32, _rh: u32, _dw: u32, _dh: u32, _q: i32) -> Option<RrFeature> { None }
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
    _rw: u32, _rh: u32, _dw: u32, _dh: u32, _jx: f32, _jy: f32, _reset: bool, _delta: f32,
) -> bool { false }

#[cfg(not(have_ngx))]
pub struct FgFeature;
#[cfg(not(have_ngx))]
pub fn create_fg(_d: u64, _c: u64, _w: u32, _h: u32, _fmt: i32) -> Option<FgFeature> { None }
#[cfg(not(have_ngx))]
pub fn release_fg(_fg: FgFeature) {}
#[cfg(not(have_ngx))]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_fg(
    _cmd: u64, _fg: &FgFeature, _color: RrImage, _mvec: RrImage, _depth: RrImage,
    _oi: RrImage, _or: RrImage, _w: u32, _h: u32, _mfc: u32, _m: [f32; 16], _m2: [f32; 16], _n: f32, _f: f32, _r: bool,
) -> bool { false }
