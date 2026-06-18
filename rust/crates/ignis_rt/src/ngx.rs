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
    use std::os::raw::c_void;

    // NVSDK_NGX_Result: Success = 0x1, failures are 0xBAD0000X. Treated as i32.
    pub const NGX_SUCCESS: i32 = 0x1;
    // NVSDK_NGX_Version_API (nvsdk_ngx_defs.h: 0x0000015 = 1.5.0).
    pub const NGX_VERSION_API: i32 = 0x0000015;

    pub type VkHandle = *const c_void; // dispatchable Vulkan handle (instance/physdev/device)

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
    }
}

/// Initialize NGX for the given Vulkan handles (raw `as_raw()` pointers). Logs the outcome.
/// Returns true if NGX initialized successfully (RTX + driver + runtime DLLs present).
#[cfg(have_ngx)]
pub fn init(instance: u64, phys_device: u64, device: u64, app_data_path: &str) -> bool {
    use ffi::*;
    // App data path as a NUL-terminated UTF-16 string (NGX writes logs/feature data here).
    let mut path16: Vec<u16> = app_data_path.encode_utf16().collect();
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
            std::ptr::null(), // gipa — NGX uses the loaded Vulkan loader
            std::ptr::null(), // gdpa
            std::ptr::null(), // feature_info
            NGX_VERSION_API,
        )
    };
    if res == NGX_SUCCESS {
        log("NGX: NVSDK_NGX_VULKAN_Init OK (DLSS runtime available)");
        true
    } else {
        log(&format!("NGX: init FAILED (result {res:#x}) — DLSS unavailable on this device"));
        false
    }
}

#[cfg(have_ngx)]
pub fn shutdown(device: u64) {
    unsafe { ffi::NVSDK_NGX_VULKAN_Shutdown1(device as ffi::VkHandle) };
}

// ── Stubs when the NGX SDK wasn't available at build time ──
#[cfg(not(have_ngx))]
pub fn init(_instance: u64, _phys_device: u64, _device: u64, _app_data_path: &str) -> bool {
    log("NGX: built without the NGX SDK — DLSS disabled");
    false
}

#[cfg(not(have_ngx))]
pub fn shutdown(_device: u64) {}
