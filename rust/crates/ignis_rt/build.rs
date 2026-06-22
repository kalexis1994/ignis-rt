//! Compile our Rust-side GLSL compute shaders to SPIR-V at build time using the Vulkan
//! SDK's glslangValidator, writing .spv into OUT_DIR for include_bytes!. Runs on the build
//! machine only (which has the Vulkan SDK); the deploy target never builds.

use std::path::Path;
use std::process::Command;

fn main() {
    link_ngx();

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    println!("cargo:rerun-if-changed=shaders");

    let glslang = find_glslang();

    for entry in std::fs::read_dir(&shader_dir).expect("shaders/ dir") {
        let path = entry.unwrap().path();
        let ext = path.extension().and_then(|e| e.to_str());
        // .comp = compute (ray query). .rgen = ray generation (ray query + Shader Execution
        // Reordering). .vert/.frag = the hybrid-raster primary-visibility pipeline. glslang infers
        // the stage from the extension.
        if matches!(ext, Some("comp") | Some("rgen") | Some("vert") | Some("frag")) {
            let name = path.file_name().unwrap().to_str().unwrap();
            let out_spv = format!("{out_dir}/{name}.spv");
            println!("cargo:rerun-if-changed={}", path.display());
            // target-env vulkan1.2 -> SPIR-V 1.5, required for GL_EXT_ray_query / ray tracing.
            let status = Command::new(&glslang)
                .args(["-V", "--target-env", "vulkan1.2", path.to_str().unwrap(), "-o", &out_spv])
                .status()
                .unwrap_or_else(|e| panic!("failed to run {glslang}: {e}"));
            assert!(status.success(), "glslangValidator failed for {name}");
        }
    }

    // Second variant of trace.rgen WITH Shader Execution Reordering (-DUSE_SER -> trace_ser.spv).
    // The plain trace.rgen.spv (compiled above without the define) carries no SER capability and
    // runs on any RT GPU; the renderer picks trace_ser.spv only when the SER extension is available.
    let trace_rgen = shader_dir.join("trace.rgen");
    if trace_rgen.exists() {
        let out_ser = format!("{out_dir}/trace_ser.spv");
        let status = Command::new(&glslang)
            .args([
                "-V", "--target-env", "vulkan1.2", "-DUSE_SER",
                trace_rgen.to_str().unwrap(), "-o", &out_ser,
            ])
            .status()
            .unwrap_or_else(|e| panic!("failed to run {glslang}: {e}"));
        assert!(status.success(), "glslangValidator failed for trace_ser");
    }
}

/// Link the NVIDIA NGX SDK (nvsdk_ngx_d.lib) for the DLSS Ray Reconstruction FFI. The lib is a
/// static loader that pulls in the nvngx_dlss*.dll runtime at execution time (present on the RTX
/// box). Path from IGNIS_NGX_ROOT (CMake-compatible: <root>/lib) or the cloned DLSS SDK fallback.
fn link_ngx() {
    println!("cargo:rustc-check-cfg=cfg(have_ngx)"); // declare the cfg (Rust 1.80+ lint)
    println!("cargo:rerun-if-env-changed=IGNIS_NGX_ROOT");
    let root = std::env::var("IGNIS_NGX_ROOT").ok();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_os == "windows" {
        // Windows: nvsdk_ngx_d.lib is a static loader that pulls in nvngx_dlss*.dll at runtime.
        let candidates = [
            root.as_deref().map(|r| format!("{r}/lib")),
            Some(r"C:/Users/Administrator/DLSS/lib/Windows_x86_64/x64".to_string()),
        ];
        for cand in candidates.into_iter().flatten() {
            if Path::new(&cand).join("nvsdk_ngx_d.lib").exists() {
                println!("cargo:rustc-link-search=native={cand}");
                println!("cargo:rustc-link-lib=static=nvsdk_ngx_d");
                // System libs the NGX static loader pulls in (registry path lookup + message box).
                println!("cargo:rustc-link-lib=dylib=advapi32");
                println!("cargo:rustc-link-lib=dylib=user32");
                println!("cargo:rustc-cfg=have_ngx");
                return;
            }
        }
    } else {
        // Linux: libnvsdk_ngx.a is the static loader; it dlopen's libnvidia-ngx-dlss*.so at runtime.
        // The DLSS SDK ships it under lib/Linux_x86_64.
        let candidates = [
            root.as_deref().map(|r| format!("{r}/lib/Linux_x86_64")),
            root.as_deref().map(|r| format!("{r}/lib")),
        ];
        for cand in candidates.into_iter().flatten() {
            if Path::new(&cand).join("libnvsdk_ngx.a").exists() {
                println!("cargo:rustc-link-search=native={cand}");
                println!("cargo:rustc-link-lib=static=nvsdk_ngx");
                // The NGX static loader pulls in libdl (dlopen/dladdr), pthread, and the C++ runtime.
                println!("cargo:rustc-link-lib=dylib=dl");
                println!("cargo:rustc-link-lib=dylib=pthread");
                println!("cargo:rustc-link-lib=dylib=stdc++");
                println!("cargo:rustc-cfg=have_ngx");
                return;
            }
        }
    }
    println!("cargo:warning=NGX SDK not found — DLSS RR disabled in this build");
}

fn find_glslang() -> String {
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let cand = format!("{sdk}/Bin/glslangValidator.exe");
        if Path::new(&cand).exists() {
            return cand;
        }
    }
    "glslangValidator".to_string() // fall back to PATH
}
