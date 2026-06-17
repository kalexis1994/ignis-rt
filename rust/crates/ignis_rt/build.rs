//! Compile our Rust-side GLSL compute shaders to SPIR-V at build time using the Vulkan
//! SDK's glslangValidator, writing .spv into OUT_DIR for include_bytes!. Runs on the build
//! machine only (which has the Vulkan SDK); the deploy target never builds.

use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    println!("cargo:rerun-if-changed=shaders");

    let glslang = find_glslang();

    for entry in std::fs::read_dir(&shader_dir).expect("shaders/ dir") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("comp") {
            let name = path.file_name().unwrap().to_str().unwrap();
            let out_spv = format!("{out_dir}/{name}.spv");
            println!("cargo:rerun-if-changed={}", path.display());
            // target-env vulkan1.2 -> SPIR-V 1.5, required for GL_EXT_ray_query.
            let status = Command::new(&glslang)
                .args(["-V", "--target-env", "vulkan1.2", path.to_str().unwrap(), "-o", &out_spv])
                .status()
                .unwrap_or_else(|e| panic!("failed to run {glslang}: {e}"));
            assert!(status.success(), "glslangValidator failed for {name}");
        }
    }
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
