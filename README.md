<p align="center">
  <img src="icons/ignis_logo.png" width="128" alt="Ignis RT">
</p>

# Ignis RT

Real-time Vulkan ray tracing renderer designed as a Blender viewport render engine. Built on top of hardware-accelerated ray tracing with NVIDIA NRD temporal denoising and DLSS upscaling.

<p align="center">
  <img src="icons/ignis_rt_preview.png" alt="Ignis RT viewport render">
</p>

![Vulkan](https://img.shields.io/badge/Vulkan-1.2-red) ![Platform](https://img.shields.io/badge/Platform-Windows-blue) ![Blender](https://img.shields.io/badge/Blender-4.0%2B-orange)

## Features

### Rendering
- **Vulkan RT pipeline** with ray query and hardware acceleration
- **Path tracing** with configurable bounce count (1-8)
- **Point/spot light support** via Next Event Estimation (NEE) direct sampling with soft shadows
- **Emissive materials** with indirect lighting contribution (GI from emissive surfaces)
- **Shadow demodulation** (CP2077-style: unshadowed radiance + separate SIGMA shadow denoising)
- **Procedural sky** with Preetham/Rayleigh+Mie atmospheric scattering
- **PBR materials** (Cook-Torrance GGX microfacet BRDF)

### Denoising & Upscaling
- **Wavefront path tracing** (optional compute-based multi-kernel pipeline)
- **NVIDIA DLSS Ray Reconstruction** (replaces NRD on all RTX GPUs)
- **NVIDIA NRD** (ReLAX diffuse+specular denoiser + SIGMA shadow denoiser, fallback for RTX 20/30)
- **NVIDIA DLSS** upscaling (Ultra Performance to Ultra Quality)
- **Auto-exposure** with GPU histogram and EMA smoothing
- **Triangular dithering** to eliminate 8-bit banding

### Blender Integration
- **Viewport render engine** (set render engine to "Ignis RT", then Viewport Shading > Rendered)
- **Staged loading** with animated loading screen (no UI freeze on complex scenes)
- **Per-frame sync** for object transforms and lights (move/rotate without full reload)
- **Material support**: Principled BSDF, Diffuse, Glossy, Glass, Emission, Mix Shader
- **Blackbody node** support for physically-based light color temperatures
- **Linked .blend** material support (libraries, collections, appended assets)
- **Crash-safe logging** with per-mesh diagnostics and fsync

### Shader Architecture
- Modular GLSL includes (`common.glsl`, `sampling.glsl`, `pbr_brdf.glsl`, `nrd_encode.glsl`, `tonemap.glsl`)
- Multiple tonemapping curves (AgX, ACES, Reinhard, Hable, Khronos Neutral)
- GGX VNDF sampling (Heitz 2018)
- ReSTIR GI reservoir infrastructure (prepared, not yet active)

## Requirements

- **GPU**: NVIDIA RTX series (ray tracing hardware required)
- **OS**: Windows 10/11
- **Vulkan SDK**: 1.2+
- **Blender**: 4.0+ (GPU backend must be set to **OpenGL**: Edit > Preferences > System > GPU Backend)
- **Build tools**: CMake 3.20+, Visual Studio 2022

### Optional SDKs
- **NVIDIA NRD SDK** for denoising (`-DIGNIS_USE_NRD=ON -DIGNIS_NRD_ROOT=<path>`)
- **NVIDIA NGX SDK** for DLSS (`-DIGNIS_USE_DLSS=ON -DIGNIS_NGX_ROOT=<path>`)

## Building

```bash
# Configure
cmake -S . -B build -DIGNIS_USE_NRD=ON -DIGNIS_USE_DLSS=ON \
  -DIGNIS_NRD_ROOT="path/to/NRD" \
  -DIGNIS_NGX_ROOT="path/to/NGX"

# Build
cmake --build build --config Release
```

## Deploying to Blender

```powershell
# Full build + deploy
.\deploy_blender.ps1

# Skip build, just copy files
.\deploy_blender.ps1 -NoBuild

# Dev mode (symlink — Python changes are instant)
.\deploy_blender.ps1 -Symlink
```

Then in Blender:
1. Edit > Preferences > Add-ons > search "Ignis" > Enable
2. Set render engine to **Ignis RT**
3. Viewport Shading > Rendered (or press Z > Rendered)

## Project Structure

```
ignis-rt/
  include/           # Public headers (C API, config, NRD/DLSS integration)
  src/
    vk/              # Vulkan core (context, renderer, RT pipeline, interop)
    *.cpp             # API implementation, NRD integration, sky model
  shaders/
    include/          # Shared GLSL modules (BRDF, sampling, tonemapping)
    raygen_blender.rgen  # Main Blender path tracer
    nrd_composite.comp   # NRD output compositing + tonemapping
    tonemap.comp         # Post-DLSS HDR to LDR conversion
  blender/ignis_rt/   # Blender addon (Python)
  tests/              # Test executable
```

## Known Limitations & Roadmap

### Current Limitations
- **Specular reflections** can appear noisy during camera motion (converges when static)
- **SHARC radiance cache** disabled pending stability investigation on complex scenes
- **ReSTIR GI** infrastructure ready but disabled (buffer access needs validation)
- **Sun-only SIGMA shadows** (point light shadows use inline path tracing, not SIGMA)
- **No area lights** (point lights only, soft shadows via stochastic sampling)
- **No texture filtering** beyond hardware bilinear

### Future Plans
- **Wavefront ray sorting** (sort rays by direction for better BVH cache coherence)
- **Ray Reconstruction polish** (jitter tuning, specular MVs, hit distance inputs)
- **ReSTIR DI** for many-light scenarios
- **Area light** support
- **Texture LOD / mipmap** streaming
- **Final render** (F12) support
- **Multi-GPU** support

## Contributing

All contributions are welcome! Whether it's bug fixes, new features, performance improvements, or documentation, feel free to open an issue or submit a pull request.

Areas where contributions would be especially valuable:
- Specular reflection quality and temporal stability
- Additional Blender material node support
- Performance optimization for complex scenes
- Ray Reconstruction refinement (specular MVs, hit distance, jitter stability)

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
