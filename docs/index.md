<div style="text-align: center; margin-bottom: 1em;">
  <img src="assets/ignis_512.png" alt="Ignis RT" style="width: 128px; height: 128px;">
</div>

# Ignis RT

**Real-time Vulkan ray tracing path tracer for Blender**

Ignis RT is a viewport render engine for Blender that uses hardware-accelerated ray tracing (RTX) to produce physically-based renders in real-time. It aims to match Cycles' output quality while maintaining interactive frame rates.

## Key Features

- **Full path tracing** — multi-bounce GI with importance sampling
- **DLSS 4 integration** — Super Resolution, Ray Reconstruction, DLAA
- **Cycles-compatible materials** — Node VM evaluates Blender shader nodes per-pixel
- **Volumetric rendering** — ray marching with Beer-Lambert + Henyey-Greenstein
- **Blender Color Management** — runtime OCIO LUT bake for any view transform
- **Procedural textures** — Cycles-exact Perlin noise (Jenkins Lookup3 hash)

## Architecture at a Glance

```mermaid
flowchart TB
    RT((Ignis RT))

    RT --- ADDON & VULKAN & SHADERS

    subgraph ADDON[Blender Addon]
        direction LR
        SE[Scene Export<br/>Meshes · Materials · Lights]
        SYNC[Engine Sync<br/>Camera · Transforms · OCIO]
    end

    subgraph VULKAN[Vulkan Renderer]
        direction LR
        RAYS[Ray Tracing<br/>Opaque BVH · Hybrid Alpha]
        PT[Path Tracing<br/>NEE · BRDF · Russian Roulette]
        DL[DLSS 4<br/>Ray Reconstruction · SR · DLAA]
    end

    subgraph SHADERS[Shader System]
        direction LR
        VM[Node VM<br/>86 opcodes · 32 regs · 64 instrs]
        TEX[Procedural Textures<br/>Perlin · Voronoi · Gradient]
        BRDF[PBR BRDF<br/>GGX · Fresnel · F82-Tint]
        VOL[Volumetrics<br/>Ray March · Beer-Lambert · HG]
    end

    style RT fill:#E06030,color:#fff,stroke:#B34A24
    style ADDON fill:#4A6380,color:#fff,stroke:#4A6380
    style VULKAN fill:#2E2E30,color:#fff,stroke:#636366
    style SHADERS fill:#2E2E30,color:#fff,stroke:#636366
```

## Render Pipeline

```mermaid
flowchart LR
    A[Blender Scene] --> B[Python Export]
    B --> C[BLAS/TLAS Build]
    C --> D[Path Tracing]
    D --> E[DLSS RR Denoise]
    E --> F[DLSS Upscale]
    F --> G[Tonemap LUT]
    G --> H[Display]

    style D fill:#E06030,color:white
    style E fill:#4A6380,color:white
    style F fill:#4A6380,color:white
    style G fill:#00695C,color:white
```

## Quick Start

1. Download the latest release from [GitHub Releases](https://github.com/kalexis1994/ignis-rt/releases)
2. In Blender: **Edit → Preferences → Add-ons → Install from Disk**
3. Select the downloaded `.zip` file
4. Set render engine to **Ignis RT**
5. Switch to **Rendered** viewport shading (Z → Rendered)

!!! info "Requirements"
    - NVIDIA RTX GPU (20/30/40/50 series)
    - Windows 10/11
    - Blender 4.0+
    - Latest NVIDIA drivers (560+)
