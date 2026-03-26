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
mindmap
  root((Ignis RT))
    Blender Addon
      Scene Export
        Meshes → BLAS
        Materials → Node VM
        Lights → NEE
        World → Sky/HDRI
      Engine Sync
        Camera
        Transforms → TLAS
        Color Management
    Vulkan Renderer
      Ray Tracing
        Opaque Ray Queries
        Hybrid Alpha Queries
        BVH Traversal
      Path Tracing
        NEE Sun/Lights
        BRDF Sampling
        Russian Roulette
      DLSS 4
        Ray Reconstruction
        Super Resolution
        DLAA
      Post-Processing
        OCIO Tonemap LUT
        Exposure
        Saturation
    Shader System
      Node VM
        86 Opcodes
        32 Registers
        64 Instructions max
      Procedural Textures
        Perlin/FBM
        Voronoi
        Gradient
      PBR BRDF
        GGX Specular
        Dielectric Fresnel
        F82-Tint Metals
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

    style D fill:#e65100,color:white
    style E fill:#1565c0,color:white
    style F fill:#1565c0,color:white
    style G fill:#2e7d32,color:white
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
