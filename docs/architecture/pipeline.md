# Render Pipeline

## Path Tracing Loop

Each pixel traces a ray through multiple bounces, accumulating lighting contributions:

```mermaid
flowchart TD
    START[Camera Ray] --> TRACE[Trace Ray - Opaque BVH]
    TRACE --> MISS{Hit?}
    MISS -->|No| SKY[Evaluate Sky/HDRI]
    MISS -->|Yes| VOLUME{Volume Material?}

    VOLUME -->|Yes| MARCH[Ray March 48 Steps]
    MARCH --> BEER[Beer-Lambert Extinction]
    BEER --> INSCATTER[In-Scatter from All Lights]
    INSCATTER --> CONTINUE

    VOLUME -->|No| ALPHA{Alpha Test?}
    ALPHA -->|Pass Through| BACKWARD[Step Back Ray]
    BACKWARD --> NONOPAQUE[Non-Opaque Query]
    NONOPAQUE --> TRACE

    ALPHA -->|Shade| EVAL[Evaluate Material - Node VM]
    EVAL --> NEE[Next Event Estimation]
    NEE --> SUN[Sun Direct Light]
    NEE --> POINT[Point/Spot/Area Lights]
    NEE --> EMISSIVE[Emissive Triangles]

    NEE --> BRDF[BRDF Sample Next Direction]
    BRDF --> RR{Russian Roulette}
    RR -->|Survive| CONTINUE[Next Bounce]
    RR -->|Terminate| OUTPUT

    CONTINUE --> TRACE
    SKY --> OUTPUT[Accumulate Radiance]

    style TRACE fill:#E06030,color:white
    style EVAL fill:#00695C,color:white
    style NEE fill:#4A6380,color:white
    style MARCH fill:#2E2E30,color:white
```

## Material Evaluation

The Node VM evaluates Blender's shader node tree per-pixel:

```mermaid
flowchart LR
    subgraph VM["Node VM (32 registers, 64 instructions)"]
        TEX[Sample Texture] --> MIX[Mix/Blend]
        NOISE[Procedural Noise] --> MIX
        MIX --> HSV[Hue/Sat/Value]
        HSV --> RAMP[ColorRamp]
        RAMP --> OUT_C[OUTPUT_COLOR]
        TEX2[Roughness Tex] --> OUT_R[OUTPUT_ROUGH]
        TEX3[Alpha Tex] --> OUT_A[OUTPUT_ALPHA]
    end

    OUT_C --> PBR[PBR BRDF]
    OUT_R --> PBR
    OUT_A --> ALPHA[Alpha Test]
    PBR --> SHADE[Final Shading]
```

## Denoising & Display

```mermaid
flowchart LR
    RAW[Raw Path Traced] --> SPLIT{DLSS RR?}
    SPLIT -->|Yes| RR[Ray Reconstruction]
    SPLIT -->|No| NRD[NRD RELAX Denoise]
    RR --> SR[DLSS Super Resolution]
    NRD --> SR
    SR --> TONE[Tonemap - OCIO LUT]
    TONE --> DITHER[8-bit Dither]
    DITHER --> DISPLAY[Blender Viewport]

    style RR fill:#00695C,color:white
    style SR fill:#4A6380,color:white
    style TONE fill:#E06030,color:white
```

## Supported Blender Nodes

| Category | Nodes | Status |
|----------|-------|--------|
| **BSDF** | Principled, Diffuse, Glossy, Transparent, Glass | Full |
| **Shader** | Mix Shader, Add Shader | Full (per-pixel blend) |
| **Texture** | Image, Noise, Voronoi, Gradient, Wave, Checker, Magic, Brick, White Noise | Full |
| **Color** | Mix (13 blend modes), ColorRamp, Hue/Sat/Value, Bright/Contrast, Invert, Gamma | Full |
| **Color** | RGB Curves | Passthrough (not evaluated) |
| **Math** | Math (21 ops), Vector Math (18 ops), Map Range, Clamp | Full |
| **Channel** | Separate RGB/XYZ | Full |
| **Channel** | Combine RGB/XYZ | Passthrough (special-case UV only) |
| **Input** | Texture Coordinate (UV, Object, Generated), Object Info (Random), Vertex Color | Full |
| **Input** | Geometry (Position, Normal, Incoming, Backfacing), Layer Weight, Fresnel | Full |
| **Input** | Light Path | Simplified (Is Camera Ray = 1, rest = 0) |
| **Vector** | Mapping (2D UV + 3D position), Normal Map | Full |
| **Vector** | Bump | TEX_NOISE height only |
| **Volume** | Volume Scatter, Principled Volume, Volume Absorption | Ray March (48 steps) |
| **Utility** | Reroute, Frame, Value, RGB | Full |
