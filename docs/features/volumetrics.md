# Volumetric Rendering

Ignis RT supports Volume Scatter and Principled Volume nodes with fixed-step ray marching.

## Algorithm

```mermaid
flowchart TD
    HIT[Ray hits volume front face] --> EXIT[Find exit distance]
    EXIT --> MARCH[Ray march 48 steps]
    MARCH --> DENSITY[Evaluate density at step]
    DENSITY --> HETERO{Heterogeneous?}
    HETERO -->|Yes| NOISE[fbm3D noise at position]
    HETERO -->|No| CONST[Constant density]
    NOISE --> SIGMA[sigma_t = color × density]
    CONST --> SIGMA
    SIGMA --> BEER[Beer-Lambert: T *= exp(-sigma_t × stepSize)]
    BEER --> SCATTER[In-scatter from lights]
    SCATTER --> EARLY{T < 0.001?}
    EARLY -->|Yes| DONE[Apply results]
    EARLY -->|No| NEXT[Next step]
    NEXT --> DENSITY

    style MARCH fill:#6a1b9a,color:white
    style NOISE fill:#e65100,color:white
```

## Cycles-Accurate Coefficients

Following Cycles' Principled Volume formula:

```
sigma_s = Color × Density     (scattering coefficient)
sigma_a = (1 - Color) × Density × AbsorptionColor  (absorption)
sigma_t = sigma_s + sigma_a   (total extinction)
```

- **Color high** → more scattering (bright, cloud-like)
- **Color low** → more absorption (dark, smoke-like)
- **Density** is constant; Color varies spatially via noise

## In-Scattering

At each march step, light is gathered from ALL sources:

| Source | Method |
|--------|--------|
| Sun | Phase-weighted directional (HG) |
| Point/Spot/Area | 1 random sample, scaled by count |
| Ambient | Isotropic (1/4π) |
| Sky | `evaluateSky(rayDir)` |

### Henyey-Greenstein Phase Function

Controls scattering directionality:

- `g > 0` → forward scattering (fog/haze)
- `g = 0` → isotropic (smoke)
- `g < 0` → backward scattering

## Heterogeneous Volumes

Noise-based density variation extracted from the Blender node tree:

```mermaid
flowchart LR
    TC[Texture Coord:Object] --> MAP[Mapping 3D]
    MAP --> NOISE[Noise Texture]
    NOISE --> BC[Bright/Contrast]
    BC --> PV[Principled Volume:Color]
```

Parameters extracted at export:
- Noise scale, detail, roughness, lacunarity
- Mapping offset (3D)
- Brightness, contrast

## Volume-Only Materials

When Material Output has Volume connected but Surface empty:

1. Front face hit → enter volume, ray march
2. Back face hit → pass through (attenuation done on entry)
3. Shadow rays → pass through volume boundaries
