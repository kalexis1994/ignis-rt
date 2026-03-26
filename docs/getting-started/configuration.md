# Configuration

## Render Properties Panels

### Sampling

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| Max Bounces | 2 | 2-8 | Path tracing bounce depth. 2 = one GI bounce. Higher = better indirect lighting, slower. |
| Samples/Pixel | 1 | 1-10 | SPP per frame. 1-3 = realtime, 4-6 = quality, 8-10 = offline. |
| Backface Culling | Off | — | Skip back-facing triangles. Faster but breaks single-sided geometry. |

### DLSS

| Setting | Default | Description |
|---------|---------|-------------|
| DLSS Quality | Quality (1.5x) | Upscaling preset. Higher = sharper but slower. DLAA = native resolution. |
| Ray Reconstruction | On | AI denoiser replacing NRD. Requires RTX GPU + driver 535+. |

### Color Management

Ignis RT reads Blender's standard Color Management settings (in Render Properties):

| Setting | Effect in Ignis |
|---------|----------------|
| **View Transform** | Selects tonemap LUT (AgX, Filmic, Standard, ACES, etc.) — baked at startup via OCIO |
| **Exposure** | Applied as 2^EV multiplier before tonemapping |
| **Look** | Mapped to contrast parameter (None, Medium Contrast, High Contrast, etc.) |
| **Display Device** | sRGB, Display P3, Rec.1886, Rec.2020 — affects LUT bake |

!!! info "All View Transforms Supported"
    Ignis RT bakes the active view transform into a 3D LUT at startup using Blender's PyOpenColorIO. Any view transform available in your Blender installation will work correctly.

### Color

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| Saturation | 1.0 | 0.0-3.0 | Post-tonemap color saturation adjustment |

## World Settings

Ignis RT reads lighting from Blender's World node tree:

### Sky Texture (Nishita/Hosek)
Sun direction is extracted automatically from `sun_elevation` and `sun_rotation` using Cycles' exact coordinate conversion formula. Background Strength scales sun intensity.

### HDRI Environment
Environment textures are loaded as equirectangular HDR maps. Sun position is extracted from the brightest pixels for NEE direct lighting.

### SUN Light
If a SUN light object exists in the scene, it takes priority over World settings for sun direction and intensity.

## Performance Tips

| Tip | Impact |
|-----|--------|
| Use DLSS Quality (1.5x) | ~2x FPS vs native |
| Keep SPP at 1-3 | Linear FPS scaling |
| Max Bounces = 2 | Sufficient for most scenes |
| Avoid > 32 point lights | NEE cost scales with light count |
| Use DLAA for screenshots | Best quality, ~0.7x FPS vs Quality |
