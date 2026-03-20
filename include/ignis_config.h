#pragma once

#include <cstdint>

namespace acpt {

// Renderer configuration — standalone version (standalone Blender renderer)
struct PathTracerConfig {
    // Exposure & tonemap
    float ptExposure         = 0.55f;
    int   ptTonemapMode      = 0;       // 0=ACES, 1=Reinhard, 2=Filmic, etc.
    float ptSaturation       = 1.0f;
    float ptContrast         = 1.0f;

    // Auto-exposure
    bool  ptAutoExposure     = false;
    float ptAutoExposureKey  = 0.18f;
    float ptAutoExposureSpeed= 2.0f;
    float ptAutoExposureMin  = 0.01f;
    float ptAutoExposureMax  = 10.0f;

    // GI / lighting tuning
    float ptGIIntensity      = 1.0f;
    float ptSkyReflIntensity = 0.5f;
    float ptAmbientMax       = 0.5f;
    float ptSunMinIntensity  = 0.1f;
    float ptSkyBounceIntensity = 0.25f;

    // Sun
    float sunAzimuth         = 30.0f;   // degrees
    float sunElevation       = 58.0f;   // degrees
    float sunIntensity       = 1.29f;
    float sunColorR          = 1.0f;
    float sunColorG          = 0.96f;
    float sunColorB          = 0.92f;
    bool  autoSkyColors      = false;

    // Ambient / sky
    float ambientColorR      = 0.5f;
    float ambientColorG      = 0.6f;
    float ambientColorB      = 0.8f;
    float ambientIntensity   = 0.5f;
    float skyColorR          = 0.5f;
    float skyColorG          = 0.6f;
    float skyColorB          = 0.8f;

    // Fog / visibility
    float cloudVisibility    = 50.0f;   // km (Koschmieder)

    // Path tracing quality
    int   maxBounces         = 2;       // 1-8 bounces (default 2)
    int   samplesPerPixel    = 1;       // 1-4 SPP (higher = cleaner but slower)

    // Debug
    int   debugView          = 0;       // 0 = off, >0 = debug view mode

    // NRD
    bool  nrdEnabled         = true;
    float nrdMaxAccumFrames  = 32.0f;    // higher = smoother when static (was 24)
    float nrdFastAccumFrames = 6.0f;    // fast history for disoccluded regions (was 4)
    float nrdLobeAngleFraction  = 0.85f;  // [0..1] higher = more permissive lobe matching (smoother reflections)
    float nrdRoughnessFraction  = 0.5f;   // [0..1] higher = more permissive roughness matching
    float nrdMinHitDistanceWeight = 0.05f;
    float nrdDisocclusionThreshold = 0.01f;
    float nrdDiffusePrepassBlur  = 0.0f;   // [0..75] pre-pass blur radius for diffuse (0=disabled, recommended)
    float nrdSpecularPrepassBlur = 8.0f;   // [0..75] pre-pass blur radius for specular
    int   nrdAtrousIterations    = 5;      // [2..8] A-trous wavelet iterations (2^(N-1) pixel radius)
    bool  nrdAntiFirefly         = true;   // clamp outlier firefly pixels
    int   nrdHistoryFixFrameNum  = 1;      // [0..6] spatial reconstruction frames for disoccluded pixels (lower=faster recovery)
    float nrdDepthThreshold      = 0.004f; // [0.001..0.1] depth-based edge stopping
    float nrdDiffusePhiLuminance = 2.0f;   // [0.5..8.0] luminance edge stopping for diffuse A-trous
    float nrdSpecularPhiLuminance= 1.0f;   // [0.5..8.0] luminance edge stopping for specular A-trous (lower = sharper reflections)

    // DLSS
    bool  dlssEnabled        = false;
    int   dlssQualityMode    = 0;       // maps to DLSSQualityMode enum
    bool  dlssRREnabled      = false;   // Use Ray Reconstruction when available (replaces NRD)

    // Rain (kept for shader uniforms, but not driven by weather engine)
    bool  rainEnabled        = false;
    float rainIntensity      = 0.0f;
    float rainWetness        = 0.0f;
    float rainWaterLevel     = 0.0f;

    // Cloud params (kept for shader uniforms, but not driven by weather engine)
    bool  cloudsEnabled      = false;
    float cloudCoverage      = 0.15f;
    float cloudStartHeight   = 2000.0f;
    float cloudEndHeight     = 4500.0f;
    float cloudAbsorption    = 0.012f;
    float cloudDensity       = 0.8f;
    float cloudWindSpeed     = 2.0f;
    float cloudWindDir       = 0.0f;
    float cloudBaseScale     = 1.0f;
    float cloudDetailScale   = 1.0f;
    float cloudPhaseForward  = 0.85f;
    float cloudPhaseBackward = -0.15f;
    float cloudPhaseWeight   = 0.5f;
    float cloudSunMultiplier = 5.0f;
    float cloudAmbientMultiplier = 0.6f;

    // Weather (kept for compatibility but not auto-driven)
    int   weatherType        = 0;
    bool  autoWeather        = false;

    // Shader mode: 0=AC (raygen.rgen), 1=Blender GI (raygen_blender.rgen)
    int   shaderMode         = 0;

    // Wavefront path tracing (experimental — compute-based multi-kernel pipeline)
    bool  useWavefront       = false;

    // Backface culling
    bool  backfaceCulling    = false;

    // HDRI environment map
    int   hdriTexIndex       = -1;     // texture index (-1 = no HDRI, use procedural sky)
    float hdriStrength       = 1.0f;   // environment intensity multiplier
};

} // namespace acpt
