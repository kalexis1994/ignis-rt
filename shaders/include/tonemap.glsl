// ============================================================
// tonemap.glsl — Tonemap Curves
// Shared between raygen_blender.rgen and nrd_composite.comp
// Sources: github.com/dmnsgn/glsl-tone-map, Khronos PBR Neutral, Three.js AgX
// ============================================================

#ifndef TONEMAP_GLSL
#define TONEMAP_GLSL

// Rec.2020 <-> sRGB matrices (Three.js / ITU-R BT.2407)
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
    0.6274, 0.0691, 0.0164,
    0.3293, 0.9195, 0.0880,
    0.0433, 0.0113, 0.8956
);
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
    1.6605, -0.1246, -0.0182,
   -0.5876,  1.1329, -0.1006,
   -0.0728, -0.0083,  1.1187
);
const mat3 AgXInsetMatrix = mat3(
    0.856627153315983, 0.137318972929847, 0.11189821299995,
    0.0951212405381588, 0.761241990602591, 0.0767994186031903,
    0.0482516061458583, 0.101439036467562, 0.811302368396859
);
const mat3 AgXOutsetMatrix = mat3(
    1.1271005818144368, -0.1413297634984383, -0.14132976349843826,
   -0.11060664309660323, 1.157823702216272, -0.11060664309660294,
   -0.016493938717834573, -0.016493938717834257, 1.2519364065950405
);

// 0: AgX Punchy (Troy Sobotka / Blender 4 — natural colors + contrast)
vec3 agxTonemap(vec3 color) {
    color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
    color = AgXInsetMatrix * color;
    color = max(color, vec3(1e-10));
    color = clamp(log2(color), -12.47393, 4.026069);
    color = (color - (-12.47393)) / (4.026069 - (-12.47393));
    vec3 x2 = color * color;
    vec3 x4 = x2 * x2;
    color = 15.5 * x4 * x2 - 40.14 * x4 * color + 31.96 * x4
          - 6.868 * x2 * color + 0.4298 * x2 + 0.1191 * color - 0.00232;
    color = pow(max(vec3(0.0), color), vec3(1.35));
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = luma + 1.4 * (color - luma);
    color = AgXOutsetMatrix * color;
    color = pow(max(vec3(0.0), color), vec3(2.2));
    color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
    return clamp(color, 0.0, 1.0);
}

// 1: ACES filmic (Narkowicz 2015 fit)
vec3 acesTonemap(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// 2: Reinhard (luminance-based)
vec3 reinhardTonemap(vec3 v) {
    float l = dot(v, vec3(0.2126, 0.7152, 0.0722));
    float lNew = l / (1.0 + l);
    return v * (lNew / max(l, 1e-6));
}

// 3: Hable / Uncharted 2
vec3 hablePartial(vec3 x) {
    const float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
vec3 hableTonemap(vec3 color) {
    vec3 curr = hablePartial(color);
    vec3 whiteScale = vec3(1.0) / hablePartial(vec3(11.2));
    return curr * whiteScale;
}

// 4: Khronos PBR Neutral
vec3 neutralTonemap(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;
    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;
    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;
    float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;
    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3(newPeak), g);
}

// Dispatch by mode index
vec3 applyTonemap(vec3 hdr, uint mode) {
    if (mode == 1u) return acesTonemap(hdr);
    if (mode == 2u) return reinhardTonemap(hdr);
    if (mode == 3u) return hableTonemap(hdr);
    if (mode == 4u) return neutralTonemap(hdr);
    return agxTonemap(hdr);
}

#endif // TONEMAP_GLSL
