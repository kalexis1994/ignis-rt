// ============================================================
// common.glsl — Constants, RNG, TBN computation
// Shared between raygen_blender.rgen, raygen.rgen, hybrid.rgen
// ============================================================

#ifndef COMMON_GLSL
#define COMMON_GLSL

const float PI = 3.14159265359;
const float INV_PI = 1.0 / PI;
const int MAX_BOUNCES_LIMIT = 8;
const float MIN_HIT_DIST = 0.001;
const float MAX_RADIANCE = 30.0;  // clamp fireflies but allow HDR headroom for bright lights

// ============================================================
// ============================================================
// Sampling — Hybrid quasi-random + PCG fallback
//
// Uses R2 low-discrepancy sequence with Cranley-Patterson rotation
// for deterministic, noise-free sampling. Each pixel gets a unique
// offset (from blue noise or hash), and each frame advances the
// sequence index. This covers the sample space uniformly without
// the random noise of pure Monte Carlo.
//
// PCG is kept as fallback for operations that need true randomness
// (Russian Roulette, stochastic decisions).
// ============================================================

uint rngState;

uint pcg(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// R2 quasi-random sequence (generalized golden ratio)
// Produces low-discrepancy points in [0,1)² — much better coverage than random
const float R2_G = 1.32471795724;  // plastic constant
const float R2_A1 = 1.0 / R2_G;
const float R2_A2 = 1.0 / (R2_G * R2_G);

// Per-pixel state for blue-noise-scrambled sampling
uint _sampleDimension;
uvec2 _pixel;

void initDeterministicSampling(uvec2 pixel, uint frameIndex) {
    _sampleDimension = 0u;
    _pixel = pixel;
    // PCG with blue noise scramble: random numbers with better spatial distribution.
    // Blue noise decorrelates neighboring pixels → smoother noise for the denoiser.
    rngState = pixel.x * 1973u + pixel.y * 9277u + frameIndex * 26699u;
}

// Random sample in [0,1) — PCG base + blue noise spatial scramble
// Produces random numbers that are spatially well-distributed across pixels.
// DLSS RR works better with this than pure random (smoother input noise).
float rand01() {
    uint dim = _sampleDimension++;
    float pcgVal = float(pcg(rngState)) / 4294967296.0;
    // Blue noise scramble: shift PCG output by a per-pixel blue noise value
    // This decorrelates neighboring pixels without creating structured patterns
    ivec2 bnCoord = ivec2((_pixel.x + dim * 7u) & 63, (_pixel.y + dim * 11u) & 63);
    float bn = texelFetch(blueNoiseTexture, bnCoord, 0).r;
    return fract(pcgVal + bn);
}

// Random 2D sample with blue noise spatial scramble
vec2 rand2() {
    uint dim = _sampleDimension++;
    float pcg1 = float(pcg(rngState)) / 4294967296.0;
    float pcg2 = float(pcg(rngState)) / 4294967296.0;
    ivec2 bn1 = ivec2((_pixel.x + dim * 7u) & 63, (_pixel.y + dim * 11u) & 63);
    ivec2 bn2 = ivec2((_pixel.y + dim * 5u) & 63, (_pixel.x + dim * 3u) & 63);
    return vec2(
        fract(pcg1 + texelFetch(blueNoiseTexture, bn1, 0).r),
        fract(pcg2 + texelFetch(blueNoiseTexture, bn2, 0).r)
    );
}

// Pure PCG random (no blue noise) for stochastic decisions
float randPCG() {
    return float(pcg(rngState)) / 4294967296.0;
}

vec2 r2Sample(uint idx) {
    return fract(vec2(R2_A1 * float(idx), R2_A2 * float(idx)));
}

vec2 stratifiedSample(uint sampleIdx, vec2 randomOffset) {
    return fract(r2Sample(sampleIdx) + randomOffset);
}

// Clamp contribution to prevent fireflies
vec3 clampRadiance(vec3 r) {
    return min(r, vec3(MAX_RADIANCE));
}

// Shadow terminator fix (Appleseed/Cycles):
// Smoothly reduces shading near the terminator line where interpolated
// normals diverge from geometric normals, preventing dark edges.
float shadowTerminatorFactor(float cosIn, float frequencyMult) {
    if (frequencyMult <= 1.0 || cosIn >= 1.0) return 1.0;
    float angle = acos(clamp(cosIn, -1.0, 1.0));
    return max(cos(angle * frequencyMult), 0.0) / max(cosIn, 0.001);
}

// ============================================================
// TBN computation
// ============================================================

void computeTBN(vec3 v0, vec3 v1, vec3 v2, vec2 uv0, vec2 uv1, vec2 uv2,
                vec3 N, out vec3 T, out vec3 B) {
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec2 duv1 = uv1 - uv0;
    vec2 duv2 = uv2 - uv0;

    float det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) > 1e-6) {
        float invDet = 1.0 / det;
        T = normalize((e1 * duv2.y - e2 * duv1.y) * invDet);
        B = normalize((e2 * duv1.x - e1 * duv2.x) * invDet);
    } else {
        T = normalize(e1);
        B = normalize(cross(N, T));
    }
    T = normalize(T - N * dot(N, T));
    B = normalize(cross(N, T));
}

vec3 applyNormalMap(vec3 texNormal, vec3 T, vec3 N, vec3 B, float strength) {
    vec3 n = vec3(texNormal.x * 2.0 - 1.0,
                  1.0 - texNormal.y * 2.0,   // DX green channel convention
                  texNormal.z * 2.0 - 1.0);
    n.xy *= strength;
    n = normalize(n);
    return normalize(T * n.x + B * n.y + N * n.z);
}

#endif // COMMON_GLSL
