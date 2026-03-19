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
// RNG — PCG hash + R2 quasi-random sequence for low-discrepancy sampling
// ============================================================

uint rngState;

uint pcg(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand01() {
    return float(pcg(rngState)) / 4294967296.0;
}

vec2 rand2() {
    return vec2(rand01(), rand01());
}

// R2 quasi-random sequence (Robertstheta) — better stratification than PCG
// for importance sampling. Based on generalized golden ratio.
// Use: r2Sample(sampleIndex) returns vec2 in [0,1)²
const float R2_G = 1.32471795724;  // plastic constant
const float R2_A1 = 1.0 / R2_G;
const float R2_A2 = 1.0 / (R2_G * R2_G);

vec2 r2Sample(uint idx) {
    return fract(vec2(R2_A1 * float(idx), R2_A2 * float(idx)));
}

// Cranley-Patterson rotation: combine R2 with random offset for each pixel
vec2 stratifiedSample(uint sampleIdx, vec2 randomOffset) {
    return fract(r2Sample(sampleIdx) + randomOffset);
}

// Clamp contribution to prevent fireflies
vec3 clampRadiance(vec3 r) {
    return min(r, vec3(MAX_RADIANCE));
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
