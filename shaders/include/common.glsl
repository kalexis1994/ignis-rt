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
const float MAX_RADIANCE = 250.0;

// ============================================================
// RNG — PCG hash
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
