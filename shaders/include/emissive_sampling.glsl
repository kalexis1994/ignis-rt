// ============================================================
// emissive_sampling.glsl — Emissive triangle MIS sampling
// Requires: emissiveTris SSBO bound at binding 26
// ============================================================

#ifndef EMISSIVE_SAMPLING_GLSL
#define EMISSIVE_SAMPLING_GLSL

struct EmissiveSample {
    vec3 position;
    vec3 emission;
    vec3 normal;
    float pdf;
    float area;
};

// Sample an emissive triangle using CDF binary search
EmissiveSample sampleEmissiveTriangle(uint triCount, float rnd, vec2 baryRnd) {
    EmissiveSample s;
    s.pdf = 0.0;
    s.emission = vec3(0.0);
    s.normal = vec3(0, 1, 0);
    s.position = vec3(0.0);
    s.area = 0.0;

    if (triCount == 0u) return s;

    // Binary search CDF to select triangle
    uint lo = 0u, hi = triCount - 1u;
    while (lo < hi) {
        uint mid = (lo + hi) / 2u;
        float cdf = emissiveTris.data[mid * 4u + 1u].w;
        if (rnd <= cdf)
            hi = mid;
        else
            lo = mid + 1u;
    }
    uint idx = lo;
    uint base = idx * 4u;

    vec4 d0 = emissiveTris.data[base + 0u];
    vec4 d1 = emissiveTris.data[base + 1u];
    vec4 d2 = emissiveTris.data[base + 2u];
    vec4 d3 = emissiveTris.data[base + 3u];

    vec3 v0 = d0.xyz;
    vec3 v1 = d1.xyz;
    vec3 v2 = d2.xyz;
    s.area = d0.w;
    s.emission = d3.xyz;

    float prevCdf = (idx > 0u) ? emissiveTris.data[(idx - 1u) * 4u + 1u].w : 0.0;
    float triProb = d1.w - prevCdf;

    float u = baryRnd.x;
    float v = baryRnd.y;
    if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
    s.position = v0 * (1.0 - u - v) + v1 * u + v2 * v;

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    s.normal = normalize(cross(edge1, edge2));

    s.pdf = (s.area > 0.0) ? triProb / s.area : 0.0;

    return s;
}

#endif // EMISSIVE_SAMPLING_GLSL
