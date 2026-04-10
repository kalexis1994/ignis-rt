// ============================================================
// sampling.glsl — BRDF Sampling Functions
// Cosine-weighted hemisphere, GGX VNDF (Heitz 2018)
// ============================================================

#ifndef SAMPLING_GLSL
#define SAMPLING_GLSL

// Cosine-weighted hemisphere sampling (for diffuse)
vec3 sampleCosineHemisphere(vec2 u, vec3 N) {
    float phi = 2.0 * PI * u.x;
    float cosTheta = sqrt(u.y);
    float sinTheta = sqrt(1.0 - u.y);

    // Build orthonormal basis around N
    vec3 up = abs(N.y) < 0.999 ? vec3(0, 1, 0) : vec3(1, 0, 0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    return normalize(tangent * (cos(phi) * sinTheta) +
                     bitangent * (sin(phi) * sinTheta) +
                     N * cosTheta);
}

// GGX VNDF sampling (Heitz 2018 — "Sampling the GGX Distribution of Visible Normals")
vec3 sampleGGX_VNDF(vec2 u, vec3 V, vec3 N, float alpha) {
    // Build orthonormal basis around N
    vec3 up = abs(N.y) < 0.999 ? vec3(0, 1, 0) : vec3(1, 0, 0);
    vec3 T1 = normalize(cross(up, N));
    vec3 T2 = cross(N, T1);

    // Transform view direction to local space
    vec3 Vh = normalize(vec3(
        dot(V, T1) * alpha,
        dot(V, N),
        dot(V, T2) * alpha
    ));

    // Orthonormal basis in hemisphere space
    float lensq = Vh.x * Vh.x + Vh.z * Vh.z;
    vec3 T1h = lensq > 0.0 ? vec3(-Vh.z, 0.0, Vh.x) / sqrt(lensq) : vec3(1, 0, 0);
    vec3 T2h = cross(Vh, T1h);

    // Uniform disk sample
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.y);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    // Microfacet normal in hemisphere space
    vec3 Nh = t1 * T1h + t2 * T2h + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    // Transform back to world space
    vec3 H = normalize(
        T1 * (alpha * Nh.x) +
        N * max(0.0, Nh.y) +
        T2 * (alpha * Nh.z)
    );

    return H;
}

// Anisotropic GGX VNDF sampling (Cycles microfacet_ggx_sample_vndf)
// T, B, N = tangent frame. alpha_x, alpha_y = anisotropic roughness
vec3 sampleGGX_VNDF_aniso(vec2 u, vec3 V, vec3 T, vec3 B, vec3 N, float alpha_x, float alpha_y) {
    // Transform V to local frame and stretch by alpha
    vec3 V_local = vec3(dot(V, T), dot(V, B), dot(V, N));
    vec3 Vh = normalize(vec3(alpha_x * V_local.x, alpha_y * V_local.y, V_local.z));

    // Orthonormal basis in hemisphere space
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1h = lensq > 1e-7 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1, 0, 0);
    vec3 T2h = cross(Vh, T1h);

    // Sample disk and project to hemisphere
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    // Microfacet normal in stretched hemisphere
    vec3 Nh = t1 * T1h + t2 * T2h + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    // Un-stretch and transform back to world space
    vec3 H_local = normalize(vec3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0, Nh.z)));
    return normalize(T * H_local.x + B * H_local.y + N * H_local.z);
}

#endif // SAMPLING_GLSL
