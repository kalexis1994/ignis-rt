// ============================================================
// nrd_encode.glsl — NRD G-buffer Packing Functions
// Pack normal+roughness, radiance+hitDist for NRD input
// ============================================================

#ifndef NRD_ENCODE_GLSL
#define NRD_ENCODE_GLSL

// NRD Oct-encoded normal + roughness (matches NRD_FrontEnd_PackNormalAndRoughness)
vec4 NRD_PackNormalRoughness(vec3 N, float roughness) {
    N /= (abs(N.x) + abs(N.y) + abs(N.z));
    vec3 r;
    r.y = N.y * 0.5 + 0.5;
    r.x = N.x * 0.5 + r.y;
    r.y -= N.x * 0.5;
    roughness = max(roughness, 1.5 / 512.0);
    float s = N.z < 0.0 ? -roughness : roughness;
    r.z = s * 0.5 + 0.5;
    return vec4(r, 0.0);
}

// Pack radiance + hit distance for NRD input
vec4 NRD_PackRadiance(vec3 radiance, float hitDist) {
    // Sanitize: NaN/Inf from degenerate paths would poison the denoiser
    radiance = max(radiance, vec3(0.0));
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = vec3(0.0);
    radiance = min(radiance, vec3(MAX_RADIANCE));
    hitDist = clamp(hitDist, 0.0, 65504.0);
    if (isnan(hitDist) || isinf(hitDist)) hitDist = 0.0;
    return vec4(radiance, hitDist);
}

#endif // NRD_ENCODE_GLSL
