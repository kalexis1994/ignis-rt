// ============================================================
// pbr_brdf.glsl — PBR Functions (GGX Microfacet Model)
// Fresnel, Normal Distribution, Geometry (Smith), Cook-Torrance
// ============================================================

#ifndef PBR_BRDF_GLSL
#define PBR_BRDF_GLSL

// Real dielectric Fresnel (exact, NOT Schlick approximation)
// Matches Cycles' fresnel_dielectric_cos() from bsdf_util.h
float fresnel_dielectric(float cosi, float eta) {
    float c = abs(cosi);
    float g = eta * eta - 1.0 + c * c;
    if (g > 0.0) {
        g = sqrt(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
        return 0.5 * A * A * (1.0 + B * B);
    }
    return 1.0;  // Total internal reflection
}

// Fresnel — Real dielectric + F82-Tint (matches Cycles' Principled BSDF exactly)
// Uses real Fresnel equations remapped to [F0, F90] range, NOT Schlick pow(5).
// This produces smoother, less aggressive reflections at grazing angles.
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    // Compute real dielectric Fresnel and remap to [F0_real, 1.0] range
    // Cycles: s = inverse_lerp(F0_real, 1.0, F_real)
    // Recover eta from F0: eta = (1+sqrt(F0))/(1-sqrt(F0))
    float avgF0 = max((F0.r + F0.g + F0.b) / 3.0, 1e-6);
    float sqrtF0 = sqrt(clamp(avgF0, 0.0, 0.999));
    float eta = (1.0 + sqrtF0) / max(1.0 - sqrtF0, 1e-6);
    float F0_real = ((eta - 1.0) / (eta + 1.0)) * ((eta - 1.0) / (eta + 1.0));
    float F_real = fresnel_dielectric(cosTheta, eta);
    float s = clamp((F_real - F0_real) / max(1.0 - F0_real, 1e-6), 0.0, 1.0);

    // Interpolate between tinted F0 and white F90 using remapped real Fresnel
    vec3 F = mix(F0, vec3(1.0), s);

    // F82-Tint correction for metals (darkens reflections around ~82°)
    vec3 B = (1.0 - F0) * (25.0 / 24.0);
    float t = 1.0 - cosTheta;
    float t2 = t * t;
    float t6 = t2 * t2 * t2;
    return max(F - B * cosTheta * t6, vec3(0.0));
}

// GGX Normal Distribution Function
float GGX_D(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith G1 (height-correlated)
float SmithG1(float NdotV, float alpha) {
    float a2 = alpha * alpha;
    return 2.0 * NdotV / (NdotV + sqrt(a2 + (1.0 - a2) * NdotV * NdotV));
}

// Smith G2 (height-correlated, used in BRDF evaluation)
float SmithG2(float NdotL, float NdotV, float alpha) {
    return SmithG1(NdotL, alpha) * SmithG1(NdotV, alpha);
}

// Cook-Torrance BRDF Evaluation (for NEE / direct light)
void evaluateCookTorrance(
    vec3 N, vec3 V, vec3 L, vec3 baseColor, float roughness, float metallic, float specularLevel,
    float ior,
    out vec3 diffuseContrib, out vec3 specularContrib
) {
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);

    if (NdotL <= 0.0) {
        diffuseContrib = vec3(0.0);
        specularContrib = vec3(0.0);
        return;
    }

    vec3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    float alpha = roughness * roughness;

    // F0 from IOR for dielectrics, baseColor for metals
    float f0_dielectric = (ior - 1.0) / (ior + 1.0);
    f0_dielectric = f0_dielectric * f0_dielectric;
    vec3 F0 = mix(vec3(f0_dielectric * specularLevel * 2.0), baseColor, metallic);
    vec3 F = fresnelSchlick(VdotH, F0);

    // Specular: GGX Cook-Torrance
    float D = GGX_D(NdotH, alpha);
    float G = SmithG2(NdotL, NdotV, alpha);
    vec3 spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    // Diffuse: Lambertian (energy-conserving with metallic)
    vec3 kd = (1.0 - F) * (1.0 - metallic);
    vec3 diff = kd * baseColor * INV_PI;

    diffuseContrib = diff * NdotL;
    specularContrib = spec * NdotL;
}

#endif // PBR_BRDF_GLSL
