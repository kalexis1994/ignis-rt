// ============================================================
// pbr_brdf.glsl — PBR Functions (GGX Microfacet Model)
// Fresnel, Normal Distribution, Geometry (Smith), Cook-Torrance
// ============================================================

#ifndef PBR_BRDF_GLSL
#define PBR_BRDF_GLSL

// Fresnel (Schlick approximation)
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    float t = 1.0 - cosTheta;
    float t2 = t * t;
    return F0 + (1.0 - F0) * (t2 * t2 * t);
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
