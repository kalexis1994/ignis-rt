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

// Fresnel — Generalized Schlick with real Fresnel interpolation
// Matches Cycles' Principled BSDF: uses actual IOR for the Fresnel curve,
// remaps to [F0, F90] range. F82-Tint only applied for metals.
vec3 fresnelGeneralized(float cosTheta, vec3 F0, float ior, float metallic) {
    // Real dielectric Fresnel at this angle, using actual material IOR
    float F0_real = ((ior - 1.0) / (ior + 1.0)) * ((ior - 1.0) / (ior + 1.0));
    float F_real = fresnel_dielectric(cosTheta, ior);
    float s = clamp((F_real - F0_real) / max(1.0 - F0_real, 1e-6), 0.0, 1.0);

    // Interpolate between tinted F0 and white F90 using remapped real Fresnel
    vec3 F = mix(F0, vec3(1.0), s);

    // F82-Tint correction (only significant for metals — darkens at ~82°)
    if (metallic > 0.001) {
        vec3 B = (1.0 - F0) * (25.0 / 24.0);
        float t = 1.0 - cosTheta;
        float t2 = t * t;
        float t6 = t2 * t2 * t2;
        F = max(F - B * cosTheta * t6, vec3(0.0));
    }
    return F;
}

// Legacy wrapper for compatibility (recovers IOR from F0)
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    float avgF0 = max((F0.r + F0.g + F0.b) / 3.0, 1e-6);
    float sqrtF0 = sqrt(clamp(avgF0, 0.0, 0.999));
    float eta = (1.0 + sqrtF0) / max(1.0 - sqrtF0, 1e-6);
    return fresnelGeneralized(cosTheta, F0, eta, 1.0);
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

// Approximate GGX single-scatter directional albedo E(mu, alpha).
// Analytical fit avoiding LUT — Lazarov 2013 / Filament.
// Returns how much energy single-scatter GGX reflects at a given angle.
float approxGGXAlbedo(float NdotX, float alpha) {
    // Curve-fit to the numerical integral of GGX BRDF over hemisphere.
    // Smooth surfaces: E → 1 (all energy reflected in single bounce).
    // Rough surfaces:  E → 0.5-0.7 (significant energy lost to multi-scatter).
    vec4 r = alpha * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotX)) * r.x + r.y;
    return clamp(-1.04 * a004 + r.z, 0.0, 1.0);
}

// Cook-Torrance BRDF Evaluation (for NEE / direct light)
// Includes Kulla-Conty multi-scatter energy compensation to match Cycles.
// ============================================================
// Coat (Clearcoat) Layer — dielectric GGX on top of base BRDF
// ============================================================
void evaluateCoatLayer(
    vec3 N, vec3 V, vec3 L,
    float coatWeight, float coatRoughness, float coatIOR,
    out vec3 coatSpecular, out float baseAttenuation
) {
    coatSpecular = vec3(0.0);
    baseAttenuation = 1.0;
    if (coatWeight <= 0.001) return;

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    if (NdotL <= 0.0) return;

    vec3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    float coatAlpha = coatRoughness * coatRoughness;
    float coatF = fresnel_dielectric(VdotH, 1.0 / coatIOR);
    float D = GGX_D(NdotH, coatAlpha);
    float G = SmithG2(NdotL, NdotV, coatAlpha);
    float specDenom = max(4.0 * NdotV * NdotL, 0.001);

    coatSpecular = vec3(coatWeight * coatF * D * G / specDenom) * NdotL;
    baseAttenuation = 1.0 - coatWeight * coatF;
}

// ============================================================
// Full PBR evaluation: Cook-Torrance + Coat layer
// ============================================================
void evaluateCookTorrance(
    vec3 N, vec3 V, vec3 L, vec3 baseColor, float roughness, float metallic, float specularLevel,
    float ior, float transmission,
    out vec3 diffuseContrib, out vec3 specularContrib,
    float coatWeight, float coatRoughness, float coatIOR
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
    vec3 dielectricF0 = vec3(f0_dielectric * specularLevel * 2.0);
    float D = GGX_D(NdotH, alpha);
    float G = SmithG2(NdotL, NdotV, alpha);
    float specDenom = max(4.0 * NdotV * NdotL, 0.001);

    // Multi-scatter GGX energy compensation (Kulla-Conty 2017, analytical approx).
    // Single-scatter GGX loses energy on rough surfaces; this recovers it.
    // Cycles ref: bsdf_microfacet.h microfacet_ggx_preserve_energy()
    float Eo = approxGGXAlbedo(NdotV, alpha);  // outgoing directional albedo
    float Ei = approxGGXAlbedo(NdotL, alpha);  // incoming directional albedo
    float msScale = 1.0 + (1.0 - Eo) * (1.0 - Ei) / max(Eo + Ei - Eo * Ei, 0.01);

    // Fast common case: opaque Principled without transmission.
    if (transmission <= 0.001) {
        vec3 F0 = mix(dielectricF0, baseColor, metallic);
        vec3 F = fresnelGeneralized(VdotH, F0, ior, metallic);
        vec3 spec = (D * G * F) / specDenom * msScale;
        // Diffuse uses directional albedo for energy-conserving kd (Cycles-matching)
        // Using Eo (integrated GGX albedo) instead of pointwise F avoids over-darkening
        // at grazing angles where specular D*G goes to zero but F is high
        float specAlbedo = mix(Eo * f0_dielectric * specularLevel * 2.0,
                               Eo * (F0.r + F0.g + F0.b) / 3.0, metallic);
        vec3 kd = vec3(1.0 - specAlbedo) * (1.0 - metallic);
        vec3 diff = kd * baseColor * INV_PI;
        diffuseContrib = diff * NdotL;
        specularContrib = spec * NdotL;

        // Coat layer attenuation + contribution
        if (coatWeight > 0.001) {
            vec3 coatSpec; float baseAtten;
            evaluateCoatLayer(N, V, L, coatWeight, coatRoughness, coatIOR, coatSpec, baseAtten);
            diffuseContrib *= baseAtten;
            specularContrib = specularContrib * baseAtten + coatSpec;
        }
        return;
    }

    // With transmission: same approach, weighted by lobe contributions.
    float dielectricWeight = (1.0 - metallic) * (1.0 - transmission);
    vec3 dielectricF = fresnelGeneralized(VdotH, dielectricF0, ior, 0.0);

    vec3 spec;
    if (metallic <= 0.001) {
        spec = (D * G * dielectricF) / specDenom * dielectricWeight * msScale;
    } else if (metallic >= 0.999) {
        vec3 metalF = fresnelGeneralized(VdotH, baseColor, ior, 1.0);
        spec = (D * G * metalF) / specDenom * msScale;
    } else {
        vec3 metalF = fresnelGeneralized(VdotH, baseColor, ior, 1.0);
        vec3 dielectricSpec = (D * G * dielectricF) / specDenom;
        vec3 metalSpec = (D * G * metalF) / specDenom;
        spec = (dielectricSpec * dielectricWeight + metalSpec * metallic) * msScale;
    }

    vec3 kd = (1.0 - dielectricF) * dielectricWeight;
    vec3 diff = kd * baseColor * INV_PI;

    diffuseContrib = diff * NdotL;
    specularContrib = spec * NdotL;

    // Coat layer attenuation + contribution
    if (coatWeight > 0.001) {
        vec3 coatSpec; float baseAtten;
        evaluateCoatLayer(N, V, L, coatWeight, coatRoughness, coatIOR, coatSpec, baseAtten);
        diffuseContrib *= baseAtten;
        specularContrib = specularContrib * baseAtten + coatSpec;
    }
}


#endif // PBR_BRDF_GLSL
