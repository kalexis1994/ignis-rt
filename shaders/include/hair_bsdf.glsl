// hair_bsdf.glsl — Karis (2016) Simplified Marschner Hair BSDF
// Based on "Physically Based Hair Shading in Unreal" (Epic Games, SIGGRAPH 2016)
// R (primary specular) + TT (transmission) + TRT (colored secondary specular)
// Uses Gaussian longitudinal scattering instead of Bessel I0 — 3-4x faster than Chiang
// Future: VK_NV_ray_tracing_linear_swept_spheres (RTX 50+ Blackwell)

#define HAIR_PI 3.14159265358979
#define HAIR_2PI 6.28318530717959
#define HAIR_INV_PI 0.31830988618379
#define HAIR_INV_2PI 0.15915494309190

// ── Fresnel (Schlick approximation for speed) ──
float hair_fresnel_schlick(float cosTheta, float F0) {
    float x = 1.0 - cosTheta;
    float x2 = x * x;
    return F0 + (1.0 - F0) * x2 * x2 * x;
}

// ── Gaussian longitudinal scattering (replaces Bessel I0 — much cheaper) ──
// M_p(theta_h, beta) = exp(-theta_h^2 / (2 * beta^2)) / (sqrt(2*pi) * beta)
float hair_gaussian_M(float theta_h, float beta) {
    float b2 = beta * beta;
    if (b2 < 1e-8) return 0.0;
    float invBeta = 1.0 / beta;
    return exp(-0.5 * theta_h * theta_h / b2) * 0.3989422804 * invBeta;
}

// ── Wrapped cosine azimuthal scattering (replaces logistic distribution) ──
float hair_azimuthal_N(float phi, float delta_phi, float exponent) {
    float dphi = phi - delta_phi;
    // Wrap to [-pi, pi]
    dphi = dphi - HAIR_2PI * floor((dphi + HAIR_PI) / HAIR_2PI);
    // Normalized cosine power
    float cosVal = cos(dphi * 0.5);
    return max(cosVal, 0.0) * pow(max(cosVal, 0.0), exponent) * (exponent + 1.0) * HAIR_INV_2PI;
}

// ── Main evaluation: Karis Simplified Marschner Hair BSDF ──
// wi = incoming direction (toward surface), wo = outgoing (toward light/bounce)
// tangent = hair fiber direction
// hairColor = base color of the hair
// roughness = longitudinal roughness [0,1]
// radialRoughness = azimuthal roughness [0,1]
// shift = cuticle tilt in radians (~0.035 for human hair)
// ior = index of refraction (1.55 for hair)
// h = position across fiber [-1,1] (from DOTS v parameter)
vec3 hair_eval(vec3 wi, vec3 wo, vec3 tangent,
               vec3 hairColor, float roughness, float radialRoughness, float shift, float ior, float h) {

    vec3 T = normalize(tangent);

    // Longitudinal angles (along fiber)
    float sin_theta_i = dot(wi, T);
    float sin_theta_o = dot(wo, T);
    float cos_theta_i = sqrt(max(0.0, 1.0 - sin_theta_i * sin_theta_i));
    float cos_theta_o = sqrt(max(0.0, 1.0 - sin_theta_o * sin_theta_o));

    // Half-angle
    float theta_h = (asin(clamp(sin_theta_i, -1.0, 1.0)) + asin(clamp(sin_theta_o, -1.0, 1.0))) * 0.5;

    // Azimuthal angle (around fiber)
    // Build local 2D frame perpendicular to T
    vec3 Y = normalize(cross(T, wi));
    if (length(Y) < 1e-6) Y = normalize(cross(T, vec3(0, 1, 0)));
    vec3 Z = cross(T, Y);
    float phi_i = atan(dot(wi, Z), dot(wi, Y));
    float phi_o = atan(dot(wo, Z), dot(wo, Y));
    float phi = phi_o - phi_i;

    // ── Roughness per lobe (Karis/Marschner) ──
    float beta = max(roughness, 0.05);
    float beta_R   = beta;          // R: primary roughness
    float beta_TT  = beta * 0.5;    // TT: tighter (light passes straight through)
    float beta_TRT = beta * 2.0;    // TRT: broader (internal bounce spreads it)
    float betaPhi = max(radialRoughness, 0.05);

    // ── Cuticle tilt per lobe ──
    float alpha = shift;
    float alpha_R   = -2.0 * alpha;   // R: shifted toward tip
    float alpha_TT  =  1.0 * alpha;   // TT: shifted toward root
    float alpha_TRT = -4.0 * alpha;   // TRT: double-shifted toward tip

    // ── Fresnel at fiber surface ──
    float cos_gamma_o = sqrt(max(0.0, 1.0 - h * h));
    float F0 = ((1.0 - ior) / (1.0 + ior));
    F0 = F0 * F0;
    float F = hair_fresnel_schlick(cos_theta_o * cos_gamma_o, F0);

    // ── Azimuthal shifts (Snell's law at fiber surface) ──
    float gamma_o = asin(clamp(h, -1.0, 1.0));
    float eta_sq_sin2 = ior * ior - sin_theta_o * sin_theta_o;
    float sin_gamma_t = (eta_sq_sin2 > 0.0) ? h * cos_theta_o / sqrt(eta_sq_sin2) : 0.0;
    float gamma_t = asin(clamp(sin_gamma_t, -1.0, 1.0));
    float cos_gamma_t = cos(gamma_t);

    // ── Attenuation per lobe (Beer-Lambert absorption, matching Cycles) ──
    // Convert base color to absorption coefficient sigma using Cycles' formula:
    //   sigma = (ln(color) / roughness_factor)^2
    // Then apply Beer-Lambert: T = exp(-sigma * path_length)
    // Ref: intern/cycles/kernel/closure/bsdf_principled_hair_chiang.h
    float roughness_fac = (((((0.245 * betaPhi) + 5.574) * betaPhi - 10.73) * betaPhi + 2.532) * betaPhi - 0.215) * betaPhi + 5.969;
    vec3 sigma = vec3(0.0);
    for (int ch = 0; ch < 3; ch++) {
        float c = max(hairColor[ch], 1e-5);
        float ls = log(c) / max(roughness_fac, 1e-5);
        sigma[ch] = ls * ls;
    }

    // Beer-Lambert transmittance through the fiber
    float path_len = 2.0 * cos_gamma_t / max(cos_theta_o, 0.01);
    vec3 T_fiber = exp(-sigma * path_len);

    vec3 A_R   = vec3(F);                                  // R: surface reflection (white)
    vec3 A_TT  = (1.0 - F) * (1.0 - F) * T_fiber;         // TT: one pass through fiber
    vec3 A_TRT = (1.0 - F) * (1.0 - F) * T_fiber * T_fiber * F;  // TRT: two passes through fiber
    float dphi_R   = -2.0 * gamma_o;
    float dphi_TT  =  2.0 * gamma_t - 2.0 * gamma_o + HAIR_PI;
    float dphi_TRT =  4.0 * gamma_t - 2.0 * gamma_o;

    // ── Sum R + TT + TRT ──
    // Azimuthal exponents control highlight sharpness (lower = softer, wider)
    // R: primary specular (white highlight)
    float M_R = hair_gaussian_M(theta_h - alpha_R, beta_R);
    float N_R = hair_azimuthal_N(phi, dphi_R, mix(48.0, 3.0, betaPhi));
    vec3 F_total = A_R * M_R * N_R;

    // TT: transmission (backlit glow through fiber)
    float M_TT = hair_gaussian_M(theta_h - alpha_TT, beta_TT);
    float N_TT = hair_azimuthal_N(phi, dphi_TT, mix(18.0, 1.5, betaPhi));
    F_total += A_TT * M_TT * N_TT;

    // TRT: secondary colored specular (warm glint)
    float M_TRT = hair_gaussian_M(theta_h - alpha_TRT, beta_TRT);
    float N_TRT = hair_azimuthal_N(phi, dphi_TRT, mix(28.0, 2.0, betaPhi));
    F_total += A_TRT * M_TRT * N_TRT;

    // Normalization: Cycles' Chiang model includes 1/(2*cos_theta) in the longitudinal
    // scattering which our Gaussian M doesn't have. Scale by cos_theta to match energy.
    // Also divide by PI for hemisphere normalization (M*N integrates over the fiber
    // coordinate system, not the rendering hemisphere).
    F_total *= cos_theta_o * HAIR_INV_PI;

    // Clamp cos_theta_i to prevent blackout at grazing angles
    return max(F_total, vec3(0.0)) * max(abs(cos_theta_i), 0.15);
}

vec3 hair_eval(vec3 wi, vec3 wo, vec3 tangent,
               vec3 hairColor, float roughness, float shift, float ior, float h) {
    return hair_eval(wi, wo, tangent, hairColor, roughness, roughness, shift, ior, h);
}
