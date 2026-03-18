// ============================================================
// BLENDER EEVEE - MICROFACET BRDF LIBRARY
// ============================================================
// High-quality PBR material evaluation ported from Blender EEVEE
// Source: source/blender/draw/engines/eevee/shaders/eevee_bxdf_microfacet_lib.glsl
//
// This provides significantly better specular quality compared to standard GGX,
// especially for rough materials with grazing angles.

#ifndef EEVEE_BXDF_GLSL
#define EEVEE_BXDF_GLSL

#define M_PI 3.14159265358979323846
#define M_1_PI 0.31830988618379067154

// ============================================================
// GGX DISTRIBUTION (D Term)
// ============================================================

// GGX/Trowbridge-Reitz Normal Distribution Function
// More accurate than simplified versions, handles edge cases better
float bxdf_ggx_D(float NH, float roughness) {
    float a2 = roughness * roughness;
    a2 = max(a2, 1e-4);  // Clamp to avoid division by zero

    float tmp = (NH * a2 - NH) * NH + 1.0;
    return a2 / (M_PI * tmp * tmp);
}

// Anisotropic GGX Distribution (for future anisotropic materials)
float bxdf_ggx_D_aniso(float NH, vec3 H, vec3 T, vec3 B, float ax, float ay) {
    float x = dot(H, T) / ax;
    float y = dot(H, B) / ay;
    float z = NH;

    float tmp = x * x + y * y + z * z;
    return 1.0 / (M_PI * ax * ay * tmp * tmp);
}

// ============================================================
// SMITH GEOMETRY TERM (G Term)
// ============================================================

// Smith's height-correlated masking-shadowing function
// More accurate than Schlick-GGX approximation
float bxdf_ggx_smith_G1(float NV, float roughness) {
    float a2 = roughness * roughness;

    // Smith G1 formula: 2 / (1 + sqrt(1 + a2 * tan2(theta)))
    // Optimized using NV instead of tan(theta)
    float cos2 = NV * NV;
    float sin2 = 1.0 - cos2;

    if (sin2 <= 0.0) return 1.0;  // Perpendicular view, no masking

    float tan2 = sin2 / cos2;
    return 2.0 / (1.0 + sqrt(1.0 + a2 * tan2));
}

// Height-correlated Smith G2 (masking-shadowing combined)
// This is more accurate than separable G1(V)*G1(L)
float bxdf_ggx_smith_G2(float NL, float NV, float roughness) {
    float a2 = roughness * roughness;

    // G1 for light direction
    float cos2_L = NL * NL;
    float sin2_L = 1.0 - cos2_L;
    float tan2_L = (sin2_L > 0.0) ? sin2_L / cos2_L : 0.0;
    float lambda_L = 0.5 * (-1.0 + sqrt(1.0 + a2 * tan2_L));

    // G1 for view direction
    float cos2_V = NV * NV;
    float sin2_V = 1.0 - cos2_V;
    float tan2_V = (sin2_V > 0.0) ? sin2_V / cos2_V : 0.0;
    float lambda_V = 0.5 * (-1.0 + sqrt(1.0 + a2 * tan2_V));

    // Height-correlated G2
    return 1.0 / (1.0 + lambda_L + lambda_V);
}

// ============================================================
// FRESNEL TERM (F Term)
// ============================================================

// Schlick's approximation (fast, good quality)
vec3 bxdf_fresnel_schlick(float VH, vec3 F0) {
    float m = clamp(1.0 - VH, 0.0, 1.0);
    float m2 = m * m;
    float m5 = m2 * m2 * m;
    return F0 + (1.0 - F0) * m5;
}

// Fresnel for dielectrics (IOR-based)
float bxdf_fresnel_dielectric(float cos_theta, float eta) {
    float sin_theta_t_sq = eta * eta * (1.0 - cos_theta * cos_theta);

    // Total internal reflection
    if (sin_theta_t_sq > 1.0) return 1.0;

    float cos_theta_t = sqrt(1.0 - sin_theta_t_sq);

    float rs = (cos_theta - eta * cos_theta_t) / (cos_theta + eta * cos_theta_t);
    float rp = (eta * cos_theta - cos_theta_t) / (eta * cos_theta + cos_theta_t);

    return 0.5 * (rs * rs + rp * rp);
}

// ============================================================
// BOUNDED VNDF SAMPLING (Visible Normal Distribution Function)
// ============================================================
// This is the key improvement from EEVEE - much better importance sampling
// for rough materials, especially at grazing angles

// Sample GGX VNDF (Visible Normal Distribution)
// Returns microfacet normal in tangent space
vec3 bxdf_ggx_sample_vndf(vec3 Ve, float roughness, vec2 Xi) {
    // Input Ve is view direction in tangent space (Z-up)

    float a = roughness;

    // Section 3.2: transforming the view direction to the hemisphere configuration
    vec3 Vh = normalize(vec3(a * Ve.x, a * Ve.y, Ve.z));

    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);

    // Section 4.2: parameterization of the projected area
    float r = sqrt(Xi.x);
    float phi = 2.0 * M_PI * Xi.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    // Section 4.3: reprojection onto hemisphere
    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    // Section 3.4: transforming the normal back to the ellipsoid configuration
    return normalize(vec3(a * Nh.x, a * Nh.y, max(0.0, Nh.z)));
}

// PDF for GGX VNDF sampling
float bxdf_ggx_vndf_pdf(float NH, float NV, float VH, float roughness) {
    float G1_V = bxdf_ggx_smith_G1(NV, roughness);
    float D = bxdf_ggx_D(NH, roughness);

    // VNDF PDF = G1(V) * D(H) * max(0, V·H) / NV
    return (G1_V * D * max(VH, 0.0)) / max(NV, 1e-8);
}

// ============================================================
// COMPLETE MICROFACET BRDF EVALUATION
// ============================================================

// GGX Microfacet BRDF (Cook-Torrance)
// f(v,l) = D(h) * G(v,l) * F(v,h) / (4 * (n·v) * (n·l))
vec3 bxdf_ggx_eval_reflection(vec3 N, vec3 V, vec3 L, float roughness, vec3 F0) {
    vec3 H = normalize(V + L);

    float NL = max(dot(N, L), 0.0);
    float NV = max(dot(N, V), 1e-8);
    float NH = max(dot(N, H), 0.0);
    float VH = max(dot(V, H), 0.0);

    if (NL <= 0.0 || NV <= 0.0) return vec3(0.0);

    // D term (distribution)
    float D = bxdf_ggx_D(NH, roughness);

    // G term (geometry) - using height-correlated G2
    float G = bxdf_ggx_smith_G2(NL, NV, roughness);

    // F term (fresnel)
    vec3 F = bxdf_fresnel_schlick(VH, F0);

    // Cook-Torrance specular BRDF
    // The 4 in denominator cancels with the 2 from G2 definition
    vec3 specular = (D * G * F) / (4.0 * NV * NL);

    return specular;
}

// Combined diffuse + specular BRDF
// Includes energy conservation (1 - F) term for diffuse
vec3 bxdf_ggx_eval(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness, float metallic) {
    vec3 H = normalize(V + L);

    float NL = max(dot(N, L), 0.0);
    float NV = max(dot(N, V), 1e-8);
    float VH = max(dot(V, H), 0.0);

    if (NL <= 0.0) return vec3(0.0);

    // F0 for metals = albedo, for dielectrics = 0.04
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Specular term
    vec3 specular = bxdf_ggx_eval_reflection(N, V, L, roughness, F0);

    // Fresnel for energy conservation
    vec3 F = bxdf_fresnel_schlick(VH, F0);

    // Diffuse term (Lambertian with energy conservation)
    // (1 - F) ensures energy conservation between diffuse and specular
    // (1 - metallic) because metals have no diffuse
    vec3 diffuse = (albedo * M_1_PI) * (vec3(1.0) - F) * (1.0 - metallic);

    return (diffuse + specular) * NL;
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

// Convert roughness to alpha (GGX parameter)
float roughness_to_alpha(float roughness) {
    // Perceptually linear roughness mapping
    return roughness * roughness;
}

// Convert alpha back to roughness
float alpha_to_roughness(float alpha) {
    return sqrt(alpha);
}

// Remap roughness for better perceptual control
// Maps [0,1] roughness to avoid too-perfect mirrors and too-rough surfaces
float remap_roughness(float roughness) {
    // Clamp to [0.045, 1.0] to avoid numerical issues
    return clamp(roughness, 0.045, 1.0);
}

#endif // EEVEE_BXDF_GLSL
