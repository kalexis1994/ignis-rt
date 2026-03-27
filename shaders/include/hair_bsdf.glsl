// hair_bsdf.glsl — Chiang et al. (2015) Principled Hair BSDF
// "A Practical and Controllable Hair and Fur Model for Production Path Tracing"
// R (primary reflection) + TT (transmission) + TRT (secondary reflection) lobes
// Future: VK_NV_ray_tracing_linear_swept_spheres (RTX 50+ Blackwell)

#define HAIR_PI 3.14159265358979
#define HAIR_2PI 6.28318530717959
#define HAIR_INV_2PI 0.15915494309190

// ── Bessel I0 approximation ──
float hair_bessel_I0(float x) {
    float xsq = x * x;
    float val = 1.0;
    float pow_x_2i = xsq * xsq;
    float i_fac_2 = 1.0;
    float pow_4_i = 16.0;
    val += 0.25 * xsq;
    for (int i = 2; i <= 9; i++) {
        i_fac_2 *= float(i * i);
        float term = pow_x_2i / (pow_4_i * i_fac_2);
        val += term;
        if (term < 1e-8 * val) break;
        pow_x_2i *= xsq;
        pow_4_i *= 4.0;
    }
    return val;
}

float hair_log_bessel_I0(float x) {
    if (x > 12.0)
        return x + 0.5 * (1.0 / (8.0 * x) - log(HAIR_2PI) - log(x));
    return log(hair_bessel_I0(x));
}

// ── Logistic distribution ──
float hair_logistic(float x, float s) {
    float v = exp(-abs(x) / s);
    return v / (s * (1.0 + v) * (1.0 + v));
}

float hair_logistic_cdf(float x, float s) {
    float arg = -x / s;
    if (arg > 80.0) return 0.0;
    return 1.0 / (1.0 + exp(arg));
}

float hair_trimmed_logistic(float x, float s) {
    float scaling = 1.0 - 2.0 * hair_logistic_cdf(-HAIR_PI, s);
    return (scaling > 1e-8) ? hair_logistic(x, s) / scaling : HAIR_INV_2PI;
}

// ── Angle wrapping to [-pi, pi] ──
float hair_wrap_angle(float a) {
    return a - HAIR_2PI * floor((a + HAIR_PI) / HAIR_2PI) - HAIR_PI;
}

// ── Longitudinal scattering (Mp) ──
float hair_Mp(float sin_ti, float cos_ti, float sin_to, float cos_to, float v) {
    float inv_v = 1.0 / v;
    float cos_arg = cos_ti * cos_to * inv_v;
    float sin_arg = sin_ti * sin_to * inv_v;

    if (v <= 0.1) {
        float logI0 = hair_log_bessel_I0(cos_arg);
        float val = logI0 - sin_arg - inv_v + 0.6931 + log(0.5 * inv_v);
        return exp(val) / (2.0 * v * sinh(inv_v));
    }
    float I0 = hair_bessel_I0(cos_arg);
    float denom = 2.0 * v * sinh(inv_v);
    return (denom > 1e-10) ? exp(-sin_arg) * I0 / denom : 0.0;
}

// ── Azimuthal scattering (Np) ──
float hair_Np(float phi, int p, float s, float gamma_o, float gamma_t) {
    float delta_phi = 2.0 * float(p) * gamma_t - 2.0 * gamma_o + float(p) * HAIR_PI;
    float phi_o = hair_wrap_angle(phi - delta_phi);
    return hair_trimmed_logistic(phi_o, s);
}

// ── Fresnel dielectric ──
float hair_fresnel(float cos_theta, float eta) {
    float sin2 = 1.0 - cos_theta * cos_theta;
    float sin2_t = sin2 / (eta * eta);
    if (sin2_t > 1.0) return 1.0;
    float cos_t = sqrt(1.0 - sin2_t);
    float a = (eta * cos_t - cos_theta) / (eta * cos_t + cos_theta);
    float b = (cos_theta - eta * cos_t) / (cos_theta + eta * cos_t);
    return 0.5 * (a * a + b * b);
}

// ── Hair attenuation (Ap lobes) ──
void hair_attenuation(float f, vec3 T, out vec3 Ap[4]) {
    vec3 oneMinusF = vec3(1.0 - f);
    Ap[0] = vec3(f);                                    // R
    Ap[1] = oneMinusF * oneMinusF * T;                  // TT
    Ap[2] = Ap[1] * T * f;                              // TRT
    vec3 Tprod = T * f;
    vec3 denom = max(vec3(1.0) - Tprod, vec3(1e-6));
    Ap[3] = Ap[1] * T * Tprod / denom;                 // TRRT+
}

// ── Main evaluation: Chiang Principled Hair BSDF ──
// wi = incoming ray direction (toward surface), wo = outgoing (toward light/bounce)
// tangent = hair fiber direction, sigma = absorption coefficient (RGB)
// roughness = longitudinal roughness [0,1], azimuthal_roughness = [0,1]
// alpha = cuticle tilt (radians), eta = IOR (1.55 default)
// h = offset across fiber [-1,1] (from cylindrical normal v parameter)
vec3 hair_chiang_eval(vec3 wi, vec3 wo, vec3 tangent,
                      vec3 sigma, float roughness, float azimuthal_roughness,
                      float alpha, float eta, float h) {

    // Remap roughness to internal variance (Cycles mapping)
    float v_r = roughness;
    float v = v_r * v_r;  // simplified: square mapping
    v = clamp(v, 0.001, 1.0);
    float s = clamp(azimuthal_roughness, 0.001, 1.0);
    s = (0.265 * s + 1.194 * s * s + 5.372 * pow(s, 22.0)) * sqrt(HAIR_PI / 8.0);
    float m0_roughness = clamp(0.5 * v, 0.001, 1.0);

    // Build local frame from tangent and incident direction
    vec3 X = normalize(tangent);
    vec3 Y = cross(X, wi);
    float yLen = length(Y);
    if (yLen < 1e-6) return vec3(0.0);
    Y /= yLen;
    vec3 Z = cross(X, Y);

    // Local coordinates
    float sin_theta_o = dot(wi, X);
    float cos_theta_o = sqrt(max(0.0, 1.0 - sin_theta_o * sin_theta_o));
    float phi_o = atan(dot(wi, Z), dot(wi, Y));

    float sin_theta_i = dot(wo, X);
    float cos_theta_i = sqrt(max(0.0, 1.0 - sin_theta_i * sin_theta_i));
    float phi_i = atan(dot(wo, Z), dot(wo, Y));
    float phi = phi_i - phi_o;

    // Refraction inside fiber
    float sin_theta_t = sin_theta_o / eta;
    float cos_theta_t = sqrt(max(0.0, 1.0 - sin_theta_t * sin_theta_t));

    // Azimuth geometry
    float sin_gamma_o = clamp(h, -1.0, 1.0);
    float cos_gamma_o = sqrt(max(0.0, 1.0 - sin_gamma_o * sin_gamma_o));
    float gamma_o = asin(sin_gamma_o);

    float eta2_sin2 = eta * eta - sin_theta_o * sin_theta_o;
    float sin_gamma_t = (eta2_sin2 > 0.0)
        ? sin_gamma_o * cos_theta_o / sqrt(eta2_sin2) : 0.0;
    sin_gamma_t = clamp(sin_gamma_t, -1.0, 1.0);
    float cos_gamma_t = sqrt(max(0.0, 1.0 - sin_gamma_t * sin_gamma_t));
    float gamma_t = asin(sin_gamma_t);

    // Transmittance through fiber
    float path = (cos_theta_t > 1e-6) ? 2.0 * cos_gamma_t / cos_theta_t : 0.0;
    vec3 T = exp(-sigma * path);

    // Fresnel and attenuation lobes
    float f = hair_fresnel(cos_theta_o * cos_gamma_o, eta);
    vec3 Ap[4];
    hair_attenuation(f, T, Ap);

    // Cuticle tilt angles
    float sin_1a = sin(alpha);
    float cos_1a = cos(alpha);
    float sin_2a = 2.0 * sin_1a * cos_1a;
    float cos_2a = cos_1a * cos_1a - sin_1a * sin_1a;
    float sin_4a = 2.0 * sin_2a * cos_2a;
    float cos_4a = cos_2a * cos_2a - sin_2a * sin_2a;

    // Tilted angles per lobe [sin_theta_o', cos_theta_o'] for R, TT, TRT
    float angles[6];
    angles[0] = sin_theta_o * cos_2a - cos_theta_o * sin_2a;   // R sin
    angles[1] = abs(cos_theta_o * cos_2a + sin_theta_o * sin_2a); // R cos
    angles[2] = sin_theta_o * cos_1a + cos_theta_o * sin_1a;   // TT sin
    angles[3] = abs(cos_theta_o * cos_1a - sin_theta_o * sin_1a); // TT cos
    angles[4] = sin_theta_o * cos_4a + cos_theta_o * sin_4a;   // TRT sin
    angles[5] = abs(cos_theta_o * cos_4a - sin_theta_o * sin_4a); // TRT cos

    // Sum R + TT + TRT
    vec3 F = vec3(0.0);
    float roughnesses[3];
    roughnesses[0] = m0_roughness;   // R: tighter
    roughnesses[1] = 0.25 * v;       // TT: sharper
    roughnesses[2] = 4.0 * v;        // TRT: broader

    for (int p = 0; p < 3; p++) {
        float Mp = hair_Mp(sin_theta_i, cos_theta_i,
                           angles[2*p], angles[2*p+1], roughnesses[p]);
        float Np = hair_Np(phi, p, s, gamma_o, gamma_t);
        F += Ap[p] * Mp * Np;
    }

    // Residual TRRT+ (uniform azimuthal distribution)
    float Mp_res = hair_Mp(sin_theta_i, cos_theta_i,
                           sin_theta_o, cos_theta_o, 4.0 * v);
    F += Ap[3] * Mp_res * HAIR_INV_2PI;

    // Absorption factor for cos_theta_i
    return max(F, vec3(0.0)) * abs(cos_theta_i);
}
