// ============================================================
// rtxpt_firefly.glsl — RTX Path Tracing v1.8.1 firefly filter port
//
// Reference: NVIDIA RTXPT PathTracer/PathTracerHelpers.hlsli:180-219.
// Per-path scalar `firefly_k` starts at 1.0 and shrinks after every BSDF
// scatter by a factor derived from the ray-cone spread angle implied by
// that scatter's pdf. Low-pdf (wide, near-diffuse) lobes collapse k;
// high-pdf (narrow) lobes preserve it. Downstream emission /
// sky / NEE contributions are capped at FIREFLY_THRESHOLD * k so single
// high-variance paths can't spike pixels — the ReSTIR reservoirs never
// lock in an unfiltered spike for later temporal reuse.
//
// Uses RGB average (not luminance) per the RTXPT source comment —
// luminance-based clamp causes a hue shift toward blue under clamp.
// ============================================================

#ifndef RTXPT_FIREFLY_GLSL
#define RTXPT_FIREFLY_GLSL

// Self-contained constants so the header can be included before common.glsl.
const float RTXPT_PI     = 3.141592653589793;
const float RTXPT_TWO_PI = 6.283185307179586;

// Firefly-filter contribution ceiling before the per-path k modulator.
// Renamed to RTXPT_FIREFLY_THRESHOLD so wf_output can keep its own
// distinct FIREFLY_THRESHOLD (50.0) without colliding.
const float RTXPT_FIREFLY_THRESHOLD = 8.0;

// Handbook polynomial acos — max error ~7e-3 rad. The firefly heuristic
// is already empirical, precision beyond this point is noise.
float rtxpt_fast_acos(float x) {
    float a = abs(x);
    float r = -0.0187293 * a + 0.0742610;
    r = r * a - 0.2121144;
    r = r * a + 1.5707288;
    r = r * sqrt(max(0.0, 1.0 - a));
    return (x >= 0.0) ? r : (RTXPT_PI - r);
}

// Plane-angle half-cone spread implied by a BSDF pdf. Solid-angle
// omega = 1/pdf, cap angle alpha = 2·acos(1 − omega / 2π).
float rtxpt_ray_cone_spread_by_pdf(float pdf) {
    if (pdf <= 0.0) return 0.0;
    return 2.0 * rtxpt_fast_acos(max(-1.0, 1.0 - (1.0 / pdf) / RTXPT_TWO_PI));
}

// Shrink firefly_k after a BSDF scatter with the given pdf + lobe weight.
// For single-lobe sampling (e.g. Lambertian cosine), pass lobe_p = 1.0.
// For multi-lobe sampling (diffuse/spec/transmission), pass the discrete
// probability of the selected lobe.
float rtxpt_update_firefly_k(float current_k, float bounce_pdf, float lobe_p) {
    float angle = rtxpt_ray_cone_spread_by_pdf(bounce_pdf);
    const float k_emp = 32.0;
    float p = k_emp / (k_emp + angle * angle);
    p *= sqrt(max(lobe_p, 0.0));
    return max(1e-5, current_k * p);
}

// Clamp an emission / sky / NEE contribution at threshold * k, scaled
// down uniformly to preserve hue. Returns the (possibly scaled) input.
vec3 rtxpt_firefly_filter(vec3 signal, float threshold, float k) {
    float t = threshold * k;
    float avg = (signal.x + signal.y + signal.z) * (1.0 / 3.0);
    if (avg > t) return signal * (t / avg);
    return signal;
}

#endif // RTXPT_FIREFLY_GLSL
