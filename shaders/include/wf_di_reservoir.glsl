// ============================================================
// wf_di_reservoir.glsl — ReSTIR DI reservoir (polymorphic lights)
//
// RTXPT-aligned DI reservoir. Carries a `lightUV` field so
// TRIANGLE / AREA / ENVIRONMENT lights can be resampled by the
// same pipeline that handles POINT / SPOT / DIRECTIONAL. Light
// data lives in the polymorphic `polyLights` buffer at
// set=0 binding=52 (see wf_lights.glsl and ignis_lights.h for
// the 96 B struct).
//
// Storage layout — 32 bytes (2 × vec4) per pixel:
//   [0] = (uintBitsToFloat(lightIdx), lightUV.x, lightUV.y, M)
//   [1] = (W, targetPdf, uintBitsToFloat(flags), uintBitsToFloat(stableId))
// `wSum` is NOT persisted; it is scratch during a single combine
// pass and gets folded into `W` by `wf_di_reservoir_finalize`.
//
// flags bit layout:
//   bit 0 = visibility tested this frame
//   bit 1 = occluded (only meaningful when bit 0 is set)
// Reading occlusion when bit 0 is clear is undefined — callers
// should treat "untested" as "assume visible".
// ============================================================

#ifndef WF_DI_RESERVOIR_GLSL
#define WF_DI_RESERVOIR_GLSL

const uint WF_DI_INVALID_LIGHT = 0xFFFFFFFFu;

// Flag bits for WFDIReservoir.flags
const uint WF_DI_FLAG_VIS_TESTED = 1u << 0;
const uint WF_DI_FLAG_OCCLUDED   = 1u << 1;

struct WFDIReservoir {
    uint  lightIdx;    // index into polyLights; WF_DI_INVALID_LIGHT = empty
    vec2  lightUV;     // sample parameters (triangle barycentric, area uv, ...)
    float M;           // effective sample count (float for clamp)
    float W;           // finalized RIS weight = wSum / (M · targetPdf_selected)
    float targetPdf;   // target pdf at the selected sample
    uint  flags;       // see WF_DI_FLAG_*
    uint  stableId;    // stable light identity used to reject remapped indices
    float wSum;        // scratch during combine; NOT persisted
};

WFDIReservoir wf_di_reservoir_empty() {
    WFDIReservoir r;
    r.lightIdx  = WF_DI_INVALID_LIGHT;
    r.lightUV   = vec2(0.0);
    r.M         = 0.0;
    r.W         = 0.0;
    r.targetPdf = 0.0;
    r.flags     = 0u;
    r.stableId  = 0u;
    r.wSum      = 0.0;
    return r;
}

bool wf_di_reservoir_is_valid(WFDIReservoir r) {
    return r.lightIdx != WF_DI_INVALID_LIGHT && r.M > 0.0;
}

// Streaming RIS update (Bitterli 2020 Alg. 3). `wi = targetPdf / srcPdf`
// — the caller precomputes this because srcPdf depends on how the
// candidate was drawn (uniform over N, CDF-based, ReGIR tile sampled,
// BSDF-proportional, etc.). If `wi <= 0` the candidate is rejected but
// M is still incremented (a zero-contribution draw still counts as a
// drawn sample for RIS normalization).
bool wf_di_reservoir_update(inout WFDIReservoir r,
                              uint  lightIdx,
                              uint  stableId,
                              vec2  lightUV,
                              float targetPdf,
                              float wi,
                              float rnd) {
    r.wSum += wi;
    r.M    += 1.0;
    bool take = (wi > 0.0) && (rnd * r.wSum < wi);
    if (take) {
        r.lightIdx  = lightIdx;
        r.stableId  = stableId;
        r.lightUV   = lightUV;
        r.targetPdf = targetPdf;
        // flags carry over from the caller; update-time flags (e.g.
        // visibility tested) are decided at finalize or by a follow-up
        // store, not here.
    }
    return take;
}

// Merge another reservoir into this one using the biased-combination
// rule (ReSTIR 2020 Alg. 6). `otherTargetPdfAtThis` must be evaluated
// at THIS pixel's surface — the resampling weight for the other
// reservoir's selected sample under our target function is
//     w_other = otherTargetPdfAtThis · other.W · other.M
// Cheaper unbiased variants (MIS, Talbot 2005) are introduced in
// Phase 5c once spatial + temporal reuse lands.
bool wf_di_reservoir_combine(inout WFDIReservoir r,
                               WFDIReservoir other,
                               float otherTargetPdfAtThis,
                               float rnd) {
    if (other.lightIdx == WF_DI_INVALID_LIGHT || other.M <= 0.0) {
        return false;
    }
    float wOther = otherTargetPdfAtThis * other.W * other.M;
    r.wSum += wOther;
    r.M    += other.M;
    bool take = (wOther > 0.0) && (rnd * r.wSum < wOther);
    if (take) {
        r.lightIdx  = other.lightIdx;
        r.stableId  = other.stableId;
        r.lightUV   = other.lightUV;
        r.targetPdf = otherTargetPdfAtThis;
        r.flags     = other.flags;
    }
    return take;
}

// Close the reservoir: pack the RIS normalization constant into W.
// Call AFTER all update/combine calls for this pixel. If the reservoir
// is empty or targetPdf is zero (degenerate draws, back-facing lights,
// etc.) W collapses to 0 so the contribution drops out cleanly.
void wf_di_reservoir_finalize(inout WFDIReservoir r) {
    if (r.M > 0.0 && r.targetPdf > 0.0) {
        r.W = r.wSum / (r.M * r.targetPdf);
    } else {
        r.W = 0.0;
    }
}

// ── Pack / Unpack ──
// Persistent 32 B layout. wSum is dropped (not needed across passes).

void wf_di_reservoir_pack(WFDIReservoir r, out vec4 v0, out vec4 v1) {
    v0 = vec4(uintBitsToFloat(r.lightIdx), r.lightUV.x, r.lightUV.y, r.M);
    v1 = vec4(r.W, r.targetPdf, uintBitsToFloat(r.flags), uintBitsToFloat(r.stableId));
}

WFDIReservoir wf_di_reservoir_unpack(vec4 v0, vec4 v1) {
    WFDIReservoir r;
    r.lightIdx  = floatBitsToUint(v0.x);
    r.lightUV   = v0.yz;
    r.M         = v0.w;
    r.W         = v1.x;
    r.targetPdf = v1.y;
    r.flags     = floatBitsToUint(v1.z);
    r.stableId  = floatBitsToUint(v1.w);
    r.wSum      = 0.0;
    return r;
}

// ── Tunables (RTXPT-aligned defaults: NVIDIA RtxdiApplicationSettings.cpp,
// ReSTIRDITemporalResamplingParams / SpatialResamplingParams) ──

// Initial-sample count per pixel. RTXPT `numPrimaryLocalLightSamples`
// default is 8 at "Medium" quality, 16 at "High". Start at 8 — cheap
// and enough for the boiling filter + temporal reuse to denoise.
const uint  WF_DI_M_INITIAL          = 8u;

// Temporal M clamp — cap inherited history so a single hot pixel
// doesn't lock in stale samples. RTXPT `maxHistoryLength` = 20 for DI.
const float WF_DI_M_CLAMP            = 20.0;

// Surface similarity thresholds for both temporal reprojection and
// spatial neighbour acceptance. RTXPT defaults are looser
// (temporalNormalThreshold = 0.5, temporalDepthThreshold = 0.1);
// ours keep RTXPT's depth but tighten the normal gate to ~25° because
// our pairwise-MIS is the biased variant (no Raytraced ray-visibility
// bias correction yet), so a looser normal gate leaks more noise than
// it's worth.
const float WF_DI_NORMAL_VALID_COS   = 0.906;
const float WF_DI_DEPTH_VALID_RELERR = 0.10;

// Spatial kernel. Base K = 2 (RTXPT steady-state numSpatialSamples is
// 1; our +1 buys noise reduction at modest extra ray cost). At the
// disocclusion branch we bump to RTXPT's numDisocclusionBoostSamples
// = 8 — see the boost logic in wf_di_spatial.
const uint  WF_DI_SPATIAL_K_BOOST    = 8u;
const float WF_DI_SPATIAL_RADIUS     = 32.0;

// Material-similarity gates for spatial reuse. Matches the spirit of
// RTXPT's RAB_AreMaterialsSimilar check — reject neighbours whose
// BRDF shape differs enough to produce a hard RIS-weight step at our
// target pdf even after the normal / depth / albedo gates accepted
// them. Absolute diffs because roughness / metallic / specularLevel
// all live in [0, 1].
const float WF_DI_ROUGHNESS_MAX_DIFF = 0.25;
const float WF_DI_METALLIC_MAX_DIFF  = 0.30;
const float WF_DI_SPECULAR_MAX_DIFF  = 0.30;

// RTXDI boiling filter strength for the DI temporal pass.
// RTXPT default for DI is 0.2 → multiplier = 10/0.2 − 9 = 41, i.e.
// any reservoir whose targetPdf · W exceeds 41× the tile non-zero
// mean gets reset. Less aggressive than the GI filter (0.35 → ~19.6)
// because DI reservoirs carry shadowed contribution and outliers
// tend to be rarer but larger.
const float WF_DI_BOILING_STRENGTH   = 0.2;

// Firefly clamp at the shading consumer. Matches the legacy inline
// DI path's 200/channel cap — rare hot light picks can't propagate
// through the denoiser.
const float WF_DI_FIREFLY_CLAMP      = 200.0;

#endif // WF_DI_RESERVOIR_GLSL
