// ============================================================
// pt_reservoir.glsl — ReSTIR PT Reservoir + Path Record
//
// Reservoir-based path resampling for indirect lighting.
// Reference: GRIS (Lin et al. 2022), NVIDIA RTXDI SDK (MIT)
//
// Usage: include this file, then declare buffers in each shader:
//   layout(binding=15, set=1, std430) buffer PTResCurrBuf { vec4 data[]; } ptResCurr;
//   layout(binding=16, set=1, std430) buffer PTResPrevBuf { vec4 data[]; } ptResPrev;
//   layout(binding=17, set=1, std430) buffer PTPathBuf   { vec4 data[]; } ptPathRec;
// ============================================================

#ifndef PT_RESERVOIR_GLSL
#define PT_RESERVOIR_GLSL

// ── Structs ──

struct PTReservoir {
    vec3  rcVertexPos;         // reconnection vertex world position
    vec3  rcVertexNormal;      // reconnection vertex normal
    vec3  rcRadiance;          // suffix radiance (from rcVertex onwards)
    float weightSum;           // RIS weight sum (becomes W after finalize)
    float M;                   // sample count
    float targetFunction;      // p_hat of selected sample
    uint  rcBounceDepth;       // bounce depth of reconnection (2 = first indirect)
    uint  rngSeed;             // RNG seed for random replay
    uint  lightType;           // 0=emission, 1=NEE, 2=envmap
    uint  lightIndex;          // light index for NEE replay
    vec3  primaryPos;          // primary hit position (for shift validation)
    vec3  primaryNormal;       // primary hit normal
    float primaryRoughness;    // roughness at primary
    float age;                 // frames since sample created
    float partialJacobian;     // cached 1/(cos_old/dist_old^2) for Jacobian computation
    float rcWiPdf;             // BRDF PDF at rcVertex (for invertibility validation)
    uint  pathLength;          // total path bounces (camera to light)
};

struct PTPathRecord {
    vec3  primaryPos;
    vec3  primaryNormal;
    float primaryRoughness;
    uint  primaryInstanceId;
    vec3  rcPos;
    vec3  rcNormal;
    vec3  rcRadiance;
    float rcDist;
    uint  rcBounce;       // 0 = no reconnection found
    bool  rcValid;
    uint  rngSeed;
    uint  lightType;
    uint  lightIndex;
    float pathRadianceLum;
};

// ── Helpers ──

PTReservoir ptEmptyReservoir() {
    PTReservoir r;
    r.rcVertexPos = vec3(0); r.rcVertexNormal = vec3(0,1,0); r.rcRadiance = vec3(0);
    r.weightSum = 0.0; r.M = 0.0; r.targetFunction = 0.0;
    r.rcBounceDepth = 0u; r.rngSeed = 0u; r.lightType = 0u; r.lightIndex = 0u;
    r.primaryPos = vec3(0); r.primaryNormal = vec3(0,1,0); r.primaryRoughness = 1.0;
    r.age = 0.0; r.partialJacobian = 0.0; r.rcWiPdf = 0.0; r.pathLength = 0u;
    return r;
}

vec2 ptOctEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return n.xy * 0.5 + 0.5;
}

vec3 ptOctDecode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

// Pack/unpack is done inline in each shader (GLSL doesn't support
// unsized array parameters). See wf_pt_temporal.comp for reference.

// ── Algorithm functions ──

float ptTargetPDF(PTReservoir r) {
    return max(dot(r.rcRadiance, vec3(0.2126, 0.7152, 0.0722)), 0.0);
}

PTReservoir ptCreateFromPathRecord(PTPathRecord pr, float cachedPartialJ, float rcPdf) {
    PTReservoir r = ptEmptyReservoir();
    if (!pr.rcValid) return r;
    r.rcVertexPos = pr.rcPos; r.rcVertexNormal = pr.rcNormal;
    r.rcRadiance = min(pr.rcRadiance, vec3(20.0));
    r.rcBounceDepth = pr.rcBounce; r.rngSeed = pr.rngSeed;
    r.lightType = pr.lightType; r.lightIndex = pr.lightIndex;
    r.primaryPos = pr.primaryPos; r.primaryNormal = pr.primaryNormal;
    r.primaryRoughness = pr.primaryRoughness; r.age = 0.0;
    r.partialJacobian = cachedPartialJ;
    r.rcWiPdf = rcPdf;
    r.pathLength = pr.rcBounce + 1u;
    float pHat = ptTargetPDF(r);
    if (pHat > 0.001) { r.weightSum = 1.0; r.M = 1.0; r.targetFunction = pHat; }
    return r;
}

bool ptReservoirUpdate(inout PTReservoir r, PTReservoir cand, float weight, inout float rng) {
    r.weightSum += weight;
    r.M += cand.M;
    bool sel = (rng * r.weightSum < weight);
    if (sel) {
        r.rcVertexPos = cand.rcVertexPos; r.rcVertexNormal = cand.rcVertexNormal;
        r.rcRadiance = cand.rcRadiance; r.rcBounceDepth = cand.rcBounceDepth;
        r.rngSeed = cand.rngSeed; r.lightType = cand.lightType; r.lightIndex = cand.lightIndex;
        r.primaryPos = cand.primaryPos; r.primaryNormal = cand.primaryNormal;
        r.primaryRoughness = cand.primaryRoughness; r.targetFunction = cand.targetFunction;
        r.age = cand.age; r.partialJacobian = cand.partialJacobian;
        r.rcWiPdf = cand.rcWiPdf; r.pathLength = cand.pathLength;
    }
    rng = fract(rng * 747.6513 + 0.3713);
    return sel;
}

void ptReservoirMerge(inout PTReservoir dest, PTReservoir src, float tgtPDF, inout float rng) {
    float w = tgtPDF * src.weightSum * src.M;
    if (w <= 0.0 || isnan(w) || isinf(w)) return;
    src.targetFunction = tgtPDF;
    float oldM = dest.M;
    ptReservoirUpdate(dest, src, w, rng);
    dest.M = oldM + src.M;
}

void ptFinalizeReservoir(inout PTReservoir r) {
    float pHat = ptTargetPDF(r);
    r.weightSum = (pHat > 0.001 && r.M > 0.0) ? min(r.weightSum / (r.M * pHat), 10.0) : 0.0;
    r.targetFunction = pHat;
}

bool ptSurfaceSimilar(vec3 n1, vec3 n2, float r1, float r2, vec3 p1, vec3 p2) {
    if (dot(n1, n2) < 0.85) return false;
    float d = length(p1 - p2);
    if (d / max(length(p1), 0.1) > 0.15) return false;
    if (abs(r1 - r2) > 0.3) return false;
    return true;
}

// ── Partial Jacobian computation (per RTXDI CalculatePartialJacobian) ──
// Returns cos(theta) / dist^2 from sample to receiver
float ptPartialJacobian(vec3 rcPos, vec3 rcNormal, vec3 receiverPos) {
    vec3 v = receiverPos - rcPos;
    float distSq = dot(v, v);
    if (distSq < 1e-8) return 0.0;
    float cosine = max(dot(rcNormal, v * inversesqrt(distSq)), 0.0);
    return cosine / distSq;
}

// ── Jacobian using cached partial (per RTXDI CalculateJacobianWithCachedJacobian) ──
// reservoir.partialJacobian stores 1/(cos_old/dist_old^2) from initial sampling.
// Returns newPartial * oldInversePartial = (cos_new/dist_new^2) * (dist_old^2/cos_old)
float ptJacobianCached(vec3 receiverNew, inout PTReservoir r) {
    float newPartial = ptPartialJacobian(r.rcVertexPos, r.rcVertexNormal, receiverNew);
    float J = newPartial * r.partialJacobian;  // partialJacobian = 1/oldPartial
    // Update cached value for next reuse (daisy chain)
    if (newPartial > 1e-8) r.partialJacobian = 1.0 / newPartial;
    return (isnan(J) || isinf(J)) ? 0.0 : clamp(J, 0.0, 100.0);
}

// ── Legacy Jacobian (when no cached partial available) ──
float ptJacobian(vec3 rcPos, vec3 rcN, vec3 recvNew, vec3 recvOrig) {
    float pNew = ptPartialJacobian(rcPos, rcN, recvNew);
    float pOrig = ptPartialJacobian(rcPos, rcN, recvOrig);
    if (pOrig < 1e-8) return 0.0;
    float J = pNew / pOrig;
    return (isnan(J) || isinf(J)) ? 0.0 : clamp(J, 0.0, 100.0);
}

// ── Invertibility validation (per RTXDI ValidateInvertibilityCondition) ──
// Checks that the shift would also be valid in the reverse direction.
// Without this, pairwise MIS cannot be unbiased.
bool ptValidateInvertibility(vec3 shiftedSurfacePos, vec3 shiftedSurfaceNormal,
                              float shiftedRoughness, PTReservoir r) {
    // Both surfaces must be rough enough
    if (shiftedRoughness < 0.25) return false;
    // Distance between shifted surface and rcVertex must exceed threshold
    float dist = length(r.rcVertexPos - shiftedSurfacePos);
    if (dist < 0.1) return false;
    // Normal of shifted surface must face the reconnection vertex
    vec3 toRC = normalize(r.rcVertexPos - shiftedSurfacePos);
    if (dot(shiftedSurfaceNormal, toRC) < 0.01) return false;
    return true;
}

// ── Reconnection vertex criteria ──
bool ptIsValidReconnection(float rPrev, float rCurr, vec3 pPrev, vec3 pCurr) {
    return rPrev >= 0.25 && rCurr >= 0.25 && length(pCurr - pPrev) > 0.1;
}

#endif
