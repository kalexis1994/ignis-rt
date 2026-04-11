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
    vec3  rcVertexPos;
    vec3  rcVertexNormal;
    vec3  rcRadiance;
    float weightSum;
    float M;
    float targetFunction;
    uint  rcBounceDepth;
    uint  rngSeed;
    uint  lightType;     // 0=emission, 1=NEE, 2=envmap
    uint  lightIndex;
    vec3  primaryPos;
    vec3  primaryNormal;
    float primaryRoughness;
    float age;
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
    r.age = 0.0;
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

// ── Pack/Unpack (operate on raw vec4[] arrays) ──
// Reservoir: 8 vec4s = 128 bytes per pixel

void ptPackReservoir(int pixelIdx, PTReservoir r, inout vec4 buf[]) {
    int b = pixelIdx * 8;
    vec2 rcN = ptOctEncode(r.rcVertexNormal);
    vec2 pN  = ptOctEncode(r.primaryNormal);
    buf[b+0] = vec4(r.rcVertexPos, r.weightSum);
    buf[b+1] = vec4(r.rcRadiance, r.M);
    buf[b+2] = vec4(rcN, r.targetFunction, uintBitsToFloat(r.rcBounceDepth));
    buf[b+3] = vec4(uintBitsToFloat(r.rngSeed), uintBitsToFloat(r.lightType), uintBitsToFloat(r.lightIndex), r.age);
    buf[b+4] = vec4(r.primaryPos, r.primaryRoughness);
    buf[b+5] = vec4(pN, 0.0, 0.0);
    buf[b+6] = vec4(0); buf[b+7] = vec4(0);
}

PTReservoir ptUnpackReservoir(int pixelIdx, vec4 buf[]) {
    PTReservoir r;
    int b = pixelIdx * 8;
    r.rcVertexPos = buf[b+0].xyz; r.weightSum = buf[b+0].w;
    r.rcRadiance = buf[b+1].xyz; r.M = buf[b+1].w;
    r.rcVertexNormal = ptOctDecode(buf[b+2].xy); r.targetFunction = buf[b+2].z;
    r.rcBounceDepth = floatBitsToUint(buf[b+2].w);
    r.rngSeed = floatBitsToUint(buf[b+3].x); r.lightType = floatBitsToUint(buf[b+3].y);
    r.lightIndex = floatBitsToUint(buf[b+3].z); r.age = buf[b+3].w;
    r.primaryPos = buf[b+4].xyz; r.primaryRoughness = buf[b+4].w;
    r.primaryNormal = ptOctDecode(buf[b+5].xy);
    return r;
}

// Path record: 6 vec4s = 96 bytes per pixel

void ptPackPathRecord(int pixelIdx, PTPathRecord pr, inout vec4 buf[]) {
    int b = pixelIdx * 6;
    buf[b+0] = vec4(pr.primaryPos, pr.primaryRoughness);
    buf[b+1] = vec4(ptOctEncode(pr.primaryNormal), uintBitsToFloat(pr.primaryInstanceId), uintBitsToFloat(pr.rngSeed));
    buf[b+2] = vec4(pr.rcPos, pr.rcDist);
    buf[b+3] = vec4(pr.rcRadiance, pr.pathRadianceLum);
    buf[b+4] = vec4(ptOctEncode(pr.rcNormal), uintBitsToFloat(pr.rcBounce), pr.rcValid ? 1.0 : 0.0);
    buf[b+5] = vec4(uintBitsToFloat(pr.lightType), uintBitsToFloat(pr.lightIndex), 0.0, 0.0);
}

PTPathRecord ptUnpackPathRecord(int pixelIdx, vec4 buf[]) {
    PTPathRecord pr;
    int b = pixelIdx * 6;
    pr.primaryPos = buf[b+0].xyz; pr.primaryRoughness = buf[b+0].w;
    pr.primaryNormal = ptOctDecode(buf[b+1].xy);
    pr.primaryInstanceId = floatBitsToUint(buf[b+1].z); pr.rngSeed = floatBitsToUint(buf[b+1].w);
    pr.rcPos = buf[b+2].xyz; pr.rcDist = buf[b+2].w;
    pr.rcRadiance = buf[b+3].xyz; pr.pathRadianceLum = buf[b+3].w;
    pr.rcNormal = ptOctDecode(buf[b+4].xy);
    pr.rcBounce = floatBitsToUint(buf[b+4].z); pr.rcValid = (buf[b+4].w > 0.5);
    pr.lightType = floatBitsToUint(buf[b+5].x); pr.lightIndex = floatBitsToUint(buf[b+5].y);
    return pr;
}

// ── Algorithm functions ──

float ptTargetPDF(PTReservoir r) {
    return max(dot(r.rcRadiance, vec3(0.2126, 0.7152, 0.0722)), 0.0);
}

PTReservoir ptCreateFromPathRecord(PTPathRecord pr) {
    PTReservoir r = ptEmptyReservoir();
    if (!pr.rcValid) return r;
    r.rcVertexPos = pr.rcPos; r.rcVertexNormal = pr.rcNormal;
    r.rcRadiance = min(pr.rcRadiance, vec3(20.0));
    r.rcBounceDepth = pr.rcBounce; r.rngSeed = pr.rngSeed;
    r.lightType = pr.lightType; r.lightIndex = pr.lightIndex;
    r.primaryPos = pr.primaryPos; r.primaryNormal = pr.primaryNormal;
    r.primaryRoughness = pr.primaryRoughness; r.age = 0.0;
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
        r.age = cand.age;
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

float ptJacobian(vec3 rcPos, vec3 rcN, vec3 recvNew, vec3 recvOrig) {
    vec3 tN = recvNew - rcPos, tO = recvOrig - rcPos;
    float dN = length(tN), dO = length(tO);
    if (dN < 0.001 || dO < 0.001) return 0.0;
    float cN = max(dot(rcN, tN/dN), 0.0), cO = max(dot(rcN, tO/dO), 1e-4);
    float J = (cN * dO*dO) / (cO * dN*dN);
    return (isnan(J) || isinf(J)) ? 0.0 : clamp(J, 0.0, 100.0);
}

bool ptIsValidReconnection(float rPrev, float rCurr, vec3 pPrev, vec3 pCurr) {
    return rPrev >= 0.25 && rCurr >= 0.25 && length(pCurr - pPrev) > 0.1;
}

#endif
