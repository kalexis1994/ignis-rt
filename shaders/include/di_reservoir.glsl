// ============================================================
// di_reservoir.glsl — ReSTIR DI (Direct Illumination) Reservoir
// Weighted Reservoir Sampling for efficient light selection.
// Reference: "Spatiotemporal Reservoir Resampling for Real-Time
//            Ray Tracing with Dynamic Direct Lighting" (Bitterli et al., 2020)
// ============================================================

#ifndef DI_RESERVOIR_GLSL
#define DI_RESERVOIR_GLSL

struct DIReservoir {
    uint  lightIndex;     // light index (0..lightCount-1 for local, 0xFFFFu for emissive)
    vec3  lightSamplePos; // sampled position on the light surface
    float targetPDF;      // p-hat: unshadowed contribution luminance at this pixel
    float weightSum;      // RIS running weight sum (W_sum)
    float M;              // number of candidates processed
    float age;            // frames since this sample was created
};

DIReservoir emptyDIReservoir() {
    DIReservoir r;
    r.lightIndex = 0xFFFFFFFFu;
    r.lightSamplePos = vec3(0);
    r.targetPDF = 0.0;
    r.weightSum = 0.0;
    r.M = 0.0;
    r.age = 0.0;
    return r;
}

// Weighted Reservoir Sampling update: possibly accept a new candidate
bool diReservoirUpdate(inout DIReservoir r, uint lightIdx, vec3 samplePos,
                       float targetPDF, float sourcePDF, inout uint rng) {
    float weight = (sourcePDF > 0.0) ? targetPDF / sourcePDF : 0.0;
    r.weightSum += weight;
    r.M += 1.0;

    // Accept with probability weight / weightSum
    rng ^= rng << 13u; rng ^= rng >> 17u; rng ^= rng << 5u;
    float xi = float(rng) / 4294967295.0;

    if (xi * r.weightSum < weight) {
        r.lightIndex = lightIdx;
        r.lightSamplePos = samplePos;
        r.targetPDF = targetPDF;
        return true;
    }
    return false;
}

// Merge another reservoir into this one (for temporal/spatial reuse)
void diReservoirMerge(inout DIReservoir r, DIReservoir other, float otherTargetPDF,
                      inout uint rng) {
    // Treat the other reservoir as M candidates with combined weight
    float otherWeight = otherTargetPDF * other.weightSum * other.M;
    if (other.targetPDF > 0.0) {
        otherWeight = otherTargetPDF / other.targetPDF * other.weightSum;
    } else {
        otherWeight = 0.0;
    }

    float prevWeightSum = r.weightSum;
    r.weightSum += otherWeight;
    r.M += other.M;

    // Accept other sample with probability otherWeight / new weightSum
    rng ^= rng << 13u; rng ^= rng >> 17u; rng ^= rng << 5u;
    float xi = float(rng) / 4294967295.0;

    if (r.weightSum > 0.0 && xi * r.weightSum < otherWeight) {
        r.lightIndex = other.lightIndex;
        r.lightSamplePos = other.lightSamplePos;
        r.targetPDF = otherTargetPDF;
    }
}

// Compute final RIS weight for shading (clamped to prevent fireflies)
float diReservoirWeight(DIReservoir r) {
    if (r.targetPDF <= 0.0 || r.M <= 0.0) return 0.0;
    float W = r.weightSum / (r.M * r.targetPDF);
    return min(W, 5.0);  // clamp to prevent firefly explosions
}

// Pack/unpack functions use raw vec4 values (caller handles SSBO access)
vec4 diPackV0(DIReservoir r) {
    return vec4(uintBitsToFloat(r.lightIndex), r.lightSamplePos);
}
vec4 diPackV1(DIReservoir r) {
    return vec4(r.targetPDF, r.weightSum, r.M, r.age);
}

DIReservoir diUnpack(vec4 d0, vec4 d1) {
    DIReservoir r;
    r.lightIndex = floatBitsToUint(d0.x);
    r.lightSamplePos = d0.yzw;
    r.targetPDF = d1.x;
    r.weightSum = d1.y;
    r.M = d1.z;
    r.age = d1.w;
    return r;
}

// Compute target PDF (p-hat): luminance of unshadowed contribution
// This is evaluated WITHOUT shadow ray — just BRDF * lightRadiance * geometric term
float diTargetPDF(vec3 N, vec3 V, vec3 lightDir, float NdotL,
                  vec3 baseColor, float roughness, float metallic,
                  vec3 lightRadiance) {
    if (NdotL <= 0.0) return 0.0;
    // Use luminance of full BRDF contribution as target PDF
    vec3 diffC, specC;
    evaluateCookTorrance(N, V, lightDir, baseColor, roughness, metallic, 0.5, 1.45,
                         diffC, specC);
    vec3 contribution = (diffC + specC) * lightRadiance * NdotL;
    return dot(contribution, vec3(0.2126, 0.7152, 0.0722)); // luminance
}

#endif // DI_RESERVOIR_GLSL
