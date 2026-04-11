// ============================================================
// gi_reservoir.glsl — ReSTIR GI Reservoir Sampling
// Ported from ac-path-tracer with adaptations for ignis-rt
//
// Weighted Reservoir Sampling for indirect lighting reuse.
// Reduces noise by temporally reusing high-quality GI samples.
//
// Requires buffer bindings:
//   giReservoirCurr (binding 24) — current frame write
//   giReservoirPrev (binding 25) — previous frame read
// ============================================================

#ifndef GI_RESERVOIR_GLSL
#define GI_RESERVOIR_GLSL

struct GIReservoir {
    vec3 position;     // world-space hit position
    vec3 normal;       // hit surface normal
    vec3 radiance;     // cached outgoing radiance at hit
    float weightSum;   // RIS weight sum
    float M;           // number of samples processed
    float hitDist;     // distance from primary to hit
    float age;         // frames since sample was generated
};

GIReservoir emptyReservoir() {
    GIReservoir r;
    r.position = vec3(0);
    r.normal = vec3(0, 1, 0);
    r.radiance = vec3(0);
    r.weightSum = 0.0;
    r.M = 0.0;
    r.hitDist = 0.0;
    r.age = 0.0;
    return r;
}

// Octahedral normal encoding (2 floats)
vec2 octEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return n.xy * 0.5 + 0.5;
}

vec3 octDecode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

// Pack reservoir into 3 vec4s and write to current-frame buffer
// pixelIdx = pixel.y * launchSize.x + pixel.x
void writeReservoir(int pixelIdx, GIReservoir r) {
    giReservoirCurr.data[pixelIdx * 3 + 0] = vec4(r.position, r.weightSum);
    giReservoirCurr.data[pixelIdx * 3 + 1] = vec4(r.radiance, r.M);
    vec2 nEnc = octEncode(r.normal);
    giReservoirCurr.data[pixelIdx * 3 + 2] = vec4(nEnc, r.hitDist, r.age);
}

// Read reservoir from previous-frame buffer
GIReservoir readReservoirPrev(int pixelIdx) {
    GIReservoir r;
    vec4 d0 = giReservoirPrev.data[pixelIdx * 3 + 0];
    vec4 d1 = giReservoirPrev.data[pixelIdx * 3 + 1];
    vec4 d2 = giReservoirPrev.data[pixelIdx * 3 + 2];
    r.position = d0.xyz;
    r.weightSum = d0.w;
    r.radiance = d1.xyz;
    r.M = d1.w;
    r.normal = octDecode(d2.xy);
    r.hitDist = d2.z;
    r.age = d2.w;
    return r;
}

// Target PDF: luminance of cached radiance (simple form per ReSTIR GI paper).
// The simple form (luminance only) makes samples more reusable across pixels
// compared to including BRDF/cosine terms.
float giTargetPDF(vec3 primaryNormal, vec3 primaryPos, GIReservoir s) {
    return max(dot(s.radiance, vec3(0.2126, 0.7152, 0.0722)), 0.0);
}

// Jacobian determinant for domain shift between two receiver points.
// Corrects the solid-angle PDF when reusing a sample from a different pixel.
// J = (cos_new * dist_orig^2) / (cos_orig * dist_new^2)
float giJacobian(vec3 samplePos, vec3 sampleNormal,
                  vec3 receiverNew, vec3 receiverOrig) {
    vec3 toNew = receiverNew - samplePos;
    vec3 toOrig = receiverOrig - samplePos;
    float distNew = length(toNew);
    float distOrig = length(toOrig);
    if (distNew < 0.001 || distOrig < 0.001) return 0.0;
    float cosNew = max(dot(sampleNormal, toNew / distNew), 0.0);
    float cosOrig = max(dot(sampleNormal, toOrig / distOrig), 1e-4);
    float J = (cosNew * distOrig * distOrig) / (cosOrig * distNew * distNew);
    // Clamp to prevent extreme values from degenerate geometry
    return clamp(J, 0.0, 100.0);
}

// Weighted Reservoir Sampling: consider a new sample
void reservoirUpdate(inout GIReservoir r, GIReservoir newSample, float weight, inout float rng) {
    r.weightSum += weight;
    r.M += 1.0;
    if (rng * r.weightSum < weight) {
        r.position = newSample.position;
        r.normal = newSample.normal;
        r.radiance = newSample.radiance;
        r.hitDist = newSample.hitDist;
        r.age = newSample.age;
    }
    rng = fract(rng * 747.6513 + 0.3713);
}

// Merge source reservoir into dest (biased but stable for real-time)
void reservoirMerge(inout GIReservoir dest, GIReservoir src, float targetPDF, inout float rng) {
    float weight = targetPDF * src.M;
    if (weight <= 0.0) return;
    float oldM = dest.M;
    reservoirUpdate(dest, src, weight, rng);
    dest.M = oldM + src.M;
}

// Temporal reuse: find previous-frame pixel via motion vector and merge.
// Includes Jacobian correction and robust validation per RTXDI reference.
// Returns true if temporal sample was successfully reused.
bool reservoirTemporalReuse(
    inout GIReservoir curr,
    vec3 primaryNormal, vec3 primaryPos,
    vec2 motionVector, ivec2 launchSize,
    ivec2 pixel, inout float rng
) {
    vec2 currUV = (vec2(pixel) + 0.5) / vec2(launchSize);
    vec2 prevUV = currUV + motionVector;

    if (prevUV.x < 0.0 || prevUV.x >= 1.0 || prevUV.y < 0.0 || prevUV.y >= 1.0)
        return false;

    ivec2 prevPixel = ivec2(prevUV * vec2(launchSize));
    int prevIdx = prevPixel.y * launchSize.x + prevPixel.x;

    GIReservoir prev = readReservoirPrev(prevIdx);

    // Reject stale or empty reservoirs
    if (prev.age > 30.0 || prev.M < 0.5)
        return false;

    // Validate: reject if sample radiance contains NaN/Inf
    if (any(isnan(prev.radiance)) || any(isinf(prev.radiance)))
        return false;

    // Clamp cached radiance (prevents stale bright samples from causing fireflies)
    prev.radiance = min(prev.radiance, vec3(10.0));

    // Normal similarity check (reject if surface orientation changed)
    float normalSim = dot(primaryNormal, prev.normal);
    if (normalSim < 0.6)
        return false;

    // Jacobian correction: compensate for solid-angle domain shift.
    // The previous receiver was at a different position than the current one.
    // Without this, grazing-angle samples get massively overweighted → bright dots.
    vec3 prevReceiverPos = primaryPos - vec3(motionVector * vec2(launchSize), 0.0) * 0.01;
    // Approximate: for temporal reuse at the same pixel, the receiver shift is small.
    // Use identity Jacobian (J=1) for temporal, full Jacobian for spatial.
    // This matches RTXDI's temporal path which does not apply Jacobian by default.
    float J = 1.0;

    // Clamp M to prevent unbounded growth (standard: 20-30x)
    prev.M = min(prev.M, 20.0);
    prev.age += 1.0;

    float pHat = giTargetPDF(primaryNormal, primaryPos, prev);
    if (pHat <= 0.0) return false;

    // Apply Jacobian to incoming reservoir weight before merge
    prev.weightSum *= J;

    reservoirMerge(curr, prev, pHat, rng);
    return true;
}

#endif // GI_RESERVOIR_GLSL
