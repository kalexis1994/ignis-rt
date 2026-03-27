// hair_intersect.glsl — Ray-cylinder intersection for procedural hair curves
// Used by the raygen shader to handle AABB candidates from hair BLAS geometry.
// Each AABB primitive = one curve segment (between consecutive subdivision keys).

// Curve keys buffer: float4 per key (xyz = position, w = radius)
layout(buffer_reference, scalar) readonly buffer CurveKeys { vec4 k[]; };

// Per-ray hair hit state (stored in local vars, used after ray query loop)
struct HairHitInfo {
    float t;
    vec3  normal;
    vec3  tangent;
    float u;       // along strand [0,1]
    float v;       // across ribbon [-1,1] (for shading)
    uint  strandId;
};

// PCG hash for per-strand random
uint hairPcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Ray-cylinder intersection for one hair segment.
// A, B = segment endpoints; R = radius; rayO, rayD = ray origin/direction
// Returns true if hit, fills tHit, hitNormal, hitParam (0..1 along segment)
bool intersectCylinder(vec3 rayO, vec3 rayD, vec3 A, vec3 B, float R,
                       float tMin, float tMax,
                       out float tHit, out vec3 hitNormal, out float hitParam)
{
    vec3 AB = B - A;
    float ABlen2 = dot(AB, AB);
    if (ABlen2 < 1e-6) return false;

    float invABlen2 = 1.0 / ABlen2;
    vec3 AO = rayO - A;
    float dAB = dot(rayD, AB);
    float oAB = dot(AO, AB);

    vec3 dPerp = rayD - (dAB * invABlen2) * AB;
    vec3 oPerp = AO  - (oAB * invABlen2) * AB;

    float qa = dot(dPerp, dPerp);
    float qb = 2.0 * dot(dPerp, oPerp);
    float qc = dot(oPerp, oPerp) - R * R;

    if (qa < 1e-8) return false;

    float disc = qb * qb - 4.0 * qa * qc;
    if (disc < 0.0) return false;

    float sqrtDisc = sqrt(disc);
    float invQa2 = 1.0 / (2.0 * qa);

    for (int i = 0; i < 2; i++) {
        float t = (i == 0) ? (-qb - sqrtDisc) * invQa2
                           : (-qb + sqrtDisc) * invQa2;
        if (isnan(t) || isinf(t)) continue;
        if (t < tMin || t > tMax) continue;

        vec3 hitPt = rayO + t * rayD;
        float s = dot(hitPt - A, AB) * invABlen2;
        if (s < 0.0 || s > 1.0) continue;

        vec3 axisPoint = A + s * AB;
        vec3 rawN = hitPt - axisPoint;
        float rawNLen = length(rawN);
        if (rawNLen < 1e-8) continue;

        hitNormal = rawN / rawNLen;
        tHit = t;
        hitParam = s;
        return true;
    }
    return false;
}

// Main entry: test ray against a hair curve segment (identified by AABB primitiveID).
// geo = GeometryMetadata for this BLAS; primId = AABB index within the BLAS.
// vertexBufferAddress → CurveKeys buffer (float4 per key)
// indexCount → subdivided keys per strand (S)
bool intersectHairSegment(vec3 rayO, vec3 rayD, float tMin, float tMax,
                          GeometryMetadata geo, int primId,
                          inout HairHitInfo hit)
{
    uint keysPerStrand = geo.indexCount;
    if (keysPerStrand < 2u) return false;
    uint segsPerStrand = keysPerStrand - 1u;

    uint strandIdx = uint(primId) / segsPerStrand;
    uint segIdx    = uint(primId) % segsPerStrand;
    uint keyIdx    = strandIdx * keysPerStrand + segIdx;

    CurveKeys keys = CurveKeys(geo.vertexBufferAddress);
    vec4 k0 = keys.k[keyIdx];
    vec4 k1 = keys.k[keyIdx + 1u];

    vec3 A = k0.xyz;
    vec3 B = k1.xyz;
    float R = max(k0.w, k1.w);

    if (any(isnan(A)) || any(isnan(B)) || isnan(R) || R <= 0.0) return false;

    float tHit;
    vec3 hitN;
    float hitS;
    if (!intersectCylinder(rayO, rayD, A, B, R, tMin, tMax, tHit, hitN, hitS))
        return false;

    if (isnan(tHit) || any(isnan(hitN))) return false;

    // Tangent along the curve at this segment
    vec3 tang = normalize(B - A);

    // Ribbon-like smooth normal (Cycles approach):
    // Compute v = position across ribbon [-1,1] for rounded appearance
    vec3 bitangent = cross(tang, -rayD);
    float btLen = length(bitangent);
    if (btLen > 1e-6) bitangent /= btLen;
    else bitangent = hitN;

    float sine = dot(hitN, bitangent);
    sine = clamp(sine, -1.0, 1.0);
    float cosine = sqrt(max(0.0, 1.0 - sine * sine));

    // Smooth normal: interpolate between radial and view-facing
    vec3 smoothN = normalize(sine * bitangent - cosine * cross(tang, bitangent));

    // Fill hit info
    hit.t = tHit;
    hit.normal = smoothN;
    hit.tangent = tang;
    hit.u = (float(segIdx) + hitS) / float(segsPerStrand);
    hit.v = sine;
    hit.strandId = strandIdx;
    return true;
}

// Helper: call from inside rayQueryProceedEXT loop for AABB candidates.
// Keys are pre-transformed to Vulkan world space (identity TLAS transform).
// World-space ray intersects world-space curve data directly — no transforms needed.
bool handleHairAABBCandidate(rayQueryEXT rq, vec3 rayO, vec3 rayD,
                             float tMin, inout HairHitInfo bestHit)
{
    uint custIdx = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false);
    GeometryMetadata geo = geometryMetadata.geometries[custIdx];

    // Hair geometry is identified by indexBufferAddress == 0
    if (geo.indexBufferAddress != 0u) return false;

    // DIAGNOSTIC: minimal hit — no buffer reads, no math
    float safet = tMin + 0.1;
    if (safet >= bestHit.t) return false;

    bestHit.t = safet;
    bestHit.normal = vec3(0, 1, 0);
    bestHit.tangent = vec3(1, 0, 0);
    bestHit.u = 0.5;
    bestHit.v = 0.0;
    bestHit.strandId = 0u;
    rayQueryGenerateIntersectionEXT(rq, safet);
    return true;
}
