// ============================================================
// stable_planes.glsl — Multi-layer denoising for delta surfaces
//
// Per NVIDIA RTXPT: decomposes each pixel into up to 3 stable
// planes for surfaces seen through glass/mirror chains. Each
// plane stores independent guide buffers for DLSS RR.
//
// Architecture:
//   BUILD pass: deterministic delta tracing, no RR, no noise.
//     Splits at glass (reflection + refraction) into separate planes.
//   FILL pass: standard noisy path tracing. Deposits radiance
//     into the correct plane using stableBranchID matching.
//
// Buffer layout (per pixel):
//   stablePlaneHeader: uvec4 per pixel
//     [0].x = plane 0 branchID
//     [0].y = plane 1 branchID
//     [0].z = plane 2 branchID
//     [0].w = (firstHitRayLength:30 bits | dominantPlaneIdx:2 bits)
//   stablePlaneData: 8 vec4s per plane per pixel (3 planes = 24 vec4s)
//     [0] = worldPos.xyz + sceneLength
//     [1] = rayDir.xyz + lastRayT
//     [2] = throughput.xyz + roughness
//     [3] = normal (oct-encoded) + viewZ + pad
//     [4] = motionVector.xy + pad.xy
//     [5] = imageXform row0 (packed)
//     [6] = imageXform row1 (packed)
//     [7] = reserved
// ============================================================

#ifndef STABLE_PLANES_GLSL
#define STABLE_PLANES_GLSL

const uint SP_MAX_PLANES = 3u;
const uint SP_DATA_STRIDE = 8u;  // vec4s per plane
const uint SP_PIXEL_STRIDE = SP_MAX_PLANES * SP_DATA_STRIDE;  // 24 vec4s per pixel

// ── stableBranchID encoding ──
// Camera = 1. Each delta scatter: shift left 2, OR lobe index.
// Lobe 0 = transmission, 1 = reflection.
uint spAdvanceBranchID(uint prevID, uint deltaLobeIndex) {
    return (prevID << 2u) | (deltaLobeIndex & 3u);
}

uint spVertexIndexFromBranchID(uint branchID) {
    return findMSB(branchID) / 2u + 1u;
}

bool spIsOnPlane(uint planeBranchID, uint vertexBranchID) {
    return planeBranchID == vertexBranchID;
}

bool spIsOnStablePath(uint planeBranchID, uint planeVertexIdx,
                       uint vertexBranchID, uint vertexIdx) {
    if (vertexIdx > planeVertexIdx) return false;
    return (planeBranchID >> ((planeVertexIdx - vertexIdx) * 2u)) == vertexBranchID;
}

// ── Per-plane data pack/unpack ──

struct StablePlaneData {
    vec3  worldPos;
    float sceneLength;
    vec3  rayDir;
    float lastRayT;
    vec3  throughput;
    float roughness;
    vec3  normal;
    float viewZ;
    vec2  motionVector;
    mat3  imageXform;
    bool  valid;
};

StablePlaneData spEmptyPlane() {
    StablePlaneData sp;
    sp.worldPos = vec3(0); sp.sceneLength = 99999.0;
    sp.rayDir = vec3(0,0,-1); sp.lastRayT = 0.0;
    sp.throughput = vec3(0); sp.roughness = 1.0;
    sp.normal = vec3(0,1,0); sp.viewZ = 0.0;
    sp.motionVector = vec2(0);
    sp.imageXform = mat3(1.0);
    sp.valid = false;
    return sp;
}

// ── Householder reflection matrix ──
mat3 spMirrorMatrix(vec3 N) {
    return mat3(1.0) - 2.0 * outerProduct(N, N);
}

// ── Rotation from direction A to direction B ──
mat3 spRotateFromTo(vec3 from, vec3 to) {
    vec3 v = cross(from, to);
    float c = dot(from, to);
    if (c < -0.9999) {
        // Nearly opposite — use perpendicular axis
        vec3 perp = abs(from.x) < 0.9 ? vec3(1,0,0) : vec3(0,1,0);
        vec3 axis = normalize(cross(from, perp));
        return mat3(-1.0) + 2.0 * outerProduct(axis, axis);
    }
    float k = 1.0 / (1.0 + c);
    return mat3(
        v.x*v.x*k + c,    v.x*v.y*k - v.z,  v.x*v.z*k + v.y,
        v.y*v.x*k + v.z,  v.y*v.y*k + c,    v.y*v.z*k - v.x,
        v.z*v.x*k - v.y,  v.z*v.y*k + v.x,  v.z*v.z*k + c
    );
}

// ── Oct-encode/decode for normals (reuse from pt_reservoir) ──
vec2 spOctEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return n.xy * 0.5 + 0.5;
}

vec3 spOctDecode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

#endif // STABLE_PLANES_GLSL
