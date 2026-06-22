// SHARC (Spatially Hashed Radiance Cache) — pure-GLSL port of NVIDIA SHARC v1.6.5 (Apache 2.0).
// World-space radiance cache: hash (worldPos, normal, distance-LOD) -> a slot; the path tracer
// accumulates radiance and queries it to terminate paths early. Read-only half (hash/find/query +
// the resolve helpers); the atomic UPDATE half (insert/accumulate) is added in SHARC-1 along with
// GL_EXT_shader_atomic_int64. Bindings: 25 = uint64 hash keys, 26 = data [accum: cap*4][resolved: cap*4].
// Requires GL_EXT_shader_explicit_arithmetic_types_int64 (already enabled by the includers) + the pc block.
#ifndef SHARC_GLSL
#define SHARC_GLSL

layout(binding = 25, std430) buffer SharcKeys { uint64_t keys[]; } _sharcKeys;
layout(binding = 26, std430) buffer SharcData { uint data[]; } _sharcData;

#define SHARC_RADIANCE_SCALE    1000.0  // float<->uint quantization for the atomic-add accumulation
#define SHARC_BUCKET_SIZE       16u     // linear-probe window for hash collisions
#define SHARC_LOG_BASE          2.0     // logarithmic LOD base (voxel size doubles per level)
#define SHARC_PROPAGATION_DEPTH 4u      // path vertices kept for backpropagation (update pass)

// Camera-relative logarithmic-LOD grid params, from the push constants.
struct SharcGrid { vec3 cameraPosition; float sceneScale; };
SharcGrid sharcGrid() {
    SharcGrid g;
    g.cameraPosition = pc.camPos.xyz;
    g.sceneScale = pc.sharcSceneScale;
    return g;
}

// Voxel LOD level: grows logarithmically with distance from the camera (fine near, coarse far).
uint sharcGetLevel(vec3 worldPos, SharcGrid grid) {
    float dist = length(worldPos - grid.cameraPosition);
    float level = log2(max(dist * grid.sceneScale, 1.0)) / log2(SHARC_LOG_BASE);
    return uint(max(level, 0.0));
}
float sharcVoxelSize(uint level, SharcGrid grid) {
    return pow(SHARC_LOG_BASE, float(level)) / max(grid.sceneScale, 1e-6);
}

// 64-bit hash key: 17b per grid axis + 10b level + 3b dominant-normal direction.
uint64_t sharcHashKey(vec3 worldPos, vec3 normal, SharcGrid grid) {
    uint level = sharcGetLevel(worldPos, grid);
    float voxelSize = sharcVoxelSize(level, grid);
    ivec3 gridPos = ivec3(floor(worldPos / voxelSize));
    uint64_t key = (uint64_t(gridPos.x & 0x1FFFF) << 0)
                 | (uint64_t(gridPos.y & 0x1FFFF) << 17)
                 | (uint64_t(gridPos.z & 0x1FFFF) << 34)
                 | (uint64_t(level & 0x3FF) << 51);
    uint normalBits;
    vec3 absN = abs(normal);
    if (absN.x > absN.y && absN.x > absN.z) normalBits = (normal.x > 0.0) ? 0u : 1u;
    else if (absN.y > absN.z)                normalBits = (normal.y > 0.0) ? 2u : 3u;
    else                                      normalBits = (normal.z > 0.0) ? 4u : 5u;
    key |= uint64_t(normalBits & 0x7) << 61;
    return (key == uint64_t(0u)) ? uint64_t(1u) : key; // 0 = empty sentinel
}

uint sharcJenkins(uint a) {
    a = (a + 0x7ed55d16u) + (a << 12); a = (a ^ 0xc761c23cu) ^ (a >> 19);
    a = (a + 0x165667b1u) + (a << 5);  a = (a + 0xd3a2646cu) ^ (a << 9);
    a = (a + 0xfd7046c5u) + (a << 3);  a = (a ^ 0xb55a4f09u) ^ (a >> 16);
    return a;
}
uint sharcHash64to32(uint64_t key) { return sharcJenkins(uint(key & 0xFFFFFFFFu)) ^ sharcJenkins(uint(key >> 32)); }
uint sharcGetSlot(uint64_t key, uint capacity) { return sharcHash64to32(key) % capacity; }
// The resolved region lives in the second half of the data buffer (after accum: cap*4).
uint sharcResolvedOffset(uint slot, uint capacity) { return capacity * 4u + slot * 4u; }

// Read-only probe: find an existing key's slot (or 0xFFFFFFFF). Stops at the first empty slot.
uint sharcFind(uint64_t key, uint capacity) {
    uint baseSlot = sharcGetSlot(key, capacity);
    for (uint i = 0u; i < SHARC_BUCKET_SIZE; i++) {
        uint slot = (baseSlot + i) % capacity;
        uint64_t stored = _sharcKeys.keys[slot];
        if (stored == key) return slot;
        if (stored == uint64_t(0u)) return 0xFFFFFFFFu;
    }
    return 0xFFFFFFFFu;
}

// Query the resolved cache for radiance at (worldPos, normal). Returns false if the cell is missing
// or has no samples yet. resolved slot = [Rf, Gf, Bf, meta] (RGB float-bits; meta low16 = sampleCount).
bool SharcGetCachedRadiance(SharcGrid grid, vec3 worldPos, vec3 normal, uint capacity, out vec3 radiance) {
    radiance = vec3(0.0);
    if (capacity == 0u) return false;
    uint64_t key = sharcHashKey(worldPos, normal, grid);
    uint slot = sharcFind(key, capacity);
    if (slot == 0xFFFFFFFFu) return false;
    uint rOff = sharcResolvedOffset(slot, capacity);
    uint sampleCount = _sharcData.data[rOff + 3u] & 0xFFFFu;
    if (sampleCount > 0u) {
        radiance = vec3(uintBitsToFloat(_sharcData.data[rOff + 0u]),
                        uintBitsToFloat(_sharcData.data[rOff + 1u]),
                        uintBitsToFloat(_sharcData.data[rOff + 2u]));
        return true;
    }
    return false;
}

// Per-path state for the update pass (populated in SHARC-1). Carries the recent path vertices so a
// later hit/miss can backpropagate its radiance to them.
struct SharcState {
    uint cacheIndices[SHARC_PROPAGATION_DEPTH];
    vec3 sampleWeights[SHARC_PROPAGATION_DEPTH];
    uint pathLength;
};
void SharcInit(out SharcState s) { s.pathLength = 0u; }

#endif // SHARC_GLSL
