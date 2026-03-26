// ============================================================
// Surfel GI Cache — surface-based irradiance cache
// Based on EA SEED's GIBS (SIGGRAPH 2021) architecture.
// Reuses SHARC spatial hash infrastructure for code reuse.
// ============================================================

#ifndef SURFEL_GI_GLSL
#define SURFEL_GI_GLSL

// ============================================================
// Configuration
// ============================================================
#define SURFEL_RADIANCE_SCALE    1000.0  // float→uint quantization for atomicAdd
#define SURFEL_BUCKET_SIZE       16u     // linear probe window for hash collisions
#define SURFEL_SAMPLE_THRESHOLD  4u      // min samples before cache is trusted
#define SURFEL_LOG_BASE          2.0     // logarithmic LOD base
#define SURFEL_GRID_SCALE_MULT   2.0     // finer grid than SHARC (2x resolution)

// ============================================================
// Grid parameters (identical to SHARC for code reuse)
// ============================================================

struct SurfelGridParams {
    vec3  cameraPosition;
    float sceneScale;
};

// Compute voxel level based on distance from camera
uint surfelGetLevel(vec3 worldPos, SurfelGridParams grid) {
    float dist = length(worldPos - grid.cameraPosition);
    float level = log2(max(dist * grid.sceneScale, 1.0)) / log2(SURFEL_LOG_BASE);
    return uint(max(level, 0.0));
}

// Voxel size at a given level
float surfelVoxelSize(uint level, SurfelGridParams grid) {
    return pow(SURFEL_LOG_BASE, float(level)) / max(grid.sceneScale, 1e-6);
}

// 64-bit hash key from world position + level + normal octant
uint64_t surfelHashKey(vec3 worldPos, vec3 normal, SurfelGridParams grid) {
    uint level = surfelGetLevel(worldPos, grid);
    float voxelSize = surfelVoxelSize(level, grid);
    ivec3 gridPos = ivec3(floor(worldPos / voxelSize));

    // Pack: 17 bits per axis + 10 bits level + 3 bits normal = 64 bits
    uint64_t key = (uint64_t(gridPos.x & 0x1FFFF) << 0)
                 | (uint64_t(gridPos.y & 0x1FFFF) << 17)
                 | (uint64_t(gridPos.z & 0x1FFFF) << 34)
                 | (uint64_t(level & 0x3FF) << 51);

    // Encode dominant normal direction (3 bits)
    uint normalBits = 0u;
    vec3 absN = abs(normal);
    if (absN.x > absN.y && absN.x > absN.z) normalBits = (normal.x > 0.0) ? 0u : 1u;
    else if (absN.y > absN.z)                normalBits = (normal.y > 0.0) ? 2u : 3u;
    else                                      normalBits = (normal.z > 0.0) ? 4u : 5u;
    key |= uint64_t(normalBits & 0x7) << 61;

    return (key == 0u) ? uint64_t(1u) : key;
}

// Jenkins 32-bit hash
uint surfelJenkins(uint a) {
    a = (a + 0x7ed55d16u) + (a << 12);
    a = (a ^ 0xc761c23cu) ^ (a >> 19);
    a = (a + 0x165667b1u) + (a << 5);
    a = (a + 0xd3a2646cu) ^ (a << 9);
    a = (a + 0xfd7046c5u) + (a << 3);
    a = (a ^ 0xb55a4f09u) ^ (a >> 16);
    return a;
}

uint surfelHash64to32(uint64_t key) {
    return surfelJenkins(uint(key & 0xFFFFFFFFu)) ^ surfelJenkins(uint(key >> 32));
}

uint surfelGetSlot(uint64_t key, uint capacity) {
    return surfelHash64to32(key) % capacity;
}

// ============================================================
// Hash Map Operations — uses independent surfel buffers (bindings 32-33)
// ============================================================

// Insert a hash key, return slot index (or 0xFFFFFFFF if failed)
uint surfelInsert(uint64_t key, uint capacity) {
    uint baseSlot = surfelGetSlot(key, capacity);
    for (uint i = 0u; i < SURFEL_BUCKET_SIZE; i++) {
        uint slot = (baseSlot + i) % capacity;
        uint64_t existing = atomicCompSwap(
            _surfelHashEntries.keys[slot], uint64_t(0u), key);
        if (existing == uint64_t(0u) || existing == key) {
            return slot;
        }
    }
    return 0xFFFFFFFFu;
}

// Find a hash key, return slot index (or 0xFFFFFFFF if not found)
uint surfelFind(uint64_t key, uint capacity) {
    uint baseSlot = surfelGetSlot(key, capacity);
    for (uint i = 0u; i < SURFEL_BUCKET_SIZE; i++) {
        uint slot = (baseSlot + i) % capacity;
        uint64_t stored = _surfelHashEntries.keys[slot];
        if (stored == key) return slot;
        if (stored == uint64_t(0u)) return 0xFFFFFFFFu;
    }
    return 0xFFFFFFFFu;
}

// Resolved data offset (second region of combined buffer)
uint surfelResolvedOffset(uint slot, uint capacity) {
    return capacity * 4u + slot * 4u;
}

// Add radiance to accumulation buffer (region 0 of surfel data)
void surfelAccumulate(uint slot, vec3 radiance, uint capacity) {
    vec3 scaled = radiance * float(SURFEL_RADIANCE_SCALE);
    uvec3 uScaled = uvec3(max(scaled, vec3(0.0)));
    if (uScaled.x > 0u) atomicAdd(_surfelData.data[slot * 4u + 0u], uScaled.x);
    if (uScaled.y > 0u) atomicAdd(_surfelData.data[slot * 4u + 1u], uScaled.y);
    if (uScaled.z > 0u) atomicAdd(_surfelData.data[slot * 4u + 2u], uScaled.z);
    atomicAdd(_surfelData.data[slot * 4u + 3u], 1u);
}

// ============================================================
// Surfel Query — get cached irradiance at a world position
// Returns true if valid cached data found, writes to outIrradiance
// ============================================================

bool surfelGetCachedIrradiance(vec3 worldPos, vec3 normal,
                                SurfelGridParams grid, uint capacity,
                                out vec3 outIrradiance) {
    outIrradiance = vec3(0.0);

    uint64_t key = surfelHashKey(worldPos, normal, grid);
    uint slot = surfelFind(key, capacity);
    if (slot == 0xFFFFFFFFu) return false;

    // Read resolved irradiance (region 1 of surfel data)
    uint rOff = surfelResolvedOffset(slot, capacity);
    float r = uintBitsToFloat(_surfelData.data[rOff + 0u]);
    float g = uintBitsToFloat(_surfelData.data[rOff + 1u]);
    float b = uintBitsToFloat(_surfelData.data[rOff + 2u]);
    uint meta = _surfelData.data[rOff + 3u];

    // Check sample count (low 16 bits of meta)
    uint sampleCount = meta & 0xFFFFu;
    if (sampleCount < SURFEL_SAMPLE_THRESHOLD) return false;

    outIrradiance = vec3(r, g, b);
    return true;
}

// ============================================================
// Sparse update pixel selection (4% of pixels update surfels)
// ============================================================

bool surfelIsUpdatePixel(ivec2 pixel, uint frameIndex) {
    // 5x5 grid = 25 pixels, 1 per frame → 4% update rate
    uint hash = uint(pixel.x) * 1973u + uint(pixel.y) * 9277u + frameIndex * 26699u;
    hash = (hash ^ (hash >> 16u)) * 0x45d9f3bu;
    return (hash % 25u) == 0u;
}

#endif // SURFEL_GI_GLSL
