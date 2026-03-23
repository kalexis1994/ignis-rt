// SHARC for Ignis RT — Pure GLSL port of NVIDIA's SHARC algorithm
// Based on: https://github.com/NVIDIA-RTX/SHARC (v1.6.5)
// Original: Copyright (c) 2023-2025 NVIDIA CORPORATION. Apache 2.0 License.
// Ported to pure GLSL to avoid HLSL buffer_reference compatibility issues.

#ifndef SHARC_SETUP_GLSL
#define SHARC_SETUP_GLSL

// ============================================================
// Configuration
// ============================================================
#define SHARC_PROPAGATION_DEPTH     4       // vertices stored for backpropagation
#define SHARC_RADIANCE_SCALE        1000.0  // float→uint quantization for atomicAdd
#define SHARC_SAMPLE_THRESHOLD      0u      // min samples before cache is trusted
#define SHARC_LOG_BASE              2.0     // logarithmic LOD base
#define SHARC_BUCKET_SIZE           16u     // linear probe window for hash collisions

// ============================================================
// Hash Grid — logarithmic LOD world-space voxel grid
// Based on NVIDIA HashGridCommon.h
// ============================================================

struct SharcGridParams {
    vec3  cameraPosition;
    float sceneScale;
};

// Compute voxel level based on distance from camera (logarithmic LOD)
uint sharcGetLevel(vec3 worldPos, SharcGridParams grid) {
    float dist = length(worldPos - grid.cameraPosition);
    float level = log2(max(dist * grid.sceneScale, 1.0)) / log2(SHARC_LOG_BASE);
    return uint(max(level, 0.0));
}

// Voxel size at a given level
float sharcVoxelSize(uint level, SharcGridParams grid) {
    return pow(SHARC_LOG_BASE, float(level)) / max(grid.sceneScale, 1e-6);
}

// 64-bit hash key from world position + level + normal
uint64_t sharcHashKey(vec3 worldPos, vec3 normal, SharcGridParams grid) {
    uint level = sharcGetLevel(worldPos, grid);
    float voxelSize = sharcVoxelSize(level, grid);
    ivec3 gridPos = ivec3(floor(worldPos / voxelSize));

    // Pack: 17 bits per axis (x,y,z) + 10 bits level + 3 bits normal = 64 bits
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

    return (key == 0u) ? uint64_t(1u) : key;  // 0 = empty sentinel
}

// Jenkins 32-bit hash
uint sharcJenkins(uint a) {
    a = (a + 0x7ed55d16u) + (a << 12);
    a = (a ^ 0xc761c23cu) ^ (a >> 19);
    a = (a + 0x165667b1u) + (a << 5);
    a = (a + 0xd3a2646cu) ^ (a << 9);
    a = (a + 0xfd7046c5u) + (a << 3);
    a = (a ^ 0xb55a4f09u) ^ (a >> 16);
    return a;
}

uint sharcHash64to32(uint64_t key) {
    return sharcJenkins(uint(key & 0xFFFFFFFFu)) ^ sharcJenkins(uint(key >> 32));
}

uint sharcGetSlot(uint64_t key, uint capacity) {
    return sharcHash64to32(key) % capacity;
}

// ============================================================
// SHARC State (per-path, for update pass)
// ============================================================

struct SharcState {
    uint   cacheIndices[SHARC_PROPAGATION_DEPTH];
    vec3   sampleWeights[SHARC_PROPAGATION_DEPTH];
    uint   pathLength;
};

void SharcInit(inout SharcState state) {
    state.pathLength = 0u;
}

// ============================================================
// Hash Map Operations (using SSBOs at bindings 20-21)
// SSBO 20 = hashEntries (uint64 per slot)
// SSBO 21 = accumulation (uvec4 per slot: rgb_scaled + sampleCount)
// Resolved data stored in SSBO at binding 21 but in second half
// Actually: we use 3 separate regions via device addresses in UBO.
//
// For GLSL port: use global SSBO bindings instead of buffer_reference.
// ============================================================

// Insert a hash key, return slot index (or 0xFFFFFFFF if failed)
uint sharcInsert(uint64_t key, uint capacity) {
    uint baseSlot = sharcGetSlot(key, capacity);

    for (uint i = 0u; i < SHARC_BUCKET_SIZE; i++) {
        uint slot = (baseSlot + i) % capacity;

        // Try to claim this slot with atomic CAS on 64-bit key
        uint64_t existing = atomicCompSwap(
            _sharcHashEntries.keys[slot], uint64_t(0u), key);

        if (existing == uint64_t(0u) || existing == key) {
            return slot;  // claimed empty slot or found existing
        }
    }
    return 0xFFFFFFFFu;  // bucket full
}

// Find a hash key, return slot index (or 0xFFFFFFFF if not found)
uint sharcFind(uint64_t key, uint capacity) {
    uint baseSlot = sharcGetSlot(key, capacity);

    for (uint i = 0u; i < SHARC_BUCKET_SIZE; i++) {
        uint slot = (baseSlot + i) % capacity;
        uint64_t stored = _sharcHashEntries.keys[slot];

        if (stored == key) return slot;
        if (stored == uint64_t(0u)) return 0xFFFFFFFFu;  // empty = not found
    }
    return 0xFFFFFFFFu;
}

// Resolved data offset: second half of the combined accum/resolved buffer
uint sharcResolvedOffset(uint slot, uint capacity) {
    return capacity * 4u + slot * 4u;
}

// Add radiance to accumulation buffer (atomic uint add for each channel)
void sharcAccumulate(uint slot, vec3 radiance, vec3 weight) {
    vec3 scaled = radiance * weight * float(SHARC_RADIANCE_SCALE);
    uvec3 uScaled = uvec3(max(scaled, vec3(0.0)));

    if (uScaled.x > 0u) atomicAdd(_sharcAccum.data[slot * 4u + 0u], uScaled.x);
    if (uScaled.y > 0u) atomicAdd(_sharcAccum.data[slot * 4u + 1u], uScaled.y);
    if (uScaled.z > 0u) atomicAdd(_sharcAccum.data[slot * 4u + 2u], uScaled.z);
    atomicAdd(_sharcAccum.data[slot * 4u + 3u], 1u);  // sample count
}

// ============================================================
// SHARC API — matches NVIDIA's SharcCommon.h interface
// ============================================================

void SharcUpdateMiss(SharcGridParams grid, SharcState state, vec3 skyRadiance, uint capacity) {
    for (uint i = 0u; i < state.pathLength; i++) {
        sharcAccumulate(state.cacheIndices[i], skyRadiance, state.sampleWeights[i]);
    }
}

bool SharcUpdateHit(SharcGridParams grid, inout SharcState state,
                    vec3 worldPos, vec3 normal,
                    vec3 directLighting, float random, uint capacity) {
    uint64_t key = sharcHashKey(worldPos, normal, grid);
    uint slot = sharcInsert(key, capacity);
    if (slot == 0xFFFFFFFFu) return true;  // bucket full, continue tracing

    vec3 radiance = directLighting;

    // Check if we can early-out (cache already has good data)
    if (state.pathLength >= 1u) {
        uint rOff = sharcResolvedOffset(slot, capacity);
        float sampleNum = uintBitsToFloat(_sharcResolved.data[rOff + 3u]);
        if (sampleNum > float(SHARC_SAMPLE_THRESHOLD) && random < 0.5) {
            radiance = vec3(
                uintBitsToFloat(_sharcResolved.data[rOff + 0u]),
                uintBitsToFloat(_sharcResolved.data[rOff + 1u]),
                uintBitsToFloat(_sharcResolved.data[rOff + 2u])
            );
            // Backpropagate to previous vertices
            for (uint i = 0u; i < state.pathLength; i++) {
                sharcAccumulate(state.cacheIndices[i], radiance, state.sampleWeights[i]);
            }
            return false;  // stop tracing
        }
    }

    // Write direct lighting to this voxel
    sharcAccumulate(slot, directLighting, vec3(1.0));

    // Backpropagate to all previous vertices in the path
    for (uint i = 0u; i < state.pathLength; i++) {
        sharcAccumulate(state.cacheIndices[i], radiance, state.sampleWeights[i]);
    }

    // Shift path history and add current vertex
    for (uint i = min(state.pathLength, SHARC_PROPAGATION_DEPTH - 1u); i > 0u; i--) {
        state.cacheIndices[i] = state.cacheIndices[i - 1u];
        state.sampleWeights[i] = state.sampleWeights[i - 1u];
    }
    state.cacheIndices[0] = slot;
    state.sampleWeights[0] = vec3(1.0);
    state.pathLength = min(state.pathLength + 1u, SHARC_PROPAGATION_DEPTH - 1u);

    return true;  // continue tracing
}

void SharcSetThroughput(inout SharcState state, vec3 throughput) {
    for (uint i = 0u; i < state.pathLength; i++) {
        state.sampleWeights[i] *= throughput;
    }
}

bool SharcGetCachedRadiance(SharcGridParams grid, vec3 worldPos, vec3 normal,
                            uint capacity, out vec3 radiance) {
    uint64_t key = sharcHashKey(worldPos, normal, grid);
    uint slot = sharcFind(key, capacity);
    if (slot == 0xFFFFFFFFu) {
        radiance = vec3(0.0);
        return false;
    }

    uint rOff = sharcResolvedOffset(slot, capacity);
    float sampleNum = uintBitsToFloat(_sharcResolved.data[rOff + 3u]);
    if (sampleNum > float(SHARC_SAMPLE_THRESHOLD)) {
        radiance = vec3(
            uintBitsToFloat(_sharcResolved.data[rOff + 0u]),
            uintBitsToFloat(_sharcResolved.data[rOff + 1u]),
            uintBitsToFloat(_sharcResolved.data[rOff + 2u])
        );
        return true;
    }

    radiance = vec3(0.0);
    return false;
}

// Determine if this pixel should do SHARC update (sparse 4%)
bool isSharcUpdatePixel(uvec2 pixel, uint frameIndex) {
    uint blockX = pixel.x / 5u;
    uint blockY = pixel.y / 5u;
    uint blockHash = (blockX * 73856093u ^ blockY * 19349663u ^ frameIndex * 83492791u);
    uint selectedX = blockX * 5u + (blockHash % 5u);
    uint selectedY = blockY * 5u + ((blockHash / 5u) % 5u);
    return (pixel.x == selectedX && pixel.y == selectedY);
}

#endif // SHARC_SETUP_GLSL
