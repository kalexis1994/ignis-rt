// SHARC (Spatial Hashing of Approximate Radiance Cache)
// Shared definitions for raygen and resolve shaders

#define SHARC_GRID_SIZE 0.1           // meters per cell (finer = smoother, more collisions)
#define SHARC_TABLE_SIZE 2097152u     // 2^21 entries
#define SHARC_TABLE_MASK (SHARC_TABLE_SIZE - 1u)
#define SHARC_MAX_AGE 120u            // frames before eviction
#define SHARC_MIN_SAMPLES 4u          // minimum samples before cache is trusted
#define SHARC_BLEND_ALPHA 0.05        // exponential moving average blend (5% new)

// 32 bytes per entry (8 uints)
struct SHARCEntry {
    uint hashKey;        // 0 = empty sentinel
    uint sampleCount;
    float radianceR;
    float radianceG;
    float radianceB;
    uint lastFrame;
    float pad0;
    float pad1;
};

// Hash function for world position -> cache key
uint sharcHash(vec3 worldPos) {
    ivec3 cell = ivec3(floor(worldPos / SHARC_GRID_SIZE));
    uint h = uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u ^ uint(cell.z) * 83492791u;
    return max(h ^ (h >> 16), 1u);  // 0 is reserved as empty sentinel
}
