// nirc_hash_grid.glsl — Multiresolution Hash Grid Encoding (Instant-NGP style)
// Encodes a 3D position into a feature vector by looking up and interpolating
// trainable features from multiple resolution levels of spatial hash tables.
//
// Architecture: 12 levels × 2 features/level = 24D output
// Hash table: 2^19 entries per level, XOR prime hash

#ifndef NIRC_HASH_GRID_GLSL
#define NIRC_HASH_GRID_GLSL

// Configuration
#define NIRC_HASH_LEVELS     12
#define NIRC_FEATURES_PER_LEVEL 2
#define NIRC_HASH_TABLE_SIZE 524288  // 2^19
#define NIRC_TOTAL_FEATURES  (NIRC_HASH_LEVELS * NIRC_FEATURES_PER_LEVEL)  // 24

// Resolution progression: N_l = floor(N_min * b^l)
// N_min=16, N_max=2048, b = exp((ln(2048) - ln(16)) / 11) ≈ 1.519
#define NIRC_N_MIN   16.0
#define NIRC_N_MAX   2048.0
#define NIRC_GROWTH  1.5191565  // exp((ln(2048) - ln(16)) / 11)

// XOR prime hash (Instant-NGP)
#define NIRC_PRIME1 1u
#define NIRC_PRIME2 2654435761u
#define NIRC_PRIME3 805459861u

// Hash table SSBO — features stored as vec2 (2 features per entry)
// Total entries: NIRC_HASH_LEVELS * NIRC_HASH_TABLE_SIZE
// Layout: level 0 occupies [0, TABLE_SIZE), level 1 [TABLE_SIZE, 2*TABLE_SIZE), etc.

// Hash a 3D grid coordinate to a table index
uint nircSpatialHash(ivec3 gridPos, uint level) {
    uint h = (uint(gridPos.x) * NIRC_PRIME1) ^
             (uint(gridPos.y) * NIRC_PRIME2) ^
             (uint(gridPos.z) * NIRC_PRIME3);
    return (h % NIRC_HASH_TABLE_SIZE) + level * NIRC_HASH_TABLE_SIZE;
}

// Get the grid resolution for a given level
float nircLevelResolution(uint level) {
    return floor(NIRC_N_MIN * pow(NIRC_GROWTH, float(level)));
}

// Hash grid encoding requires the caller to define NIRC_HASH_FEATURES_BUFFER
// as a macro that expands to the SSBO accessor, e.g.:
//   #define NIRC_HASH_FEATURES_BUFFER hashGrid.features
//
// The buffer must contain NIRC_HASH_LEVELS * NIRC_HASH_TABLE_SIZE * 2 floats.

#ifdef NIRC_HASH_FEATURES_BUFFER

// Encode a position into NIRC_TOTAL_FEATURES (24) features
void nircHashGridEncode(
    in vec3 worldPos,
    in vec3 scenePosScale,
    in vec3 scenePosOffset,
    out float outFeatures[NIRC_TOTAL_FEATURES])
{
    vec3 pos = worldPos * scenePosScale + scenePosOffset;
    pos = clamp(pos, vec3(0.0), vec3(1.0));

    for (uint level = 0u; level < uint(NIRC_HASH_LEVELS); level++) {
        float res = nircLevelResolution(level);
        vec3 gridPos = pos * res;

        ivec3 p0 = ivec3(floor(gridPos));
        ivec3 p1 = p0 + ivec3(1);
        vec3 w = fract(gridPos);

        vec2 result = vec2(0.0);
        for (int dz = 0; dz < 2; dz++) {
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    ivec3 corner = ivec3(
                        dx == 0 ? p0.x : p1.x,
                        dy == 0 ? p0.y : p1.y,
                        dz == 0 ? p0.z : p1.z);

                    float wx = dx == 0 ? (1.0 - w.x) : w.x;
                    float wy = dy == 0 ? (1.0 - w.y) : w.y;
                    float wz = dz == 0 ? (1.0 - w.z) : w.z;
                    float weight = wx * wy * wz;

                    uint idx = nircSpatialHash(corner, level);
                    vec2 feat = vec2(NIRC_HASH_FEATURES_BUFFER[idx * 2],
                                     NIRC_HASH_FEATURES_BUFFER[idx * 2 + 1]);
                    result += feat * weight;
                }
            }
        }

        uint baseIdx = level * uint(NIRC_FEATURES_PER_LEVEL);
        outFeatures[baseIdx]     = result.x;
        outFeatures[baseIdx + 1] = result.y;
    }
}

#endif // NIRC_HASH_FEATURES_BUFFER

#endif // NIRC_HASH_GRID_GLSL
