// ============================================================
// wf_structs.glsl — Shared data structures for wavefront path tracing
// ============================================================

#ifndef WF_STRUCTS_GLSL
#define WF_STRUCTS_GLSL

// Per-path state surviving across bounce iterations (48 bytes = 3x vec4)
struct PathState {
    vec3 origin;        // ray origin
    uint pixelIndex;    // linear pixel index (x + y * width)
    vec3 direction;     // ray direction
    uint rngState;      // PCG RNG state
    vec3 throughput;    // path throughput
    uint flags;         // bits: [0:3]=bounce, [4]=isDiffusePath, [5]=terminated, [6]=isGlass
};

// Hit result from intersection kernel (32 bytes = 2x vec4)
struct HitResult {
    float hitT;         // intersection distance
    uint customIndex;   // BLAS geometry index (for GeometryMetadata)
    int instanceId;     // instance index (for prevTransforms)
    int primitiveId;    // triangle index
    vec2 barycentrics;  // barycentric coordinates
    uint hitType;       // 0=miss, 1=hit
    uint pad0;
};

// Shadow ray for NEE (48 bytes = 3x vec4)
struct ShadowRay {
    vec3 origin;        // shadow ray origin
    float maxDist;      // max trace distance
    vec3 direction;     // shadow ray direction
    uint pathIndex;     // index into PathState for accumulation
    vec3 radiance;      // pre-computed contribution if unshadowed
    uint channel;       // 0=diffuse, 1=specular
};

// Per-pixel radiance accumulator (32 bytes = 2x vec4)
struct PixelRadiance {
    vec3 diffuse;       // accumulated diffuse radiance
    float hitDist;      // primary hit distance (NRD)
    vec3 specular;      // accumulated specular radiance
    float pad0;
};

// Primary hit data for G-buffer output (64 bytes = 4x vec4)
struct PrimaryGBuffer {
    vec3 normal;        // shading normal
    float roughness;
    vec3 worldPos;      // world-space hit position
    float viewZ;        // view-space Z
    vec3 albedo;        // base color
    float hitDist;      // primary hit distance
    vec3 localPos;      // local-space position (for motion vectors)
    uint instanceId;    // for prevTransforms
    uint customIndex;   // BLAS index
    float metallic;
    float penumbra;     // SIGMA penumbra distance
    float primaryNdotL; // sun dot product (for SIGMA)
};

// Atomic counters for wavefront dispatch
struct WavefrontCounters {
    uint activeRayCount;      // paths to process this bounce
    uint shadowRayCount;      // shadow rays generated
    uint nextActiveRayCount;  // paths continuing to next bounce
    uint pad0;
};

// Helper: pack/unpack bounce from flags
uint wf_getBounce(uint flags) { return flags & 0xFu; }
uint wf_setBounce(uint flags, uint bounce) { return (flags & ~0xFu) | (bounce & 0xFu); }
bool wf_isDiffuse(uint flags) { return (flags & 0x10u) != 0u; }
bool wf_isTerminated(uint flags) { return (flags & 0x20u) != 0u; }
bool wf_isGlass(uint flags) { return (flags & 0x40u) != 0u; }

uint wf_setDiffuse(uint flags, bool v) { return v ? (flags | 0x10u) : (flags & ~0x10u); }
uint wf_setTerminated(uint flags) { return flags | 0x20u; }
uint wf_setGlass(uint flags, bool v) { return v ? (flags | 0x40u) : (flags & ~0x40u); }

#endif // WF_STRUCTS_GLSL
