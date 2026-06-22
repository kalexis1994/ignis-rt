// ============================================================
// wf_common.glsl — Wavefront path tracing: shared data contract (Rust renderer dialect)
// ============================================================
// Ported from the C++ wavefront (shaders/wavefront/wf_structs.glsl) but re-expressed against the
// RUST renderer's descriptor convention: set 0 is the existing trace.rgen scene layout (camera is a
// PUSH CONSTANT, not a UBO; geometry comes via buffer_reference in GeomTable), and set 1 is the new
// wavefront-private working set. The C++ wf_*.comp kernels bind a Camera UBO at set=0/binding=2 and
// their counters at set=1/binding=5 — that contract does NOT match this renderer, which is why the
// kernels are re-written here rather than copied.
//
// All wavefront kernels (#include this header) run with local_size_x = 256, one invocation per path.
#ifndef WF_COMMON_GLSL
#define WF_COMMON_GLSL

#extension GL_EXT_ray_query : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : require

#define WF_WORKGROUP_SIZE 256

// ------------------------------------------------------------
// set 0 — scene data (IDENTICAL bindings to trace.rgen, so a kernel can read the scene the same way
// the megakernel does). Declared here once; individual kernels use only the subset they need.
// ------------------------------------------------------------
layout(binding = 0, rgba32f) uniform image2D outImage;
layout(binding = 1) uniform accelerationStructureEXT tlas;

layout(buffer_reference, std430) readonly buffer Verts   { float v[]; };
layout(buffer_reference, std430) readonly buffer Norms   { float n[]; };
layout(buffer_reference, std430) readonly buffer Uvs     { float u[]; };
layout(buffer_reference, std430) readonly buffer Indices { uint  i[]; };
layout(buffer_reference, std430) readonly buffer MatIds  { uint  m[]; };
struct GeomDesc { uint64_t vtx; uint64_t idx; uint64_t nrm; uint64_t mat; uint64_t uv; uint64_t pad; };

// Full GPUMaterial (2204 bytes), scalar layout — byte-for-byte identical to trace.rgen's Material.
struct Material {
    uint diffuseTexIndex, normalTexIndex, mapsTexIndex, detailTexIndex;
    uint normalDetailTexIndex;
    float baseR, baseG, baseB;
    float roughness;
    float emissiveR, emissiveG, emissiveB;
    float fresnelC;
    float fresnelEXP, detailUVMultiplier, detailNormalBlend;
    uint flags; float alphaRef; uint shaderType;
    float fresnelMaxLevel;
    uint maskTexIndex, detailRTexIndex, detailGTexIndex, detailBTexIndex;
    uint detailATexIndex, detailNMTexIndex;
    float multR, multG, multB, multA, magicMult, detailNMMult, detailNMMultV;
    float sunSpecular, sunSpecularEXP;
    uint nodeVmHeader, nodeVmPad0, nodeVmPad1, nodeVmPad2;
    uvec4 nodeVmCode[128];
};
layout(binding = 2, std430) readonly buffer GeomTable  { GeomDesc descs[]; };
layout(binding = 3, scalar) readonly buffer Materials  { Material mats[]; };
layout(binding = 4) uniform sampler2D textures[];
struct Light { vec4 posRange; vec4 colInt; vec4 dirSizeX; vec4 tanSizeY; };
layout(binding = 5, std430) readonly buffer Lights { Light lights[]; };
layout(binding = 6, rgba32f) uniform image2D accumImage;
layout(binding = 8, std430) readonly buffer World { vec4 worldColor; vec4 hdriParams; };
// DLSS-RR guide buffers (written at the primary hit when hasTlas bit2 is set).
layout(binding = 9,  r32f)    uniform image2D gDepth;
layout(binding = 10, rgba16f) uniform image2D gNormalRough;
layout(binding = 11, rgba16f) uniform image2D gAlbedo;
layout(binding = 12, rgba16f) uniform image2D gMotion;
layout(binding = 13, rgba16f) uniform image2D gNoisy;
layout(binding = 15, rgba16f) uniform image2D gSpecAlbedo;
layout(binding = 16, std430) readonly buffer EmissiveTris { vec4 data[]; } emissiveTris;
layout(binding = 17, std430) readonly buffer EnvDist      { float data[]; } envDist;
layout(binding = 18, std430) readonly buffer PrevXforms   { vec4 rows[]; } prevXforms;

// Push constant — IDENTICAL to trace.rgen's PC block (so the host pushes the same 208-byte struct).
layout(push_constant) uniform PC {
    mat4  invViewProj;
    vec4  camPos;        // xyz = world pos, w = sun angular radius
    uvec2 dims;          // trace_width, trace_height
    uint  hasTlas;       // bitfield: 1=tlas 2=lut 4=writeGuides 8=dlssRR 16=ser 32=hdri 64=motion
    uint  numLights;
    vec4  sunDir;        // xyz dir, w intensity
    vec4  sunCol;        // rgb color, w = accumulated frame index
    mat4  prevViewProj;
    vec2  jitter;
    uint  emissiveCount;
} pc;

// ------------------------------------------------------------
// Wavefront working-set structs (ported verbatim from C++ wf_structs.glsl — binding-agnostic).
// The matching Rust #[repr(C)] structs MUST keep these byte sizes under scalar layout.
// ------------------------------------------------------------

// Per-path state surviving across bounce iterations (48 bytes).
struct PathState {
    vec3 origin;       uint pixelIndex;
    vec3 direction;    uint rngState;
    vec3 throughput;   uint flags;     // [0:3]=bounce [4]=isDiffuse [5]=terminated [6]=isGlass
};

// Intersection result from the intersect kernel (32 bytes).
struct HitResult {
    float hitT;  uint customIndex; int instanceId; int primitiveId;
    vec2 barycentrics; uint hitType; uint pad0;    // hitType: 0=miss 1=hit
};

// Deferred NEE shadow ray (48 bytes).
struct ShadowRay {
    vec3 origin;    float maxDist;
    vec3 direction; uint pathIndex;
    vec3 radiance;  uint channel;     // 0=diffuse 1=specular
};

// Per-pixel radiance accumulator (32 bytes).
struct PixelRadiance {
    vec3 diffuse;  float hitDist;
    vec3 specular; float pad0;
};

// Primary-hit G-buffer for the DLSS guides (64 bytes).
struct PrimaryGBuffer {
    vec3 normal;   float roughness;
    vec3 worldPos; float viewZ;
    vec3 albedo;   float hitDist;
    vec3 localPos; uint instanceId;
    uint customIndex; float metallic; float penumbra; float primaryNdotL;
};

// Atomic dispatch counters (16 bytes).
struct WavefrontCounters {
    uint activeRayCount;      // paths to process this bounce
    uint shadowRayCount;      // shadow rays generated this bounce
    uint nextActiveRayCount;  // paths continuing to the next bounce
    uint pad0;
};

// ------------------------------------------------------------
// set 1 — wavefront private working set (NEW; not present in the megakernel path).
// MVP layout: AoS PathState + an index queue for compaction (simpler/correct first; the C++ SoA
// double-buffer split is a later bandwidth optimisation). One PathState/Hit/GBuf entry per pixel.
// ------------------------------------------------------------
layout(set = 1, binding = 0, scalar) buffer Paths     { PathState     paths[];    };
layout(set = 1, binding = 1, scalar) buffer Hits      { HitResult     hits[];     };
layout(set = 1, binding = 2, scalar) buffer Shadows   { ShadowRay     shadows[];  };
layout(set = 1, binding = 3, scalar) buffer Radiance  { PixelRadiance radiance[]; };
layout(set = 1, binding = 4, scalar) buffer Counters  { WavefrontCounters counters; };
layout(set = 1, binding = 5, scalar) buffer ActiveIdx { uint activeIdx[]; };  // active path indices, this bounce
layout(set = 1, binding = 6, scalar) buffer NextIdx   { uint nextIdx[];   };  // compaction target, next bounce
layout(set = 1, binding = 7, scalar) buffer Indirect  { uint indirect[];  };  // VkDispatchIndirectCommand[] (x,y,z)
layout(set = 1, binding = 8, scalar) buffer GBuffer   { PrimaryGBuffer gbuf[]; }; // per-pixel primary G-buffer

// ------------------------------------------------------------
// flag bit helpers (from wf_structs.glsl).
// ------------------------------------------------------------
uint wf_getBounce(uint flags)            { return flags & 0xFu; }
uint wf_setBounce(uint flags, uint b)    { return (flags & ~0xFu) | (b & 0xFu); }
bool wf_isDiffuse(uint flags)            { return (flags & 0x10u) != 0u; }
bool wf_isTerminated(uint flags)         { return (flags & 0x20u) != 0u; }
bool wf_isGlass(uint flags)              { return (flags & 0x40u) != 0u; }
uint wf_setDiffuse(uint flags, bool v)   { return v ? (flags | 0x10u) : (flags & ~0x10u); }
uint wf_setTerminated(uint flags)        { return flags | 0x20u; }
uint wf_setGlass(uint flags, bool v)     { return v ? (flags | 0x40u) : (flags & ~0x40u); }

// PCG hash RNG (same stream the megakernel uses, so per-pixel sequences line up for A/B parity).
uint wf_pcg(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float wf_rand(inout uint state) { return float(wf_pcg(state)) * (1.0 / 4294967296.0); }

#endif // WF_COMMON_GLSL
