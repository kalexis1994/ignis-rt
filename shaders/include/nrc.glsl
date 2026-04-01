/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * GLSL/Vulkan port of NRC (Neural Radiance Cache) shader includes.
 * Combined from: NrcStructures.h, NrcHelpers.hlsli, Nrc.hlsli
 *
 * Integration pattern for Vulkan GLSL (struct types must exist before
 * SSBO declarations that reference them):
 *
 *   // Phase 1 -- pull in struct types, enums, type aliases only
 *   #define NRC_STRUCTURES_ONLY
 *   #include "nrc.glsl"
 *
 *   // Declare SSBOs using the NRC struct types
 *   layout(set=0,binding=0) buffer BufA { NrcPackedQueryPathInfo    d[]; } nrc_qpi;
 *   layout(set=0,binding=1) buffer BufB { NrcPackedTrainingPathInfo d[]; } nrc_tpi;
 *   layout(set=0,binding=2) buffer BufC { NrcPackedPathVertex       d[]; } nrc_tpv;
 *   layout(set=0,binding=3) buffer BufD { NrcRadianceParams         d[]; } nrc_qrp;
 *   layout(set=0,binding=4) buffer BufE { uint                      d[]; } nrc_cnt;
 *
 *   // Wire up accessor macros
 *   #define NRC_USE_CUSTOM_BUFFER_ACCESSORS 1
 *   #define NRC_BUFFER_QUERY_PATH_INFO        nrc_qpi.d
 *   #define NRC_BUFFER_TRAINING_PATH_INFO     nrc_tpi.d
 *   #define NRC_BUFFER_TRAINING_PATH_VERTICES nrc_tpv.d
 *   #define NRC_BUFFER_QUERY_RADIANCE_PARAMS  nrc_qrp.d
 *   #define NRC_BUFFER_QUERY_COUNTERS_DATA    nrc_cnt.d
 *
 *   // Phase 2 -- pull in helper functions and public API
 *   #define NRC_UPDATE 1          // or NRC_QUERY 1, or ENABLE_NRC 0
 *   #include "nrc.glsl"
 *
 * If NRC is disabled (ENABLE_NRC 0) and you do not need SSBOs, a single
 * #include "nrc.glsl" without NRC_STRUCTURES_ONLY is sufficient -- the
 * struct types and stub functions are all emitted in one pass.
 */

// =========================================================================
//  Phase 1: Structures, type aliases, constants
//  Guarded by NRC_GLSL_STRUCTS so they are only emitted once.
// =========================================================================

#ifndef NRC_GLSL_STRUCTS
#define NRC_GLSL_STRUCTS

#extension GL_KHR_shader_subgroup_ballot : enable

// --- Configuration defaults ---

#ifndef NRC_PACK_PATH_16BITS
#define NRC_PACK_PATH_16BITS 0
#endif

#ifndef TCNN_USES_FIXED_POINT_POSITIONS
#define TCNN_USES_FIXED_POINT_POSITIONS 0
#endif

// --- Type aliases (float precision: always fp32 in this port) ---

#define nrc_float16_t  float
#define nrc_float16_t2 vec2
#define nrc_uint       uint
#define nrc_uint2      uvec2
#define nrc_uint3      uvec3
#define nrc_float2     vec2
#define nrc_float3     vec3
#define nrc_float4     vec4

#define NrcPackableUint  uint
#define NrcPackableFloat float
#define NrcEncodedPosition vec3

// --- Enum replacements (const uint) ---

// NrcCounter
const uint NRC_COUNTER_QUERIES          = 0;
const uint NRC_COUNTER_TRAINING_RECORDS = 1;
const uint NRC_COUNTER_COUNT            = 2;

// NrcDebugPathTerminationReason
const uint NRC_DEBUG_TERM_UNSET                     = 0;
const uint NRC_DEBUG_TERM_PATH_MISS_EXIT            = 1;
const uint NRC_DEBUG_TERM_CREATE_QUERY_IMMEDIATE    = 2;
const uint NRC_DEBUG_TERM_MAX_PATH_VERTICES         = 3;
const uint NRC_DEBUG_TERM_CREATE_QUERY_AFTER_DIRECT = 4;
const uint NRC_DEBUG_TERM_RUSSIAN_ROULETTE          = 5;
const uint NRC_DEBUG_TERM_BRDF_ABSORPTION           = 6;
const uint NRC_DEBUG_TERM_COUNT                     = 7;

// NrcResolveMode
const uint NRC_RESOLVE_ADD_QUERY_RESULT_TO_OUTPUT                  = 0;
const uint NRC_RESOLVE_REPLACE_OUTPUT_WITH_QUERY_RESULT            = 1;
const uint NRC_RESOLVE_TRAINING_BOUNCE_HEATMAP                     = 2;
const uint NRC_RESOLVE_TRAINING_BOUNCE_HEATMAP_SMOOTHED            = 3;
const uint NRC_RESOLVE_PRIMARY_VERTEX_TRAINING_RADIANCE            = 4;
const uint NRC_RESOLVE_PRIMARY_VERTEX_TRAINING_RADIANCE_SMOOTHED   = 5;
const uint NRC_RESOLVE_SECONDARY_VERTEX_TRAINING_RADIANCE          = 6;
const uint NRC_RESOLVE_SECONDARY_VERTEX_TRAINING_RADIANCE_SMOOTHED = 7;
const uint NRC_RESOLVE_QUERY_INDEX                                 = 8;
const uint NRC_RESOLVE_TRAINING_QUERY_INDEX                        = 9;
const uint NRC_RESOLVE_DIRECT_CACHE_VIEW                           = 10;

// NrcMode
const uint NRC_MODE_DISABLED = 0;
const uint NRC_MODE_UPDATE   = 1;
const uint NRC_MODE_QUERY    = 2;

// NrcProgressState
const uint NRC_PROGRESS_CONTINUE                        = 0;
const uint NRC_PROGRESS_TERMINATE_IMMEDIATELY           = 1;
const uint NRC_PROGRESS_TERMINATE_AFTER_DIRECT_LIGHTING = 2;

// --- Structures ---

struct NrcRadianceParams
{
    vec3  encodedPosition;
    float roughness;
    vec2  normal;
    vec2  viewDirection;
    vec3  albedo;
    vec3  specular;
};

struct NrcTrainingPathInfo
{
    uint packedData;
    uint queryBufferIndex;
};

struct NrcPackedTrainingPathInfo
{
    uint packedData;
    uint queryBufferIndex;
};

struct NrcQueryPathInfo
{
    vec3 prefixThroughput;
    uint queryBufferIndex;
};

struct NrcPackedQueryPathInfo
{
    uint prefixThroughput;
    uint queryBufferIndex;
};

struct NrcPathVertex
{
    vec3  radiance;
    vec3  throughput;
    vec3  encodedPosition;
    float linearRoughness;
    vec3  normal;
    vec3  viewDirection;
    vec3  albedo;
    vec3  specular;
};

struct NrcPackedPathVertex
{
    uint data[7];
    uint pad0;
    vec3 encodedPosition;
    uint pad1;
};

struct NrcConstants
{
    uvec2 frameDimensions;
    uvec2 trainingDimensions;

    vec3  scenePosScale;
    uint  samplesPerPixel;

    vec3  scenePosBias;
    uint  maxPathVertices;

    uint  learnIrradiance;
    uint  radianceCacheDirect;
    float radianceUnpackMultiplier;
    uint  resolveMode;

    uint  enableTerminationHeuristic;
    uint  skipDeltaVertices;
    float terminationHeuristicThreshold;
    float trainingTerminationHeuristicThreshold;

    float proportionUnbiased;
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

struct NrcDebugTrainingPathInfo
{
    vec3 debugRadiance;
    vec3 accumulation;
};

struct NrcSurfaceAttributes
{
    vec3  encodedPosition;
    float roughness;
    vec3  specularF0;
    vec3  diffuseReflectance;
    vec3  shadingNormal;
    vec3  viewVector;
    bool  isDeltaLobe;
};

struct NrcPathState
{
    uint  packedPrefixThroughput;
    uint  queryBufferIndex;
    float primarySpreadRadius;
    float cumulSpreadRadius;
    uint  packedData;
    float brdfPdf;
};

// NrcBuffers: empty in GLSL custom-buffer mode.  Actual storage is
// accessed through NRC_BUFFER_* macros defined by the integrator.
struct NrcBuffers
{
    uint _unused;
};

struct NrcContext
{
    NrcConstants constants;
    NrcBuffers   buffers;
    uvec2        pixelIndex;
    uint         sampleIndex;
};

#endif // NRC_GLSL_STRUCTS

// =========================================================================
//  Early-out when only structures are requested
// =========================================================================

#ifdef NRC_STRUCTURES_ONLY
// Stop here.  The integrator will #undef NRC_STRUCTURES_ONLY, declare
// SSBOs, set up NRC_BUFFER_* macros, and re-include this file.
#else // full include follows

// =========================================================================
//  Phase 2: Functions (helpers + public API)
//  Guarded by NRC_GLSL_FUNCS so they are only emitted once.
// =========================================================================

#ifndef NRC_GLSL_FUNCS
#define NRC_GLSL_FUNCS

// --- Buffer accessor defaults ---

#ifndef NRC_USE_CUSTOM_BUFFER_ACCESSORS
#define NRC_USE_CUSTOM_BUFFER_ACCESSORS 1
#endif

// --- NRC mode state ---

#if defined(ENABLE_NRC)
    #if !((ENABLE_NRC == 0) || (ENABLE_NRC == 1))
        #error "If you define ENABLE_NRC, set it to 0 or 1"
    #endif
#else
    #if (defined(NRC_UPDATE) || defined(NRC_QUERY))
        #define ENABLE_NRC 1
    #else
        #define ENABLE_NRC 0
    #endif
#endif

#if defined(NRC_UPDATE) && (NRC_UPDATE == 1)
    #if !defined(NRC_QUERY)
        #define NRC_QUERY 0
    #endif
    const uint g_nrcMode = NRC_MODE_UPDATE;
#elif defined(NRC_QUERY) && (NRC_QUERY == 1)
    #if !defined(NRC_UPDATE)
        #define NRC_UPDATE 0
    #endif
    const uint g_nrcMode = NRC_MODE_QUERY;
#elif ENABLE_NRC == 1
    uint g_nrcMode = NRC_MODE_DISABLED;
#else
    #if !defined(NRC_QUERY)
        #define NRC_QUERY 0
    #endif
    #if !defined(NRC_UPDATE)
        #define NRC_UPDATE 0
    #endif
    const uint g_nrcMode = NRC_MODE_DISABLED;
#endif

// =========================================================================
//  Helpers  (ported from NrcHelpers.hlsli)
// =========================================================================

// --- Half-float helpers for R11G11B10 packing ---

float nrc_f16tof32(uint x)
{
    return unpackHalf2x16(x).x;
}

uint nrc_f32tof16(float x)
{
    return packHalf2x16(vec2(x, 0.0)) & 0xFFFFu;
}

// --- NaN / Inf checks (bit-exact, cannot be optimized away) ---

bool NrcIsNan(float x)
{
    return (floatBitsToUint(x) & 0x7FFFFFFFu) > 0x7F800000u;
}

bool NrcIsInf(float x)
{
    return (floatBitsToUint(x) & 0x7FFFFFFFu) == 0x7F800000u;
}

bool NrcIsNan(vec3 v)
{
    return NrcIsNan(v.x) || NrcIsNan(v.y) || NrcIsNan(v.z);
}

bool NrcIsInf(vec3 v)
{
    return NrcIsInf(v.x) || NrcIsInf(v.y) || NrcIsInf(v.z);
}

vec3 NrcSanitizeNansInfs(vec3 input_val)
{
    if (NrcIsInf(input_val) || NrcIsNan(input_val))
    {
        return vec3(0.0);
    }
    return input_val;
}

// --- Unorm / Snorm packing ---

uint NrcPackUnorm16Unsafe(float v)
{
    return uint(trunc(v * 65535.0 + 0.5));
}

uint NrcPackUnorm16(float v)
{
    v = NrcIsNan(v) ? 0.0 : clamp(v, 0.0, 1.0);
    return NrcPackUnorm16Unsafe(v);
}

uint NrcPackUnorm2x16(vec2 v)
{
    return (NrcPackUnorm16(v.y) << 16u) | NrcPackUnorm16(v.x);
}

vec2 NrcUnpackUnorm2x16(uint packed)
{
    return vec2(float(packed & 0xFFFFu), float(packed >> 16u)) * (1.0 / 65535.0);
}

int NrcFloatToSnorm16(float v)
{
    v = NrcIsNan(v) ? 0.0 : min(max(v, -1.0), 1.0);
    return int(trunc(v * 32767.0 + (v >= 0.0 ? 0.5 : -0.5)));
}

uint NrcPackSnorm2x16(vec2 v)
{
    return (uint(NrcFloatToSnorm16(v.x)) & 0x0000FFFFu) | (uint(NrcFloatToSnorm16(v.y)) << 16u);
}

vec2 NrcUnpackSnorm2x16(uint packed)
{
    ivec2 bits = ivec2(int(packed << 16u), int(packed)) >> 16;
    vec2 unpacked = max(vec2(bits) / 32767.0, vec2(-1.0));
    return unpacked;
}

// --- Octahedral encoding ---

vec2 NrcOctWrap(vec2 v)
{
    return vec2(
        (1.0 - abs(v.y)) * (v.x >= 0.0 ? 1.0 : -1.0),
        (1.0 - abs(v.x)) * (v.y >= 0.0 ? 1.0 : -1.0)
    );
}

vec3 NrcOctToDirection(vec2 p)
{
    vec3 n = vec3(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    vec2 tmp = (n.z < 0.0) ? NrcOctWrap(vec2(n.x, n.y)) : vec2(n.x, n.y);
    n.x = tmp.x;
    n.y = tmp.y;
    return normalize(n);
}

vec2 NrcDirectionToOct(vec3 n)
{
    vec2 p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    p = (n.z < 0.0) ? NrcOctWrap(p) : p;
    return p;
}

uint NrcEncodeNormal2x16(vec3 normal)
{
    vec2 octNormal = NrcDirectionToOct(normal);
    return NrcPackSnorm2x16(octNormal);
}

vec3 NrcDecodeNormal2x16(uint packedNormal)
{
    vec2 octNormal = NrcUnpackSnorm2x16(packedNormal);
    return NrcOctToDirection(octNormal);
}

// --- R11G11B10 packing ---
// Reference: DirectX-Graphics-Samples MiniEngine PixelPacking_R11G11B10.hlsli

vec3 NrcUnpackR11G11B10(uint packed)
{
    float r = nrc_f16tof32((packed << 4u) & 0x7FF0u);
    float g = nrc_f16tof32((packed >> 7u) & 0x7FF0u);
    float b = nrc_f16tof32((packed >> 17u) & 0x7FE0u);
    return vec3(r, g, b);
}

uint NrcPackR11G11B10(vec3 v)
{
    v = min(v, uintBitsToFloat(0x477C0000u));
    uint r = ((nrc_f32tof16(v.x) + 8u) >> 4u) & 0x000007FFu;
    uint g = ((nrc_f32tof16(v.y) + 8u) << 7u) & 0x003FF800u;
    uint b = ((nrc_f32tof16(v.z) + 16u) << 17u) & 0xFFC00000u;
    return r | g | b;
}

// --- Color space conversion ---

vec3 NrcRgbToXyzRec709(vec3 c)
{
    // GLSL is column-major; columns correspond to HLSL rows
    const mat3 M = mat3(
        0.4123907992659595, 0.2126390058715104, 0.0193308187155918,
        0.3575843393838780, 0.7151686787677559, 0.1191947797946259,
        0.1804807884018343, 0.0721923153607337, 0.9505321522496608
    );
    return M * c;
}

vec3 NrcXyzToRgbRec709(vec3 c)
{
    const mat3 M = mat3(
         3.240969941904522,  -0.9692436362808803,  0.05563007969699373,
        -1.537383177570094,   1.875967501507721,  -0.2039769588889765,
        -0.4986107602930032,  0.04155505740717569, 1.056971514242878
    );
    return M * c;
}

// --- LogLuv HDR encoding ---

uint NrcEncodeLogLuvHdr(vec3 color)
{
    vec3 XYZ = NrcRgbToXyzRec709(color);

    float logY = 409.6 * (log2(XYZ.y) + 20.0);
    uint Le = uint(clamp(logY, 0.0, 16383.0));

    if (Le == 0u)
    {
        return 0u;
    }

    float invDenom = 1.0 / (-2.0 * XYZ.x + 12.0 * XYZ.y + 3.0 * (XYZ.x + XYZ.y + XYZ.z));
    vec2 uv = vec2(4.0, 9.0) * XYZ.xy * invDenom;

    uvec2 uve = uvec2(clamp(820.0 * uv, vec2(0.0), vec2(511.0)));

    return (Le << 18u) | (uve.x << 9u) | uve.y;
}

vec3 NrcDecodeLogLuvHdr(uint packedColor)
{
    uint Le = packedColor >> 18u;
    if (Le == 0u)
    {
        return vec3(0.0);
    }

    float logY = (float(Le) + 0.5) / 409.6 - 20.0;
    float Y = pow(2.0, logY);

    uvec2 uve = uvec2(packedColor >> 9u, packedColor) & uvec2(0x1FFu);
    vec2 uv = (vec2(uve) + 0.5) / 820.0;

    float invDenom = 1.0 / (6.0 * uv.x - 16.0 * uv.y + 12.0);
    vec2 xy = vec2(9.0, 4.0) * uv * invDenom;

    float s = Y / xy.y;
    vec3 XYZ = vec3(s * xy.x, Y, s * (1.0 - xy.x - xy.y));

    return max(NrcXyzToRgbRec709(XYZ), vec3(0.0));
}

// --- Spherical coordinate conversion ---

vec2 NrcCartesianToSphericalUnorm(vec3 p)
{
    const float kOneOverPi    = 0.318309886183790671538;
    const float kOneOverTwoPi = 0.159154943091895335769;

    p = normalize(p);
    vec2 sph;
    sph.x = acos(p.z) * kOneOverPi;
    sph.y = atan(-p.y, -p.x) * kOneOverTwoPi + 0.5;
    return sph;
}

vec2 NrcSafeCartesianToSphericalUnorm(vec3 p)
{
    if (any(isinf(p)) || any(isnan(p)))
    {
        return vec2(0.0);
    }
    return NrcCartesianToSphericalUnorm(p);
}

// --- Buffer index calculation ---

uint NrcCalculateQueryPathIndex(const uvec2 frameDimensions, const uvec2 pixel, const uint sampleIndex, const uint samplesPerPixel)
{
    uint index  = sampleIndex;
    uint stride = samplesPerPixel;
    index  += pixel.x * stride;
    stride *= frameDimensions.x;
    index  += pixel.y * stride;
    return index;
}

uint NrcCalculateTrainingPathIndex(const uvec2 trainingDimensions, const uvec2 pixel)
{
    return (trainingDimensions.x * pixel.y) + pixel.x;
}

uint NrcCalculateTrainingPathVertexIndex(uvec2 trainingDimensions, uvec2 pixel, const uint vertexIdx, const uint maxPathVertices)
{
    uint trainingPathIndex = NrcCalculateTrainingPathIndex(trainingDimensions, pixel);
    return trainingPathIndex * maxPathVertices + vertexIdx;
}

// --- Packed path vertex helpers ---

NrcPackedPathVertex NrcUpdateTrainingPathVertex(NrcPackedPathVertex packedPathVertex, const vec3 radiance, const vec3 throughput)
{
    packedPathVertex.data[0] = NrcEncodeLogLuvHdr(NrcSanitizeNansInfs(radiance));
    packedPathVertex.data[1] = NrcEncodeLogLuvHdr(NrcSanitizeNansInfs(throughput));
    return packedPathVertex;
}

NrcPathVertex NrcUnpackPathVertexWithRT(const NrcPackedPathVertex packed, const vec3 radiance, const vec3 throughput)
{
    NrcPathVertex vertex;
    vertex.radiance        = NrcSanitizeNansInfs(radiance);
    vertex.throughput      = NrcSanitizeNansInfs(throughput);
    vertex.linearRoughness = uintBitsToFloat(packed.data[2]);
    vertex.normal          = NrcDecodeNormal2x16(packed.data[3]);
    vertex.viewDirection   = NrcDecodeNormal2x16(packed.data[4]);
    vertex.albedo          = NrcUnpackR11G11B10(packed.data[5]);
    vertex.specular        = NrcUnpackR11G11B10(packed.data[6]);
    vertex.encodedPosition = packed.encodedPosition;
    return vertex;
}

NrcPathVertex NrcUnpackPathVertex(const NrcPackedPathVertex packed)
{
    NrcPathVertex vertex;
    vertex.radiance        = NrcDecodeLogLuvHdr(packed.data[0]);
    vertex.throughput      = NrcDecodeLogLuvHdr(packed.data[1]);
    vertex.linearRoughness = uintBitsToFloat(packed.data[2]);
    vertex.normal          = NrcDecodeNormal2x16(packed.data[3]);
    vertex.viewDirection   = NrcDecodeNormal2x16(packed.data[4]);
    vertex.albedo          = NrcUnpackR11G11B10(packed.data[5]);
    vertex.specular        = NrcUnpackR11G11B10(packed.data[6]);
    vertex.encodedPosition = packed.encodedPosition;
    return vertex;
}

// Overload: unpack with explicit radiance/throughput
NrcPathVertex NrcUnpackPathVertex(const NrcPackedPathVertex packed, const vec3 radiance, const vec3 throughput)
{
    return NrcUnpackPathVertexWithRT(packed, radiance, throughput);
}

// --- Counter increment (subgroup-optimized) ---

#if ENABLE_NRC

uint NrcIncrementCounter(uint counter)
{
    uint laneCount   = subgroupBallotBitCount(subgroupBallot(true));
    uint laneOffset  = subgroupBallotExclusiveBitCount(subgroupBallot(true));
    uint originalValue;
    if (subgroupElect())
    {
        originalValue = atomicAdd(NRC_BUFFER_QUERY_COUNTERS_DATA[counter], laneCount);
    }
    originalValue = subgroupBroadcastFirst(originalValue);
    return originalValue + laneOffset;
}

#endif // ENABLE_NRC

// --- Position encoding ---

vec3 NrcEncodePosition(vec3 worldSpacePosition, NrcConstants nrcConstants)
{
    return fma(worldSpacePosition, nrcConstants.scenePosScale, nrcConstants.scenePosBias);
}

vec3 NrcEncodePosition(vec3 localPositionOffset, vec3 localOrigin, NrcConstants nrcConstants)
{
    return fma(localOrigin + localPositionOffset, nrcConstants.scenePosScale, nrcConstants.scenePosBias);
}

// --- Radiance params creation ---

NrcRadianceParams NrcCreateRadianceParams(const NrcPathVertex vertex)
{
    NrcRadianceParams params;
    params.encodedPosition = vertex.encodedPosition;
    params.roughness       = vertex.linearRoughness;
    params.normal          = NrcSafeCartesianToSphericalUnorm(vertex.normal);
    params.viewDirection   = NrcSafeCartesianToSphericalUnorm(vertex.viewDirection);
    params.albedo          = vertex.albedo;
    params.specular        = vertex.specular;
    return params;
}

// --- Packed path vertex initialization ---

NrcPackedPathVertex NrcInitializePackedPathVertex(
    const float linearRoughness,
    const vec3  shadingNormal,
    const vec3  viewDirection,
    const vec3  diffuseAlbedo,
    const vec3  specularAlbedo,
    const vec3  encodedPosition)
{
    NrcPackedPathVertex v;
    v.data[0] = NrcEncodeLogLuvHdr(vec3(0.0));
    v.data[1] = NrcEncodeLogLuvHdr(vec3(1.0));
    v.data[2] = floatBitsToUint(linearRoughness);
    v.data[3] = NrcEncodeNormal2x16(shadingNormal);
    v.data[4] = NrcEncodeNormal2x16(viewDirection);
    v.data[5] = NrcPackR11G11B10(diffuseAlbedo);
    v.data[6] = NrcPackR11G11B10(specularAlbedo);
    v.encodedPosition = encodedPosition;
    v.pad0 = 0u;
    v.pad1 = 0u;
    return v;
}

// --- Path info pack / unpack ---

NrcPackedTrainingPathInfo NrcPackTrainingPathInfo(const NrcTrainingPathInfo pathInfo)
{
    NrcPackedTrainingPathInfo packedPathInfo;
    packedPathInfo.packedData       = pathInfo.packedData;
    packedPathInfo.queryBufferIndex = pathInfo.queryBufferIndex;
    return packedPathInfo;
}

NrcTrainingPathInfo NrcUnpackTrainingPathInfo(const NrcPackedTrainingPathInfo packedPathInfo)
{
    NrcTrainingPathInfo pathInfo;
    pathInfo.packedData       = packedPathInfo.packedData;
    pathInfo.queryBufferIndex = packedPathInfo.queryBufferIndex;
    return pathInfo;
}

NrcQueryPathInfo NrcUnpackQueryPathInfo(const NrcPackedQueryPathInfo packedPathInfo)
{
    NrcQueryPathInfo pathInfo;
    pathInfo.prefixThroughput = NrcDecodeLogLuvHdr(packedPathInfo.prefixThroughput);
    pathInfo.queryBufferIndex = packedPathInfo.queryBufferIndex;
    return pathInfo;
}

vec3 NrcUnpackQueryRadiance(const NrcConstants nrcConstants, vec3 packedQueryRadiance)
{
    return packedQueryRadiance * nrcConstants.radianceUnpackMultiplier;
}

// =========================================================================
//  NRC Public API  (ported from Nrc.hlsli)
// =========================================================================

// --- Mode queries ---

bool NrcIsEnabled()
{
    return g_nrcMode != NRC_MODE_DISABLED;
}

bool NrcIsUpdateMode()
{
    return g_nrcMode == NRC_MODE_UPDATE;
}

bool NrcIsQueryMode()
{
    return g_nrcMode == NRC_MODE_QUERY;
}

#if ENABLE_NRC

// -------------------------------------------------------------------------
//  Internal helpers
// -------------------------------------------------------------------------

bool NrcEvaluateTerminationHeuristic(const NrcPathState pathState, float threshold)
{
    return (pathState.primarySpreadRadius > 0.0) && (pathState.cumulSpreadRadius > (threshold * pathState.primarySpreadRadius));
}

// packedData bit layout:
//   [6:0]   Vertex Count
//   [11:7]  Termination Reason
//   [15:12] Flags
const uint nrcTerminationReasonShift          = 7u;
const uint nrcPathFlagsShift                  = 12u;
const uint nrcPathFlagHasExitedScene          = (1u << (nrcPathFlagsShift + 0u));
const uint nrcPathFlagIsUnbiased              = (1u << (nrcPathFlagsShift + 1u));
const uint nrcPathFlagPreviousHitWasDeltaLobe = (1u << (nrcPathFlagsShift + 2u));
const uint nrcPathFlagHeuristicReset          = (1u << (nrcPathFlagsShift + 3u));
const uint nrcVertexCountMask                 = ((1u << nrcTerminationReasonShift) - 1u);
const uint nrcTerminationReasonMask           = (((1u << nrcPathFlagsShift) - 1u) & ~nrcVertexCountMask);

void NrcSetFlag(inout uint packedData, in uint flag)
{
    packedData |= flag;
}

void NrcClearFlag(inout uint packedData, in uint flag)
{
    packedData &= ~flag;
}

void NrcSetFlagValue(inout uint packedData, in uint flag, in bool value)
{
    packedData &= ~flag;
    packedData |= value ? flag : 0u;
}

bool NrcGetFlag(in uint packedData, in uint flag)
{
    return (packedData & flag) != 0u;
}

uint NrcGetVertexCount(in uint packedData)
{
    return (packedData & nrcVertexCountMask);
}

void NrcSetVertexCount(inout uint packedData, uint vertexCount)
{
    packedData &= ~nrcVertexCountMask;
    packedData |= vertexCount;
}

// -------------------------------------------------------------------------
//  Public NRC Shader API
// -------------------------------------------------------------------------

void NrcSetDebugPathTerminationReason(inout NrcPathState pathState, uint reason)
{
    pathState.packedData &= ~nrcTerminationReasonMask;
    pathState.packedData |= reason << nrcTerminationReasonShift;
}

uint NrcGetDebugPathTerminationReason(in NrcPathState pathState)
{
    return (pathState.packedData & nrcTerminationReasonMask) >> nrcTerminationReasonShift;
}

NrcContext NrcCreateContext(in NrcConstants constants, in NrcBuffers buffers, in uvec2 pixelIndex)
{
    NrcContext context;
    context.constants   = constants;
    context.buffers     = buffers;
    context.pixelIndex  = pixelIndex;
    context.sampleIndex = 0u;
    return context;
}

NrcPathState NrcCreatePathState(in NrcConstants constants, float rand0to1)
{
    NrcPathState pathState;
    pathState.queryBufferIndex       = 0xFFFFFFFFu;
    pathState.packedPrefixThroughput = 0u;
    pathState.cumulSpreadRadius      = 0.0;
    pathState.primarySpreadRadius    = 0.0;
    pathState.packedData             = 0u;
    pathState.brdfPdf                = 0.0;

    bool isUnbiased = NrcIsUpdateMode() && (rand0to1 < constants.proportionUnbiased);
    NrcSetFlagValue(pathState.packedData, nrcPathFlagIsUnbiased, isUnbiased);

    return pathState;
}

void NrcSetSampleIndex(inout NrcContext context, in uint sampleIndex)
{
    context.sampleIndex = sampleIndex;
}

bool NrcCanUseRussianRoulette(in NrcPathState pathState)
{
    return !NrcGetFlag(pathState.packedData, nrcPathFlagIsUnbiased);
}

uint NrcUpdateOnHit(
    in NrcContext context,
    inout NrcPathState pathState,
    NrcSurfaceAttributes surfaceAttributes,
    float hitDistance,
    uint bounce,
    inout vec3 throughput,
    inout vec3 radiance)
{
    if (!NrcIsEnabled())
    {
        return NRC_PROGRESS_CONTINUE;
    }

    // Update the path spread approximation
    const float cosGamma = abs(dot(surfaceAttributes.viewVector, surfaceAttributes.shadingNormal));
    if (pathState.primarySpreadRadius == 0.0)
    {
        const float kOneOverFourPI = 0.079577471545947667884;
        pathState.primarySpreadRadius = hitDistance / sqrt(cosGamma * kOneOverFourPI);
    }
    else if (!NrcGetFlag(pathState.packedData, nrcPathFlagPreviousHitWasDeltaLobe))
    {
        pathState.cumulSpreadRadius += hitDistance / sqrt(cosGamma * pathState.brdfPdf);
    }
    NrcSetFlagValue(pathState.packedData, nrcPathFlagPreviousHitWasDeltaLobe, surfaceAttributes.isDeltaLobe);

    // Determine if we want to skip querying NRC at this bounce
    const bool skipVertex = (context.constants.skipDeltaVertices != 0u || context.constants.enableTerminationHeuristic != 0u) && surfaceAttributes.isDeltaLobe;
    if (skipVertex)
    {
        return NRC_PROGRESS_CONTINUE;
    }

    uint vertexCount = NrcGetVertexCount(pathState.packedData);
    if (NrcIsUpdateMode())
    {
        // Write training path vertex information
        const uint trainingPathVertexIndex = NrcCalculateTrainingPathVertexIndex(context.constants.trainingDimensions, context.pixelIndex, vertexCount, context.constants.maxPathVertices);
        if (vertexCount > 0u)
        {
            const uint previousTrainingPathVertexIndex = trainingPathVertexIndex - 1u;
            NRC_BUFFER_TRAINING_PATH_VERTICES[previousTrainingPathVertexIndex] = NrcUpdateTrainingPathVertex(NRC_BUFFER_TRAINING_PATH_VERTICES[previousTrainingPathVertexIndex], radiance, throughput);
        }

        vertexCount++;
        NrcSetVertexCount(pathState.packedData, vertexCount);

        // Reset the path tracer's throughput and radiance for the next path segment
        throughput = vec3(1.0);
        radiance   = vec3(0.0);

        // Store path vertex
        NRC_BUFFER_TRAINING_PATH_VERTICES[trainingPathVertexIndex] = NrcInitializePackedPathVertex(
            surfaceAttributes.roughness, surfaceAttributes.shadingNormal, surfaceAttributes.viewVector,
            surfaceAttributes.diffuseReflectance, surfaceAttributes.specularF0, surfaceAttributes.encodedPosition);

        bool terminate = (bounce == context.constants.maxPathVertices - 1u);
        if (!NrcGetFlag(pathState.packedData, nrcPathFlagIsUnbiased))
        {
            if (NrcEvaluateTerminationHeuristic(pathState, context.constants.trainingTerminationHeuristicThreshold))
            {
                terminate = terminate || NrcGetFlag(pathState.packedData, nrcPathFlagHeuristicReset);
                NrcSetFlag(pathState.packedData, nrcPathFlagHeuristicReset);
                pathState.cumulSpreadRadius = 0.0;
            }
        }

        if (terminate)
        {
            return (context.constants.radianceCacheDirect != 0u) ? NRC_PROGRESS_TERMINATE_IMMEDIATELY : NRC_PROGRESS_TERMINATE_AFTER_DIRECT_LIGHTING;
        }
    }
    else
    {
        // Query mode
        bool createQuery = false;
        if (context.constants.enableTerminationHeuristic != 0u)
        {
            createQuery = NrcEvaluateTerminationHeuristic(pathState, context.constants.terminationHeuristicThreshold);
        }
        else
        {
            createQuery = (vertexCount == 0u);
        }

        vertexCount++;
        NrcSetVertexCount(pathState.packedData, vertexCount);

        if (createQuery)
        {
            vec3 prefixThroughput;
            if (context.constants.learnIrradiance != 0u)
            {
                prefixThroughput = throughput * (surfaceAttributes.specularF0 + surfaceAttributes.diffuseReflectance);
            }
            else
            {
                prefixThroughput = throughput;
            }
            prefixThroughput = max(vec3(0.0), NrcSanitizeNansInfs(prefixThroughput));
            pathState.packedPrefixThroughput = NrcEncodeLogLuvHdr(prefixThroughput);

            pathState.queryBufferIndex = NrcIncrementCounter(NRC_COUNTER_QUERIES);

            NrcRadianceParams params;
            params.encodedPosition = surfaceAttributes.encodedPosition;
            params.roughness       = surfaceAttributes.roughness;
            params.normal          = NrcSafeCartesianToSphericalUnorm(surfaceAttributes.shadingNormal);
            params.viewDirection   = NrcSafeCartesianToSphericalUnorm(surfaceAttributes.viewVector);
            params.albedo          = surfaceAttributes.diffuseReflectance;
            params.specular        = surfaceAttributes.specularF0;

            NRC_BUFFER_QUERY_RADIANCE_PARAMS[pathState.queryBufferIndex] = params;

            if (context.constants.radianceCacheDirect != 0u)
            {
                NrcSetDebugPathTerminationReason(pathState, NRC_DEBUG_TERM_CREATE_QUERY_IMMEDIATE);
                return NRC_PROGRESS_TERMINATE_IMMEDIATELY;
            }
            else
            {
                NrcSetDebugPathTerminationReason(pathState, NRC_DEBUG_TERM_CREATE_QUERY_AFTER_DIRECT);
                return NRC_PROGRESS_TERMINATE_AFTER_DIRECT_LIGHTING;
            }
        }
    }
    return NRC_PROGRESS_CONTINUE;
}

void NrcUpdateOnMiss(inout NrcPathState pathState)
{
    NrcSetDebugPathTerminationReason(pathState, NRC_DEBUG_TERM_PATH_MISS_EXIT);
    NrcSetFlag(pathState.packedData, nrcPathFlagHasExitedScene);
}

void NrcSetBrdfPdf(inout NrcPathState pathState, in float brdfPdf)
{
    pathState.brdfPdf = brdfPdf;
}

void NrcWriteFinalPathInfo(
    in    NrcContext context,
    inout NrcPathState pathState,
    in    vec3 throughput,
    in    vec3 radiance)
{
    if (!NrcIsEnabled())
    {
        return;
    }

    if (NrcIsUpdateMode())
    {
        // Training pass
        uint vertexCount = NrcGetVertexCount(pathState.packedData);
        if (vertexCount > 0u)
        {
            const uint vertexIndex = vertexCount - 1u;
            const uint arrayIndex = NrcCalculateTrainingPathVertexIndex(
                context.constants.trainingDimensions, context.pixelIndex, vertexIndex, context.constants.maxPathVertices);
            NRC_BUFFER_TRAINING_PATH_VERTICES[arrayIndex] = NrcUpdateTrainingPathVertex(NRC_BUFFER_TRAINING_PATH_VERTICES[arrayIndex], radiance, throughput);

            // Create self-training records
            if (!NrcGetFlag(pathState.packedData, nrcPathFlagHasExitedScene) && (context.constants.maxPathVertices > 1u))
            {
                NrcPathVertex vertex = NrcUnpackPathVertex(NRC_BUFFER_TRAINING_PATH_VERTICES[arrayIndex], radiance, throughput);
                pathState.queryBufferIndex = NrcIncrementCounter(NRC_COUNTER_QUERIES);
                NRC_BUFFER_QUERY_RADIANCE_PARAMS[pathState.queryBufferIndex] = NrcCreateRadianceParams(vertex);
            }
        }

        NrcTrainingPathInfo unpackedPathInfo;
        unpackedPathInfo.packedData       = pathState.packedData;
        unpackedPathInfo.queryBufferIndex = pathState.queryBufferIndex;

        const uint trainingPathIndex = NrcCalculateTrainingPathIndex(context.constants.trainingDimensions, context.pixelIndex);
        NRC_BUFFER_TRAINING_PATH_INFO[trainingPathIndex] = NrcPackTrainingPathInfo(unpackedPathInfo);
    }
    else
    {
        // Query pass
        const uint queryPathIndex = NrcCalculateQueryPathIndex(context.constants.frameDimensions, context.pixelIndex, context.sampleIndex, context.constants.samplesPerPixel);

        NrcPackedQueryPathInfo packedQueryPathInfo;
        packedQueryPathInfo.prefixThroughput = pathState.packedPrefixThroughput;
        packedQueryPathInfo.queryBufferIndex = pathState.queryBufferIndex;

        NRC_BUFFER_QUERY_PATH_INFO[queryPathIndex] = packedQueryPathInfo;
    }
}

#else // !ENABLE_NRC

// -------------------------------------------------------------------------
//  Stub functions when NRC is disabled
// -------------------------------------------------------------------------

void NrcSetDebugPathTerminationReason(inout NrcPathState pathState, uint reason)
{
}

NrcContext NrcCreateContext(in NrcConstants constants, in NrcBuffers buffers, in uvec2 pixelIndex)
{
    NrcContext context;
    context.constants   = constants;
    context.buffers     = buffers;
    context.pixelIndex  = pixelIndex;
    context.sampleIndex = 0u;
    return context;
}

NrcPathState NrcCreatePathState(in NrcConstants constants, float rand0to1)
{
    NrcPathState pathState;
    pathState.queryBufferIndex       = 0u;
    pathState.packedPrefixThroughput = 0u;
    pathState.cumulSpreadRadius      = 0.0;
    pathState.primarySpreadRadius    = 0.0;
    pathState.packedData             = 0u;
    pathState.brdfPdf                = 0.0;
    return pathState;
}

void NrcSetSampleIndex(inout NrcContext context, in uint sampleIndex)
{
}

bool NrcCanUseRussianRoulette(in NrcPathState pathState)
{
    return true;
}

uint NrcUpdateOnHit(in NrcContext context, inout NrcPathState pathState, NrcSurfaceAttributes surfaceAttributes, float hitDistance, uint bounce, inout vec3 throughput, inout vec3 radiance)
{
    return NRC_PROGRESS_CONTINUE;
}

void NrcUpdateOnMiss(inout NrcPathState pathState)
{
}

void NrcSetBrdfPdf(inout NrcPathState pathState, in float brdfPdf)
{
}

void NrcWriteFinalPathInfo(in NrcContext context, inout NrcPathState pathState, in vec3 throughput, in vec3 radiance)
{
}

#endif // ENABLE_NRC

#endif // NRC_GLSL_FUNCS
#endif // !NRC_STRUCTURES_ONLY
