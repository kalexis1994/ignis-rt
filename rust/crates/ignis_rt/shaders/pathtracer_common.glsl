// pathtracer_common.glsl - shared bindings, structs and functions for the path tracer. Included
// by the megakernel (trace.rgen) and the wavefront stages (wf_*.comp). No #version/#extension/main
// here; each shader declares those. See the wavefront plan in memory.
layout(binding = 0, rgba32f) uniform image2D outImage;
layout(binding = 1) uniform accelerationStructureEXT tlas;

layout(buffer_reference, std430) readonly buffer Verts { float v[]; };
layout(buffer_reference, std430) readonly buffer Norms { float n[]; };
layout(buffer_reference, std430) readonly buffer Uvs   { float u[]; };
layout(buffer_reference, std430) readonly buffer Indices { uint i[]; };
layout(buffer_reference, std430) readonly buffer MatIds  { uint m[]; };
struct GeomDesc { uint64_t vtx; uint64_t idx; uint64_t nrm; uint64_t mat; uint64_t uv; uint64_t pad; };
// Full GPUMaterial (2204 bytes), scalar layout to match the C++ byte-for-byte (incl. the
// Node VM bytecode at offset 156). Field names follow the C++ struct; base color is stored in
// ksAmbient/ksDiffuse/ksSpecular (offset 20), roughness in ksSpecularEXP, metallic in fresnelC.
struct Material {
    uint diffuseTexIndex, normalTexIndex, mapsTexIndex, detailTexIndex;
    uint normalDetailTexIndex;
    float baseR, baseG, baseB;                 // base color (offset 20)
    float roughness;                           // (offset 32)
    float emissiveR, emissiveG, emissiveB;     // (offset 36)
    float fresnelC;                            // metallic (offset 48)
    float fresnelEXP, detailUVMultiplier, detailNormalBlend;
    uint flags; float alphaRef; uint shaderType;
    float fresnelMaxLevel;                     // emission strength (offset 76)
    uint maskTexIndex, detailRTexIndex, detailGTexIndex, detailBTexIndex;
    uint detailATexIndex, detailNMTexIndex;
    float multR, multG, multB, multA, magicMult, detailNMMult, detailNMMultV;
    float sunSpecular, sunSpecularEXP;
    uint nodeVmHeader, nodeVmPad0, nodeVmPad1, nodeVmPad2;
    uvec4 nodeVmCode[128];                     // (offset 156)
};
layout(binding = 2, std430) readonly buffer GeomTable { GeomDesc descs[]; };
layout(binding = 3, scalar) readonly buffer Materials { Material mats[]; };
layout(binding = 4) uniform sampler2D textures[];
struct Light { vec4 posRange; vec4 colInt; vec4 dirSizeX; vec4 tanSizeY; };
layout(binding = 5, std430) readonly buffer Lights { Light lights[]; };
layout(binding = 6, rgba32f) uniform image2D accumImage; // persistent radiance accumulator
layout(binding = 7) uniform sampler3D lutTex;            // OCIO view-transform LUT (Blender)
// binding 8: background color + HDRI params (hdriParams.x = env texture index or -1, .y = strength).
layout(binding = 8, std430) readonly buffer World { vec4 worldColor; vec4 hdriParams; };
// DLSS guide buffers, written at the primary hit (hasTlas bit2). r32f depth = linear view-space.
layout(binding = 9,  r32f)    uniform image2D gDepth;
layout(binding = 10, rgba16f) uniform image2D gNormalRough; // rgb = world normal, a = roughness
layout(binding = 11, rgba16f) uniform image2D gAlbedo;      // diffuse albedo
layout(binding = 12, rgba16f) uniform image2D gMotion;      // pixel-space backward motion vectors
layout(binding = 13, rgba16f) uniform image2D gNoisy;       // noisy 1-spp linear color (DLSS-RR input)
layout(binding = 15, rgba16f) uniform image2D gSpecAlbedo;  // specular albedo (EnvBRDFApprox, DLSS-RR)
// Emissive triangles for NEE (area lights from geometry); 4 vec4 per tri, CDF baked by the addon:
// [0]=(v0.xyz, area) [1]=(v1.xyz, cdf) [2]=(v2.xyz, totalPower) [3]=(emission.xyz, pad).
layout(binding = 16, std430) readonly buffer EmissiveTris { vec4 data[]; } emissiveTris;
// Environment (HDRI) luminance distribution for importance sampling — built by env_cdf.comp. Layout
// matches that shader: [0]=funcInt, marginal CDF, conditional CDFs, func values.
layout(binding = 17, std430) readonly buffer EnvDist { float data[]; } envDist;
// Previous-frame object-to-world transforms, one per TLAS instance (3 vec4 = 3x4 row-major), indexed
// by gl_InstanceID. Lets writeGuide place a moving object's surface point where it was last frame so
// its motion vector tracks the object, not just the camera. Valid only when hasTlas bit6 is set.
layout(binding = 18, std430) readonly buffer PrevXforms { vec4 rows[]; } prevXforms;

layout(push_constant) uniform PC {
    mat4  invViewProj;
    vec4  camPos;
    uvec2 dims;
    uint  hasTlas;
    uint  numLights;
    vec4  sunDir; // xyz dir, w intensity
    vec4  sunCol; // rgb color, w = accumulated frame index
    mat4  prevViewProj; // forward view-proj of the previous frame (DLSS motion vectors)
    vec2  jitter;       // DLSS-RR: deterministic sub-pixel offset from pixel center [-0.5,0.5]
    uint  emissiveCount; // number of emissive triangles for NEE (0 = none)
    uint  wfIn;          // wavefront compaction: read side (0/1) of the ping-pong live-path queue
    uint  wfOut;         // wavefront compaction: write side (0/1)
    uint  maxBounces;    // path-tracing bounce budget (Max Bounces selector; was the const MAX_BOUNCES)
    uint  spp;           // samples per pixel per frame (Samples/Pixel selector; wavefront loops this)
    uint  sampleIdx;     // current sample 0..spp-1 within this frame
    uint  sharcCapacity;   // SHARC radiance-cache slots; 0 = cache disabled
    float sharcSceneScale; // SHARC world-space -> voxel scale (camera-relative logarithmic-LOD grid)
} pc;

// Per-instance raster data (binding 23): objectToWorld (3 vec4 rows) + customIndex, indexed by the
// draw-order instanceId. Used only to resolve a rasterized primary hit (hybrid raster); the megakernel
// and the ray-traced wavefront paths never touch it.
struct RasterInst { vec4 o2w0; vec4 o2w1; vec4 o2w2; uint customIndex; uint rpad0, rpad1, rpad2; };
layout(binding = 23, std430) readonly buffer RasterInsts { RasterInst data[]; } rasterInsts;

// SHARC world-space radiance cache (bindings 25-26). Declared after the pc block (uses pc.camPos +
// pc.sharcSceneScale + pc.sharcCapacity). Read-only here; the atomic update half comes in SHARC-1.
#include "sharc.glsl"

// objectToWorld (mat4x3, matching the ray query's) for raster instance instId.
mat4x3 rasterO2W(uint instId) {
    RasterInst ri = rasterInsts.data[instId];
    return mat4x3(vec3(ri.o2w0.x, ri.o2w1.x, ri.o2w2.x),
                  vec3(ri.o2w0.y, ri.o2w1.y, ri.o2w2.y),
                  vec3(ri.o2w0.z, ri.o2w1.z, ri.o2w2.z),
                  vec3(ri.o2w0.w, ri.o2w1.w, ri.o2w2.w));
}

// Inverse of an affine objectToWorld (mat4x3) -> worldToObject, for the raster hit's motion vectors.
mat4x3 invertAffine(mat4x3 m) {
    mat3 Rinv = inverse(mat3(m[0], m[1], m[2]));
    return mat4x3(Rinv[0], Rinv[1], Rinv[2], -Rinv * m[3]);
}

// Write the DLSS guide buffers for the primary hit at pixel p. depth = linear view-space distance
// (projection of the hit onto the camera forward); motion = where this world point was last frame.
// EnvBRDFApprox — Epic/Karis split-sum specular albedo. DLSS-RR demodulates the specular lobe by
// this; without it (or with diffuse albedo as a stand-in) specular/glass reflections shimmer.
vec3 EnvBRDFApprox(vec3 F0, float roughness, float NdotV) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    vec2 AB = vec2(-1.04, 1.04) * a004 + r.zw;
    return F0 * AB.x + vec3(AB.y);
}

// Householder reflection matrix I - 2*n*n^T (symmetric, so column/row major agree). Accumulated
// across mirror bounces in PSR so the reflected surface's normal is transformed into virtual space.
mat3 mirrorMatrix(vec3 n) {
    return mat3(
        1.0 - 2.0*n.x*n.x,      -2.0*n.x*n.y,      -2.0*n.x*n.z,
             -2.0*n.x*n.y, 1.0 - 2.0*n.y*n.y,      -2.0*n.y*n.z,
             -2.0*n.x*n.z,      -2.0*n.y*n.z, 1.0 - 2.0*n.z*n.z);
}

// prevWorldPos = where this surface point was last frame. For a static object it equals worldPos
// (camera-only motion); for a moving object it is the point transformed by the object's PREVIOUS
// object-to-world, so the motion vector tracks the object and DLSS-RR can reproject it cleanly.
void writeGuide(uvec2 p, vec3 worldPos, vec3 prevWorldPos, vec3 N, float roughness, vec3 albedo, vec3 specAlbedo) {
    if ((pc.hasTlas & 4u) == 0u) return;
    // DLSS-RR depth: linear view-space depth (distance along the camera forward axis). The RR path
    // uses linear depth (Use_HW_Depth=0), unlike DLSS-SR which uses NDC depth. Matches the C++
    // GetViewDepthImage that EvaluateRR consumes.
    vec4 cf = pc.invViewProj * vec4(0.0, 0.0, 1.0, 1.0);
    vec3 fwd = normalize(cf.xyz / cf.w - pc.camPos.xyz);
    float depth = max(dot(worldPos - pc.camPos.xyz, fwd), 0.0);
    vec2 mv = vec2(0.0);
    vec4 pp = pc.prevViewProj * vec4(prevWorldPos, 1.0);
    if (pp.w > 1e-6) {
        vec2 prevNdc = pp.xy / pp.w;
        vec2 prevUV  = vec2(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);
        // currUV must be the *un-jittered* projection of worldPos — which, since worldPos lies on
        // the primary ray cast through the jittered pixel, is exactly (p + 0.5 + jitter)/dims. Using
        // the bare pixel centre leaks the sub-pixel jitter into the motion vector, so a static camera
        // gets MV == jitter (instead of 0) and DLSS-RR reprojects every frame -> "boiling"/haze.
        vec2 currUV  = (vec2(p) + 0.5 + pc.jitter) / vec2(pc.dims);
        mv = (prevUV - currUV) * vec2(pc.dims); // pixel-space backward motion vector (jitter-free)
    }
    imageStore(gDepth,      ivec2(p), vec4(depth));
    imageStore(gNormalRough,ivec2(p), vec4(N, roughness));
    imageStore(gAlbedo,     ivec2(p), vec4(albedo, 1.0));
    imageStore(gMotion,     ivec2(p), vec4(mv, 0.0, 0.0));
    imageStore(gSpecAlbedo, ivec2(p), vec4(specAlbedo, 0.0));
}

// Apply a previous-frame instance transform (3x4 row-major, 3 vec4) to an object-space point.
vec3 prevObjToWorld(uint iid, vec3 localPos) {
    vec4 lp = vec4(localPos, 1.0);
    return vec3(dot(prevXforms.rows[iid*3u + 0u], lp),
                dot(prevXforms.rows[iid*3u + 1u], lp),
                dot(prevXforms.rows[iid*3u + 2u], lp));
}

const float PI = 3.14159265;
const int   MAX_BOUNCES = 8;   // higher than diffuse-only needs: glass enter+exit costs 2 each
// DEBUG: when true, the primary hit is painted by path class (red=glass/transmissive,
// green=opaque, blue=escaped to sky), bypassing accumulation + tonemap. Set false for normal render.
const bool  DEBUG_PATH = false;

vec3 sky(vec3 dir) {
    // HDRI environment: equirectangular sample of the uploaded env texture (same mapping as
    // raygen_blender.rgen / Cycles, Y-up). This is what lets grazing reflections pick up the
    // environment instead of flat dark, fixing the Fresnel edge darkening on smooth meshes.
    int hdriIdx = int(hdriParams.x);
    float strength = hdriParams.y;
    if (hdriIdx >= 0 && strength > 0.0) {
        float phi = atan(-dir.z, dir.x);
        float u = 0.5 - phi / (2.0 * PI);
        float v = acos(clamp(dir.y, -1.0, 1.0)) / PI;
        return texture(textures[nonuniformEXT(uint(hdriIdx))], vec2(u, v)).rgb * strength;
    }
    return worldColor.rgb; // flat background color (no HDRI)
}

// ── Environment (HDRI) importance sampling — pbrt Distribution2D over envDist ──
const uint ENV_W = 256u, ENV_H = 128u;
struct EnvSample { vec3 dir; vec3 radiance; float pdf; }; // pdf in solid-angle measure

// Inverse of the sky() equirect mapping: (u,v) in [0,1]^2 -> world direction (Y-up).
vec3 envUvToDir(float u, float v, out float sinT) {
    float theta = v * PI;
    float phi = (0.5 - u) * 2.0 * PI;
    sinT = sin(theta);
    return vec3(sinT * cos(phi), cos(theta), -sinT * sin(phi));
}

// Sample a direction proportional to environment luminance (binary-search the marginal then the
// conditional CDF). Piecewise-constant over cells. Returns the solid-angle pdf.
EnvSample sampleEnvironment(vec2 r) {
    EnvSample s; s.dir = vec3(0,1,0); s.radiance = vec3(0.0); s.pdf = 0.0;
    float funcInt = envDist.data[0];
    if (funcInt <= 0.0) return s;
    uint condBase = 1u + ENV_H;
    uint funcBase = 1u + ENV_H + ENV_H * ENV_W;
    uint lo = 0u, hi = ENV_H - 1u;
    while (lo < hi) { uint m = (lo + hi) / 2u; if (r.y <= envDist.data[1u + m]) hi = m; else lo = m + 1u; }
    uint vrow = lo;
    lo = 0u; hi = ENV_W - 1u;
    while (lo < hi) { uint m = (lo + hi) / 2u; if (r.x <= envDist.data[condBase + vrow * ENV_W + m]) hi = m; else lo = m + 1u; }
    uint ucol = lo;
    float uu = (float(ucol) + 0.5) / float(ENV_W);
    float vv = (float(vrow) + 0.5) / float(ENV_H);
    float sinT;
    s.dir = envUvToDir(uu, vv, sinT);
    s.radiance = sky(s.dir);
    float f = envDist.data[funcBase + vrow * ENV_W + ucol]; // already includes sinT
    s.pdf = (sinT > 1e-4) ? (f / funcInt) / (2.0 * PI * PI * sinT) : 0.0;
    return s;
}

// Solid-angle pdf the environment sampler would assign to an arbitrary direction (for MIS).
float envPdf(vec3 dir) {
    float funcInt = envDist.data[0];
    if (funcInt <= 0.0) return 0.0;
    float phi = atan(-dir.z, dir.x);
    float u = fract(0.5 - phi / (2.0 * PI));
    float v = clamp(acos(clamp(dir.y, -1.0, 1.0)) / PI, 0.0, 0.999999);
    uint ucol = min(uint(u * float(ENV_W)), ENV_W - 1u);
    uint vrow = min(uint(v * float(ENV_H)), ENV_H - 1u);
    float sinT = sin(v * PI);
    uint funcBase = 1u + ENV_H + ENV_H * ENV_W;
    float f = envDist.data[funcBase + vrow * ENV_W + ucol];
    return (sinT > 1e-4) ? (f / funcInt) / (2.0 * PI * PI * sinT) : 0.0;
}

vec3 hashColor(uint i) {
    return 0.35 + 0.55 * vec3(
        fract(sin(float(i) * 12.9898) * 43758.5453),
        fract(sin(float(i) * 78.2330) * 43758.5453),
        fract(sin(float(i) * 37.7190) * 43758.5453));
}

// Blender's runtime OCIO LUT: log2-encode into the LUT's domain, then sample (display-ready).
// Matches the C++ tonemap.comp runtimeLutTonemap exactly.
vec3 tonemapLut(vec3 c) {
    c = max(c, vec3(1e-10));
    c = clamp(log2(c), -12.47393, 12.52607);
    c = (c - (-12.47393)) / (12.52607 - (-12.47393));
    return clamp(texture(lutTex, c).rgb, 0.0, 1.0);
}
// Fallback when no LUT is present (e.g. before init): ACES filmic.
vec3 aces(vec3 x) {
    return clamp((x*(2.51*x+0.03)) / (x*(2.43*x+0.59)+0.14), 0.0, 1.0);
}

vec3 v3(Verts b, uint i) { return vec3(b.v[i*3u], b.v[i*3u+1u], b.v[i*3u+2u]); }
vec3 n3(Norms b, uint i) { return vec3(b.n[i*3u], b.n[i*3u+1u], b.n[i*3u+2u]); }

// Reconstruct the barycentrics (u, v) of worldPos inside triangle `prim` of geometry `id` under the
// instance transform o2w — used by the hybrid raster to feed getSurface the same bc a ray query would.
// Matches getSurface's convention: worldPos = w0*(1-u-v) + w1*u + w2*v.
vec2 reconstructBary(uint id, uint prim, mat4x3 o2w, vec3 worldPos) {
    GeomDesc gd = descs[id];
    if (gd.idx == 0ul || gd.vtx == 0ul) return vec2(0.0);
    Indices ib = Indices(gd.idx);
    Verts vb = Verts(gd.vtx);
    uint i0 = ib.i[prim*3u+0u], i1 = ib.i[prim*3u+1u], i2 = ib.i[prim*3u+2u];
    vec3 wp0 = o2w * vec4(v3(vb, i0), 1.0);
    vec3 wp1 = o2w * vec4(v3(vb, i1), 1.0);
    vec3 wp2 = o2w * vec4(v3(vb, i2), 1.0);
    vec3 e1 = wp1 - wp0, e2 = wp2 - wp0, p = worldPos - wp0;
    float d11 = dot(e1, e1), d12 = dot(e1, e2), d22 = dot(e2, e2);
    float dp1 = dot(p, e1), dp2 = dot(p, e2);
    float denom = d11 * d22 - d12 * d12;
    if (abs(denom) < 1e-20) return vec2(0.0);
    return vec2((d22 * dp1 - d12 * dp2) / denom, (d11 * dp2 - d12 * dp1) / denom);
}
vec2 t2(Uvs   b, uint i) { return vec2(b.u[i*2u], b.u[i*2u+1u]); }

// Tangent basis from the triangle's world-space positions and UVs (ported from
// common.glsl computeTBN). T/B are Gram-Schmidt-orthonormalized against N so the
// perturbed normal stays well-conditioned even on skewed UVs.
void computeTBN(vec3 v0, vec3 v1, vec3 v2, vec2 uv0, vec2 uv1, vec2 uv2,
                vec3 N, out vec3 T, out vec3 B) {
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec2 duv1 = uv1 - uv0;
    vec2 duv2 = uv2 - uv0;
    float det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) > 1e-6) {
        float invDet = 1.0 / det;
        T = normalize((e1 * duv2.y - e2 * duv1.y) * invDet);
        B = normalize((e2 * duv1.x - e1 * duv2.x) * invDet);
    } else {
        T = normalize(e1);
        B = normalize(cross(N, T));
    }
    T = normalize(T - N * dot(N, T));
    B = normalize(cross(N, T));
}

// Tangent-space normal map -> world normal. DX green-channel convention (1 - g*2).
vec3 applyNormalMap(vec3 texNormal, vec3 T, vec3 N, vec3 B, float strength) {
    vec3 n = vec3(texNormal.x * 2.0 - 1.0,
                  1.0 - texNormal.y * 2.0,
                  texNormal.z * 2.0 - 1.0);
    n.xy *= strength;
    n = normalize(n);
    return normalize(T * n.x + B * n.y + N * n.z);
}

// PCG hash (scalar) — now the source of the per-pixel, per-dimension Owen scramble seed.
uint  pcg(inout uint s) { s = s*747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }

// ── Low-discrepancy sampling: Owen-scrambled Sobol' (Burley 2020, "Practical Hash-based Owen
// Scrambling") ─────────────────────────────────────────────────────────────────────────────
// Each draw returns the next point of a (0,2)-sequence indexed by the temporal sample index
// g_sampleIdx (the accumulation frame), Owen-scrambled by a per-pixel + per-dimension seed (the
// evolving `s`). Across frames this walks a stratified sequence per pixel, so the 1-spp estimate
// DLSS-RR accumulates converges far faster and cleaner than white noise. RTXPT uses this family.
uint reverseBits32(uint x) {
    x = (x << 16) | (x >> 16);
    x = ((x & 0x00ff00ffu) << 8) | ((x & 0xff00ff00u) >> 8);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x & 0xf0f0f0f0u) >> 4);
    x = ((x & 0x33333333u) << 2) | ((x & 0xccccccccu) >> 2);
    x = ((x & 0x55555555u) << 1) | ((x & 0xaaaaaaaau) >> 1);
    return x;
}
// Sobol' second dimension — the (0,2)-sequence partner of the van der Corput first dimension.
uint sobol2nd(uint i) {
    uint r = 0u;
    uint v = 1u << 31;
    while (i != 0u) {
        if ((i & 1u) != 0u) r ^= v;
        v ^= v >> 1;
        i >>= 1;
    }
    return r;
}
// Nested-uniform (Owen) scramble via the Laine-Karras permutation.
uint owenScramble(uint x, uint seed) {
    x = reverseBits32(x);
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return reverseBits32(x);
}

uint g_sampleIdx; // temporal sample index (accumulation frame); set in main()
// Ray-cone texture LOD base (texture-independent part of the mip level): per-texture sampling adds
// 0.5*log2(texW*texH). Set in getSurface from the ray cone; read by the diffuse + Node VM samples.
// A very negative value (cone width 0) means "no LOD" -> mip 0 (sharp), the pre-ray-cones behaviour.
float g_texLodBase = -1e30;

float rnd(inout uint s) {
    uint seed = pcg(s); // advance the per-pixel, per-dimension scramble seed
    return float(owenScramble(reverseBits32(g_sampleIdx), seed)) * (1.0 / 4294967296.0);
}
vec2 rand2(inout uint s) {
    uint sx = pcg(s);
    uint sy = pcg(s);
    uint x = owenScramble(reverseBits32(g_sampleIdx), sx);
    uint y = owenScramble(sobol2nd(g_sampleIdx),      sy);
    return vec2(float(x), float(y)) * (1.0 / 4294967296.0);
}

mat3 build_onb(vec3 n) {
    vec3 up = abs(n.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 t = normalize(cross(up, n));
    vec3 b = cross(n, t);
    return mat3(t, b, n);
}

vec3 cosineHemisphere(vec3 N, vec2 u) {
    float phi = 2.0 * PI * u.x;
    float sr = sqrt(u.y);
    vec3 local = vec3(cos(phi)*sr, sin(phi)*sr, sqrt(max(0.0, 1.0 - u.y)));
    return normalize(build_onb(N) * local);
}

// GGX VNDF sampling (Heitz) — returns the microfacet normal H. Port from ignis-web.
vec3 sampleGgxVndf(vec2 u, vec3 V, vec3 N, float alpha) {
    mat3 onb = build_onb(N);
    vec3 T1 = onb[0], T2 = onb[1];
    vec3 Vh = normalize(vec3(dot(V,T1)*alpha, dot(V,T2)*alpha, dot(V,N)));
    float lensq = Vh.x*Vh.x + Vh.y*Vh.y;
    vec3 T1h = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(lensq) : vec3(1,0,0);
    vec3 T2h = cross(Vh, T1h);
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1*t1)) + s * t2;
    vec3 Nh = t1*T1h + t2*T2h + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2)) * Vh;
    return normalize(T1*(alpha*Nh.x) + T2*(alpha*Nh.y) + N*max(0.0, Nh.z));
}

float smithG1(float NdotX, float alpha) {
    float a2 = alpha * alpha;
    return 2.0 * NdotX / (NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX));
}

float ggxD(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Unpolarized dielectric Fresnel reflectance. eta = n_transmitted / n_incident at the
// interface; cosThetaI is measured against the microfacet normal on the incident side.
float fresnelDielectric(float cosThetaI, float eta) {
    cosThetaI = clamp(cosThetaI, -1.0, 1.0);
    if (cosThetaI < 0.0) { eta = 1.0 / eta; cosThetaI = -cosThetaI; }
    float sin2T = (1.0 - cosThetaI * cosThetaI) / (eta * eta);
    if (sin2T >= 1.0) return 1.0;                 // total internal reflection
    float cosThetaT = sqrt(1.0 - sin2T);
    float rs = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    float rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    return 0.5 * (rs * rs + rp * rp);
}

// Single-scatter GGX directional albedo (Karis 2014 split-sum analytic fit, evaluated for F0=1).
// ~1.0 at roughness 0 (no energy loss), dropping toward ~0.45 at roughness 1 where single-scatter
// GGX loses ~half its energy. Drives the multiscatter compensation below.
float ggxDirectionalAlbedo(float NdotV, float alpha) {
    float rough = sqrt(clamp(alpha, 0.0, 1.0));
    vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    vec2 sb = vec2(-1.04, 1.04) * a004 + r.zw;
    return clamp(sb.x + sb.y, 1e-3, 1.0);
}
// Multiscatter energy compensation (Karis/Filament): boosts the single-scatter GGX lobe to recover
// the energy it loses at higher roughness, so rough metals/glossy don't darken. For white F0 this
// equals Cycles' energy_scale = 1/E (bsdf_microfacet.h:389-440); for coloured F0 (metals) it is the
// standard real-time approximation of Cycles' coloured darkening term.
vec3 ggxEnergyComp(vec3 F0, float NdotV, float alpha) {
    float Ess = ggxDirectionalAlbedo(NdotV, alpha);
    return 1.0 + F0 * (1.0 / Ess - 1.0);
}

// Cook-Torrance BRDF value (Lambertian diffuse albedo/pi + GGX specular) for a light
// direction. Multiply by (radiance * NdotL) for the contribution. Physically consistent
// with the indirect bounce (cosine-sampled diffuse implies the same albedo/pi).
vec3 brdf(vec3 N, vec3 V, vec3 Ldir, vec3 diffAlbedo, vec3 F0, float alpha) {
    float nl = max(dot(N, Ldir), 1e-4);
    float nv = max(dot(N, V), 1e-4);
    vec3 H = normalize(V + Ldir);
    float nh = max(dot(N, H), 0.0);
    float vh = max(dot(V, H), 0.0);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - vh, 5.0);
    float G = smithG1(nv, alpha) * smithG1(nl, alpha);
    vec3 spec = ggxD(nh, alpha) * G * F / (4.0 * nv * nl);
    return diffAlbedo * (1.0 / PI) + spec * ggxEnergyComp(F0, nv, alpha);
}

// Sample a direction within a cone of half-angle acos(cosThetaMax) around `axis`.
vec3 sample_cone(vec3 axis, float cosThetaMax, vec2 u) {
    float ct = 1.0 + u.x * (cosThetaMax - 1.0);
    float st = sqrt(max(0.0, 1.0 - ct * ct));
    float phi = 2.0 * PI * u.y;
    return normalize(build_onb(axis) * vec3(cos(phi)*st, sin(phi)*st, ct));
}

// Is the material at this hit "see-through" for shadow / indirect rays? Light-path-transparent
// glass (Blender's Mix(Glass, Transparent, LightPath) window trick, flag bit 256) and clear
// transmissive glass shouldn't cast opaque shadows or block GI — Blender treats them as
// transparent for those ray types.
bool matSeeThrough(uint id, uint prim) {
    GeomDesc gd = descs[id];
    if (gd.mat == 0ul) return false;
    uint matId = MatIds(gd.mat).m[prim];
    Material mat = mats[matId];
    // Cycles: shadow transparency follows actual Transparent closures / alpha, NOT solid glass.
    // Light-path-transparent (256), transparent-weighted (multB>0), or alpha-masked (bit0) pass.
    return ((mat.flags & 257u) != 0u) || (mat.multB > 0.0);
}

// Shadow occlusion that passes through see-through glass (loops past up to 8 glass layers),
// so windows and nearby glass panes don't cast solid black shadows.
bool occluded(vec3 o, vec3 d, float tmax) {
    float t0 = 0.001;
    for (int i = 0; i < 8; i++) {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, tlas, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFFu, o, t0, d, tmax);
        while (rayQueryProceedEXT(rq)) {}
        if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT)
            return false; // reached the light unobstructed
        uint id   = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
        uint prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
        if (!matSeeThrough(id, prim)) return true; // opaque surface blocks the light
        t0 = rayQueryGetIntersectionTEXT(rq, true) + 0.001; // skip this glass layer, keep going
    }
    return false;
}

// ============================================================================
// Volumes — Cycles Principled Volume, P1: homogeneous analytic. Port of the C++
// include/volume_march.glsl (coefficients are Cycles-exact, incl. the sqrt on
// AbsorptionColor). Heterogeneous noise + OpenVDB grids come in later phases.
// Material encoding (byte-identical to the C++): flags bit2=volume, bit3=volume-only,
// bit4=hetero; density=sunSpecular, anisotropy=sunSpecularEXP, scatter colour=base
// colour, nodeVmCode[2]=absorption.rgb+emissionStrength, [3]=emission.rgb+temperature,
// [4].x=blackbody intensity.
// ============================================================================

// Henyey-Greenstein phase function (Cycles volume.h; eval == pdf, g clamped by caller).
float henyeyGreensteinPhase(float cosTheta, float g) {
    if (abs(g) < 1e-4) return 0.25 / PI;
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

// Cycles Principled Volume coefficients. NOTE the sqrt on absorptionColor (svm closure):
// sigma_a = (1-color) * (1-sqrt(absorptionColor)) * density.
void computeVolumeCoefficients(vec3 scatterColor, vec3 absorptionColor, float density,
                               out vec3 sigma_s, out vec3 sigma_a, out vec3 sigma_t) {
    sigma_s = scatterColor * density;
    sigma_a = max(vec3(1.0) - scatterColor, vec3(0.0))
            * max(vec3(1.0) - sqrt(max(absorptionColor, vec3(0.0))), vec3(0.0))
            * density;
    sigma_t = sigma_s + sigma_a;
}

// Decode a volume material's coefficients straight from the material buffer (shared by the
// shadow + guide paths, which don't run getSurface).
void volumeCoeffsFor(uint matId, out vec3 sigma_s, out vec3 sigma_a, out vec3 sigma_t) {
    // Direct member reads (no 2204-byte struct copy into registers).
    vec3 aCol = vec3(uintBitsToFloat(mats[matId].nodeVmCode[2].x), uintBitsToFloat(mats[matId].nodeVmCode[2].y),
                     uintBitsToFloat(mats[matId].nodeVmCode[2].z));
    computeVolumeCoefficients(vec3(mats[matId].baseR, mats[matId].baseG, mats[matId].baseB), aCol,
                              mats[matId].sunSpecular, sigma_s, sigma_a, sigma_t);
}

// Distance to the next boundary of the SAME instance (the volume's exit). Identity by the TLAS
// custom index (== descs[] index), always available at every call site. NOTES: (a) the BLAS is
// built with GeometryFlagsKHR::OPAQUE (renderer.rs), which auto-commits and skips candidates —
// force NoOpaque so the candidate loop actually runs; (b) accept ANY face of the same instance
// (not just back faces): from just inside the entry, the next same-instance boundary IS the exit
// for a closed mesh, and this stays correct under mirrored/negative-scale transforms where the
// winding (and so front/back) is flipped.
float traceVolumeExit(vec3 entryPos, vec3 dir, uint volumeId) {
    float exitDist = 10.0;
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsNoOpaqueEXT, 0xFFu, entryPos, 0.0015, dir, 1e5);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
            uint hid = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false));
            if (hid == volumeId) {
                rayQueryConfirmIntersectionEXT(rq);
            }
        }
    }
    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
        exitDist = rayQueryGetIntersectionTEXT(rq, true);
    }
    return max(exitDist, 0.005);
}

// Shadow transmittance: like occluded() but volume-aware — a volume boundary attenuates by
// Beer-Lambert over its inside segment instead of hard-blocking, and see-through glass still
// passes. Returns vec3(0) when an opaque surface blocks the light. Closest-hit per iteration
// (no TerminateOnFirstHit) so the t0 marching is ordered.
vec3 shadowT(vec3 o, vec3 d, float tmax) {
    vec3 T = vec3(1.0);
    float t0 = 0.001;
    for (int i = 0; i < 8; i++) {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, tlas, gl_RayFlagsOpaqueEXT, 0xFFu, o, t0, d, tmax);
        while (rayQueryProceedEXT(rq)) {}
        if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT)
            return T; // reached the light
        uint id    = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
        uint prim  = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
        float hitT = rayQueryGetIntersectionTEXT(rq, true);
        GeomDesc gd = descs[id];
        uint matId = 0u; uint mFlags = 0u;
        if (gd.mat != 0ul) { matId = MatIds(gd.mat).m[prim]; mFlags = mats[matId].flags; }
        if ((mFlags & 4u) != 0u) {
            // Volume boundary: the front face attenuates over the inside segment (clipped to the
            // light distance — the light may sit inside the volume); the back face just passes.
            if (rayQueryGetIntersectionFrontFaceEXT(rq, true)) {
                float seg = min(traceVolumeExit(o + d * (hitT + 5e-4), d, id), tmax - hitT);
                vec3 ss, sa, st;
                volumeCoeffsFor(matId, ss, sa, st);
                // Heterogeneous (noise) volumes: the shadow segment can't afford the noise march,
                // but using the CONSTANT density over-extinguishes exponentially — the fBM chain
                // averages ~0.5 (and the smoke has holes). Scale the optical depth accordingly,
                // or a room fog silently ate all NEE light (everything shadow-occluded).
                float densScale = ((mFlags & 16u) != 0u) ? 0.5 : 1.0;
                T *= exp(-st * densScale * max(seg, 0.0));
                if (max(T.r, max(T.g, T.b)) < 1e-3) return vec3(0.0);
            }
            t0 = hitT + 0.001;
            continue;
        }
        if (!matSeeThrough(id, prim)) return vec3(0.0); // opaque blocker
        t0 = hitT + 0.001; // skip this glass layer, keep going
    }
    return T;
}

// ============================================================================
// Node VM (Stage 1: textures + Mix/Math/ColorRamp; procedurals come in Stage 2).
// Ported from ignis-rt's node_vm.glsl — interprets the bytecode our addon generates.
// ============================================================================
float srgb1(float c) { return (c <= 0.04045) ? (c / 12.92) : pow((c + 0.055) / 1.055, 2.4); }
vec3 srgbToLinear(vec3 c) { return vec3(srgb1(c.r), srgb1(c.g), srgb1(c.b)); }

// ── Procedural noise (Cycles-exact, ported from include/noise.glsl) ──
// Jenkins Lookup3 integer hash (Cycles util/hash.h hash_uint3). Seed = 0xdeadbeef + (3<<2) + 13
// = 0xdeadbf08 — must match Cycles bit-for-bit or every Perlin/fBM/cell pattern desyncs.
uint hash3(uvec3 k) {
    uint a = 0xdeadbf08u + k.x;
    uint b = 0xdeadbf08u + k.y;
    uint c = 0xdeadbf08u + k.z;
    c ^= b; c -= (b << 14u) | (b >> 18u);
    a ^= c; a -= (c << 11u) | (c >> 21u);
    b ^= a; b -= (a << 25u) | (a >> 7u);
    c ^= b; c -= (b << 16u) | (b >> 16u);
    a ^= c; a -= (c << 4u)  | (c >> 28u);
    b ^= a; b -= (a << 14u) | (a >> 18u);
    c ^= b; c -= (b << 24u) | (b >> 8u);
    return c;
}
// Perlin improved-noise gradient dot (Cycles grad3).
float gradientDot3(uint hash, float x, float y, float z) {
    uint h = hash & 15u;
    float u = h < 8u ? x : y;
    float vt = (h == 12u || h == 14u) ? x : z;
    float v = h < 4u ? y : vt;
    return ((h & 1u) != 0u ? -u : u) + ((h & 2u) != 0u ? -v : v);
}
// 3D Perlin noise, quintic fade, ~[-1,1].
float perlinNoise3D(vec3 p) {
    vec3 fl = floor(p);
    ivec3 i = ivec3(fl);
    vec3  f = p - fl;
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float g000 = gradientDot3(hash3(uvec3(i)),                  f.x,       f.y,       f.z);
    float g100 = gradientDot3(hash3(uvec3(i + ivec3(1,0,0))),   f.x - 1.0, f.y,       f.z);
    float g010 = gradientDot3(hash3(uvec3(i + ivec3(0,1,0))),   f.x,       f.y - 1.0, f.z);
    float g110 = gradientDot3(hash3(uvec3(i + ivec3(1,1,0))),   f.x - 1.0, f.y - 1.0, f.z);
    float g001 = gradientDot3(hash3(uvec3(i + ivec3(0,0,1))),   f.x,       f.y,       f.z - 1.0);
    float g101 = gradientDot3(hash3(uvec3(i + ivec3(1,0,1))),   f.x - 1.0, f.y,       f.z - 1.0);
    float g011 = gradientDot3(hash3(uvec3(i + ivec3(0,1,1))),   f.x,       f.y - 1.0, f.z - 1.0);
    float g111 = gradientDot3(hash3(uvec3(i + ivec3(1,1,1))),   f.x - 1.0, f.y - 1.0, f.z - 1.0);
    float x00 = mix(g000, g100, u.x);
    float x10 = mix(g010, g110, u.x);
    float x01 = mix(g001, g101, u.x);
    float x11 = mix(g011, g111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
// Cycles snoise_3d: signed Perlin scaled to ~[-1,1] by noise_scale3 = 0.9820 (svm/noise.h:679).
// fBM and the Noise/bump opcodes must route through this, not raw perlinNoise3D, to match amplitude.
float snoise3D(vec3 p) { return 0.9820 * perlinNoise3D(p); }
// fBM (Cycles fractal_noise.h noise_fbm) with the Normalize toggle; normalized output ~[0,1].
float noise_fbm_3d(vec3 p, float detail, float roughness, float lacunarity, bool normalize) {
    float dc = clamp(detail, 0.0, 15.0);            // Cycles clamps Detail to [0,15] (noisetex.h:259)
    float rough = max(roughness, 0.0);              // and Roughness to >= 0 (noisetex.h:260)
    float fscale = 1.0, amp = 1.0, maxamp = 0.0, sum = 0.0;
    int nOctaves = int(dc);
    for (int i = 0; i <= nOctaves; i++) {
        sum += snoise3D(fscale * p) * amp;
        maxamp += amp; amp *= rough; fscale *= lacunarity;
    }
    float rmd = dc - floor(dc);
    if (rmd != 0.0) {
        float sum2 = sum + snoise3D(fscale * p) * amp;
        return normalize ? mix(0.5 * sum / maxamp + 0.5, 0.5 * sum2 / (maxamp + amp) + 0.5, rmd)
                         : mix(sum, sum2, rmd);
    }
    return normalize ? (0.5 * sum / maxamp + 0.5) : sum;
}
// Normalized fBM (the Noise Texture opcode's flavour).
float fbm3D(vec3 p, float detail, float roughness, float lacunarity) {
    return noise_fbm_3d(p, detail, roughness, lacunarity, true);
}
float gradientLinear(vec3 p)    { return clamp(p.x, 0.0, 1.0); }
float gradientQuadratic(vec3 p) { float r = max(p.x, 0.0); return clamp(r * r, 0.0, 1.0); }
float gradientSpherical(vec3 p) { return max(1.0 - length(p), 0.0); }
float gradientRadial(vec3 p)    { return atan(p.y, p.x) / (2.0 * PI) + 0.5; }
// Cycles PCG-3D integer hash (util/hash.h:50) for Voronoi cell jitter — uint logical shifts, NOT the
// Jenkins bit-extraction we used before (which gave a different point distribution → Voronoi never
// matched Cycles even for default F1). uvec3(ivec3) bit-reinterprets negative cell coords like Cycles.
uvec3 pcg3d_i(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= (v >> 16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v & 0x7FFFFFFFu;
}
vec3 hashI3F3(ivec3 k) { return vec3(pcg3d_i(uvec3(k))) * (1.0 / 2147483647.0); }
// Voronoi F1 — squared-then-sqrt nearest cell-point distance (Euclidean, Cycles-exact PCG jitter).
float voronoiF1(vec3 p, float randomness) {
    vec3 fl = floor(p);
    vec3 f  = fract(p);
    float minDist = 1e10;
    for (int k = -1; k <= 1; k++)
    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
        vec3 offset = vec3(float(i), float(j), float(k));
        vec3 jitter = hashI3F3(ivec3(fl) + ivec3(i, j, k));
        vec3 diff = (offset + randomness * jitter) - f;
        minDist = min(minDist, dot(diff, diff));
    }
    return sqrt(minDist);
}
float waveTexture(vec3 p, float scale, float distortion, uint waveType) {
    float coord = p.x * scale;
    if (distortion != 0.0) coord += distortion * perlinNoise3D(p);
    return (waveType == 0u) ? (0.5 + 0.5 * sin(coord)) : fract(coord);
}
float magicTexture(vec3 p, float distortion) {
    float x = sin((p.x + p.y + p.z) * 5.0);
    float y = sin((p.x - p.y + p.z) * 5.0);
    float z = sin((p.x + p.y - p.z) * 5.0);
    if (distortion > 0.0) { x = sin(x*distortion); y = sin(y*distortion); z = sin(z*distortion); }
    return (x + y + z) / 3.0 * 0.5 + 0.5;
}

// ── Extended noise (the volume smoke chain + future Noise-node fractal types) ──
// Jenkins hash_uint4 (Cycles util/hash.h): seed 0xdeadbeef+(4<<2)+13 = 0xdeadbf0c, mix()+final().
// NOTE: the C++ old noise.glsl used the wrong seed family here; this one matches Cycles.
uint hash4(uvec4 k) {
    uint a = 0xdeadbf0cu + k.x, b = 0xdeadbf0cu + k.y, c = 0xdeadbf0cu + k.z;
    // mix(a,b,c)
    a -= c; a ^= (c << 4u)  | (c >> 28u); c += b;
    b -= a; b ^= (a << 6u)  | (a >> 26u); a += c;
    c -= b; c ^= (b << 8u)  | (b >> 24u); b += a;
    a -= c; a ^= (c << 16u) | (c >> 16u); c += b;
    b -= a; b ^= (a << 19u) | (a >> 13u); a += c;
    c -= b; c ^= (b << 4u)  | (b >> 28u); b += a;
    a += k.w;
    // final(a,b,c)
    c ^= b; c -= (b << 14u) | (b >> 18u);
    a ^= c; a -= (c << 11u) | (c >> 21u);
    b ^= a; b -= (a << 25u) | (a >> 7u);
    c ^= b; c -= (b << 16u) | (b >> 16u);
    a ^= c; a -= (c << 4u)  | (c >> 28u);
    b ^= a; b -= (a << 14u) | (a >> 18u);
    c ^= b; c -= (b << 24u) | (b >> 8u);
    return c;
}
// Cycles grad4 (noise.h).
float gradientDot4(uint hash, float x, float y, float z, float w) {
    uint h = hash & 31u;
    float u = h < 24u ? x : y;
    float v = h < 16u ? y : z;
    float s = h < 8u  ? z : w;
    return ((h & 1u) != 0u ? -u : u) + ((h & 2u) != 0u ? -v : v) + ((h & 4u) != 0u ? -s : s);
}
// 4D Perlin — quintic fade, quadrilinear (Cycles perlin_4d).
float perlinNoise4D(vec4 p) {
    vec4 fl = floor(p);
    ivec4 i = ivec4(fl);
    vec4 f = p - fl;
    vec4 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float g0000 = gradientDot4(hash4(uvec4(i)),                  f.x,     f.y,     f.z,     f.w);
    float g1000 = gradientDot4(hash4(uvec4(i + ivec4(1,0,0,0))), f.x-1.0, f.y,     f.z,     f.w);
    float g0100 = gradientDot4(hash4(uvec4(i + ivec4(0,1,0,0))), f.x,     f.y-1.0, f.z,     f.w);
    float g1100 = gradientDot4(hash4(uvec4(i + ivec4(1,1,0,0))), f.x-1.0, f.y-1.0, f.z,     f.w);
    float g0010 = gradientDot4(hash4(uvec4(i + ivec4(0,0,1,0))), f.x,     f.y,     f.z-1.0, f.w);
    float g1010 = gradientDot4(hash4(uvec4(i + ivec4(1,0,1,0))), f.x-1.0, f.y,     f.z-1.0, f.w);
    float g0110 = gradientDot4(hash4(uvec4(i + ivec4(0,1,1,0))), f.x,     f.y-1.0, f.z-1.0, f.w);
    float g1110 = gradientDot4(hash4(uvec4(i + ivec4(1,1,1,0))), f.x-1.0, f.y-1.0, f.z-1.0, f.w);
    float g0001 = gradientDot4(hash4(uvec4(i + ivec4(0,0,0,1))), f.x,     f.y,     f.z,     f.w-1.0);
    float g1001 = gradientDot4(hash4(uvec4(i + ivec4(1,0,0,1))), f.x-1.0, f.y,     f.z,     f.w-1.0);
    float g0101 = gradientDot4(hash4(uvec4(i + ivec4(0,1,0,1))), f.x,     f.y-1.0, f.z,     f.w-1.0);
    float g1101 = gradientDot4(hash4(uvec4(i + ivec4(1,1,0,1))), f.x-1.0, f.y-1.0, f.z,     f.w-1.0);
    float g0011 = gradientDot4(hash4(uvec4(i + ivec4(0,0,1,1))), f.x,     f.y,     f.z-1.0, f.w-1.0);
    float g1011 = gradientDot4(hash4(uvec4(i + ivec4(1,0,1,1))), f.x-1.0, f.y,     f.z-1.0, f.w-1.0);
    float g0111 = gradientDot4(hash4(uvec4(i + ivec4(0,1,1,1))), f.x,     f.y-1.0, f.z-1.0, f.w-1.0);
    float g1111 = gradientDot4(hash4(uvec4(i + ivec4(1,1,1,1))), f.x-1.0, f.y-1.0, f.z-1.0, f.w-1.0);
    float x00 = mix(g0000, g1000, u.x); float x10 = mix(g0100, g1100, u.x);
    float x01 = mix(g0010, g1010, u.x); float x11 = mix(g0110, g1110, u.x);
    float x02 = mix(g0001, g1001, u.x); float x12 = mix(g0101, g1101, u.x);
    float x03 = mix(g0011, g1011, u.x); float x13 = mix(g0111, g1111, u.x);
    float y0 = mix(x00, x10, u.y); float y1 = mix(x01, x11, u.y);
    float y2 = mix(x02, x12, u.y); float y3 = mix(x03, x13, u.y);
    return mix(mix(y0, y1, u.z), mix(y2, y3, u.z), u.w);
}
// Cycles snoise_4d: noise_scale4 = 0.8344 (noise.h:686).
float snoise4D(vec4 p) { return 0.8344 * perlinNoise4D(p); }
float noise_fbm_4d(vec4 p, float detail, float roughness, float lacunarity, bool normalize) {
    float dc = clamp(detail, 0.0, 15.0);
    float rough = max(roughness, 0.0);
    float fscale = 1.0, amp = 1.0, maxamp = 0.0, sum = 0.0;
    int nOctaves = int(dc);
    for (int i = 0; i <= nOctaves; i++) {
        sum += snoise4D(fscale * p) * amp;
        maxamp += amp; amp *= rough; fscale *= lacunarity;
    }
    float rmd = dc - floor(dc);
    if (rmd != 0.0) {
        float sum2 = sum + snoise4D(fscale * p) * amp;
        return normalize ? mix(0.5 * sum / maxamp + 0.5, 0.5 * sum2 / (maxamp + amp) + 0.5, rmd)
                         : mix(sum, sum2, rmd);
    }
    return normalize ? (0.5 * sum / maxamp + 0.5) : sum;
}
// Musgrave fractal variants (Cycles fractal_noise.h; in 4.1+ merged into the Noise node's Type).
float noise_multi_fractal_3d(vec3 p, float detail, float roughness, float lacunarity) {
    float value = 1.0, pwr = 1.0;
    int nOctaves = int(clamp(detail, 0.0, 15.0));
    for (int i = 0; i <= nOctaves; i++) {
        value *= (pwr * snoise3D(p) + 1.0);
        pwr *= roughness; p *= lacunarity;
    }
    float rmd = clamp(detail, 0.0, 15.0) - floor(clamp(detail, 0.0, 15.0));
    if (rmd != 0.0) value *= (rmd * pwr * snoise3D(p) + 1.0);
    return value;
}
float noise_hetero_terrain_3d(vec3 p, float detail, float roughness, float lacunarity, float offset) {
    float pwr = roughness;
    float value = offset + snoise3D(p);
    p *= lacunarity;
    int nOctaves = int(clamp(detail, 0.0, 15.0));
    for (int i = 1; i <= nOctaves; i++) {
        value += (snoise3D(p) + offset) * pwr * value;
        pwr *= roughness; p *= lacunarity;
    }
    float rmd = clamp(detail, 0.0, 15.0) - floor(clamp(detail, 0.0, 15.0));
    if (rmd != 0.0) value += rmd * (snoise3D(p) + offset) * pwr * value;
    return value;
}
float noise_hybrid_multi_fractal_3d(vec3 p, float detail, float roughness, float lacunarity, float offset, float gain) {
    float pwr = 1.0, value = 0.0, weight = 1.0;
    int nOctaves = int(clamp(detail, 0.0, 15.0));
    for (int i = 0; (weight > 0.001) && (i <= nOctaves); i++) {
        weight = min(weight, 1.0);
        float signal_ = (snoise3D(p) + offset) * pwr;
        pwr *= roughness;
        value += weight * signal_;
        weight *= gain * signal_;
        p *= lacunarity;
    }
    float rmd = clamp(detail, 0.0, 15.0) - floor(clamp(detail, 0.0, 15.0));
    if ((rmd != 0.0) && (weight > 0.001)) {
        weight = min(weight, 1.0);
        value += rmd * weight * (snoise3D(p) + offset) * pwr;
    }
    return value;
}
float noise_ridged_multi_fractal_3d(vec3 p, float detail, float roughness, float lacunarity, float offset, float gain) {
    float pwr = roughness;
    float signal_ = offset - abs(snoise3D(p));
    signal_ *= signal_;
    float value = signal_;
    int nOctaves = int(clamp(detail, 0.0, 15.0));
    for (int i = 1; i <= nOctaves; i++) {
        p *= lacunarity;
        float weight = clamp(signal_ * gain, 0.0, 1.0);
        signal_ = offset - abs(snoise3D(p));
        signal_ *= signal_;
        signal_ *= weight;
        value += signal_ * pwr;
        pwr *= roughness;
    }
    return value;
}
// Dispatch by NodeNoiseType (0=Multifractal 1=fBM 2=Hybrid 3=Ridged 4=HeteroTerrain).
float noiseSelect3D(vec3 p, float detail, float roughness, float lacunarity,
                    float offset, float gain, int noiseType, bool normalize) {
    if (noiseType == 0) return noise_multi_fractal_3d(p, detail, roughness, lacunarity);
    if (noiseType == 2) return noise_hybrid_multi_fractal_3d(p, detail, roughness, lacunarity, offset, gain);
    if (noiseType == 3) return noise_ridged_multi_fractal_3d(p, detail, roughness, lacunarity, offset, gain);
    if (noiseType == 4) return noise_hetero_terrain_3d(p, detail, roughness, lacunarity, offset);
    return noise_fbm_3d(p, detail, roughness, lacunarity, normalize);
}
// Gradient types missing from the base set (Cycles gradient.h) + the dispatch.
float gradientEasing(vec3 p) { float r = clamp(p.x, 0.0, 1.0); float t = r * r; return 3.0 * t - 2.0 * t * r; }
float gradientDiagonal(vec3 p) { return clamp((p.x + p.y) * 0.5, 0.0, 1.0); }
float gradientQuadraticSphere(vec3 p) { float r = max(1.0 - length(p), 0.0); return r * r; }
float gradientSelect(vec3 p, int gtype) {
    if (gtype == 1) return gradientQuadratic(p);
    if (gtype == 2) return gradientEasing(p);
    if (gtype == 3) return gradientDiagonal(p);
    if (gtype == 4) return gradientSpherical(p);
    if (gtype == 5) return gradientQuadraticSphere(p);
    if (gtype == 6) return gradientRadial(p);
    return gradientLinear(p);
}
// 16-entry uniform colour-ramp LUT (the addon bakes the volume chain's ColorRamp into this).
float sampleColorRamp16(float t, vec4 r0, vec4 r1, vec4 r2, vec4 r3) {
    t = clamp(t, 0.0, 1.0);
    float ramp[16] = float[16](r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w,
                               r2.x, r2.y, r2.z, r2.w, r3.x, r3.y, r3.z, r3.w);
    float idx = t * 15.0;
    int i0 = int(floor(idx));
    int i1 = min(i0 + 1, 15);
    return mix(ramp[i0], ramp[i1], idx - float(i0));
}
vec3 brickTexture(vec3 p, float scale, float mortarSize, vec3 color1, vec3 color2, vec3 mortarColor) {
    vec2 uvB = p.xy * scale;
    if (mod(floor(uvB.y), 2.0) > 0.5) uvB.x += 0.5;
    vec2 brick = fract(uvB);
    float isBrick = step(mortarSize, brick.x) * step(brick.x, 1.0 - mortarSize)
                  * step(mortarSize, brick.y) * step(brick.y, 1.0 - mortarSize);
    vec2 brickId = floor(uvB);
    float h = fract(sin(dot(brickId, vec2(127.1, 311.7))) * 43758.5453);
    return mix(mortarColor, mix(color1, color2, step(0.5, h)), isBrick);
}

vec4 evalColorRamp(uint mi, uint dataOff, uint stopCount, float factor) {
    uvec4 d0 = mats[mi].nodeVmCode[dataOff];
    float pos0 = uintBitsToFloat(d0.x);
    vec4 col0 = vec4(uintBitsToFloat(d0.y), uintBitsToFloat(d0.z), uintBitsToFloat(d0.w), 1.0);
    if (stopCount <= 1u || factor <= pos0) return col0;
    float prevPos = pos0; vec4 prevCol = col0;
    for (uint i = 1u; i < stopCount; i++) {
        uvec4 di = mats[mi].nodeVmCode[dataOff + i];
        float curPos = uintBitsToFloat(di.x);
        vec4 curCol = vec4(uintBitsToFloat(di.y), uintBitsToFloat(di.z), uintBitsToFloat(di.w), 1.0);
        if (factor <= curPos) return mix(prevCol, curCol, (factor - prevPos) / max(curPos - prevPos, 1e-6));
        prevPos = curPos; prevCol = curCol;
    }
    return prevCol;
}

// rgb_ramp_lookup (cycles-main/src/kernel/svm/ramp.h:57-93) for a single channel.
// The baked curve table lives in mats[mi].nodeVmCode[dataOff + j] as uvec4, with
// table[j] = (R,G,B,_) packed in (.x,.y,.z); `chan` selects which of R/G/B to read.
// interpolate is always true for svm_node_curves (ramp.h:125-127).
float rampLookup1(uint mi, uint dataOff, uint tableSize, float f, bool extrapolate, int chan) {
    int size = int(tableSize);
    if ((f < 0.0 || f > 1.0) && extrapolate) {
        float t0, dy;
        if (f < 0.0) {                                                // ramp.h:67-71
            t0 = uintBitsToFloat(mats[mi].nodeVmCode[dataOff][chan]);
            dy = t0 - uintBitsToFloat(mats[mi].nodeVmCode[dataOff + 1u][chan]);
            f = -f;
        } else {                                                     // ramp.h:72-76
            t0 = uintBitsToFloat(mats[mi].nodeVmCode[dataOff + uint(size - 1)][chan]);
            dy = t0 - uintBitsToFloat(mats[mi].nodeVmCode[dataOff + uint(size - 2)][chan]);
            f = f - 1.0;
        }
        return t0 + dy * f * float(size - 1);                        // ramp.h:77
    }

    f = clamp(f, 0.0, 1.0) * float(size - 1);                        // ramp.h:80 (saturatef)
    int i = clamp(int(f), 0, size - 1);                              // ramp.h:83
    float t = f - float(i);                                          // ramp.h:84
    float a = uintBitsToFloat(mats[mi].nodeVmCode[dataOff + uint(i)][chan]); // ramp.h:86
    if (t > 0.0) {                                                   // ramp.h:88 (interpolate==true)
        float a1 = uintBitsToFloat(mats[mi].nodeVmCode[dataOff + uint(i + 1)][chan]);
        a = (1.0 - t) * a + t * a1;                                  // ramp.h:89
    }
    return a;
}

struct VmResult {
    vec3 baseColor; float roughness; float metallic; vec3 emission; float emissionStrength;
    float alpha; float ior; float transmission; float normalStrength; vec2 transformedUV;
    vec2 bumpGradient; float bumpStrength; float bumpDistance; float bumpDet;
    bool hasBaseColor; bool hasRoughness; bool hasMetallic; bool hasEmission;
    bool hasAlpha; bool hasIor; bool hasTransmission; bool hasNormalStrength; bool hasTransformedUV;
    bool hasBump;
};

// Sample a material texture at the ray-cone mip level (g_texLodBase + the per-texture term). Falls
// back to mip 0 (plain texture()) when there's no cone, matching the pre-ray-cones behaviour.
vec4 texLodSample(uint idx, vec2 uv) {
    if (g_texLodBase <= -1e29) return texture(textures[nonuniformEXT(idx)], uv);
    ivec2 d = textureSize(textures[nonuniformEXT(idx)], 0);
    return textureLod(textures[nonuniformEXT(idx)], uv, g_texLodBase + 0.5 * log2(float(d.x * d.y)));
}

VmResult executeNodeVm(uint mi, vec2 uv, vec3 worldPos, vec3 viewDir, vec3 normal, vec3 localPos,
                       vec4 vcol, uint instanceId, vec3 dPdu, vec3 dPdv) {
    VmResult res;
    res.hasBaseColor = false; res.hasRoughness = false; res.hasMetallic = false; res.hasEmission = false;
    res.hasAlpha = false; res.hasIor = false; res.hasTransmission = false;
    res.hasNormalStrength = false; res.hasTransformedUV = false; res.hasBump = false;
    res.baseColor = vec3(0.8); res.roughness = 0.5; res.metallic = 0.0;
    res.emission = vec3(0.0); res.emissionStrength = 0.0;
    res.alpha = 1.0; res.ior = 1.45; res.transmission = 0.0;
    res.normalStrength = 1.0; res.transformedUV = uv;
    res.bumpGradient = vec2(0.0); res.bumpStrength = 0.0; res.bumpDistance = 1.0; res.bumpDet = 0.0;

    uint instrCount = mats[mi].nodeVmHeader & 0xFFu;
    if (instrCount == 0u) return res; // no VM program -> caller uses constants/direct textures

    vec4 R[32];
    for (int i = 0; i < 32; i++) R[i] = vec4(0.0);
    R[0] = vec4(uv, 0.0, 1.0);

    for (uint pc = 0u; pc < min(instrCount, 128u); pc++) {
        uvec4 instr = mats[mi].nodeVmCode[pc];
        uint op  = instr.x & 0xFFu;
        uint dst = (instr.x >> 8u)  & 0x1Fu;
        uint sA  = (instr.x >> 16u) & 0x1Fu;
        uint sB  = (instr.x >> 24u) & 0x1Fu;

        if (op == 0x00u) continue;                                   // NOP
        else if (op == 0x01u) {                                      // SAMPLE_TEX
            R[dst] = texLodSample(instr.z, R[sA].xy);
            if (instr.y != 0u) R[dst].rgb = srgbToLinear(max(R[dst].rgb, vec3(0.0)));
        }
        else if (op == 0x10u) {                                      // UV_TRANSFORM
            vec2 sc = vec2(uintBitsToFloat(instr.y), uintBitsToFloat(instr.z));
            vec2 of = vec2(uintBitsToFloat(instr.w), 0.0);
            vec2 bl = vec2(R[sA].x, 1.0 - R[sA].y) * sc + of;
            R[dst] = vec4(bl.x, 1.0 - bl.y, 0.0, 1.0);
        }
        else if (op == 0x11u) {                                      // UV_ROTATE
            float a = uintBitsToFloat(instr.y); vec2 bl = vec2(R[sA].x, 1.0 - R[sA].y);
            float c = cos(a), s = sin(a);
            vec2 r = vec2(bl.x*c - bl.y*s, bl.x*s + bl.y*c);
            R[dst] = vec4(r.x, 1.0 - r.y, 0.0, 1.0);
        }
        else if (op == 0x13u) R[dst] = vec4(R[sA].x, 1.0 - R[sA].y, R[sA].z, R[sA].w); // UV_VFLIP
        else if (op == 0x20u) R[dst] = mix(R[sA], R[sB], uintBitsToFloat(instr.y));    // MIX
        else if (op == 0x21u) R[dst] = mix(R[sA], R[sB], R[instr.y & 0x1Fu].x);        // MIX_REG
        else if (op == 0x22u) R[dst] = R[sA] * R[sB];                                  // MULTIPLY
        else if (op == 0x23u) R[dst] = R[sA] + R[sB];                                  // ADD
        else if (op == 0x24u) R[dst] = R[sA] - R[sB];                                  // SUBTRACT
        else if (op == 0x25u) R[dst] = vec4(1.0) - (vec4(1.0)-R[sA])*(vec4(1.0)-R[sB]);// SCREEN
        else if (op == 0x27u) R[dst] = vec4(min(R[sA].rgb, R[sB].rgb), R[sA].a);       // DARKEN
        else if (op == 0x2Cu) R[dst] = vec4(max(R[sA].rgb, R[sB].rgb), R[sA].a);       // LIGHTEN
        else if (op == 0x28u) R[dst] = mix(R[sA], vec4(1.0)-R[sA], uintBitsToFloat(instr.y)); // INVERT
        else if (op == 0x29u) R[dst] = vec4(pow(max(R[sA].rgb, vec3(0.0)), vec3(uintBitsToFloat(instr.y))), R[sA].a); // GAMMA
        else if (op == 0x26u) {                                      // OVERLAY
            vec3 a = R[sA].rgb, b = R[sB].rgb, r;
            for (int i=0;i<3;i++) r[i] = (a[i]<0.5) ? 2.0*a[i]*b[i] : 1.0-2.0*(1.0-a[i])*(1.0-b[i]);
            R[dst] = vec4(r, R[sA].a);
        }
        else if (op == 0x2Au) {                                      // BRIGHT_CONTRAST
            float br = uintBitsToFloat(instr.y), ct = uintBitsToFloat(instr.z);
            vec3 c = R[sA].rgb + br;
            if (ct != 0.0) { float f = (ct>0.0) ? 1.0/max(1.0-ct,1e-4) : 1.0+ct; c = (c-0.5)*f + 0.5; }
            R[dst] = vec4(c, R[sA].a);
        }
        else if (op == 0x2Bu) {                                      // HUE_SAT_VAL
            float hue=uintBitsToFloat(instr.y), sat=uintBitsToFloat(instr.z), val=uintBitsToFloat(instr.w);
            vec3 c = R[sA].rgb;
            float cmax=max(c.r,max(c.g,c.b)), cmin=min(c.r,min(c.g,c.b)), d=cmax-cmin;
            float h=0.0, s=0.0, v=cmax;
            if (cmax>0.0) s=d/cmax;
            if (d>1e-6) { if(cmax==c.r) h=(c.g-c.b)/d; else if(cmax==c.g) h=2.0+(c.b-c.r)/d; else h=4.0+(c.r-c.g)/d; h/=6.0; if(h<0.0) h+=1.0; }
            h=fract(h+hue-0.5); s=clamp(s*sat,0.0,1.0); v=v*val;
            float hi=floor(h*6.0), fr=h*6.0-hi;
            float p=v*(1.0-s), q=v*(1.0-s*fr), tv=v*(1.0-s*(1.0-fr));
            int ih=int(hi)%6;
            if(ih==0)c=vec3(v,tv,p); else if(ih==1)c=vec3(q,v,p); else if(ih==2)c=vec3(p,v,tv);
            else if(ih==3)c=vec3(p,q,v); else if(ih==4)c=vec3(tv,p,v); else c=vec3(v,p,q);
            R[dst]=vec4(c, R[sA].a);
        }
        else if (op == 0x2Du) {                                      // COLOR_DODGE
            vec3 a=R[sA].rgb, b=R[sB].rgb;
            R[dst]=vec4(b.x<1.0?min(a.x/max(1.0-b.x,1e-6),1.0):1.0, b.y<1.0?min(a.y/max(1.0-b.y,1e-6),1.0):1.0, b.z<1.0?min(a.z/max(1.0-b.z,1e-6),1.0):1.0, R[sA].a);
        }
        else if (op == 0x2Eu) {                                      // COLOR_BURN
            vec3 a=R[sA].rgb, b=R[sB].rgb;
            R[dst]=vec4(b.x>0.0?max(1.0-(1.0-a.x)/b.x,0.0):0.0, b.y>0.0?max(1.0-(1.0-a.y)/b.y,0.0):0.0, b.z>0.0?max(1.0-(1.0-a.z)/b.z,0.0):0.0, R[sA].a);
        }
        else if (op == 0x2Fu) {                                      // SOFT_LIGHT
            vec3 a=R[sA].rgb, b=R[sB].rgb, r;
            for(int i=0;i<3;i++){ if(b[i]<0.5) r[i]=a[i]-(1.0-2.0*b[i])*a[i]*(1.0-a[i]); else r[i]=a[i]+(2.0*b[i]-1.0)*(sqrt(max(a[i],0.0))-a[i]); }
            R[dst]=vec4(r, R[sA].a);
        }
        else if (op == 0x90u) R[dst] = vec4(R[sA].rgb + 2.0*R[sB].rgb - 1.0, R[sA].a); // LINEAR_LIGHT
        else if (op == 0x12u) R[dst] = R[sA];                        // UV_MATRIX (passthrough)
        else if (op == 0x30u) {                                      // COLORRAMP
            R[dst] = evalColorRamp(mi, pc + 1u, instr.y, R[sA].x);
            pc += instr.y;
        }
        else if (op == 0x40u) R[dst] = vec4(dot(R[sA].rgb, vec3(0.2126,0.7152,0.0722))); // LUMINANCE
        else if (op == 0x41u) R[dst].x = R[sA].x + R[sB].x;
        else if (op == 0x42u) R[dst].x = R[sA].x * R[sB].x;
        else if (op == 0x43u) R[dst].x = R[sA].x / max(abs(R[sB].x), 1e-6);
        else if (op == 0x44u) R[dst].x = pow(max(R[sA].x, 0.0), R[sB].x);
        else if (op == 0x45u) R[dst].x = min(R[sA].x, R[sB].x);
        else if (op == 0x46u) R[dst].x = max(R[sA].x, R[sB].x);
        else if (op == 0x47u) R[dst].x = clamp(R[sA].x, uintBitsToFloat(instr.y), uintBitsToFloat(instr.z));
        else if (op == 0x49u) R[dst].x = R[sA].x - R[sB].x;
        else if (op == 0x4Au) R[dst].x = abs(R[sA].x);
        else if (op == 0x4Bu) R[dst].x = sqrt(max(R[sA].x, 0.0));
        else if (op == 0x4Cu) R[dst].x = (abs(R[sB].x) > 1e-6) ? mod(R[sA].x, R[sB].x) : 0.0;
        else if (op == 0x4Du) R[dst].x = floor(R[sA].x);
        else if (op == 0x4Eu) R[dst].x = ceil(R[sA].x);
        else if (op == 0x4Fu) R[dst].x = fract(R[sA].x);
        else if (op == 0x70u) R[dst].x = sin(R[sA].x);
        else if (op == 0x71u) R[dst].x = cos(R[sA].x);
        else if (op == 0x73u) R[dst].x = (R[sA].x < R[sB].x) ? 1.0 : 0.0;
        else if (op == 0x74u) R[dst].x = (R[sA].x > R[sB].x) ? 1.0 : 0.0;
        else if (op == 0x72u) R[dst].x = tan(R[sA].x);
        else if (op == 0x75u) R[dst].x = round(R[sA].x);
        else if (op == 0x76u) R[dst].x = sign(R[sA].x);
        else if (op == 0x77u) {                                      // SMOOTH_MIN
            float k = uintBitsToFloat(instr.y);
            if (k <= 0.0) R[dst].x = min(R[sA].x, R[sB].x);
            else { float h = max(k - abs(R[sA].x - R[sB].x), 0.0) / k; R[dst].x = min(R[sA].x, R[sB].x) - h*h*h*k*(1.0/6.0); }
        }
        else if (op == 0x48u) {                                      // MAP_RANGE (simplified)
            float fmn=uintBitsToFloat(instr.y), fmx=uintBitsToFloat(instr.z), tmn=uintBitsToFloat(instr.w);
            R[dst].x = mix(tmn, 1.0, clamp((R[sA].x-fmn)/max(fmx-fmn,1e-6), 0.0, 1.0));
        }
        else if (op == 0x86u) {                                      // VEC_MATH
            uint vop = instr.y; vec3 a = R[sA].xyz, b = R[sB].xyz;
            if (vop==0u) R[dst]=vec4(a+b,0.0); else if (vop==1u) R[dst]=vec4(a-b,0.0);
            else if (vop==2u) R[dst]=vec4(a*b,0.0);
            else if (vop==3u) R[dst]=vec4(a.x/max(abs(b.x),1e-6), a.y/max(abs(b.y),1e-6), a.z/max(abs(b.z),1e-6), 0.0);
            else if (vop==4u) R[dst]=vec4(cross(a,b),0.0); else if (vop==5u) R[dst].x=dot(a,b);
            else if (vop==6u) R[dst].x=length(a); else if (vop==7u) R[dst].x=distance(a,b);
            else if (vop==8u) R[dst]=vec4(normalize(a),0.0);
            else if (vop==9u) R[dst]=vec4(a*uintBitsToFloat(instr.z),0.0);
            else if (vop==10u) R[dst]=vec4(reflect(a,normalize(b)),0.0);
            else if (vop==11u) R[dst]=vec4(abs(a),0.0); else if (vop==12u) R[dst]=vec4(min(a,b),0.0);
            else if (vop==13u) R[dst]=vec4(max(a,b),0.0); else if (vop==14u) R[dst]=vec4(floor(a),0.0);
            else if (vop==15u) R[dst]=vec4(fract(a),0.0); else if (vop==16u) R[dst]=vec4(mod(a,b),0.0);
            else if (vop==17u) R[dst]=vec4(sign(a),0.0); else R[dst]=vec4(a,0.0);
        }
        else if (op == 0x66u) R[dst] = vec4(localPos, 1.0);          // LOAD_LOCAL_POS
        else if (op == 0x87u) {                                      // MAP_RANGE_FULL
            float fmn = uintBitsToFloat(instr.y), fmx = uintBitsToFloat(instr.z), tmn = uintBitsToFloat(instr.w);
            float tmx = uintBitsToFloat(mats[mi].nodeVmCode[pc + 1u].x);
            R[dst].x = mix(tmn, tmx, clamp((R[sA].x - fmn) / max(fmx - fmn, 1e-6), 0.0, 1.0));
            pc += 1u;
        }
        else if (op == 0x50u) R[dst] = vec4(R[sA][instr.y & 3u]);                      // SEPARATE_RGB
        else if (op == 0x51u) R[dst] = vec4(R[sA].x, R[sB].x, R[instr.y & 0xFu].x, 1.0);// COMBINE_RGB
        else if (op == 0x60u) R[dst] = vec4(uintBitsToFloat(instr.y), uintBitsToFloat(instr.z), uintBitsToFloat(instr.w), 1.0); // LOAD_CONST
        else if (op == 0x61u) R[dst] = vec4(uintBitsToFloat(instr.y));                 // LOAD_SCALAR
        else if (op == 0x62u) R[dst] = vec4(worldPos, 1.0);                            // LOAD_WORLD_POS
        else if (op == 0x63u) R[dst] = vec4(viewDir, 1.0);                             // LOAD_VIEW_DIR
        else if (op == 0x89u) R[dst] = vec4(normal, 1.0);                              // LOAD_NORMAL
        else if (op == 0x8Au) R[dst] = vec4(-viewDir, 1.0);                            // LOAD_INCOMING
        else if (op == 0x8Bu) R[dst].x = dot(normal, -viewDir) < 0.0 ? 1.0 : 0.0;      // BACKFACING
        else if (op == 0x94u) R[dst] = vcol;                                           // VERTEX_COLOR
        else if (op == 0x95u) {                                      // OBJECT_RANDOM
            uint h = instanceId; h = ((h>>16u)^h)*0x45d9f3bu; h = ((h>>16u)^h)*0x45d9f3bu; h = (h>>16u)^h;
            R[dst] = vec4(float(h)/4294967295.0, 0.0, 0.0, 1.0);
        }
        else if (op == 0x64u) {                                      // LAYER_WEIGHT
            float blend = uintBitsToFloat(instr.y); float NdotV = abs(dot(normal, -viewDir));
            if (instr.z == 0u) { float f0 = pow((1.0-blend)/(1.0+blend), 2.0); R[dst].x = f0 + (1.0-f0)*pow(1.0-NdotV, 5.0); }
            else R[dst].x = 1.0 - NdotV;
        }
        else if (op == 0x65u) {                                      // FRESNEL
            float ior = uintBitsToFloat(instr.y); float NdotV = abs(dot(normal, -viewDir));
            float f0 = pow((ior-1.0)/(ior+1.0), 2.0); R[dst].x = f0 + (1.0-f0)*pow(1.0-NdotV, 5.0);
        }
        else if (op == 0x84u) {                                      // RGB_CURVES (svm_node_curves, ramp.h:114-134)
            uint tableSize  = instr.y & 0x7FFFFFFFu;                 // table size (bit31 = extrapolate flag)
            bool extrapolate = (instr.y & 0x80000000u) != 0u;
            float minX = uintBitsToFloat(instr.z);
            float maxX = uintBitsToFloat(instr.w);
            uint dataOff = pc + 1u;                                  // baked table follows the opcode word
            vec3 color = R[sA].rgb;
            float fac = R[sB].x;
            vec3 relpos = (color - vec3(minX)) / (maxX - minX);      // ramp.h:122-123
            float r = rampLookup1(mi, dataOff, tableSize, relpos.x, extrapolate, 0); // R uses .x (ramp.h:125)
            float g = rampLookup1(mi, dataOff, tableSize, relpos.y, extrapolate, 1); // G uses .y (ramp.h:126)
            float b = rampLookup1(mi, dataOff, tableSize, relpos.z, extrapolate, 2); // B uses .z (ramp.h:127)
            R[dst] = vec4(mix(color, vec3(r, g, b), fac), R[sA].a);  // ramp.h:129
            pc += tableSize;                                         // skip the trailing table (loop does pc++)
        }
        else if (op == 0x58u) {                                      // TEX_CHECKER (Cycles 3-axis parity)
            float scale = uintBitsToFloat(instr.y);
            vec3 p = R[sA].xyz * scale;                              // full 3D co*scale (checker.h:37)
            p = (p + vec3(1e-6)) * 0.999999;                         // per-axis precision nudge (checker.h:18)
            int xi = abs(int(floor(p.x)));
            int yi = abs(int(floor(p.y)));
            int zi = abs(int(floor(p.z)));
            bool exy = ((xi & 1) == (yi & 1));
            bool ez  = ((zi & 1) != 0);
            float f = (exy == ez) ? 1.0 : 0.0;                       // ((xi%2==yi%2)==(zi%2)), checker.h:25
            // color1 = R[sB] (register, full RGB); color2 from immediates (blue==green until the encoder
            // carries a 3rd float — grey checkers are exact, a coloured-blue color2 is pending FIX-encoder).
            vec3 color2 = vec3(uintBitsToFloat(instr.z), uintBitsToFloat(instr.w), uintBitsToFloat(instr.w));
            R[dst] = vec4((f == 1.0) ? R[sB].rgb : color2, 1.0);     // f==1 -> color1 (checker.h:41)
        }
        else if (op == 0x80u) {                                      // TEX_NOISE (fBM)
            float scale = uintBitsToFloat(instr.y);
            float n = fbm3D(R[sA].xyz * scale, uintBitsToFloat(instr.z), uintBitsToFloat(instr.w), 2.0);
            R[dst] = vec4(n, n, n, 1.0);
        }
        else if (op == 0x81u) {                                      // TEX_GRADIENT
            vec3 p = R[sA].xyz; float f;
            if (instr.y == 1u) f = gradientQuadratic(p);
            else if (instr.y == 2u) f = gradientRadial(p);
            else if (instr.y == 3u) f = gradientSpherical(p);
            else f = gradientLinear(p);
            R[dst] = vec4(f, f, f, 1.0);
        }
        else if (op == 0x82u) {                                      // TEX_VORONOI (F1)
            float d = voronoiF1(R[sA].xyz * uintBitsToFloat(instr.y), 1.0);
            R[dst] = vec4(d, d, d, 1.0);
        }
        else if (op == 0x83u) {                                      // TEX_WAVE
            float w = waveTexture(R[sA].xyz, uintBitsToFloat(instr.y), uintBitsToFloat(instr.z), instr.w);
            R[dst] = vec4(w, w, w, 1.0);
        }
        else if (op == 0x88u) {                                      // TEX_WHITE_NOISE
            uvec3 ph = uvec3(floatBitsToUint(R[sA].x), floatBitsToUint(R[sA].y), floatBitsToUint(R[sA].z));
            uint h = hash3(ph);
            R[dst] = vec4(float(h)/4294967295.0, float(h*2654435761u)/4294967295.0, float(h*340573321u)/4294967295.0, 1.0);
        }
        else if (op == 0x91u) {                                      // TEX_MAGIC
            float m = magicTexture(R[sA].xyz * uintBitsToFloat(instr.z), uintBitsToFloat(instr.y));
            R[dst] = vec4(m, m, m, 1.0);
        }
        else if (op == 0x92u) {                                      // TEX_BRICK
            vec3 b = brickTexture(R[sA].xyz, uintBitsToFloat(instr.y), uintBitsToFloat(instr.z),
                                  vec3(0.8), vec3(0.4, 0.2, 0.1), vec3(0.1));
            R[dst] = vec4(b, 1.0);
        }
        else if (op == 0x93u) {                                      // NOISE_BUMP (finite-diff fBM gradient)
            float scale = uintBitsToFloat(instr.y);
            float detail = uintBitsToFloat(instr.z);
            float rough = uintBitsToFloat(instr.w);
            vec3 p = R[sA].xyz * scale;
            float eps = 0.01;
            float h_c = fbm3D(p, detail, rough, 2.0);
            float h_x = fbm3D(p + vec3(eps,0.0,0.0), detail, rough, 2.0);
            float h_y = fbm3D(p + vec3(0.0,eps,0.0), detail, rough, 2.0);
            R[dst] = vec4((h_x-h_c)/eps, (h_y-h_c)/eps, 0.0, 1.0);  // .z=0 -> noise-bump path
        }
        else if (op == 0x96u) {                                      // TEX_BUMP (Cycles svm_node_set_bump)
            vec2 bumpUV = (sA != 0u) ? R[sA].xy : uv;
            uint texBumpIdx = instr.z;
            ivec2 td = textureSize(textures[nonuniformEXT(texBumpIdx)], 0);
            float ts = 1.0 / float(max(td.x, td.y));
            const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
            float hc = dot(texture(textures[nonuniformEXT(texBumpIdx)], bumpUV).rgb, lw);
            float hx = dot(texture(textures[nonuniformEXT(texBumpIdx)], bumpUV + vec2(ts,0.0)).rgb, lw);
            float hy = dot(texture(textures[nonuniformEXT(texBumpIdx)], bumpUV + vec2(0.0,ts)).rgb, lw);
            vec3 dPx = dPdu * ts, dPy = dPdv * ts;
            vec3 Rx = cross(dPy, normal), Ry = cross(normal, dPx);
            float det = dot(dPx, Rx);
            vec3 sg = (hx-hc) * Rx + (hy-hc) * Ry;
            vec3 Tn = normalize(dPdu);
            vec3 Bn = normalize(dPdv - Tn * dot(dPdv, Tn));
            float sgSign = sign(det);
            R[dst] = vec4(sgSign * dot(sg, Tn), sgSign * dot(sg, Bn), abs(det), hc);
        }
        else if (op >= 0xE0u) {                                      // outputs
            if (op == 0xF0u) { res.baseColor = R[sA].rgb; res.hasBaseColor = true; }
            else if (op == 0xF1u) { res.roughness = R[sA].x; res.hasRoughness = true; }
            else if (op == 0xF2u) { res.metallic = R[sA].x; res.hasMetallic = true; }
            else if (op == 0xF3u) { res.emission = R[sA].rgb; res.emissionStrength = R[sA].a; res.hasEmission = true; }
            else if (op == 0xF4u) { res.alpha = R[sA].x; res.hasAlpha = true; }
            else if (op == 0xF5u) { res.normalStrength = R[sA].x; res.hasNormalStrength = true; }
            else if (op == 0xF6u) { res.ior = R[sA].x; res.hasIor = true; }
            else if (op == 0xF7u) { res.transmission = R[sA].x; res.hasTransmission = true; }
            else if (op == 0xEFu) { res.transformedUV = R[sA].xy; res.hasTransformedUV = true; }
            else if (op == 0xF8u) {                                  // OUTPUT_BUMP
                res.bumpGradient = R[sA].xy; res.bumpStrength = uintBitsToFloat(instr.y);
                res.bumpDistance = uintBitsToFloat(instr.z); res.bumpDet = R[sA].z;
                res.hasBump = true;
            }
        }
    }
    return res;
}

// Surface attributes at a committed triangle hit. coneWidth = ray-cone width at the hit (for
// texture LOD; 0 disables it -> mip 0).
void getSurface(uint id, uint prim, vec2 bc, mat4x3 o2w, vec3 rd, float coneWidth,
                out vec3 N, out vec3 albedo, out float roughness, out float metallic, out vec3 emission,
                out float ior, out float transmission, out bool frontFace, out vec3 geomN,
                out bool lightPathTransparent, out float transparentProb, out vec2 surfUV,
                out float surfAlpha) {
    g_texLodBase = -1e30; // reset per hit; set below from the ray cone (mip 0 until then)
    GeomDesc gd = descs[id];
    N = -rd;
    geomN = -rd;          // ray-facing geometric normal — used to offset secondary rays robustly
    lightPathTransparent = false;
    transparentProb = 0.0;
    surfUV = vec2(0.0);
    surfAlpha = 1.0;      // opaque unless the material is alpha-tested / has a per-pixel alpha
    roughness = 1.0; // no material -> fully rough (diffuse)
    metallic = 0.0;
    emission = vec3(0.0);
    ior = 1.45;
    transmission = 0.0;
    frontFace = true;     // ray hits the outward (front) face — refined from the geometric normal below
    vec2 uvCoord = vec2(0.0);
    float w0 = 1.0 - bc.x - bc.y;
    // Per-vertex world positions + V-flipped UVs, kept for the tangent basis (normal mapping).
    vec3 wp0 = vec3(0.0), wp1 = vec3(0.0), wp2 = vec3(0.0);
    vec2 uv0_ = vec2(0.0), uv1_ = vec2(0.0), uv2_ = vec2(0.0);
    bool hasTBN = false;
    // Interpolated positions fed to the Node VM: world space + object ("Generated"/Object) space.
    vec3 worldPos = vec3(0.0), localPos = vec3(0.0);
    vec3 dPdu = vec3(0.0), dPdv = vec3(0.0);   // UV->world differentials for texture bump
    if (gd.idx != 0ul) {
        Indices ib = Indices(gd.idx);
        uint i0 = ib.i[prim*3u+0u], i1 = ib.i[prim*3u+1u], i2 = ib.i[prim*3u+2u];
        // Geometric (face) normal from the triangle, flipped toward the ray.
        vec3 geoN = -rd;
        if (gd.vtx != 0ul) {
            Verts vb = Verts(gd.vtx);
            vec3 op0 = v3(vb,i0), op1 = v3(vb,i1), op2 = v3(vb,i2);
            wp0 = o2w * vec4(op0, 1.0);
            wp1 = o2w * vec4(op1, 1.0);
            wp2 = o2w * vec4(op2, 1.0);
            // Geometric normal from WORLD-space edges (matches raygen_blender.rgen). Computing it
            // as mat3(o2w)*cross(object edges) is WRONG under mirrored (negative-scale) or
            // non-uniformly-scaled instances — it skews/flips the normal and corrupts shading.
            vec3 rawGeoN = normalize(cross(wp1 - wp0, wp2 - wp0));
            frontFace = dot(rd, rawGeoN) < 0.0;
            geoN = frontFace ? rawGeoN : -rawGeoN;   // face toward the ray
            localPos = op0*w0 + op1*bc.x + op2*bc.y;        // object-space hit position
            worldPos = wp0*w0 + wp1*bc.x + wp2*bc.y;        // world-space hit position
        }
        if (gd.nrm != 0ul) {
            Norms nb = Norms(gd.nrm);
            N = normalize(mat3(o2w) * (n3(nb,i0)*w0 + n3(nb,i1)*bc.x + n3(nb,i2)*bc.y));
            // Glass entry/exit from the outward smooth normal (ignis-rt mirror-bug fix); glass
            // refraction itself uses glassDepth so this is only a hint.
            frontFace = dot(rd, N) < 0.0;
            // Shadow terminator fix (matches raygen_blender.rgen): if the smooth normal dips below
            // the geometric face plane, snap to the (ray-facing) geometric normal.
            if (dot(N, geoN) < 0.0) N = geoN;
        } else {
            N = geoN;
        }
        if (gd.uv != 0ul) {
            Uvs ub = Uvs(gd.uv);
            // V-flip per vertex (Blender bottom-left -> Vulkan top-left); barycentric interp commutes.
            uv0_ = vec2(t2(ub,i0).x, 1.0 - t2(ub,i0).y);
            uv1_ = vec2(t2(ub,i1).x, 1.0 - t2(ub,i1).y);
            uv2_ = vec2(t2(ub,i2).x, 1.0 - t2(ub,i2).y);
            uvCoord = uv0_*w0 + uv1_*bc.x + uv2_*bc.y;
            // Ray-cone texture LOD (Akenine-Möller 2021): the triangle's UV-area / world-area ratio,
            // the cone width at the hit, and the angle of incidence give the mip level (per texture,
            // add 0.5*log2(texW*texH) where it's sampled). Texture lookups in a ray shader have no
            // screen-space derivatives, so without this they always read mip 0 and alias at distance.
            if (coneWidth > 0.0) {
                float uvArea = abs((uv1_.x - uv0_.x) * (uv2_.y - uv0_.y) - (uv2_.x - uv0_.x) * (uv1_.y - uv0_.y));
                float wArea = length(cross(wp1 - wp0, wp2 - wp0));
                if (uvArea > 1e-12 && wArea > 1e-12) {
                    g_texLodBase = log2(coneWidth) - log2(max(abs(dot(rd, N)), 1e-3))
                                 + 0.5 * log2(uvArea / wArea);
                }
            }
        }
        hasTBN = (gd.vtx != 0ul) && (gd.uv != 0ul);
        geomN = geoN;          // ray-facing geometric normal for robust ray offsets
    }
    if (hasTBN) {
        vec2 d1 = uv1_ - uv0_, d2 = uv2_ - uv0_;
        vec3 e1 = wp1 - wp0, e2 = wp2 - wp0;
        float det = d1.x*d2.y - d1.y*d2.x;
        float inv = (abs(det) > 1e-8) ? 1.0/det : 0.0;
        dPdu = (e1*d2.y - e2*d1.y) * inv;
        dPdv = (e2*d1.x - e1*d2.x) * inv;
    }
    surfUV = uvCoord;
    albedo = hashColor(id);
    if (gd.mat != 0ul) {
        uint matId = MatIds(gd.mat).m[prim];
        Material mat = mats[matId];
        // Defaults from the material's constant fields.
        albedo = vec3(mat.baseR, mat.baseG, mat.baseB);
        roughness = clamp(mat.roughness, 0.04, 1.0);
        metallic = clamp(mat.fresnelC, 0.0, 1.0);
        emission = vec3(mat.emissiveR, mat.emissiveG, mat.emissiveB) * mat.fresnelMaxLevel;
        ior = mat.detailUVMultiplier;          // refraction index (matches wf_shade.comp)
        transmission = clamp(mat.multR, 0.0, 1.0);
        lightPathTransparent = (mat.flags & 256u) != 0u; // Mix(Glass,Transparent,LightPath) trick
        transparentProb = clamp(mat.multB, 0.0, 1.0);     // Transparent BSDF weight (architectural glass)
        // Node VM: its outputs override the constants. Falls back to the direct diffuse
        // texture slot when the program doesn't produce a base color.
        VmResult vm = executeNodeVm(matId, uvCoord, worldPos, rd, N, localPos, vec4(1.0), id, dPdu, dPdv);
        if (vm.hasBaseColor) albedo = vm.baseColor;
        else if (mat.diffuseTexIndex != 0xFFFFFFFFu) albedo = texLodSample(mat.diffuseTexIndex, uvCoord).rgb;
        if (vm.hasRoughness) roughness = clamp(vm.roughness, 0.04, 1.0);
        if (vm.hasMetallic) metallic = clamp(vm.metallic, 0.0, 1.0);
        if (vm.hasEmission) emission = vm.emission * vm.emissionStrength;
        if (vm.hasIor) ior = vm.ior;
        if (vm.hasTransmission) transmission = clamp(vm.transmission, 0.0, 1.0);
        // Per-pixel alpha for alpha-tested/blended materials (flag bit0). From the VM (texture
        // alpha) or the constant multG; opaque materials stay surfAlpha = 1.
        if (vm.hasAlpha) surfAlpha = clamp(vm.alpha, 0.0, 1.0);
        else if ((mat.flags & 1u) != 0u) surfAlpha = clamp(mat.multG, 0.0, 1.0);
        // Normal / height-map perturbation (port of wf_shade.comp). The sign of
        // detailNormalBlend selects the mode: < 0 = bump/height map, >= 0 = tangent normal map.
        if (hasTBN && mat.normalTexIndex != 0xFFFFFFFFu) {
            vec2 texUV = vm.hasTransformedUV ? vm.transformedUV : uvCoord;
            vec3 T, B;
            computeTBN(wp0, wp1, wp2, uv0_, uv1_, uv2_, N, T, B);
            float rawStrength = mat.detailNormalBlend;
            if (rawStrength < 0.0) {
                // Bump/height map, Cycles-style. Packed: floor(packed)=distance*100, frac=strength.
                float packed = abs(rawStrength);
                float bumpDistance = floor(packed) / 100.0;
                float bumpStrength = fract(packed);
                ivec2 texDim = textureSize(textures[nonuniformEXT(mat.normalTexIndex)], 0);
                float texelSize = 1.0 / float(max(texDim.x, texDim.y));
                // Height read as gamma-corrected luminance (preserves dark-range contrast).
                vec3 hc = texture(textures[nonuniformEXT(mat.normalTexIndex)], texUV).rgb;
                vec3 hx = texture(textures[nonuniformEXT(mat.normalTexIndex)], texUV + vec2(texelSize, 0.0)).rgb;
                vec3 hy = texture(textures[nonuniformEXT(mat.normalTexIndex)], texUV + vec2(0.0, texelSize)).rgb;
                const vec3 lumW = vec3(0.2126, 0.7152, 0.0722);
                float h_c = pow(max(dot(hc, lumW), 0.0), 1.0/2.2);
                float h_x = pow(max(dot(hx, lumW), 0.0), 1.0/2.2);
                float h_y = pow(max(dot(hy, lumW), 0.0), 1.0/2.2);
                float texScale = pow(float(max(texDim.x, texDim.y)), 0.65);
                vec3 surfgrad = (h_x - h_c) * normalize(T) + (h_y - h_c) * normalize(B);
                vec3 bumpN = normalize(N - bumpDistance * texScale * surfgrad);
                N = normalize(bumpStrength * bumpN + (1.0 - bumpStrength) * N);
            } else {
                vec3 texN = texture(textures[nonuniformEXT(mat.normalTexIndex)], texUV).rgb;
                float strength = rawStrength > 0.0 ? rawStrength : 1.0;
                N = applyNormalMap(texN, T, N, B, strength);
            }
        }
        // Procedural / texture bump from the VM (Cycles svm_node_set_bump). bumpDet > 0 means
        // a texture bump with a real UV->world Jacobian; 0 means a noise bump (simplified form).
        if (hasTBN && vm.hasBump) {
            vec3 T, B;
            computeTBN(wp0, wp1, wp2, uv0_, uv1_, uv2_, N, T, B);
            vec3 surfgrad = vm.bumpGradient.x * normalize(T) + vm.bumpGradient.y * normalize(B);
            vec3 bumpN = (vm.bumpDet > 1e-6)
                ? normalize(vm.bumpDet * N - vm.bumpDistance * surfgrad)
                : normalize(N - vm.bumpDistance * surfgrad);
            N = normalize(vm.bumpStrength * bumpN + (1.0 - vm.bumpStrength) * N);
        }
    }
    // Camera-side terminator fix (port of wf_shade.comp:681 / raygen_blender.rgen): on smooth-shaded
    // curved surfaces the interpolated/perturbed normal can dip toward or behind the view horizon at
    // grazing angles, which renders as black contours. Blend it back to the geometric normal there.
    float NdotV = dot(N, -rd);
    if (NdotV < 0.01) {
        float blend = clamp(-NdotV * 50.0 + 0.5, 0.0, 1.0);
        N = normalize(mix(N, geomN, blend));
    }
}

// One sampled point on an emissive triangle (area-measure pdf), for NEE area lighting.
struct EmissiveSample { vec3 position; vec3 emission; vec3 normal; float pdf; float area; };

// Importance-sample an emissive triangle by power (CDF baked by the addon), then a uniform point on
// it. pdf is in area measure (triProb / area). Ported from the C++ sampleEmissiveTriangle.
EmissiveSample sampleEmissiveTriangle(uint triCount, float r, vec2 baryRnd) {
    EmissiveSample s;
    s.pdf = 0.0; s.emission = vec3(0.0); s.normal = vec3(0,1,0); s.position = vec3(0.0); s.area = 0.0;
    if (triCount == 0u) return s;
    uint lo = 0u, hi = triCount - 1u;
    while (lo < hi) {
        uint mid = (lo + hi) / 2u;
        if (r <= emissiveTris.data[mid * 4u + 1u].w) hi = mid; else lo = mid + 1u; // .w = cdf
    }
    uint base = lo * 4u;
    vec4 d0 = emissiveTris.data[base + 0u]; // v0.xyz, area
    vec4 d1 = emissiveTris.data[base + 1u]; // v1.xyz, cdf
    vec4 d2 = emissiveTris.data[base + 2u]; // v2.xyz, totalPower
    vec3 v0 = d0.xyz, v1 = d1.xyz, v2 = d2.xyz;
    s.area = d0.w;
    s.emission = emissiveTris.data[base + 3u].xyz;
    float prevCdf = (lo > 0u) ? emissiveTris.data[(lo - 1u) * 4u + 1u].w : 0.0;
    float triProb = d1.w - prevCdf;
    float u = baryRnd.x, v = baryRnd.y;
    if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
    s.position = v0 * (1.0 - u - v) + v1 * u + v2 * v;
    s.normal = normalize(cross(v1 - v0, v2 - v0));
    s.pdf = (s.area > 0.0) ? triProb / s.area : 0.0;
    return s;
}

// Direct light arriving at a point INSIDE a volume, HG phase-weighted and shadow-tested:
// sun + scene lights (centre-sampled) + one emissive-triangle NEE sample + isotropic sky.
// Shared by the homogeneous and heterogeneous volume paths (interior fog is usually lamp-lit).
vec3 volumeDirectAt(vec3 samplePos, vec3 rd, float anisotropy, inout uint rng) {
    vec3 L_in = vec3(0.0);
    // Sun.
    if (pc.sunDir.w > 0.0) {
        vec3 sDir = normalize(pc.sunDir.xyz);
        vec3 sVis = shadowT(samplePos, sDir, 10000.0);
        if (sVis != vec3(0.0)) {
            L_in += pc.sunCol.rgb * pc.sunDir.w
                  * henyeyGreensteinPhase(dot(rd, sDir), anisotropy) * sVis;
        }
    }
    // Scene lights (point/spot/area sampled at centre — P1 approximation).
    for (uint vli = 0u; vli < pc.numLights; vli++) {
        Light VL = lights[vli];
        vec3 toL = VL.posRange.xyz - samplePos;
        float ld = length(toL);
        if (VL.posRange.w > 0.0 && ld > VL.posRange.w) continue;
        vec3 lDir = toL / max(ld, 1e-4);
        float spotAtten = 1.0;
        bool vIsSpot = (VL.posRange.w < 0.0) && (VL.tanSizeY.w < 0.0);
        if (vIsSpot) {
            float tSpot = clamp((dot(-lDir, VL.dirSizeX.xyz) - VL.dirSizeX.w) * abs(VL.tanSizeY.w), 0.0, 1.0);
            spotAtten = tSpot * tSpot * (3.0 - 2.0 * tSpot);
            if (spotAtten <= 0.0) continue;
        }
        float vAreaFactor = 1.0;
        bool vIsArea = (VL.posRange.w < 0.0) && (VL.tanSizeY.w >= 0.0);
        if (vIsArea) {
            // Same Cycles area estimator as directShade: ×A ×cosθ_e, one-sided.
            float cosE = dot(-lDir, VL.dirSizeX.xyz);
            if (cosE <= 0.0) continue;
            vAreaFactor = cosE * abs(VL.dirSizeX.w * VL.tanSizeY.w);
        }
        vec3 lVis = shadowT(samplePos, lDir, ld - 0.01);
        if (lVis == vec3(0.0)) continue;
        L_in += VL.colInt.rgb * VL.colInt.w * spotAtten * vAreaFactor
              * henyeyGreensteinPhase(dot(rd, lDir), anisotropy) * lVis / (ld * ld + 1e-3);
    }
    // Emissive triangles (mesh lamps), one NEE sample.
    if (pc.emissiveCount > 0u) {
        EmissiveSample ves = sampleEmissiveTriangle(pc.emissiveCount, rnd(rng), rand2(rng));
        if (ves.pdf > 0.0) {
            vec3 toE = ves.position - samplePos;
            float eD = length(toE);
            vec3 eDir = toE / max(eD, 1e-4);
            float eCos = abs(dot(-eDir, ves.normal)); // double-sided (Cycles mesh emission)
            if (eCos > 0.0 && eD > 0.01) {
                vec3 eVis = shadowT(samplePos, eDir, eD - 0.01);
                if (eVis != vec3(0.0)) {
                    L_in += ves.emission * eCos * henyeyGreensteinPhase(dot(rd, eDir), anisotropy)
                          * eVis / ((eD * eD) * max(ves.pdf, 1e-6));
                }
            }
        }
    }
    L_in += sky(rd) * 0.25 / PI; // isotropic sky ambient (C++ approximation)
    return L_in;
}

// Heterogeneous volume: fixed-step march with the exported noise chain driving the scatter
// colour (or emission). Port of the C++ marchVolumeHeterogeneous, but on the Cycles-exact Rust
// noise base (fixed hash seed + snoise scales) and with the lighting evaluated ONCE at a
// weighted-reservoir-picked scatter point (transmittance*sigma_s-proportional) using the full
// shadowed light set — instead of the C++'s per-step unshadowed sun + 1 random light (leaky).
void marchVolumeHetero(vec3 entryPos, vec3 rd, float marchDist, mat4x3 w2o, uint matId,
                       vec3 scatterColor, vec3 absorbCol, float density, float anisotropy,
                       vec3 emColor, float emStrength, float temperature, float blackbody,
                       inout uint rng, inout vec3 L, inout vec3 tp) {
    // Decode the noise-chain descriptor (byte-identical to the C++ layout).
    float noiseScale  = uintBitsToFloat(mats[matId].nodeVmPad0);
    float noiseDetail = uintBitsToFloat(mats[matId].nodeVmPad1);
    float noiseBright = uintBitsToFloat(mats[matId].nodeVmPad2);
    float noiseRough  = uintBitsToFloat(mats[matId].nodeVmCode[0].x);
    float noiseLac    = uintBitsToFloat(mats[matId].nodeVmCode[0].y);
    float noiseContr  = uintBitsToFloat(mats[matId].nodeVmCode[0].z);
    uint  noiseDims   = mats[matId].nodeVmCode[0].w;
    vec3  mapOffset   = vec3(uintBitsToFloat(mats[matId].nodeVmCode[1].x),
                             uintBitsToFloat(mats[matId].nodeVmCode[1].y),
                             uintBitsToFloat(mats[matId].nodeVmCode[1].z));
    float noiseW      = uintBitsToFloat(mats[matId].nodeVmCode[1].w);
    uint  typeNorm    = mats[matId].nodeVmCode[4].y;
    int   noiseType   = int(typeNorm & 0xFFFFu);
    bool  noiseNorm   = ((typeNorm >> 16u) & 0xFFFFu) != 0u;
    float noiseOffset = uintBitsToFloat(mats[matId].nodeVmCode[4].z);
    float noiseGain   = uintBitsToFloat(mats[matId].nodeVmCode[4].w);
    vec3  bboxMin     = vec3(uintBitsToFloat(mats[matId].nodeVmCode[5].x),
                             uintBitsToFloat(mats[matId].nodeVmCode[5].y),
                             uintBitsToFloat(mats[matId].nodeVmCode[5].z));
    vec3  bboxRange   = max(vec3(uintBitsToFloat(mats[matId].nodeVmCode[6].x),
                                 uintBitsToFloat(mats[matId].nodeVmCode[6].y),
                                 uintBitsToFloat(mats[matId].nodeVmCode[6].z)) - bboxMin, vec3(1e-4));
    vec4  mapRow0     = vec4(uintBitsToFloat(mats[matId].nodeVmCode[7].x), uintBitsToFloat(mats[matId].nodeVmCode[7].y),
                             uintBitsToFloat(mats[matId].nodeVmCode[7].z), uintBitsToFloat(mats[matId].nodeVmCode[7].w));
    vec4  mapRow1     = vec4(uintBitsToFloat(mats[matId].nodeVmCode[8].x), uintBitsToFloat(mats[matId].nodeVmCode[8].y),
                             uintBitsToFloat(mats[matId].nodeVmCode[8].z), uintBitsToFloat(mats[matId].nodeVmCode[8].w));
    vec4  mapRow2     = vec4(uintBitsToFloat(mats[matId].nodeVmCode[9].x), uintBitsToFloat(mats[matId].nodeVmCode[9].y),
                             uintBitsToFloat(mats[matId].nodeVmCode[9].z), uintBitsToFloat(mats[matId].nodeVmCode[9].w));
    float distortion  = uintBitsToFloat(mats[matId].nodeVmCode[10].x);
    int   gradType    = int(mats[matId].nodeVmCode[10].y);
    float mixFactor   = uintBitsToFloat(mats[matId].nodeVmCode[10].z);
    uint  volFlags2   = mats[matId].nodeVmCode[10].w;
    bool useObjCoords = (volFlags2 & 1u)  != 0u;
    bool hasFullMtx   = (volFlags2 & 2u)  != 0u;
    bool hasGradient  = (volFlags2 & 4u)  != 0u;
    bool hasMix       = (volFlags2 & 8u)  != 0u;
    bool hasRamp      = (volFlags2 & 16u) != 0u;
    bool hasDistort   = (volFlags2 & 32u) != 0u;
    uint chainTarget  = (volFlags2 >> 8u) & 0xFu;
    vec4 rampLut0 = vec4(0.0), rampLut1 = vec4(0.0), rampLut2 = vec4(0.0), rampLut3 = vec4(0.0);
    if (hasRamp) {
        rampLut0 = vec4(uintBitsToFloat(mats[matId].nodeVmCode[11].x), uintBitsToFloat(mats[matId].nodeVmCode[11].y),
                        uintBitsToFloat(mats[matId].nodeVmCode[11].z), uintBitsToFloat(mats[matId].nodeVmCode[11].w));
        rampLut1 = vec4(uintBitsToFloat(mats[matId].nodeVmCode[12].x), uintBitsToFloat(mats[matId].nodeVmCode[12].y),
                        uintBitsToFloat(mats[matId].nodeVmCode[12].z), uintBitsToFloat(mats[matId].nodeVmCode[12].w));
        rampLut2 = vec4(uintBitsToFloat(mats[matId].nodeVmCode[13].x), uintBitsToFloat(mats[matId].nodeVmCode[13].y),
                        uintBitsToFloat(mats[matId].nodeVmCode[13].z), uintBitsToFloat(mats[matId].nodeVmCode[13].w));
        rampLut3 = vec4(uintBitsToFloat(mats[matId].nodeVmCode[14].x), uintBitsToFloat(mats[matId].nodeVmCode[14].y),
                        uintBitsToFloat(mats[matId].nodeVmCode[14].z), uintBitsToFloat(mats[matId].nodeVmCode[14].w));
    }

    const int MAX_STEPS = 32;
    float stepSize = max(marchDist / float(MAX_STEPS), 0.02);
    int numSteps = min(int(marchDist / stepSize), MAX_STEPS);
    vec3 volT = vec3(1.0);
    vec3 Wsum = vec3(0.0);       // Σ T_i·sigma_s_i·dt — the total single-scatter weight
    float wRunning = 0.0;        // scalar running total for the reservoir pick
    vec3 pickPos = entryPos; bool havePick = false;
    vec3 volEmission = vec3(0.0);

    for (int s = 0; s < numSteps; s++) {
        float t = (float(s) + 0.5) * stepSize;
        vec3 samplePos = entryPos + rd * t;

        float localEmStr = emStrength;
        vec3 localScatter = scatterColor;
        if (noiseScale > 0.0) {
            vec3 objPos = w2o * vec4(samplePos, 1.0);
            vec3 baseCoord = useObjCoords ? objPos : (objPos - bboxMin) / bboxRange;
            vec3 mapped = hasFullMtx
                ? vec3(dot(mapRow0.xyz, baseCoord) + mapRow0.w,
                       dot(mapRow1.xyz, baseCoord) + mapRow1.w,
                       dot(mapRow2.xyz, baseCoord) + mapRow2.w)
                : baseCoord + mapOffset;
            float n;
            if (noiseDims == 4u) {
                vec4 np4 = vec4(mapped, noiseW) * noiseScale;
                if (hasDistort && distortion != 0.0) {
                    np4 += vec4(perlinNoise4D(np4 + vec4(0,0,0,200)), perlinNoise4D(np4 + vec4(0,0,0,100)),
                                perlinNoise4D(np4), perlinNoise4D(np4 + vec4(0,0,0,150))) * distortion;
                }
                n = noise_fbm_4d(np4, noiseDetail, noiseRough, noiseLac, noiseNorm);
            } else {
                vec3 np = mapped * noiseScale;
                if (hasDistort && distortion != 0.0) {
                    np += vec3(perlinNoise3D(np + vec3(13.5)), perlinNoise3D(np),
                               perlinNoise3D(np - vec3(13.5))) * distortion;
                }
                n = noiseSelect3D(np, noiseDetail, noiseRough, noiseLac, noiseOffset, noiseGain, noiseType, noiseNorm);
            }
            n = max((1.0 + noiseContr) * n + (noiseBright - 0.5 * noiseContr), 0.0); // BrightContrast
            float chainValue = n;
            if (hasMix && hasGradient)      chainValue = mix(n, gradientSelect(mapped, gradType), mixFactor);
            else if (hasGradient)           chainValue = gradientSelect(mapped, gradType);
            if (hasRamp)                    chainValue = sampleColorRamp16(chainValue, rampLut0, rampLut1, rampLut2, rampLut3);
            if (chainTarget == 0u)          localScatter *= chainValue;
            else if (chainTarget == 1u)     localEmStr = chainValue;
        }

        vec3 sigma_s, sigma_a, sigma_t;
        computeVolumeCoefficients(localScatter, absorbCol, density, sigma_s, sigma_a, sigma_t);
        if (max(sigma_t.r, max(sigma_t.g, sigma_t.b)) >= 1e-4) {
            // Reservoir-pick the lighting sample point ∝ this step's scatter weight.
            vec3 wi = volT * sigma_s * stepSize;
            float ws = dot(wi, vec3(0.2126, 0.7152, 0.0722));
            Wsum += wi;
            wRunning += ws;
            if (ws > 0.0 && rnd(rng) * wRunning < ws) { pickPos = samplePos; havePick = true; }
            // Emission accumulates deterministically per step.
            if (localEmStr > 0.0) volEmission += volT * emColor * localEmStr * stepSize;
            if (blackbody > 0.0 && temperature > 0.0) {
                float bbPower = blackbody * 5.67e-8 * temperature * temperature * temperature * temperature * 1e-12;
                vec3 bbColor = vec3(1.0, clamp((temperature - 1000.0) / 5000.0, 0.0, 1.0),
                                    clamp((temperature - 3000.0) / 7000.0, 0.0, 1.0));
                volEmission += volT * bbColor * bbPower * stepSize;
            }
            volT *= exp(-sigma_t * stepSize);
            if (max(volT.r, max(volT.g, volT.b)) < 0.001) break;
        }
    }

    vec3 inscatter = havePick ? Wsum * volumeDirectAt(pickPos, rd, anisotropy, rng) : vec3(0.0);
    L += tp * (inscatter + volEmission);
    tp *= volT;
}

// Direct lighting (NEE): sun + scene lights, each shadow-tested, with full diffuse+GGX BRDF
// so glossy/metal surfaces get specular highlights (glints) from the lights.
vec3 directShade(vec3 hitPos, vec3 N, vec3 geoN, vec3 V, vec3 diffAlbedo, vec3 F0, float alpha, int bounce, inout uint rng) {
    // Offset along the geometric normal (not the shading normal): on smooth-shaded silhouettes
    // the shading normal is nearly tangent, so offsetting along it leaves the shadow-ray origin
    // on the surface -> self-occlusion -> black terminator. geoN always faces the ray.
    vec3 o = hitPos + geoN * 0.002;
    vec3 sum = vec3(0.0);
    // Sun — cone-sampled within its angular radius (cam_pos.w) for soft shadows.
    vec3 sunDir = normalize(pc.sunDir.xyz);
    float sunSize = pc.camPos.w;
    if (sunSize > 1e-4) sunDir = sample_cone(sunDir, cos(sunSize), rand2(rng));
    float ndl = max(dot(N, sunDir), 0.0);
    if (ndl > 0.0) {
        vec3 sVis = shadowT(o, sunDir, 10000.0); // volume-aware: fog attenuates instead of hard-blocking
        if (sVis != vec3(0.0)) {
            vec3 Li = pc.sunCol.rgb * pc.sunDir.w; // full sun irradiance — Cycles doesn't cap it; the
                                                   // old min(...,4.0) crushed sun/shadow contrast
            sum += brdf(N, V, sunDir, diffAlbedo, F0, alpha) * Li * ndl * sVis;
        }
    }
    // Scene lights, all soft-shadowed by sampling a point on the light's surface (Cycles-style).
    // Type encoding from the addon (shared with C++): posRange.w < 0 = area OR spot, and the
    // sign of tanSizeY.w disambiguates — < 0 = spot, >= 0 = area. posRange.w >= 0 = point.
    //   Area: sample the rectangle.  Spot: cone falloff, hard shadow (matches the original).
    //   Point: sample a disk of the light's radius (shadow_soft_size in dirSizeX.w).
    for (uint li = 0u; li < pc.numLights; li++) {
        Light L = lights[li];
        vec3 lpos = L.posRange.xyz;
        bool isSpecial = L.posRange.w < 0.0;
        bool isSpot = isSpecial && (L.tanSizeY.w < 0.0);
        bool isArea = isSpecial && !isSpot;
        float spotAtten = 1.0;
        if (isArea) {
            vec3 tang = L.tanSizeY.xyz;
            vec3 bitan = cross(L.dirSizeX.xyz, tang);
            vec2 r = rand2(rng);
            lpos += tang * ((r.x - 0.5) * L.dirSizeX.w) + bitan * ((r.y - 0.5) * L.tanSizeY.w);
        } else if (isSpot) {
            // Cone attenuation (Cycles smoothstep): dirSizeX.w = cos(half angle),
            // abs(tanSizeY.w) = inverse blend width. Outside the cone -> no contribution.
            vec3 spotDir = L.dirSizeX.xyz;
            float cosHalfAngle = L.dirSizeX.w;
            float spotSmooth = abs(L.tanSizeY.w);
            vec3 Lc = normalize(lpos - o);
            float t = clamp((dot(-Lc, spotDir) - cosHalfAngle) * spotSmooth, 0.0, 1.0);
            spotAtten = t * t * (3.0 - 2.0 * t);
            if (spotAtten <= 0.0) continue;
        } else {
            float softRadius = clamp(L.dirSizeX.w * 0.5, 0.0, 2.0);
            if (softRadius > 1e-4) {
                vec3 Lc = normalize(lpos - o);
                vec3 lt = abs(Lc.y) < 0.999 ? normalize(cross(Lc, vec3(0,1,0))) : normalize(cross(Lc, vec3(1,0,0)));
                vec3 lb = cross(Lc, lt);
                vec2 ds = rand2(rng);
                float dr = sqrt(ds.x) * softRadius;
                float da = ds.y * 2.0 * PI;
                lpos += lt * (cos(da) * dr) + lb * (sin(da) * dr);
            }
        }
        vec3 toL = lpos - o;
        float dist = length(toL);
        if (L.posRange.w > 0.0 && dist > L.posRange.w) continue;
        vec3 Ldir = toL / max(dist, 1e-4);
        float nl = max(dot(N, Ldir), 0.0);
        if (nl <= 0.0) continue;
        float areaFactor = 1.0;
        if (isArea) {
            // Cycles area-light estimator: contribution = L·A·cosθ_e/d² — the uniform-rect pdf is
            // 1/A (×A) and the panel emits ONE-SIDED Lambert (×cosθ_e, reject behind). Without ×A
            // every fixture under 1m² was 1/A too bright (0.3m panel ≈ 11×); without cosθ_e it lit
            // at full strength sideways; without the facing test it lit things BEHIND the panel.
            float cosE = dot(-Ldir, L.dirSizeX.xyz);
            if (cosE <= 0.0) continue;
            areaFactor = cosE * abs(L.dirSizeX.w * L.tanSizeY.w);
        }
        vec3 lVis = shadowT(o, Ldir, dist - 0.01);
        if (lVis == vec3(0.0)) continue;
        vec3 Li = L.colInt.rgb * L.colInt.w * spotAtten * areaFactor / (dist*dist + 1e-3);
        sum += brdf(N, V, Ldir, diffAlbedo, F0, alpha) * Li * nl * lVis;
    }
    // Emissive triangle NEE (area lights from geometry), power-importance-sampled with a power-
    // heuristic MIS against the BSDF. Without this, an emissive panel is only seen when a path ray
    // randomly hits it -> very noisy. The BSDF-side emission stays unweighted (matches the C++); the
    // light-side MIS weight cancels most of the resulting double count.
    if (pc.emissiveCount > 0u) {
        EmissiveSample es = sampleEmissiveTriangle(pc.emissiveCount, rnd(rng), rand2(rng));
        if (es.pdf > 0.0) {
            vec3 toE = es.position - o;
            float eDist = length(toE);
            vec3 eDir = toE / max(eDist, 1e-4);
            float eNdotL = max(dot(N, eDir), 0.0);
            // DOUBLE-SIDED emitter cosine: Cycles mesh emission emits from BOTH faces. The old
            // one-sided dot() rejected every sample whose triangle normal faced away — a lamp
            // or window-portal mesh with normals pointing "outward" cast NO light at all.
            float eCos = abs(dot(-eDir, es.normal));
            if (eNdotL > 0.0 && eCos > 0.0 && eDist > 0.01) {
                vec3 eVis = shadowT(o, eDir, eDist - 0.01);
                if (eVis != vec3(0.0)) {
                    vec3 eRadiance = es.emission * eCos / (eDist * eDist);
                    // Light-only (no MIS weight): the forward BSDF-strategy emission is now skipped at
                    // bounce-reached emitters (see the emission gate above), so NEE carries the FULL
                    // emitter contribution. Keeping a MIS weight here would under-count it.
                    sum += brdf(N, V, eDir, diffAlbedo, F0, alpha) * eNdotL * eRadiance * eVis / max(es.pdf, 1e-6);
                }
            }
        }
    }
    // Environment (HDRI/sky) NEE — importance-sampled by luminance (env_cdf.comp). Light-sampling
    // strategy for the environment; the BSDF samples it on specular/glass escapes (the bounce loop
    // skips the sky on a diffuse-bounce escape so the env isn't double counted). funcInt>0 means the
    // distribution is built and non-black. Limited to the first couple of bounces: the env shadow
    // ray reaches to infinity (expensive), and deep-bounce env lighting is low and gets denoised.
    if (bounce <= 1 && (pc.hasTlas & 32u) != 0u && envDist.data[0] > 0.0) {
        EnvSample en = sampleEnvironment(rand2(rng));
        if (en.pdf > 0.0) {
            float eNdotL = max(dot(N, en.dir), 0.0);
            if (eNdotL > 0.0) {
                vec3 eVis = shadowT(o, en.dir, 1e30);
                if (eVis != vec3(0.0)) {
                    sum += brdf(N, V, en.dir, diffAlbedo, F0, alpha) * eNdotL * en.radiance * eVis / en.pdf;
                }
            }
        }
    }
    return sum;
}

// Primary Surface Replacement (PSR) for the DLSS-RR guides. A deterministic walk of the mirror/
// glass chain from the camera: while the surface is a smooth delta (perfect mirror or glass/
// transparent), keep following the reflection/refraction without recording; record the guides at
// the first non-delta (diffuse/rough) surface, at the *virtual* unfolded position
// (camPos + totalPathDistance * primaryRayDir). Because writeGuide() derives depth + motion from
// that world position, the denoiser then reprojects the reflected content (noisy) instead of the
// static mirror (which made reflections shimmer). Ported in spirit from RTXDI PSRUtils.hlsli.
// Kept separate from the stochastic colour path: PSR must be deterministic or it jitters itself.
void writeGuidePSR(uvec2 p, vec3 ro, vec3 rd,
                   bool rasterFirst, uint rInst, uint rPrim, vec2 rBc, vec3 rPos) {
    if ((pc.hasTlas & 4u) == 0u) return;
    vec3 primaryRd = rd;
    float pathDist = 0.0;
    mat3 mirrorMat = mat3(1.0);
    float chainRough = 0.0; // max roughness of the reflectors in the chain = the reflection's blur
    int glassDepth = 0; // entry/exit tracking for refraction (robust to flipped normals)
    for (int b = 0; b < 6; b++) {
        uint id, prim, iid; vec2 bc; float thit; mat4x3 o2w, w2o;
        if (rasterFirst && b == 0) {
            // Injected rasterized primary hit — skips the guide walk's primary ray query too, so the
            // raster offloads BOTH primary traces (colour path + guides). w2o for object motion.
            id = rasterInsts.data[rInst].customIndex; prim = rPrim; bc = rBc;
            iid = rInst; o2w = rasterO2W(rInst); w2o = invertAffine(o2w);
            thit = length(rPos - ro);
        } else {
            rayQueryEXT rq;
            rayQueryInitializeEXT(rq, tlas, gl_RayFlagsOpaqueEXT, 0xFFu, ro, 0.001, rd, 1e30);
            while (rayQueryProceedEXT(rq)) {}
            if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
                // Escaped to the sky: a far virtual surface straight along the unfolded direction.
                vec3 skyPos = pc.camPos.xyz + (pathDist + 1e5) * primaryRd;
                writeGuide(p, skyPos, skyPos, -primaryRd, 1.0, vec3(0.0), vec3(0.0));
                return;
            }
            id   = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
            prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
            bc   = rayQueryGetIntersectionBarycentricsEXT(rq, true);
            thit = rayQueryGetIntersectionTEXT(rq, true);
            o2w  = rayQueryGetIntersectionObjectToWorldEXT(rq, true);
            iid  = uint(rayQueryGetIntersectionInstanceIdEXT(rq, true)); // per-instance (prev xform)
            w2o  = rayQueryGetIntersectionWorldToObjectEXT(rq, true);
        }
        vec3 N, albedo, emission, geoN;
        float roughness, metallic, ior, transmission, transparentProb;
        bool frontFace, lightPathTransparent;
        vec2 surfUV; float surfAlpha;
        getSurface(id, prim, bc, o2w, rd, 0.0, N, albedo, roughness, metallic, emission, ior, transmission, frontFace, geoN, lightPathTransparent, transparentProb, surfUV, surfAlpha);
        vec3 hitPos = ro + rd * thit;
        pathDist += thit;

        // Volume boundary (P1): thin fog unfolds straight through (the guide describes what's
        // behind; the fog itself stays noisy colour for the denoiser). A DENSE volume records a
        // fog guide at the mean free path instead — a stable, reprojectable depth — because the
        // content behind it is invisible and a see-through guide would make RR reproject garbage.
        // Everything here is analytic/deterministic (PSR must not jitter itself).
        {
            GeomDesc vgd = descs[id];
            if (vgd.mat != 0ul) {
                uint vMatId = MatIds(vgd.mat).m[prim];
                if ((mats[vMatId].flags & 4u) != 0u) {
                    if (frontFace && mats[vMatId].sunSpecular > 0.0) {
                        float exitDist = traceVolumeExit(hitPos + rd * 5e-4, rd, id);
                        vec3 ss, sa, st;
                        volumeCoeffsFor(vMatId, ss, sa, st);
                        float sT = dot(st, vec3(1.0 / 3.0));
                        if (exp(-sT * exitDist) <= 0.5) {
                            float mfp = min(1.0 / max(sT, 1e-4), exitDist);
                            vec3 fogPos = pc.camPos.xyz + (pathDist + mfp) * primaryRd;
                            writeGuide(p, fogPos, fogPos, -primaryRd, 1.0,
                                       vec3(mats[vMatId].baseR, mats[vMatId].baseG, mats[vMatId].baseB),
                                       vec3(0.0));
                            return;
                        }
                    }
                    ro = hitPos + rd * 0.001; // thin (or back face): straight through
                    continue;
                }
            }
        }

        bool smoothSurf = roughness < 0.05;
        // PSR follows not just perfect mirrors but moderately glossy metals (the lobe centre is the
        // mirror direction): their reflection moves with the reflected content, so writing the guide
        // at the reflected (virtual) hit instead of the metal surface is what stops the glossy-metal
        // jitter. The reflection's blur is carried in chainRough and written as the guide roughness.
        bool isMirror = (roughness < 0.2) && (metallic > 0.7);
        bool isGlass  = smoothSurf && transmission > 0.5;
        // Architectural window glass / alpha-cutout: genuinely no bend, unfold straight through.
        bool seeThrough = transparentProb > 0.0 || surfAlpha < 0.5;

        if (isMirror) {
            chainRough = max(chainRough, roughness); // accumulate the reflection blur
            mirrorMat = mirrorMat * mirrorMatrix(N);
            rd = reflect(rd, N);
            ro = hitPos + rd * 0.001;
            continue;
        }
        if (isGlass) {
            // Follow the *actual* refraction so the guides describe what the colour path sees
            // through the glass. Straight-through guides made refracted content converge slowly
            // (depth/motion didn't match the bent view, so DLSS-RR couldn't reproject it). Entry/
            // exit eta via glassDepth (robust to flipped normals); TIR falls back to reflection.
            // The virtual position still unfolds straight ahead, but now at the true bent path
            // length, so depth + the reflected surface attributes line up with the colour.
            float iorC = max(ior, 1.001);
            bool entering = (glassDepth == 0);
            float etaRefract = entering ? (1.0 / iorC) : iorC;
            vec3 dir = refract(rd, N, etaRefract);
            if (dot(dir, dir) < 1e-8) {
                dir = reflect(rd, N); // total internal reflection
            } else {
                glassDepth = entering ? glassDepth + 1 : max(glassDepth - 1, 0);
            }
            ro = hitPos + ((dot(dir, N) >= 0.0) ? N : -N) * 0.001;
            rd = dir;
            continue;
        }
        if (seeThrough) {
            ro = hitPos + rd * 0.001; // straight through, keep direction
            continue;
        }
        // First non-delta surface: record the guides at the virtual (unfolded) position. The guide
        // roughness is max(surface, chain) so DLSS-RR blurs the reflection by the metal's glossiness.
        vec3 vpos = pc.camPos.xyz + pathDist * primaryRd;
        vec3 vN = normalize(mirrorMat * N);
        // Object motion: shift the (virtual) position by this surface point's object-space displacement
        // between frames, so a moving object's motion vector tracks it. For a direct hit vpos == hitPos
        // so prevVpos lands exactly on the previous-frame surface point; through a delta chain it is an
        // approximation. hasTlas bit6 = previous transforms uploaded.
        vec3 prevVpos = vpos;
        if ((pc.hasTlas & 64u) != 0u) {
            vec3 localPos = w2o * vec4(hitPos, 1.0);
            prevVpos = vpos + (prevObjToWorld(iid, localPos) - hitPos);
        }
        vec3 F0 = mix(vec3(0.04), albedo, metallic);
        float NdotV = max(dot(N, -rd), 1e-3);
        vec3 specAlb = EnvBRDFApprox(F0, roughness, NdotV);
        // Diffuse guide demodulated by metalness (RR Guide §3.4.1: diffuse reflectance only). Metals
        // have ~0 diffuse reflectance — their colour drives F0 -> the specular-albedo guide instead.
        writeGuide(p, vpos, prevVpos, vN, max(roughness, chainRough), albedo * (1.0 - metallic), specAlb);
        return;
    }
    // Too many delta bounces: fall back to the deepest virtual position (rough/far).
    vec3 deepPos = pc.camPos.xyz + pathDist * primaryRd;
    writeGuide(p, deepPos, deepPos, -primaryRd, 1.0, vec3(0.0), vec3(0.0));
}


// Diagnostic: paint the emissive-NEE chain state instead of rendering (set true + deploy, read
// the viewport as a false-colour map: WHITE=emissive surface, GREEN∝light, BLUE∝occluded,
// PURPLE∝facing away, YELLOW=broken pdf). Found the 256-tri emissive cap bug.
const bool DEBUG_EMISSIVE = false;

// Cycles-style indirect clamp (integrator sample_clamp_indirect, default 30): per-CONTRIBUTION,
// hue-preserving (scales the RGB sum down), applied only past the first bounce — direct light is
// NEVER clamped (integrator.cpp:310, light_passes.h:126). Replaces the old whole-sample
// per-channel min(L,16), which both kept 1.6-3x more indirect energy than Cycles AND turned every
// bright COLOURED contribution grey (e.g. a warm (40,8,2) lamp bounce became (16,8,2)) — a big
// part of the brighter/desaturated/flatter mismatch vs Cycles.
vec3 clampIndirect(vec3 v, int bounce) {
    if (bounce == 0) return v;
    float s = v.r + v.g + v.b;
    return (s > 30.0) ? v * (30.0 / s) : v;
}

// ── Path tracer core, factored for both the megakernel and the wavefront ────────────────────────
// PathCtx carries everything that must survive between bounces. The megakernel keeps it in locals
// (tracePath loops traceBounce); the wavefront serializes it to the PathState buffer and calls
// traceBounce once per dispatch. See the wavefront plan in memory.
struct PathCtx {
    vec3 ro;            // current ray origin
    vec3 rd;            // current ray direction
    vec3 tp;            // throughput
    vec3 L;             // accumulated radiance
    float coneWidth;    // ray-cone width (texture LOD)
    float pathRough;    // max surface roughness seen so far (path roughness regularization)
    int b;              // real (budget-spending) bounces taken so far
    int glassBounces;   // dielectric/passthrough hops (don't spend the bounce budget)
    int glassDepth;     // how many glass surfaces deep (0 = outside)
    bool isDiffuse;     // ray has scattered diffusely ("Is Diffuse Ray" for Light Path glass)
    bool lastDiffuse;   // previous bounce was a diffuse scatter (env NEE double-count)
};

// One real (diffuse/specular) bounce. The inner loop swallows transparent/alpha passthroughs and
// glass hops, which don't spend the bounce budget. Returns true while the path is alive (the caller
// should keep going), false on termination (miss / RR / max bounce / dead-end). Mutates c + rng.
bool traceBounce(inout PathCtx c, float spreadAngle, inout uint rng,
                 bool rasterFirst, uint rInst, uint rPrim, vec2 rBc, vec3 rPos,
                 inout SharcState sharc, bool sharcUpdate) {
    for (int inner = 0; inner < 128; inner++) {
        uint id, prim; vec2 bc; float thit; mat4x3 o2w; vec3 hitPos;
        if (rasterFirst && c.b == 0 && inner == 0) {
            // Injected rasterized primary hit — skip the ray query (hybrid raster). It's the same
            // closest hit the opaque-flagged ray query would return; transparency is still handled,
            // because the inner loop re-traces from here onward via the ray query.
            id = rasterInsts.data[rInst].customIndex; prim = rPrim; bc = rBc;
            o2w = rasterO2W(rInst);
            hitPos = rPos;
            thit = length(rPos - c.ro);
        } else {
            rayQueryEXT rq;
            rayQueryInitializeEXT(rq, tlas, gl_RayFlagsOpaqueEXT, 0xFFu, c.ro, 0.001, c.rd, 1e30);
            while (rayQueryProceedEXT(rq)) {}
            if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
                if (DEBUG_PATH && c.b == 0) { c.L = vec3(0.0, 0.0, 1.0); return false; } // blue = sky
                // Add the sky unless reached by a diffuse bounce while env NEE is active (avoids double count).
                if (c.b == 0 || !c.lastDiffuse || envDist.data[0] <= 0.0) c.L += clampIndirect(c.tp * sky(c.rd), c.b);
                if (sharcUpdate && pc.sharcCapacity > 0u) SharcUpdateMiss(sharc, sky(c.rd));
                return false;
            }
            id   = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
            prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
            bc   = rayQueryGetIntersectionBarycentricsEXT(rq, true);
            thit = rayQueryGetIntersectionTEXT(rq, true);
            o2w  = rayQueryGetIntersectionObjectToWorldEXT(rq, true);
            hitPos = c.ro + c.rd * thit;
        }

#ifdef USE_SER
        // Shader Execution Reordering at the primary hit: regroup warps by material before the
        // divergent Node VM / texturing / BSDF work. Only in the SER variant (megakernel raygen).
        if (c.b == 0 && (pc.hasTlas & 16u) != 0u) {
            reorderThreadNV(MatIds(descs[id].mat).m[prim], 16);
        }
#endif

        c.coneWidth += thit * spreadAngle; // ray-cone width at this hit (for texture LOD)
        vec3 N, albedo, emission, geoN;
        float roughness, metallic, ior, transmission, transparentProb;
        bool frontFace, lightPathTransparent;
        vec2 surfUV;
        float surfAlpha;
        getSurface(id, prim, bc, o2w, c.rd, c.coneWidth, N, albedo, roughness, metallic, emission, ior, transmission, frontFace, geoN, lightPathTransparent, transparentProb, surfUV, surfAlpha);

        // ── Volume materials (Cycles Principled Volume / Volume Scatter — P1: homogeneous) ──
        // Replaces the surface shading entirely (short-circuits the transparent cascade, SHARC and
        // the BSDF tail). The crossing does NOT consume a real bounce (glassBounces-guarded skip,
        // like glass passthrough). Deep bounces (c.b>=2) pass through unmarched, matching the C++.
        {
            GeomDesc vgd = descs[id];
            uint vMatId = 0u; uint vFlags = 0u;
            if (vgd.mat != 0ul) { vMatId = MatIds(vgd.mat).m[prim]; vFlags = mats[vMatId].flags; }
            if ((vFlags & 4u) != 0u) {
                float density = mats[vMatId].sunSpecular;
                float continueT = 0.002; // default: straight past the boundary (back face / deep / empty)
                if (frontFace && c.b < 2 && density > 0.0) {
                    vec3 entryPos = hitPos + c.rd * 5e-4;
                    float marchDist = traceVolumeExit(entryPos, c.rd, id);
                    // Geometry INSIDE the volume clips the march; the next loop iteration then hits
                    // and shades that surface normally (fog in front of it already accounted).
                    {
                        rayQueryEXT irq;
                        rayQueryInitializeEXT(irq, tlas, gl_RayFlagsOpaqueEXT, 0xFFu, entryPos, 0.001, c.rd, marchDist - 0.001);
                        while (rayQueryProceedEXT(irq)) {}
                        if (rayQueryGetIntersectionTypeEXT(irq, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
                            uint iId = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(irq, true));
                            if (iId != id) marchDist = rayQueryGetIntersectionTEXT(irq, true);
                        }
                    }
                    // Decode (byte-identical layout to the C++): anisotropy, absorption, emission,
                    // blackbody. Direct member reads — no 2204-byte struct copy into registers.
                    float anisotropy = clamp(mats[vMatId].sunSpecularEXP, -0.99, 0.99);
                    vec3 scatterCol = vec3(mats[vMatId].baseR, mats[vMatId].baseG, mats[vMatId].baseB);
                    vec3 absorbCol = vec3(uintBitsToFloat(mats[vMatId].nodeVmCode[2].x),
                                          uintBitsToFloat(mats[vMatId].nodeVmCode[2].y),
                                          uintBitsToFloat(mats[vMatId].nodeVmCode[2].z));
                    float emStrength = uintBitsToFloat(mats[vMatId].nodeVmCode[2].w);
                    vec3 emColor = vec3(uintBitsToFloat(mats[vMatId].nodeVmCode[3].x),
                                        uintBitsToFloat(mats[vMatId].nodeVmCode[3].y),
                                        uintBitsToFloat(mats[vMatId].nodeVmCode[3].z));
                    float temperature = uintBitsToFloat(mats[vMatId].nodeVmCode[3].w);
                    float blackbody = uintBitsToFloat(mats[vMatId].nodeVmCode[4].x);

                    if ((vFlags & 16u) != 0u) {
                        // Heterogeneous (noise-driven smoke): fixed-step march with the exported
                        // noise chain. Radiance goes straight to c.L (never into SHARC).
                        mat4x3 vw2o = invertAffine(o2w);
                        marchVolumeHetero(entryPos, c.rd, marchDist, vw2o, vMatId,
                                          scatterCol, absorbCol, density, anisotropy,
                                          emColor, emStrength, temperature, blackbody,
                                          rng, c.L, c.tp);
                    } else {
                        // Homogeneous (analytic, no stepping). In-scatter: ONE stochastic scatter
                        // point, distance-sampled proportional to transmittance (the closed-form
                        // (1-T)/sigma_t weight is the matching estimator) — the sun-shadow boundary
                        // becomes noise the accumulation/DLSS smooths instead of a razor-hard cut.
                        vec3 sigma_s, sigma_a, sigma_t;
                        computeVolumeCoefficients(scatterCol, absorbCol, density, sigma_s, sigma_a, sigma_t);
                        vec3 T = exp(-sigma_t * marchDist);
                        vec3 invSigmaT = vec3(1.0) / max(sigma_t, vec3(1e-4));
                        float sBar = max(dot(sigma_t, vec3(1.0 / 3.0)), 1e-4);
                        float tS = -log(1.0 - rnd(rng) * (1.0 - exp(-sBar * marchDist))) / sBar;
                        vec3 samplePos = entryPos + c.rd * clamp(tS, 0.0, marchDist);
                        vec3 L_in = volumeDirectAt(samplePos, c.rd, anisotropy, rng);
                        vec3 inscatter = sigma_s * L_in * (vec3(1.0) - T) * invSigmaT;

                        // Volume emission (+ approximate blackbody — Planck-exact comes in P2).
                        vec3 volEmission = vec3(0.0);
                        if (emStrength > 0.0) volEmission = emColor * emStrength * (vec3(1.0) - T) * invSigmaT;
                        if (blackbody > 0.0 && temperature > 0.0) {
                            float bbPower = blackbody * 5.67e-8 * temperature * temperature * temperature * temperature * 1e-12;
                            vec3 bbColor = vec3(1.0, clamp((temperature - 1000.0) / 5000.0, 0.0, 1.0),
                                                clamp((temperature - 3000.0) / 7000.0, 0.0, 1.0));
                            volEmission += bbColor * bbPower * (vec3(1.0) - T) * invSigmaT;
                        }

                        // Direct to c.L — deliberately NOT into SHARC (a volume event has no surface
                        // position/normal; caching it would poison neighbouring cells).
                        c.L += c.tp * (inscatter + volEmission);
                        c.tp *= T;
                    }
                    continueT = max(marchDist - 1e-3, 0.002); // continue from just before the exit/inner hit
                    c.ro = entryPos;
                } else {
                    c.ro = hitPos;
                }
                c.ro += c.rd * continueT;
                if (++c.glassBounces >= 32) return false;
                continue;
            }
        }

        // Stochastic transparent passthrough (Cycles Transparent BSDF / architectural glass): a
        // Transparent-weighted surface passes light straight through with no refraction.
        {
            float tProb = transparentProb;
            if (tProb > 0.0) {
                float cosTheta = abs(dot(N, -c.rd));
                float iorG = max(ior, 1.01);
                float f0 = (iorG - 1.0) / (iorG + 1.0); f0 *= f0;
                float fresnelRefl = f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
                tProb = 1.0 - fresnelRefl;   // Fresnel transmission probability
            }
            if (tProb > 0.0 && rnd(rng) < tProb) {
                c.ro = hitPos + c.rd * 0.002;    // straight through, no bend, no shading
                if (++c.glassBounces >= 32) return false;
                continue;
            }
        }
        // Light-path-transparent glass: secondary specular/reflection rays pass through cheaply.
        if (lightPathTransparent && c.b > 0 && !c.isDiffuse) {
            c.ro = hitPos + c.rd * 0.002;
            if (++c.glassBounces >= 32) return false;
            continue;
        }
        // Alpha transparency (Cycles BLEND, stochastic): probability (1 - alpha) passes through.
        if (surfAlpha < 0.999 && rnd(rng) >= surfAlpha) {
            c.ro = hitPos + c.rd * 0.0005;
            if (++c.glassBounces >= 64) return false;
            continue;
        }

        // DEBUG: visualize the surface UV of the primary hit (red=U, green=V, tiled).
        if (DEBUG_PATH && c.b == 0) {
            c.L = vec3(fract(surfUV.x), fract(surfUV.y), 0.0);
            return false;
        }

        // ── SHARC ── At a secondary opaque hit: non-update pixels read the cached outgoing radiance and
        // terminate (skipping the shade + the deeper bounces); update pixels keep tracing to fill it.
        SharcGrid sharcG = sharcGrid();
        bool sharcOn = pc.sharcCapacity > 0u;
        // The cache stores view-INDEPENDENT diffuse radiance, so only use it on diffuse surfaces (skip
        // glass/transmission + sharp specular), and only when the segment exceeds the voxel size (closer
        // than that the coarse cache leaks a neighbour's lighting across surfaces).
        bool sharcDiffuse = transmission < 0.3 && roughness > 0.3;
        if (sharcOn && !sharcUpdate && sharcDiffuse && c.b >= 1
            && thit > sharcVoxelSize(sharcGetLevel(hitPos, sharcG), sharcG)) {
            vec3 cached;
            if (SharcGetCachedRadiance(sharcG, hitPos, N, pc.sharcCapacity, cached)) {
                c.L += clampIndirect(c.tp * cached, c.b); // cached outgoing radiance (always indirect) + terminate
                return false;
            }
        }

        // Emitted radiance from this surface. Light-sampling strategy: only add the forward (BSDF)
        // term when the camera sees the emitter directly (b==0) or there is no emissive NEE. At a
        // bounce-reached emitter (b>0) with NEE active, the emitter is already counted by the previous
        // vertex's (now-unweighted) emissive NEE; adding the forward term too double-counts it (lamps +
        // their bounce light were up to ~2x too bright, washing the scene). Mirrors the env diffuse-skip.
        if (c.b == 0u || pc.emissiveCount == 0u) {
            c.L += clampIndirect(c.tp * emission, c.b);
        }

        // Transmissive surfaces have no Lambertian lobe.
        vec3 diffAlbedo = albedo * (1.0 - metallic) * (1.0 - transmission);
        vec3 V = -c.rd;
        vec3 F0 = mix(vec3(0.04), albedo, metallic);
        // Dielectric specular->diffuse energy layering (Cycles closure_layering_weight, bsdf_util.h:491):
        // the specular lobe reflects part of the incoming energy, so the diffuse below must lose it.
        // Without this the spec was added ON TOP at full strength, so every dielectric surface read too
        // bright / flat / washed (the lift grows toward grazing angles). Attenuating diffAlbedo here
        // covers BOTH the direct (brdf) and the indirect bounce, which reuse this same diffAlbedo.
        {
            float NdotV_lay = max(dot(N, V), 1e-3);
            vec3 specAlb_lay = EnvBRDFApprox(F0, roughness, NdotV_lay);
            diffAlbedo *= 1.0 - max(specAlb_lay.r, max(specAlb_lay.g, specAlb_lay.b));
        }
        // Path roughness regularization (RTXPT-style): a glossy lobe after a rougher vertex is widened.
        float regRough = max(roughness, c.pathRough);
        c.pathRough = max(c.pathRough, roughness);
        float alpha = max(regRough * regRough, 1e-3);

        // ── DEBUG_EMISSIVE v2: 16-sample false-colour map of the emissive-NEE chain ──
        // WHITE = this surface itself is emissive (in the renderer's set!) | RED = zero triangles |
        // GREEN ∝ real emissive light arriving | BLUE ∝ fraction shadow-occluded |
        // PURPLE ∝ fraction facing away | YELLOW ∝ fraction with broken pdf.
        if (DEBUG_EMISSIVE && c.b == 0) {
            vec3 dbg = vec3(0.0);
            if (pc.emissiveCount == 0u) {
                dbg = vec3(1.0, 0.0, 0.0);
            } else if (dot(emission, vec3(1.0)) > 0.0) {
                dbg = vec3(1.0); // WHITE: the renderer sees THIS mesh as emissive
            } else {
                vec3 so = hitPos + geoN * 0.002;
                vec3 lightSum = vec3(0.0);
                float occ = 0.0, away = 0.0, badPdf = 0.0;
                const int NDBG = 16;
                for (int di = 0; di < NDBG; di++) {
                    EmissiveSample es = sampleEmissiveTriangle(pc.emissiveCount, rnd(rng), rand2(rng));
                    if (es.pdf <= 0.0) { badPdf += 1.0; continue; }
                    vec3 toE = es.position - so;
                    float eDist = length(toE);
                    vec3 eDir = toE / max(eDist, 1e-4);
                    float eNdotL = max(dot(N, eDir), 0.0);
                    float eCos = abs(dot(-eDir, es.normal));
                    if (eNdotL <= 0.0) { away += 1.0; continue; }
                    vec3 eVis = shadowT(so, eDir, eDist - 0.01);
                    if (eVis == vec3(0.0)) { occ += 1.0; continue; }
                    lightSum += es.emission * eCos * eNdotL * eVis / max(es.pdf, 1e-6) / (eDist * eDist);
                }
                float g = clamp(dot(lightSum, vec3(0.333)) / float(NDBG) * 4.0, 0.0, 1.0);
                dbg = vec3(0.0, g, 0.0)
                    + vec3(0.0, 0.0, 0.6) * (occ / float(NDBG))
                    + vec3(0.5, 0.0, 0.5) * (away / float(NDBG))
                    + vec3(0.6, 0.6, 0.0) * (badPdf / float(NDBG));
            }
            c.L = dbg;
            return false;
        }

        // Direct lighting (NEE: diffuse + specular glints from sun/lights, soft shadows).
        vec3 directLight = directShade(hitPos, N, geoN, V, diffAlbedo, F0, alpha, c.b, rng);
        c.L += clampIndirect(c.tp * directLight, c.b); // b==0 -> direct, never clamped (Cycles)
        // SHARC update: write this voxel's outgoing direct radiance + backpropagate to prior vertices.
        // Only cache diffuse surfaces (glass/specular are view-dependent; the path passes through them).
        if (sharcOn && sharcUpdate && sharcDiffuse) {
            SharcUpdateHit(sharcG, sharc, hitPos, N, emission + directLight, rnd(rng), pc.sharcCapacity, thit);
        }

        // BRDF bounce. A transmissive (glass) interaction is chosen with probability pTrans.
        c.lastDiffuse = false;
        vec2 u = rand2(rng);
        float pTrans = clamp(transmission, 0.0, 1.0) * (1.0 - metallic);
        if (pTrans > 0.001 && rnd(rng) < pTrans) {
            // Rough dielectric: sample a microfacet half-vector, split reflect/refract by Fresnel.
            float iorC = max(ior, 1.001);
            bool entering = (c.glassDepth == 0); // camera is always outside -> first glass hit enters
            vec3 Hn = (alpha > 1.5e-3) ? sampleGgxVndf(u, V, N, alpha) : N;
            float etaFresnel = entering ? iorC : (1.0 / iorC);
            float F = fresnelDielectric(dot(V, Hn), etaFresnel);
            vec3 dir;
            vec3 tint = vec3(1.0);
            bool crossed = false;
            if (rnd(rng) < F) {
                dir = reflect(c.rd, Hn);                          // surface reflection (untinted)
            } else {
                float etaRefract = entering ? (1.0 / iorC) : iorC;
                dir = refract(c.rd, Hn, etaRefract);
                if (dot(dir, dir) < 1e-8) dir = reflect(c.rd, Hn); // total internal reflection
                else { tint = sqrt(max(albedo, vec3(0.0))); crossed = true; }
            }
            c.glassDepth = crossed ? (entering ? c.glassDepth + 1 : max(c.glassDepth - 1, 0)) : c.glassDepth;
            float w = (alpha > 1.5e-3) ? smithG1(max(abs(dot(N, dir)), 1e-4), alpha) : 1.0;
            c.tp *= tint * w / pTrans;
            c.ro = hitPos + ((dot(dir, N) >= 0.0) ? N : -N) * 0.001;
            c.rd = dir;
            // Glass interfaces don't spend the bounce budget; cap total dielectric hops instead.
            if (++c.glassBounces >= 32) return false;
            continue;   // not a real bounce -> keep tracing in the inner loop
        }

        // Opaque diffuse/specular: this IS a real bounce.
        float invOpaque = 1.0 / max(1.0 - pTrans, 1e-3);
        float lumD = dot(diffAlbedo, vec3(0.299, 0.587, 0.114));
        float lumS = dot(F0, vec3(0.299, 0.587, 0.114));
        float pSpec = clamp(lumS / (lumS + lumD + 1e-4), 0.1, 0.9);
        c.ro = hitPos + geoN * 0.002;   // offset along geometric normal (silhouette-safe)
        vec3 sharcTpBefore = c.tp;      // SHARC: snapshot to derive this bounce's throughput factor
        if (rnd(rng) < pSpec) {
            vec3 H = sampleGgxVndf(u, V, N, alpha);
            vec3 Ldir = reflect(c.rd, H);
            if (dot(Ldir, N) <= 0.0) return false;
            float VdotH = max(dot(V, H), 0.0);
            vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
            c.tp *= F * smithG1(max(dot(N, Ldir), 0.0), alpha)
                  * ggxEnergyComp(F0, max(dot(N, V), 1e-4), alpha) * invOpaque / pSpec;
            c.rd = Ldir;
        } else {
            c.rd = cosineHemisphere(N, u);
            c.tp *= diffAlbedo * invOpaque / (1.0 - pSpec);
            c.isDiffuse = true;
            c.lastDiffuse = (c.b <= 1) && ((pc.hasTlas & 32u) != 0u);
        }

        // Russian roulette after a couple of real bounces (uses the entering bounce count).
        if (c.b >= 2) {
            float q = clamp(max(c.tp.r, max(c.tp.g, c.tp.b)), 0.05, 1.0);
            if (rnd(rng) > q) return false;
            c.tp /= q;
        }
        // SHARC: fold this bounce's throughput (BSDF * Russian roulette) into the stored vertex weights.
        if (sharcOn && sharcUpdate) SharcSetThroughput(sharc, c.tp / max(sharcTpBefore, vec3(1e-6)));
        c.b++;
        if (c.b >= int(pc.maxBounces)) return false;
        return true;   // real bounce taken; the path continues
    }
    return false;       // inner loop overflow (pathological glass/alpha stack)
}

// Megakernel entry: run the whole path by looping traceBounce. Same image as before the split.
vec3 tracePath(vec3 ro, vec3 rd, float spreadAngle, inout uint rng, bool sharcUpdate) {
    PathCtx c;
    c.ro = ro; c.rd = rd; c.tp = vec3(1.0); c.L = vec3(0.0);
    c.coneWidth = 0.0; c.pathRough = 0.0;
    c.b = 0; c.glassBounces = 0; c.glassDepth = 0;
    c.isDiffuse = false; c.lastDiffuse = false;
    SharcState sharc; SharcInit(sharc); // megakernel keeps the SHARC update state in a local
    for (int guard = 0; guard < 512; guard++) {
        if (!traceBounce(c, spreadAngle, rng, false, 0u, 0u, vec2(0.0), vec3(0.0), sharc, sharcUpdate)) break;
    }
    return c.L;
}
