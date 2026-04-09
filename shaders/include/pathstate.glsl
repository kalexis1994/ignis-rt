// ============================================================
// pathstate.glsl — PathState SoA (Structure of Arrays) layout
// 4 separate buffers per field group, double-buffered (ping-pong)
//
// Bindings (set 1):
//   0:  originDir READ (6 floats/path: origin.xyz + direction.xyz)
//   7:  originDir WRITE
//   9:  pixelRng READ (2 uints/path: pixelIndex + rngState)
//   10: pixelRng WRITE
//   11: throughput READ (3 floats/path: throughput.xyz)
//   12: throughput WRITE
//   13: flags READ (1 uint/path)
//   14: flags WRITE
// ============================================================

#ifndef PATHSTATE_GLSL
#define PATHSTATE_GLSL

#ifdef PATHSTATE_WRITE_TO_READ
// ---- K0 (camera rays): writes to READ-side buffers ----

layout(binding = 0,  set = 1, scalar) buffer OriginDirBuf   { float data[]; } psOriginDir;
layout(binding = 9,  set = 1, scalar) buffer PixelRngBuf    { uint  data[]; } psPixelRng;
layout(binding = 11, set = 1, scalar) buffer ThroughputBuf  { float data[]; } psThroughput;
layout(binding = 13, set = 1, scalar) buffer FlagsBuf       { uint  data[]; } psFlags;

void psInitOrigin(uint idx, vec3 v)      { uint b = idx*6u; psOriginDir.data[b]=v.x; psOriginDir.data[b+1]=v.y; psOriginDir.data[b+2]=v.z; }
void psInitDirection(uint idx, vec3 v)   { uint b = idx*6u; psOriginDir.data[b+3]=v.x; psOriginDir.data[b+4]=v.y; psOriginDir.data[b+5]=v.z; }
void psInitPixelIndex(uint idx, uint v)  { psPixelRng.data[idx*2u] = v; }
void psInitRngState(uint idx, uint v)    { psPixelRng.data[idx*2u+1u] = v; }
void psInitThroughput(uint idx, vec3 v)  { uint b = idx*3u; psThroughput.data[b]=v.x; psThroughput.data[b+1]=v.y; psThroughput.data[b+2]=v.z; }
void psInitFlags(uint idx, uint v)       { psFlags.data[idx] = v; }

void psInitAll(uint idx, vec3 origin, uint pixelIndex, vec3 dir, uint rng, vec3 throughput, uint flags) {
    psInitOrigin(idx, origin);
    psInitDirection(idx, dir);
    psInitPixelIndex(idx, pixelIndex);
    psInitRngState(idx, rng);
    psInitThroughput(idx, throughput);
    psInitFlags(idx, flags);
}

#else
// ---- All other kernels: READ from [R] bindings, WRITE to [W] bindings ----

layout(binding = 0,  set = 1, scalar) buffer OriginDirRead    { float data[]; } psOriginDirR;
layout(binding = 7,  set = 1, scalar) buffer OriginDirWrite   { float data[]; } psOriginDirW;
layout(binding = 9,  set = 1, scalar) buffer PixelRngRead     { uint  data[]; } psPixelRngR;
layout(binding = 10, set = 1, scalar) buffer PixelRngWrite    { uint  data[]; } psPixelRngW;
layout(binding = 11, set = 1, scalar) buffer ThroughputRead   { float data[]; } psThroughputR;
layout(binding = 12, set = 1, scalar) buffer ThroughputWrite  { float data[]; } psThroughputW;
layout(binding = 13, set = 1, scalar) buffer FlagsRead        { uint  data[]; } psFlagsR;
layout(binding = 14, set = 1, scalar) buffer FlagsWrite       { uint  data[]; } psFlagsW;

// ---- Read helpers (from READ buffers) ----
vec3  psReadOrigin(uint idx)      { uint b = idx*6u; return vec3(psOriginDirR.data[b], psOriginDirR.data[b+1], psOriginDirR.data[b+2]); }
vec3  psReadDirection(uint idx)   { uint b = idx*6u; return vec3(psOriginDirR.data[b+3], psOriginDirR.data[b+4], psOriginDirR.data[b+5]); }
uint  psReadPixelIndex(uint idx)  { return psPixelRngR.data[idx*2u]; }
uint  psReadRngState(uint idx)    { return psPixelRngR.data[idx*2u+1u]; }
vec3  psReadThroughput(uint idx)  { uint b = idx*3u; return vec3(psThroughputR.data[b], psThroughputR.data[b+1], psThroughputR.data[b+2]); }
uint  psReadFlags(uint idx)       { return psFlagsR.data[idx]; }

// ---- Write helpers (to WRITE buffers) ----
void psWriteOrigin(uint idx, vec3 v)      { uint b = idx*6u; psOriginDirW.data[b]=v.x; psOriginDirW.data[b+1]=v.y; psOriginDirW.data[b+2]=v.z; }
void psWriteDirection(uint idx, vec3 v)   { uint b = idx*6u; psOriginDirW.data[b+3]=v.x; psOriginDirW.data[b+4]=v.y; psOriginDirW.data[b+5]=v.z; }
void psWritePixelIndex(uint idx, uint v)  { psPixelRngW.data[idx*2u] = v; }
void psWriteRngState(uint idx, uint v)    { psPixelRngW.data[idx*2u+1u] = v; }
void psWriteThroughput(uint idx, vec3 v)  { uint b = idx*3u; psThroughputW.data[b]=v.x; psThroughputW.data[b+1]=v.y; psThroughputW.data[b+2]=v.z; }
void psWriteFlags(uint idx, uint v)       { psFlagsW.data[idx] = v; }

void psWriteAll(uint idx, vec3 origin, uint pixelIndex, vec3 dir, uint rng, vec3 throughput, uint flags) {
    psWriteOrigin(idx, origin);
    psWriteDirection(idx, dir);
    psWritePixelIndex(idx, pixelIndex);
    psWriteRngState(idx, rng);
    psWriteThroughput(idx, throughput);
    psWriteFlags(idx, flags);
}

// ---- Terminate: write to READ buffer (in-place flag update) ----
void psTerminate(uint idx, uint flags) {
    psFlagsR.data[idx] = flags | 0x20u;
}

// ---- Read from WRITE buffers (used by sort_count, sort_scatter) ----
uint psWriteReadFlags(uint idx) { return psFlagsW.data[idx]; }

// Copy full path from WRITE to READ buffers (used by sort_scatter)
void psCopyWriteToRead(uint srcIdx, uint dstIdx) {
    // originDir (6 floats)
    uint s6 = srcIdx * 6u, d6 = dstIdx * 6u;
    for (uint i = 0u; i < 6u; i++)
        psOriginDirR.data[d6 + i] = psOriginDirW.data[s6 + i];
    // pixelRng (2 uints)
    uint s2 = srcIdx * 2u, d2 = dstIdx * 2u;
    psPixelRngR.data[d2]     = psPixelRngW.data[s2];
    psPixelRngR.data[d2 + 1] = psPixelRngW.data[s2 + 1];
    // throughput (3 floats)
    uint s3 = srcIdx * 3u, d3 = dstIdx * 3u;
    for (uint i = 0u; i < 3u; i++)
        psThroughputR.data[d3 + i] = psThroughputW.data[s3 + i];
    // flags (1 uint)
    psFlagsR.data[dstIdx] = psFlagsW.data[srcIdx];
}

#endif // PATHSTATE_WRITE_TO_READ

#endif // PATHSTATE_GLSL
