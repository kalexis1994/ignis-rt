// ── Wavefront PathState buffer (binding 19) ─────────────────────────────────────────────────────
// Serializes the PathCtx (from pathtracer_common.glsl) + rng + spreadAngle + flags so a path can be
// carried between per-bounce dispatches. 5 vec4 = 80 bytes per path; one path per pixel, so the
// thread/pixel index IS the path index (no pixel field needed). Included by the wavefront stages
// only — the megakernel never references binding 19. Include AFTER pathtracer_common.glsl.

// 7 vec4 = 112 bytes. s0..s4 = PathCtx + rng + spreadAngle + flags; s5..s6 = the injected raster
// primary hit (instanceId, primId, barycentrics, worldPos), present only under hybrid raster.
struct PathState { vec4 s0; vec4 s1; vec4 s2; vec4 s3; vec4 s4; vec4 s5; vec4 s6; };
layout(binding = 19, std430) buffer PathStates { PathState s[]; } paths;

// Compaction (Phase 2): a ping-pong queue of still-alive pixel indices + a control buffer. The extend
// stage reads queue half pc.wfIn, appends survivors to half pc.wfOut, and atomic-bumps the live count.
layout(binding = 20, std430) buffer WfQueue { uint q[]; } wfq;  // 2 halves of N pixels
layout(binding = 21, std430) buffer WfCtrl  { uint c[]; } wfc;  // [count0, count1, argsX, argsY, argsZ]

// Hybrid-raster G-buffer (read by wf_gen): the rasterized primary hit + world position.
layout(binding = 22, rg32ui)  uniform readonly uimage2D gHit; // (instanceId, primId), 0xFFFFFFFF = miss
layout(binding = 24, rgba32f) uniform readonly image2D  gPos; // world position of the hit

// Store / load the injected raster primary hit (s5..s6). s6.w = 1 marks it present.
void storeRasterHit(uint idx, uint rInst, uint rPrim, vec2 rBc, vec3 rPos) {
    paths.s[idx].s5 = vec4(uintBitsToFloat(rInst), uintBitsToFloat(rPrim), rBc.x, rBc.y);
    paths.s[idx].s6 = vec4(rPos, 1.0);
}
void noRasterHit(uint idx) { paths.s[idx].s6 = vec4(0.0); }
bool loadRasterHit(uint idx, out uint rInst, out uint rPrim, out vec2 rBc, out vec3 rPos) {
    vec4 a = paths.s[idx].s5, b = paths.s[idx].s6;
    rInst = floatBitsToUint(a.x); rPrim = floatBitsToUint(a.y); rBc = a.zw; rPos = b.xyz;
    return b.w > 0.5;
}

const uint PATH_DEAD = 4u; // flag bit (bit0 = isDiffuse, bit1 = lastDiffuse, bit2 = dead)

void storePath(uint idx, PathCtx c, uint rng, float spreadAngle, bool dead) {
    uint flags = (c.isDiffuse ? 1u : 0u) | (c.lastDiffuse ? 2u : 0u) | (dead ? PATH_DEAD : 0u);
    paths.s[idx].s0 = vec4(c.ro, c.coneWidth);
    paths.s[idx].s1 = vec4(c.rd, c.pathRough);
    paths.s[idx].s2 = vec4(c.tp, spreadAngle);
    paths.s[idx].s3 = vec4(c.L, uintBitsToFloat(rng));
    paths.s[idx].s4 = vec4(float(c.b), float(c.glassBounces), float(c.glassDepth), uintBitsToFloat(flags));
}

PathCtx loadPath(uint idx, out uint rng, out float spreadAngle, out bool dead) {
    PathState st = paths.s[idx];
    PathCtx c;
    c.ro = st.s0.xyz; c.coneWidth = st.s0.w;
    c.rd = st.s1.xyz; c.pathRough = st.s1.w;
    c.tp = st.s2.xyz; spreadAngle = st.s2.w;
    c.L  = st.s3.xyz; rng = floatBitsToUint(st.s3.w);
    c.b = int(st.s4.x); c.glassBounces = int(st.s4.y); c.glassDepth = int(st.s4.z);
    uint flags = floatBitsToUint(st.s4.w);
    c.isDiffuse   = (flags & 1u) != 0u;
    c.lastDiffuse = (flags & 2u) != 0u;
    dead = (flags & PATH_DEAD) != 0u;
    return c;
}
