#version 460
// ── Hybrid rasterization — primary-visibility fragment shader ────────────────────────────────────
// Writes (instanceId, primId) to the hit G-buffer + the interpolated world position to the position
// G-buffer; depth is written automatically by the depth test. The wavefront resolves the primary hit
// (geometry + barycentrics) from these instead of casting the primary ray.

layout(location = 0) in vec3 vWorldPos;
layout(location = 0) out uvec2 outHit;
layout(location = 1) out vec4 outPos;

layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 o2w0; vec4 o2w1; vec4 o2w2;
    uint instanceId;
} pc;

void main() {
    outHit = uvec2(pc.instanceId, uint(gl_PrimitiveID));
    outPos = vec4(vWorldPos, 1.0);
}
