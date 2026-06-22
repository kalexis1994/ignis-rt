#version 460
// ── Hybrid rasterization — primary-visibility fragment shader ────────────────────────────────────
// Writes (instanceId, primId) to the hit G-buffer; the depth attachment is written automatically by
// the depth test, so the closest surface per pixel wins. The wavefront ray-gen reconstructs the
// primary hit from this instead of tracing the primary ray.

layout(location = 0) out uvec2 outHit;

layout(push_constant) uniform PC {
    mat4 mvp;
    uint instanceId;
} pc;

void main() {
    outHit = uvec2(pc.instanceId, uint(gl_PrimitiveID));
}
