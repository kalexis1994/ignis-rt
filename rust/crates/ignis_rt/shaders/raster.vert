#version 460
// ── Hybrid rasterization — primary-visibility vertex shader ──────────────────────────────────────
// Transforms an object-space vertex by the per-instance MVP (viewProj * objectToWorld) and passes the
// world-space position to the fragment, which records it into the G-buffer so the wavefront can
// resolve the primary hit without a ray query. One draw per TLAS instance.

layout(location = 0) in vec3 inPos;
layout(location = 0) out vec3 vWorldPos;

layout(push_constant) uniform PC {
    mat4 mvp;                       // viewProj * objectToWorld
    vec4 o2w0; vec4 o2w1; vec4 o2w2; // objectToWorld rows (row-major 3x4)
    uint instanceId;                // stored in the hit G-buffer
} pc;

void main() {
    vec4 p = vec4(inPos, 1.0);
    vWorldPos = vec3(dot(pc.o2w0, p), dot(pc.o2w1, p), dot(pc.o2w2, p));
    gl_Position = pc.mvp * p;
}
