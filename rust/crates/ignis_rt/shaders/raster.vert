#version 460
// ── Hybrid rasterization — primary-visibility vertex shader ──────────────────────────────────────
// Transforms an object-space vertex position by the per-instance MVP (CPU-computed viewProj *
// objectToWorld). One draw per TLAS instance; the fragment shader records the hit into the G-buffer
// that the wavefront ray-gen reads instead of casting the primary ray. See the wavefront plan in memory.

layout(location = 0) in vec3 inPos;

layout(push_constant) uniform PC {
    mat4 mvp;        // viewProj * objectToWorld for this instance
    uint instanceId; // stored in the G-buffer for the wavefront to resolve the surface
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPos, 1.0);
}
