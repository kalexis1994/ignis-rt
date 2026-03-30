#version 460

// Hybrid G-buffer rasterization: vertex shader
// Position-only vertex input (float3, 12-byte stride from BLAS vertex buffers).

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform PushConstants {
    mat4 mvp;              // proj * view * instanceModel
    vec4 camPosAndPad;     // camera world position (xyz), pad (w)
    uint instanceIndex;    // TLAS instance index
    uint blasIndex;        // BLAS custom index
};

layout(location = 0) flat out uint outInstanceIndex;
layout(location = 1) flat out uint outBlasIndex;
layout(location = 2) out float outLinearDist;

void main() {
    vec4 clipPos = mvp * vec4(inPosition, 1.0);
    gl_Position = clipPos;
    // Linear view distance = clip.w for standard perspective projection
    // (clip.w = dot(modelViewRow3, vertex) which equals -viewZ for RH)
    outLinearDist = abs(clipPos.w);
    outInstanceIndex = instanceIndex;
    outBlasIndex = blasIndex;
}
