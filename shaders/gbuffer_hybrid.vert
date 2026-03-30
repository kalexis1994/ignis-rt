#version 460

// Hybrid G-buffer rasterization: vertex shader
// Outputs clip-space position + flat instance/BLAS metadata.
// Position-only vertex input (float3, 12-byte stride from BLAS vertex buffers).

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform PushConstants {
    mat4 mvp;              // proj * view * instanceModel (with DLSS jitter)
    uint instanceIndex;    // TLAS instance index
    uint blasIndex;        // BLAS custom index (geometryMetadata lookup)
};

layout(location = 0) flat out uint outInstanceIndex;
layout(location = 1) flat out uint outBlasIndex;

void main() {
    gl_Position = mvp * vec4(inPosition, 1.0);
    outInstanceIndex = instanceIndex;
    outBlasIndex = blasIndex;
}
