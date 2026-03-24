#version 460

// Hybrid rasterization: G-buffer vertex shader
// Uses MVP matrix from push constants (computed per-instance on CPU)

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform PushConstants {
    mat4 mvp;           // proj * view * model (pre-computed per instance)
    uint customIndex;   // BLAS geometry index (maps to geometryMetadata)
    uint instanceId;    // Instance index (for motion vectors via prevTransforms)
};

// Pass to fragment shader
layout(location = 0) flat out uint vCustomIndex;
layout(location = 1) flat out uint vInstanceId;

void main() {
    gl_Position = mvp * vec4(inPosition, 1.0);
    vCustomIndex = customIndex;
    vInstanceId = instanceId;
}
