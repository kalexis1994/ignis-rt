#version 460

// Hybrid rasterization: G-buffer vertex shader
// Transforms vertices and passes through instance IDs for visibility buffer

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform PushConstants {
    mat4 model;         // object-to-world transform
    uint customIndex;   // BLAS geometry index (maps to geometryMetadata)
    uint instanceId;    // Instance index (for motion vectors via prevTransforms)
};

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 projection;
};

// Pass to fragment shader
layout(location = 0) flat out uint vCustomIndex;
layout(location = 1) flat out uint vInstanceId;

void main() {
    vec4 worldPos = model * vec4(inPosition, 1.0);
    gl_Position = projection * view * worldPos;
    vCustomIndex = customIndex;
    vInstanceId = instanceId;
}
