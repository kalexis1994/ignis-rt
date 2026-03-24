#version 460
#extension GL_EXT_fragment_shader_barycentric : require

// Hybrid rasterization: G-buffer fragment shader
// Outputs visibility data (customIndex, primitiveId, instanceId, barycentrics)
// for the raygen shader to read instead of tracing the primary ray.

layout(location = 0) flat in uint vCustomIndex;
layout(location = 1) flat in uint vInstanceId;

// MRT outputs — must match visibility buffer image formats
layout(location = 0) out uvec2 outVisibility;    // RG32UI: customIndex, primitiveId
layout(location = 1) out uint  outInstanceId;     // R32UI: instance ID
layout(location = 2) out vec2  outBarycentric;    // RG16F: barycentric UV

void main() {
    outVisibility = uvec2(vCustomIndex, gl_PrimitiveID);
    outInstanceId = vInstanceId;
    // VK_KHR_fragment_shader_barycentric provides gl_BaryCoordEXT
    outBarycentric = gl_BaryCoordEXT.yz;  // (u, v) — w = 1-u-v computed in raygen
}
