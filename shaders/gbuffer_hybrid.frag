#version 460

// Hybrid G-buffer rasterization: fragment shader
// Outputs primitive ID, instance metadata, and linear distance from camera.

layout(location = 0) flat in uint outInstanceIndex;
layout(location = 1) flat in uint outBlasIndex;
layout(location = 2) in float outLinearDist;

layout(location = 0) out uint fragPrimID;
layout(location = 1) out uvec2 fragInstanceInfo;
layout(location = 2) out float fragLinearDepth;

void main() {
    fragPrimID = uint(gl_PrimitiveID);
    fragInstanceInfo = uvec2(outInstanceIndex, outBlasIndex);
    fragLinearDepth = outLinearDist;
}
