#version 460

// Hybrid G-buffer rasterization: fragment shader
// Outputs primitive ID + instance metadata to uint render targets.
// Depth is written automatically via the depth attachment (D32_SFLOAT).
// The raygen shader reconstructs hit position from depth + camera matrices.

layout(location = 0) flat in uint outInstanceIndex;
layout(location = 1) flat in uint outBlasIndex;

layout(location = 0) out uint fragPrimID;
layout(location = 1) out uvec2 fragInstanceInfo;

void main() {
    fragPrimID = uint(gl_PrimitiveID);
    fragInstanceInfo = uvec2(outInstanceIndex, outBlasIndex);
}
