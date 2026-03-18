#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 projection;
    vec4 lightDir;
    vec4 cameraPos;
    mat4 lightViewProj;
};

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
};

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragWorldPos;

void main() {
    vec4 worldPos = model * vec4(inPosition, 1.0);
    gl_Position = projection * view * worldPos;

    fragNormal = mat3(transpose(inverse(model))) * inNormal;
    fragUV = inUV;
    fragWorldPos = worldPos.xyz;
}
