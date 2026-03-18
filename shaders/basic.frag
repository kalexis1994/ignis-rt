#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragWorldPos;

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

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(-lightDir.xyz);

    float NdotL = max(dot(N, L), 0.0);
    float ambient = 0.15;
    float diffuse = NdotL;

    vec3 lit = color.rgb * (ambient + diffuse);
    outColor = vec4(lit, color.a);
}
