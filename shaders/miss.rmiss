#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec4 hitValue;      // Primary rays (location 0)
layout(location = 1) rayPayloadInEXT vec4 shadowValue;   // Shadow rays (location 1)

void main() {
    // Sky gradient for primary rays
    vec3 skyTop = vec3(3.0, 4.0, 6.0);           // Blue sky
    vec3 skyHorizon = vec3(6.0, 7.0, 8.0);       // Bright horizon
    vec3 skyBottom = vec3(2.0, 2.5, 3.0);        // Ground reflection

    // Use ray direction Y component for gradient
    float t = clamp(gl_WorldRayDirectionEXT.y * 0.5 + 0.5, 0.0, 1.0);

    // Two-part gradient: below and above horizon
    vec3 skyColor;
    if (t < 0.5) {
        // Below horizon (ground reflection)
        skyColor = mix(skyBottom, skyHorizon, t * 2.0);
    } else {
        // Above horizon (sky)
        skyColor = mix(skyHorizon, skyTop, (t - 0.5) * 2.0);
    }

    // Primary rays (location 0): return sky color
    hitValue.rgb = skyColor;

    // Shadow rays (location 1): return bright = not occluded
    shadowValue.rgb = vec3(1.0);
}
