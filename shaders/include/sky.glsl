// ============================================================
// sky.glsl — Shared procedural sky evaluation
// Requires: CameraProperties 'cam' uniform at binding 2
// ============================================================

#ifndef SKY_GLSL
#define SKY_GLSL

vec3 proceduralSky(vec3 dir, vec3 sunDirL, vec3 ambColor, vec3 sunTint) {
    float sunAlt = sunDirL.y;
    float cosTheta = max(dir.y, 0.001);
    float cosGamma = dot(dir, sunDirL);

    float rayleighPhase = 0.75 * (1.0 + cosGamma * cosGamma);
    vec3 rayleighColor = vec3(0.30, 0.42, 0.68);

    float sunsetShift = clamp(1.0 - sunAlt * 2.0, 0.0, 1.0);
    rayleighColor = mix(rayleighColor, vec3(0.7, 0.35, 0.15), sunsetShift * 0.6);

    float opticalDepth = 1.0 / (cosTheta + 0.15 * pow(93.885 - degrees(acos(cosTheta)), -1.253));
    opticalDepth = min(opticalDepth, 40.0);

    vec3 rayleigh = rayleighColor * rayleighPhase * exp(-opticalDepth * 0.1);

    float g = 0.76;
    float miePhase = 1.5 * ((1.0 - g*g) / (2.0 + g*g))
                   * (1.0 + cosGamma*cosGamma) / pow(1.0 + g*g - 2.0*g*cosGamma, 1.5);
    vec3 mie = sunTint * miePhase * 0.02;

    float sunInt = clamp(sunAlt * 3.0 + 0.3, 0.05, 1.0);
    vec3 clearSky = (rayleigh + mie) * sunInt * ambColor * 2.0;

    clearSky += sunTint * pow(max(cosGamma, 0.0), 512.0) * 5.0;
    clearSky += sunTint * pow(max(cosGamma, 0.0), 16.0) * 0.4;

    vec3 fogColor = ambColor * 0.6 + sunTint * 0.3 + vec3(0.1);
    fogColor = mix(fogColor, vec3(dot(fogColor, vec3(0.33))), 0.3);
    float hazeT = 1.0 - smoothstep(0.0, 0.25, dir.y);
    clearSky = mix(clearSky, fogColor * sunInt, hazeT * 0.8);

    return max(clearSky, vec3(0.0));
}

vec3 evaluateSky(vec3 dir) {
    float sunIntensity = cam.sunLight.w;

    // World background color (from Blender's World → Background node)
    // Packed in windParams.zw (R, G) and rainParams.w (B)
    vec3 worldBg = vec3(cam.windParams.z, cam.windParams.w, cam.rainParams.w);

    if (sunIntensity <= 0.0) {
        // No sun light — use world background color as uniform environment
        return max(worldBg, vec3(0.0));
    }

    vec3 sunDir = normalize(cam.sunLight.xyz);
    vec3 ambColor = cam.ambientLight.rgb;
    vec3 sunTint = cam.skyLight.rgb;

    if (dot(ambColor, ambColor) < 0.001) ambColor = vec3(0.5, 0.6, 0.8);
    if (dot(sunTint, sunTint) < 0.001) sunTint = vec3(1.0, 0.92, 0.85);

    if (dir.y > 0.0) {
        return proceduralSky(dir, sunDir, ambColor, sunTint);
    } else {
        vec3 fogColor = ambColor * 0.6 + sunTint * 0.3 + vec3(0.1);
        fogColor = mix(fogColor, vec3(dot(fogColor, vec3(0.33))), 0.3);
        float sunI = clamp(sunDir.y * 3.0 + 0.3, 0.05, 1.0);
        float groundT = clamp(-dir.y * 5.0, 0.0, 1.0);
        return mix(fogColor * sunI, ambColor * 0.15, groundT);
    }
}

#endif // SKY_GLSL
