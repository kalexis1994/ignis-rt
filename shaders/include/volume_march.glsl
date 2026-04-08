// volume_march.glsl — Volume rendering utilities (Cycles Principled Volume)
// Shared between monolithic raygen and wavefront pipeline
// Requires: common.glsl, noise.glsl, topLevelAS, CameraProperties cam

#ifndef VOLUME_MARCH_GLSL
#define VOLUME_MARCH_GLSL

// ── Henyey-Greenstein phase function ──

float henyeyGreensteinPhase(float cosTheta, float g) {
    if (abs(g) < 1e-4) return 0.25 * INV_PI;
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

vec3 sampleHenyeyGreenstein(vec3 inDir, float g, vec2 xi) {
    float cosTheta;
    if (abs(g) < 1e-4) {
        cosTheta = 1.0 - 2.0 * xi.x;
    } else {
        float k = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi.x);
        cosTheta = (1.0 + g * g - k * k) / (2.0 * g);
    }
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 2.0 * PI * xi.y;
    vec3 w = inDir;
    vec3 u = abs(w.y) < 0.999 ? normalize(cross(w, vec3(0, 1, 0))) : normalize(cross(w, vec3(1, 0, 0)));
    vec3 v = cross(w, u);
    return normalize(u * (cos(phi) * sinTheta) + v * (sin(phi) * sinTheta) + w * cosTheta);
}

// ── Cycles Principled Volume coefficients ──

void computeVolumeCoefficients(vec3 scatterColor, vec3 absorptionColor, float density,
                               out vec3 sigma_s, out vec3 sigma_a, out vec3 sigma_t) {
    sigma_s = scatterColor * density;
    sigma_a = max(vec3(1.0) - scatterColor, vec3(0.0))
            * max(vec3(1.0) - sqrt(max(absorptionColor, vec3(0.0))), vec3(0.0))
            * density;
    sigma_t = sigma_s + sigma_a;
}

// ── Find volume exit distance (back face of same instance) ──

float traceVolumeExit(accelerationStructureEXT tlas, vec3 entryPos, vec3 dir, int volumeInstanceId) {
    float exitDist = 10.0;
    rayQueryEXT exitRQ;
    rayQueryInitializeEXT(exitRQ, tlas, gl_RayFlagsNoneEXT,
                          0xFF, entryPos, 0.0, dir, 100000.0);
    while (rayQueryProceedEXT(exitRQ)) {
        if (rayQueryGetIntersectionTypeEXT(exitRQ, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
            int hitInstId = rayQueryGetIntersectionInstanceIdEXT(exitRQ, false);
            bool isFront = rayQueryGetIntersectionFrontFaceEXT(exitRQ, false);
            if (!isFront && hitInstId == volumeInstanceId) {
                rayQueryConfirmIntersectionEXT(exitRQ);
            }
        }
    }
    if (rayQueryGetIntersectionTypeEXT(exitRQ, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
        exitDist = rayQueryGetIntersectionTEXT(exitRQ, true);
    }
    return max(exitDist, 0.005);
}

// ── Volume shadow transmittance (for shadow rays through volumes) ──

vec3 evaluateVolumeShadowTransmittance(accelerationStructureEXT tlas,
    vec3 scatterColor, vec3 absorptionColor, float density,
    vec3 origin, vec3 dir, float hitT, int hitInstanceId) {
    if (density <= 0.0) return vec3(1.0);

    vec3 vEntry = origin + dir * (hitT + 0.001);
    float vDist = traceVolumeExit(tlas, vEntry, dir, hitInstanceId);

    vec3 sigma_s, sigma_a, sigma_t;
    computeVolumeCoefficients(scatterColor, absorptionColor, density, sigma_s, sigma_a, sigma_t);
    return exp(-sigma_t * vDist);
}

// ── Homogeneous volume (analytic, no stepping) ──

void marchVolumeHomogeneous(
    vec3 entryPos, vec3 rayDir, float marchDist,
    vec3 scatterColor, vec3 absorptionColor, float density, float anisotropy,
    vec3 emissionColor, float emissionStrength, float temperature, float blackbodyIntensity,
    vec3 sunDir, float sunIntensity, vec3 sunColor, vec3 skyColor,
    inout vec3 radiance, inout vec3 throughput)
{
    vec3 sigma_s, sigma_a, sigma_t;
    computeVolumeCoefficients(scatterColor, absorptionColor, density, sigma_s, sigma_a, sigma_t);

    vec3 T = exp(-sigma_t * marchDist);

    // In-scattering at midpoint
    vec3 L_in = vec3(0.0);
    if (sunIntensity > 0.0) {
        float phase = henyeyGreensteinPhase(dot(rayDir, sunDir), anisotropy);
        L_in += sunColor * sunIntensity * phase;
    }
    L_in += skyColor * 0.25 * INV_PI;

    vec3 inscatter = sigma_s * L_in * (vec3(1.0) - T) / max(sigma_t, vec3(0.0001));

    // Volume emission
    vec3 volEmission = vec3(0.0);
    if (emissionStrength > 0.0) {
        volEmission = emissionColor * emissionStrength;
        volEmission *= (vec3(1.0) - T) / max(sigma_t, vec3(0.0001));
    }

    // Blackbody emission
    if (blackbodyIntensity > 0.0 && temperature > 0.0) {
        float bbPower = blackbodyIntensity * 5.67e-8 * temperature * temperature * temperature * temperature * 1e-12;
        vec3 bbColor = vec3(1.0, clamp((temperature - 1000.0) / 5000.0, 0.0, 1.0),
                             clamp((temperature - 3000.0) / 7000.0, 0.0, 1.0));
        volEmission += bbColor * bbPower * (vec3(1.0) - T) / max(sigma_t, vec3(0.0001));
    }

    radiance += (inscatter + volEmission) * throughput;
    throughput *= T;
}

// ── Heterogeneous volume (ray march with noise) ──

void marchVolumeHeterogeneous(
    vec3 entryPos, vec3 rayDir, float marchDist, mat4x3 worldToObj,
    vec3 scatterColor, vec3 absorptionColor, float density, float anisotropy,
    vec3 emissionColor, float emissionStrength, float temperature, float blackbodyIntensity,
    // Noise params
    float noiseScale, float noiseDetail, float noiseBrightness, float noiseRoughness,
    float noiseLacunarity, float noiseContrast, vec3 mappingOffset,
    int noiseType, bool noiseNormalize, float noiseOffset, float noiseGain,
    uint noiseDimensions, float noiseW, float noiseDistortion,
    bool hasDistortion, bool hasFullMatrix, vec4 mapRow0, vec4 mapRow1, vec4 mapRow2,
    int gradientType, bool hasGradient, float mixFactor, bool hasMix,
    bool hasRamp, vec4 rampLut0, vec4 rampLut1, vec4 rampLut2, vec4 rampLut3,
    uint chainTarget, bool useObjectCoords, vec3 volBboxMin, vec3 volBboxRange,
    // Lighting
    vec3 sunDir, float sunIntensity, vec3 sunColor, vec3 skyColor,
    uint lightCount, vec4 lights[128],
    // Output
    inout vec3 radiance, inout vec3 throughput)
{
    // Adaptive step count: shorter volumes need fewer steps
    // Max 32 for wavefront (vs 48 monolithic) to balance quality vs perf
    const int MAX_STEPS = 32;
    float stepSize = max(marchDist / float(MAX_STEPS), 0.02);
    int numSteps = min(int(marchDist / stepSize), MAX_STEPS);
    vec3 volTransmittance = vec3(1.0);
    vec3 inscatter = vec3(0.0);
    vec3 volEmission = vec3(0.0);

    for (int step = 0; step < numSteps; step++) {
        float t = (float(step) + 0.5) * stepSize;
        vec3 samplePos = entryPos + rayDir * t;

        float localEmissionStr = emissionStrength;
        vec3 localScatterColor = scatterColor;

        // Evaluate noise chain if active
        if (noiseScale > 0.0) {
            vec3 objPos = (worldToObj * vec4(samplePos, 1.0));
            vec3 baseCoord = useObjectCoords ? objPos : (objPos - volBboxMin) / volBboxRange;

            vec3 mappedCoord;
            if (hasFullMatrix) {
                mappedCoord = vec3(
                    dot(mapRow0.xyz, baseCoord) + mapRow0.w,
                    dot(mapRow1.xyz, baseCoord) + mapRow1.w,
                    dot(mapRow2.xyz, baseCoord) + mapRow2.w);
            } else {
                mappedCoord = baseCoord + mappingOffset;
            }

            float n;
            if (noiseDimensions == 4u) {
                vec4 noisePos4 = vec4(mappedCoord, noiseW) * noiseScale;
                if (hasDistortion && noiseDistortion != 0.0) {
                    vec4 distR;
                    distR.x = perlinNoise4D(noisePos4 + vec4(0,0,0,200)) * noiseDistortion;
                    distR.y = perlinNoise4D(noisePos4 + vec4(0,0,0,100)) * noiseDistortion;
                    distR.z = perlinNoise4D(noisePos4) * noiseDistortion;
                    distR.w = perlinNoise4D(noisePos4 + vec4(0,0,0,150)) * noiseDistortion;
                    noisePos4 += distR;
                }
                n = noise_fbm_4d(noisePos4, noiseDetail, noiseRoughness, noiseLacunarity, noiseNormalize);
            } else {
                vec3 noisePos = mappedCoord * noiseScale;
                if (hasDistortion && noiseDistortion != 0.0) {
                    vec3 distR;
                    distR.x = perlinNoise3D(noisePos + vec3(13.5)) * noiseDistortion;
                    distR.y = perlinNoise3D(noisePos) * noiseDistortion;
                    distR.z = perlinNoise3D(noisePos - vec3(13.5)) * noiseDistortion;
                    noisePos += distR;
                }
                n = noiseSelect3D(noisePos, noiseDetail, noiseRoughness, noiseLacunarity,
                                  noiseOffset, noiseGain, noiseType, noiseNormalize);
            }

            // BrightContrast
            float bcA = 1.0 + noiseContrast;
            float bcB = noiseBrightness - 0.5 * noiseContrast;
            n = max(bcA * n + bcB, 0.0);

            // Gradient + Mix
            float chainValue = n;
            if (hasMix && hasGradient) {
                float grad = gradientSelect(mappedCoord, gradientType);
                chainValue = mix(n, grad, mixFactor);
            } else if (hasGradient) {
                chainValue = gradientSelect(mappedCoord, gradientType);
            }

            // Color ramp
            if (hasRamp) {
                chainValue = sampleColorRamp16(chainValue, rampLut0, rampLut1, rampLut2, rampLut3);
            }

            if (chainTarget == 0u) localScatterColor *= chainValue;
            else if (chainTarget == 1u) localEmissionStr = chainValue;
        }

        // Volume coefficients
        vec3 sigma_s, sigma_a, sigma_t;
        computeVolumeCoefficients(localScatterColor, absorptionColor, density, sigma_s, sigma_a, sigma_t);
        if (max(sigma_t.r, max(sigma_t.g, sigma_t.b)) < 0.0001) continue;

        // In-scattering: sun
        vec3 L_in = vec3(0.0);
        if (sunIntensity > 0.0) {
            float phase = henyeyGreensteinPhase(dot(rayDir, sunDir), anisotropy);
            L_in += sunColor * sunIntensity * phase;
        }

        // 1 random local light
        uint vLightCount = min(lightCount, 32u);
        if (vLightCount > 0u) {
            uint li = min(uint(rand01() * float(vLightCount)), vLightCount - 1u);
            vec4 lPosRange = lights[li * 4u];
            vec4 lColInt = lights[li * 4u + 1u];
            vec3 toLight = lPosRange.xyz - samplePos;
            float dist = length(toLight);
            float lRange = abs(lPosRange.w);
            if (dist < lRange && dist > 0.01) {
                float atten = 1.0 / (dist * dist);
                float falloff = clamp(1.0 - pow(dist / lRange, 4.0), 0.0, 1.0);
                falloff *= falloff;
                float phase = henyeyGreensteinPhase(dot(rayDir, toLight / dist), anisotropy);
                L_in += lColInt.rgb * lColInt.w * atten * falloff * phase * float(vLightCount);
            }
        }

        // Sky/HDRI
        L_in += skyColor * 0.25 * INV_PI;

        inscatter += volTransmittance * sigma_s * L_in * stepSize;

        // Volume emission
        if (localEmissionStr > 0.0) {
            volEmission += volTransmittance * emissionColor * localEmissionStr * stepSize;
        }
        if (blackbodyIntensity > 0.0 && temperature > 0.0) {
            float bbPower = blackbodyIntensity * 5.67e-8 * temperature * temperature * temperature * temperature * 1e-12;
            vec3 bbColor = vec3(1.0, clamp((temperature - 1000.0) / 5000.0, 0.0, 1.0),
                                 clamp((temperature - 3000.0) / 7000.0, 0.0, 1.0));
            volEmission += volTransmittance * bbColor * bbPower * stepSize;
        }

        // Beer-Lambert
        volTransmittance *= exp(-sigma_t * stepSize);
        if (max(volTransmittance.r, max(volTransmittance.g, volTransmittance.b)) < 0.001) break;
    }

    radiance += (inscatter + volEmission) * throughput;
    throughput *= volTransmittance;
}

#endif // VOLUME_MARCH_GLSL
