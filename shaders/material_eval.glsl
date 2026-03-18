// ============================================================
// ASSETTO CORSA PATH TRACER - MATERIAL EVALUATION
// ============================================================
// Uber shader for evaluating all AC material types
// This is where the magic happens - translates AC materials to path tracing

#ifndef MATERIAL_EVAL_GLSL
#define MATERIAL_EVAL_GLSL

#include "common.glsl"

// Forward declarations (these must be provided by the including shader)
// - MaterialData materials[]
// - sampler2D textures[]

// ============================================================
// MATERIAL SAMPLING
// ============================================================

// Sample material textures and build MaterialResult
MaterialResult evaluateMaterialTextures(MaterialData material, HitInfo hit) {
    MaterialResult result;

    // Unpack shared parameters
    float carDirtLevel = uintBitsToFloat(cam.parameters.z);

    // Initialize defaults
    result.albedo = material.ksDiffuse.rgb;
    result.emission = vec3(0.0);
    result.roughness = 0.5;
    result.metallic = 0.0;
    result.specular = 1.0;
    result.specularExp = material.ksSpecularEXP;
    result.alpha = 1.0;

    // Sample diffuse texture
    if (material.txDiffuse >= 0) {
        vec4 diffuseSample = texture(textures[nonuniformEXT(material.txDiffuse)], hit.uv);
        result.albedo = diffuseSample.rgb;
        result.alpha = diffuseSample.a;

        // Alpha test
        if (material.blendMode == BLEND_MODE_ALPHA_TEST) {
            if (result.alpha < material.alphaRef) {
                result.alpha = 0.0;
                return result;
            }
        }
    }

    // Sample normal map and update hit normal
    if ((material.flags & MAT_FLAG_HAS_NORMAL_MAP) != 0 && material.txNormal >= 0) {
        vec4 normalSample = texture(textures[nonuniformEXT(material.txNormal)], hit.uv);
        vec3 tangentNormal = decodeNormalMap(normalSample);
        // Note: hit.normal will be updated by caller after this returns
    }

    // Sample txMaps (R=Specular, G=Gloss, B=Detail blend)
    vec3 txMapsValue = vec3(1.0);
    if (material.txMaps >= 0) {
        txMapsValue = texture(textures[nonuniformEXT(material.txMaps)], hit.uv).rgb;
        applyTxMaps(result, txMapsValue);
    }

    // Sample detail texture
    if ((material.flags & MAT_FLAG_HAS_DETAIL_MAP) != 0 && material.txDetail >= 0) {
        vec2 detailUV = hit.uv * 8.0;  // Detail texture tiles at higher frequency
        vec3 detailColor = texture(textures[nonuniformEXT(material.txDetail)], detailUV).rgb;

        // Apply detail based on material type
        if (material.type == MAT_TYPE_PERPIXEL_MULTIMAP || material.type == MAT_TYPE_PERPIXEL_MULTIMAP_EMISSIVE) {
            // Use txMaps.b as blend factor
            result.albedo = applyDetailTexture(result.albedo, detailColor, txMapsValue.b);
        } else if (material.type == MAT_TYPE_MULTILAYER || material.type == MAT_TYPE_MULTILAYER_NM) {
            // Fixed blend for multilayer
            result.albedo = applyDetailTexture(result.albedo, detailColor, 0.3);
        }
    }

    // NEW: Apply dirt logic for multimap shaders
    if ((material.type == MAT_TYPE_PERPIXEL_MULTIMAP || material.type == MAT_TYPE_PERPIXEL_MULTIMAP_EMISSIVE) &&
        material.txDust >= 0 && carDirtLevel > 0.0) {
        
        vec4 txDustValue = texture(textures[nonuniformEXT(material.txDust)], hit.uv);
        
        // Logic from carDirt.hlsl
        float dirtAmount = carDirtLevel * txDustValue.a;
        
        // Blend albedo with dirt color
        result.albedo = mix(result.albedo, txDustValue.rgb, dirtAmount);
        
        // Reduce reflections/specular in dirty areas
        float mapsMultiplier = clamp(1.0 - dirtAmount * 10.0, 0.0, 1.0);
        result.specular *= mapsMultiplier; // Reduce specular highlight
        
        // Increase roughness to make it look matte
        result.roughness = mix(result.roughness, 1.0, dirtAmount);
    }

    // Convert specular exponent to roughness for PBR
    result.roughness = max(result.roughness, specularExpToRoughness(result.specularExp));

    // Calculate F0 (reflectance at normal incidence)
    // Non-metals: F0 ~ 0.04, metals: F0 = albedo
    result.F0 = mix(vec3(0.04), result.albedo, result.metallic);

    // Apply fresnel params from material
    // AC uses custom fresnel with exp, C, maxLevel parameters
    // We'll use this in the lighting calculation

    return result;
}

// ============================================================
// MATERIAL TYPE EVALUATION
// ============================================================

// Evaluate ksPerPixel (basic diffuse/specular)
vec3 evaluatePerPixel(MaterialData material, MaterialResult matResult, HitInfo hit, vec3 V, vec3 L, float NdotL) {
    vec3 diffuse = lambertianBRDF(matResult.albedo) * NdotL;

    // Blinn-Phong specular
    float spec = blinnPhongSpecular(hit.normal, V, L, matResult.specularExp);
    vec3 specular = material.ksSpecular.rgb * spec * matResult.specular;

    return diffuse + specular;
}

// Evaluate ksPerPixelMultiMap (car paint, detailed surfaces)
vec3 evaluatePerPixelMultiMap(MaterialData material, MaterialResult matResult, HitInfo hit, vec3 V, vec3 L, float NdotL) {
    // More advanced lighting with Cook-Torrance BRDF
    vec3 diffuse = lambertianBRDF(matResult.albedo) * NdotL;

    // Cook-Torrance specular (already includes Schlick fresnel internally)
    vec3 specular = cookTorranceBRDF(hit.normal, V, L, matResult.roughness, matResult.F0) * NdotL;

    return diffuse + specular;
}

// Evaluate ksMultilayer (car paint with fresnel layers)
vec3 evaluateMultilayer(MaterialData material, MaterialResult matResult, HitInfo hit, vec3 V, vec3 L, float NdotL) {
    // Multilayer car paint: diffuse base + specular coat
    vec3 diffuse = lambertianBRDF(matResult.albedo) * NdotL;

    // Cook-Torrance specular (already includes Schlick fresnel internally)
    vec3 specular = cookTorranceBRDF(hit.normal, V, L, matResult.roughness * 0.5, vec3(0.08)) * NdotL;

    return diffuse + specular;
}

// Evaluate ksTyres (rubber with dirt, blur, wetness)
vec3 evaluateTyres(MaterialData material, MaterialResult matResult, HitInfo hit, vec3 V, vec3 L, float NdotL) {
    // Tyres are mostly diffuse with subtle specular
    vec3 diffuse = lambertianBRDF(matResult.albedo) * NdotL;

    // Very tight specular for rubber
    float spec = blinnPhongSpecular(hit.normal, V, L, 200.0);
    vec3 specular = vec3(0.1) * spec;

    // TODO: Add dirt, blur, wetness effects (need additional textures)

    return diffuse + specular;
}

// Evaluate ksWindscreen (glass with refraction)
vec3 evaluateWindscreen(MaterialData material, MaterialResult matResult, HitInfo hit, vec3 V, vec3 L, float NdotL) {
    // Windscreen is mostly transparent with strong specular
    vec3 diffuse = lambertianBRDF(matResult.albedo) * NdotL * 0.1;  // Very little diffuse

    // Cook-Torrance specular (already includes Schlick fresnel internally)
    vec3 specular = cookTorranceBRDF(hit.normal, V, L, 0.02, vec3(0.04)) * NdotL;

    return diffuse + specular;
}

// Evaluate ksTree subsurface scattering (leaf translucency)
// Returns SSS irradiance contribution (without albedo, for NRD compatibility)
vec3 evaluateTreeSSS(vec3 N, vec3 V, vec3 L) {
    float backLight = max(dot(-N, L), 0.0);
    float viewBackScatter = pow(max(dot(V, -L), 0.0), 3.0) * 0.3;
    return vec3(backLight * 0.2 + viewBackScatter) * 0.15;
}

// ============================================================
// MAIN UBER SHADER EVALUATION
// ============================================================

// Evaluate material for direct lighting (sun/sky)
vec3 evaluateMaterialDirect(MaterialData material, MaterialResult matResult, HitInfo hit, vec3 V, vec3 L) {
    float NdotL = max(dot(hit.normal, L), 0.0);

    // Trees can receive SSS even when back-facing
    if (NdotL <= 0.0 && material.type != MAT_TYPE_TREE) return vec3(0.0);

    // Add emissive (always)
    if ((material.flags & MAT_FLAG_HAS_EMISSIVE) != 0 || material.type == MAT_TYPE_PERPIXEL_MULTIMAP_EMISSIVE) {
        matResult.emission = material.ksSpecular.rgb * 5.0;
    }

    // Switch based on material type
    vec3 lighting = vec3(0.0);

    switch (material.type) {
        case MAT_TYPE_PERPIXEL:
        case MAT_TYPE_PERPIXEL_NM:
            lighting = evaluatePerPixel(material, matResult, hit, V, L, NdotL);
            break;

        case MAT_TYPE_PERPIXEL_MULTIMAP:
        case MAT_TYPE_PERPIXEL_MULTIMAP_EMISSIVE:
            lighting = evaluatePerPixelMultiMap(material, matResult, hit, V, L, NdotL);
            break;

        case MAT_TYPE_MULTILAYER:
        case MAT_TYPE_MULTILAYER_NM:
            lighting = evaluateMultilayer(material, matResult, hit, V, L, NdotL);
            break;

        case MAT_TYPE_TYRES:
            lighting = evaluateTyres(material, matResult, hit, V, L, NdotL);
            break;

        case MAT_TYPE_WINDSCREEN:
            lighting = evaluateWindscreen(material, matResult, hit, V, L, NdotL);
            break;

        case MAT_TYPE_TREE: {
            // PBR foliage: Cook-Torrance BRDF + leaf subsurface scattering
            vec3 albedo = matResult.albedo * material.ksDiffuse.rgb;
            float roughness = max(specularExpToRoughness(matResult.specularExp), 0.3);
            vec3 F0 = vec3(0.04); // Dielectric foliage

            // Energy-conserving PBR
            vec3 H = normalize(L + V);
            float VdotH = max(dot(V, H), EPSILON);
            vec3 F = fresnelSchlickVec3(VdotH, F0);

            vec3 diffuse = lambertianBRDF(albedo) * (vec3(1.0) - F) * NdotL;
            vec3 specular = cookTorranceBRDF(hit.normal, V, L, roughness, F0) * NdotL
                          * material.ksSpecular.r * matResult.specular;

            // Subsurface scattering (thin leaf translucency)
            vec3 sss = lambertianBRDF(albedo) * evaluateTreeSSS(hit.normal, V, L);

            lighting = diffuse + specular + sss;
            break;
        }

        default:
            // Fallback: simple Lambertian
            lighting = lambertianBRDF(matResult.albedo) * NdotL;
            break;
    }

    return lighting + matResult.emission;
}

// Sample material BRDF for indirect lighting (path tracing)
// Returns: outgoing direction and attenuation (BRDF * cosTheta / PDF)
vec3 sampleMaterialBRDF(MaterialData material, MaterialResult matResult, HitInfo hit, vec3 V, inout uint seed, out vec3 direction) {
    // Determine if we sample diffuse or specular lobe
    float specularProbability = luminance(matResult.F0) / (luminance(matResult.F0) + luminance(matResult.albedo));
    specularProbability = clamp(specularProbability, 0.1, 0.9);

    bool sampleSpecular = random(seed) < specularProbability;

    if (sampleSpecular) {
        // Sample GGX specular lobe
        vec3 H = sampleGGX(hit.normal, matResult.roughness, seed);
        direction = reflect(-V, H);

        // If sampled direction is below surface, fall back to diffuse
        if (dot(direction, hit.normal) <= 0.0) {
            direction = sampleHemisphereCosine(hit.normal, seed);
            return matResult.albedo;  // Lambertian BRDF * cosTheta / PDF = albedo
        }

        // Cook-Torrance BRDF evaluation
        vec3 brdf = cookTorranceBRDF(hit.normal, V, direction, matResult.roughness, matResult.F0);
        float NdotL = max(dot(hit.normal, direction), 0.0);

        // Account for sampling probability
        return brdf * NdotL / specularProbability;

    } else {
        // Sample cosine-weighted hemisphere (diffuse)
        direction = sampleHemisphereCosine(hit.normal, seed);

        // Lambertian BRDF * cosTheta / PDF = albedo (they cancel out)
        return matResult.albedo / (1.0 - specularProbability);
    }
}

// Evaluate material for direct lighting with separated diffuse/specular (for NRD)
// outDiffuse: diffuse component (Lambert term)
// outSpecular: specular component (Blinn-Phong/Cook-Torrance)
void evaluateMaterialDirectSeparated(MaterialData material, MaterialResult matResult, HitInfo hit, vec3 V, vec3 L,
                                     out vec3 outDiffuse, out vec3 outSpecular) {
    float NdotL = max(dot(hit.normal, L), 0.0);

    // Initialize
    outDiffuse = vec3(0.0);
    outSpecular = vec3(0.0);

    // Trees can receive SSS even when back-facing
    if (NdotL <= 0.0 && material.type != MAT_TYPE_TREE) return;

    // Diffuse component - IRRADIANCE ONLY (no albedo, no BRDF)
    // NRD expects incoming light energy (irradiance), not final radiance
    // Albedo will be re-applied after denoising in composite shader
    outDiffuse = vec3(NdotL);

    // Specular component (depends on material type)
    switch (material.type) {
        case MAT_TYPE_PERPIXEL:
        case MAT_TYPE_PERPIXEL_NM:
        case MAT_TYPE_TYRES:
            // Blinn-Phong specular
            float spec = blinnPhongSpecular(hit.normal, V, L, matResult.specularExp);
            outSpecular = material.ksSpecular.rgb * spec * matResult.specular;
            break;

        case MAT_TYPE_PERPIXEL_MULTIMAP:
        case MAT_TYPE_PERPIXEL_MULTIMAP_EMISSIVE:
        case MAT_TYPE_MULTILAYER:
        case MAT_TYPE_MULTILAYER_NM:
            // Cook-Torrance specular (already includes Schlick fresnel internally)
            // Do NOT multiply by fresnelAC — that causes double-fresnel (wet look)
            vec3 specBRDF = cookTorranceBRDF(hit.normal, V, L, matResult.roughness, matResult.F0) * NdotL;
            outSpecular = specBRDF;
            break;

        case MAT_TYPE_WINDSCREEN:
        case MAT_TYPE_PERPIXEL_REFLECTION:
            // Glass-like: strong specular, weak diffuse
            // Cook-Torrance already includes Schlick fresnel internally
            outDiffuse *= 0.1;  // Very little diffuse for glass
            vec3 glassBRDF = cookTorranceBRDF(hit.normal, V, L, 0.02, vec3(0.04)) * NdotL;
            outSpecular = glassBRDF;
            break;

        case MAT_TYPE_TREE: {
            // Billboard foliage: no separated specular (billboard normals are flat,
            // Cook-Torrance on billboards produces harsh per-face highlights).
            // All lighting goes through diffuse channel for smooth NRD denoising.
            outSpecular = vec3(0.0);

            // Subsurface scattering (works even for back-facing normals)
            outDiffuse += evaluateTreeSSS(hit.normal, V, L);
            break;
        }

        default:
            // Fallback: pure diffuse
            outSpecular = vec3(0.0);
            break;
    }

    // Add emissive to diffuse channel
    if ((material.flags & MAT_FLAG_HAS_EMISSIVE) != 0 || material.type == MAT_TYPE_PERPIXEL_MULTIMAP_EMISSIVE) {
        outDiffuse += material.ksSpecular.rgb * 5.0;
    }
}

// ============================================================
// MATERIAL DEMODULATION FOR NRD
// ============================================================
// Returns irradiance (incoming light) without BRDF/albedo applied
// This is what NRD expects - demodulated lighting that will be
// re-modulated with albedo AFTER denoising

void evaluateMaterialIrradiance(MaterialData material, MaterialResult matResult, HitInfo hit, vec3 V, vec3 L,
                                 out vec3 outIrradiance, out vec3 outAlbedo) {
    float NdotL = max(dot(hit.normal, L), 0.0);

    // Initialize
    outIrradiance = vec3(0.0);
    outAlbedo = matResult.albedo;  // Store albedo for later application

    // Trees can receive SSS even when back-facing
    if (NdotL <= 0.0 && material.type != MAT_TYPE_TREE) return;

    // Trees: proper PBR irradiance with SSS
    if (material.type == MAT_TYPE_TREE) {
        outAlbedo = matResult.albedo * material.ksDiffuse.rgb;
        outIrradiance = vec3(NdotL);
        // Subsurface scattering (works even for back-facing normals)
        outIrradiance += evaluateTreeSSS(hit.normal, V, L);
        return;
    }

    // IRRADIANCE = incoming light intensity * geometric term
    // No BRDF, no albedo - just the raw incoming light
    outIrradiance = vec3(NdotL);  // Geometric term only

    // Albedo will be applied after denoising: finalColor = denoisedIrradiance * albedo / PI
    // (The /PI is part of the Lambertian BRDF)

    // Add emissive to irradiance (emissive is independent of lighting)
    if ((material.flags & MAT_FLAG_HAS_EMISSIVE) != 0 || material.type == MAT_TYPE_PERPIXEL_MULTIMAP_EMISSIVE) {
        outIrradiance += material.ksSpecular.rgb * 5.0;
    }
}

#endif // MATERIAL_EVAL_GLSL
