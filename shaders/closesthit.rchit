#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) rayPayloadInEXT vec4 hitValue;  // Incoming payload from primary ray
hitAttributeEXT vec2 attribs;

// For tracing secondary rays (bounces)
layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

// Random number generation (same as raygen)
uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

float random(inout uint seed) {
    seed = hash(seed);
    return float(seed) / 4294967295.0;
}

// Generate random direction in hemisphere around normal (cosine-weighted)
vec3 randomHemisphereDirection(vec3 normal, inout uint seed) {
    // Random angles
    float r1 = random(seed);
    float r2 = random(seed);

    float phi = 2.0 * 3.14159265359 * r1;
    float cosTheta = sqrt(r2);
    float sinTheta = sqrt(1.0 - r2);

    // Local tangent space
    vec3 tangent = abs(normal.y) < 0.999 ? normalize(cross(normal, vec3(0, 1, 0))) : normalize(cross(normal, vec3(1, 0, 0)));
    vec3 bitangent = cross(normal, tangent);

    // Transform to world space
    return normalize(tangent * (cos(phi) * sinTheta) +
                     bitangent * (sin(phi) * sinTheta) +
                     normal * cosTheta);
}

// Material types
#define MAT_TYPE_PERPIXEL          0
#define MAT_TYPE_PERPIXEL_NM       1
#define MAT_TYPE_MULTIMAP          2
#define MAT_TYPE_MULTIMAP_EMISSIVE 3
#define MAT_TYPE_MULTILAYER        4

// Material flags
#define MAT_HAS_NORMAL_MAP    (1 << 0)
#define MAT_HAS_DETAIL_MAP    (1 << 1)
#define MAT_HAS_EMISSIVE      (1 << 2)
#define MAT_HAS_REFLECTION    (1 << 3)
#define MAT_MULTILAYER        (1 << 4)
#define MAT_USE_UV2           (1 << 5)

// Geometry metadata structure (matches C++ struct)
struct GeometryMetadata {
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
    uint64_t normalBufferAddress;
    uint64_t uvBufferAddress;
    uint vertexCount;
    uint indexCount;
    uint padding[2];
};

// Material data structure (matches C++ struct)
struct MaterialData {
    uint type;
    uint flags;
    uint blendMode;      // 0=Opaque, 1=AlphaBlend, 2=AlphaTest
    float alphaRef;      // Alpha test threshold
    vec4 ksAmbient;
    vec4 ksDiffuse;
    vec4 ksSpecular;
    float ksSpecularEXP;
    float padding2[3];
    vec4 fresnelParams;
    // Texture indices into textures[] array
    int txDiffuse;   // -1 = no texture
    int txNormal;
    int txDetail;
    int txMaps;      // For ksPerPixelMultiMap: packed spec/gloss/detail
};

// Blend mode constants
#define BLEND_MODE_OPAQUE      0
#define BLEND_MODE_ALPHA_BLEND 1
#define BLEND_MODE_ALPHA_TEST  2

// Storage buffer with geometry metadata for all meshes
layout(binding = 3, set = 0, scalar) buffer GeometryMetadataBuffer {
    GeometryMetadata geometries[];
} geometryMetadata;

// Storage buffer with material data for all meshes
layout(binding = 4, set = 0, scalar) buffer MaterialBuffer {
    MaterialData materials[];
} materialBuffer;

// Texture array (bindless textures)
layout(binding = 5, set = 0) uniform sampler2D textures[];

// Buffer references for accessing vertex data
layout(buffer_reference, scalar) buffer Vertices { float v[]; };
layout(buffer_reference, scalar) buffer Indices { uint i[]; };
layout(buffer_reference, scalar) buffer Normals { float n[]; };
layout(buffer_reference, scalar) buffer UVs { float uv[]; };

void main() {
    // Get mesh ID from instance custom index
    uint meshId = gl_InstanceCustomIndexEXT;

    // Get metadata and material for this mesh
    GeometryMetadata meta = geometryMetadata.geometries[meshId];
    MaterialData material = materialBuffer.materials[meshId];

    // Get indices of the hit triangle
    Indices indices = Indices(meta.indexBufferAddress);
    uint idx0 = indices.i[gl_PrimitiveID * 3 + 0];
    uint idx1 = indices.i[gl_PrimitiveID * 3 + 1];
    uint idx2 = indices.i[gl_PrimitiveID * 3 + 2];

    // Barycentric coordinates for interpolation
    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    // Calculate shading normal (smooth or flat)
    vec3 shadingNormal;
    if (meta.normalBufferAddress != 0) {
        // Smooth shading: interpolate vertex normals
        Normals normals = Normals(meta.normalBufferAddress);
        vec3 n0 = vec3(normals.n[idx0 * 3 + 0], normals.n[idx0 * 3 + 1], normals.n[idx0 * 3 + 2]);
        vec3 n1 = vec3(normals.n[idx1 * 3 + 0], normals.n[idx1 * 3 + 1], normals.n[idx1 * 3 + 2]);
        vec3 n2 = vec3(normals.n[idx2 * 3 + 0], normals.n[idx2 * 3 + 1], normals.n[idx2 * 3 + 2]);
        shadingNormal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);
    } else {
        // Flat shading: geometric normal
        Vertices vertices = Vertices(meta.vertexBufferAddress);
        vec3 v0 = vec3(vertices.v[idx0 * 3 + 0], vertices.v[idx0 * 3 + 1], vertices.v[idx0 * 3 + 2]);
        vec3 v1 = vec3(vertices.v[idx1 * 3 + 0], vertices.v[idx1 * 3 + 1], vertices.v[idx1 * 3 + 2]);
        vec3 v2 = vec3(vertices.v[idx2 * 3 + 0], vertices.v[idx2 * 3 + 1], vertices.v[idx2 * 3 + 2]);
        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;
        shadingNormal = normalize(cross(edge1, edge2));
    }

    // Get interpolated UV coordinates
    vec2 uv = vec2(0.5); // Default UV if no UV buffer
    if (meta.uvBufferAddress != 0) {
        UVs uvs = UVs(meta.uvBufferAddress);
        vec2 uv0 = vec2(uvs.uv[idx0 * 2 + 0], uvs.uv[idx0 * 2 + 1]);
        vec2 uv1 = vec2(uvs.uv[idx1 * 2 + 0], uvs.uv[idx1 * 2 + 1]);
        vec2 uv2 = vec2(uvs.uv[idx2 * 2 + 0], uvs.uv[idx2 * 2 + 1]);
        uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    }

    // ========== TEXTURE SAMPLING ==========

    // Sample diffuse texture
    vec3 baseColor = material.ksDiffuse.rgb;
    float alpha = 1.0;

    if (material.txDiffuse >= 0) {
        vec4 texColor = texture(textures[nonuniformEXT(material.txDiffuse)], uv);
        baseColor = texColor.rgb;
        alpha = texColor.a;

        // Alpha test: discard transparent pixels
        if (material.blendMode == BLEND_MODE_ALPHA_TEST) {
            if (alpha < material.alphaRef) {
                hitValue.rgb = vec3(0.0);
                return;
            }
        }
    }

    // Sample txMaps for ksPerPixelMultiMap (spec, gloss, detail blend)
    vec3 txMapsValue = vec3(1.0);  // Default: full spec, full gloss, no detail
    if (material.type == MAT_TYPE_MULTIMAP && material.txMaps >= 0) {
        // R = Specular multiplier, G = Gloss multiplier, B = Detail blend
        txMapsValue = texture(textures[nonuniformEXT(material.txMaps)], uv).rgb;
    }

    // Sample detail texture for ksPerPixelMultiMap and ksMultilayer_objsp
    vec3 detailColor = vec3(1.0);
    if (material.txDetail >= 0) {
        // Detail textures usually tile at higher frequency
        vec2 detailUV = uv * 8.0;  // 8x tiling for detail
        detailColor = texture(textures[nonuniformEXT(material.txDetail)], detailUV).rgb;

        // Apply detail texture based on txMaps.b or use full detail for multilayer
        if (material.type == MAT_TYPE_MULTIMAP) {
            // Blend detail based on txMaps.b
            float detailBlend = txMapsValue.b;
            baseColor = mix(baseColor, baseColor * detailColor, detailBlend * 0.5);
        } else if (material.type == MAT_TYPE_MULTILAYER) {
            // For multilayer, detail is blended more aggressively
            baseColor = baseColor * mix(vec3(1.0), detailColor, 0.3);
        }
    }

    // ========== HYBRID PATH TRACING (Direct + 1 Indirect Bounce) ==========

    // Get random seed from payload
    uint seed = floatBitsToUint(hitValue.w);

    // Calculate world hit position
    vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    vec3 finalColor = vec3(0.0);

    // 1. EMISSIVE (self-emission)
    if (material.type == MAT_TYPE_MULTIMAP_EMISSIVE || (material.flags & MAT_HAS_EMISSIVE) != 0) {
        finalColor += material.ksSpecular.rgb * 5.0;
    }

    // 2. DIRECT LIGHTING (sun at afternoon angle ~4pm) WITH SHADOWS
    vec3 sunDir = normalize(vec3(0.6, 0.5, 0.6));  // Side angle, not zenith
    vec3 sunColor = vec3(3.5, 3.2, 3.0);  // Reduced intensity
    float NdotL = max(dot(shadingNormal, sunDir), 0.0);

    // Only add diffuse if facing sun
    if (NdotL > 0.0) {
        // NO SHADOWS - any shadow test (traceRayEXT or rayQuery) causes VK_ERROR_DEVICE_LOST
        // This appears to be a driver/hardware limitation with this specific configuration
        float visibility = 1.0;
        finalColor += baseColor * sunColor * NdotL * visibility;
    }

    // 3. SKY AMBIENT (subtle fill light)
    float skyFactor = max(shadingNormal.y, 0.0);  // Only upward facing surfaces
    vec3 skyAmbient = vec3(0.1, 0.12, 0.15) * skyFactor;  // Slightly brighter for visibility
    finalColor += baseColor * skyAmbient;

    // 4. INDIRECT LIGHTING (DISABLED - testing without bounces first)
    // vec3 bounceDir = randomHemisphereDirection(shadingNormal, seed);
    // vec4 indirectPayload = vec4(0.0, 0.0, 0.0, uintBitsToFloat(seed));
    // uint rayFlags = gl_RayFlagsOpaqueEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
    // traceRayEXT(topLevelAS, rayFlags, 0xff, 0, 0, 0,
    //             hitPos + shadingNormal * 0.001, 0.001, bounceDir, 10000.0, 0);
    //
    // vec3 indirectLight = indirectPayload.rgb;
    // finalColor += baseColor * indirectLight * 0.5;  // Scale down indirect

    // Pack hit position and normal into payload.w for shadow test in raygen
    // We'll use the .w component to store a packed value
    // For now, just return the color
    hitValue = vec4(finalColor, hitValue.w);
}
