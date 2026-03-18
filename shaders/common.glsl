// ============================================================
// ASSETTO CORSA PATH TRACER - COMMON SHADER UTILITIES
// ============================================================
// Uber shader system for all AC material types
// Translated from HLSL/DirectX to GLSL/Vulkan

#ifndef COMMON_GLSL
#define COMMON_GLSL

// ============================================================
// CONSTANTS
// ============================================================

#define PI 3.14159265359
#define INV_PI 0.31830988618
#define EPSILON 0.0001

// Material types (matches C++ enum)
#define MAT_TYPE_PERPIXEL                0
#define MAT_TYPE_PERPIXEL_NM             1
#define MAT_TYPE_PERPIXEL_MULTIMAP       2
#define MAT_TYPE_PERPIXEL_MULTIMAP_EMISSIVE 3
#define MAT_TYPE_MULTILAYER              4
#define MAT_TYPE_MULTILAYER_NM           5
#define MAT_TYPE_TYRES                   6
#define MAT_TYPE_WINDSCREEN              7
#define MAT_TYPE_BRAKE_DISC              8
#define MAT_TYPE_SKINNED_MESH            9
#define MAT_TYPE_TREE                    10
#define MAT_TYPE_PERPIXEL_REFLECTION     11  // Glass-like materials with reflection + alpha transparency

// Material flags (bitfield)
#define MAT_FLAG_HAS_NORMAL_MAP    (1 << 0)
#define MAT_FLAG_HAS_DETAIL_MAP    (1 << 1)
#define MAT_FLAG_HAS_EMISSIVE      (1 << 2)
#define MAT_FLAG_HAS_REFLECTION    (1 << 3)
#define MAT_FLAG_MULTILAYER        (1 << 4)
#define MAT_FLAG_USE_UV2           (1 << 5)
#define MAT_FLAG_ALPHA_TEST        (1 << 6)
#define MAT_FLAG_TWO_SIDED         (1 << 7)

// Blend modes
#define BLEND_MODE_OPAQUE          0
#define BLEND_MODE_ALPHA_BLEND     1
#define BLEND_MODE_ALPHA_TEST      2

// ============================================================
// STRUCTURES
// ============================================================

// Note: MaterialData is defined in the main shader file (raygen.rgen)
// to avoid duplication and compilation issues

// Hit information (built from ray query)
struct HitInfo {
    vec3 position;      // World space hit position
    vec3 normal;        // Shading normal (from vertex normals or normal map)
    vec3 geometricNormal; // Geometric normal (from triangle)
    vec3 tangent;       // Tangent vector for normal mapping
    vec3 bitangent;     // Bitangent vector for normal mapping
    vec2 uv;            // Texture coordinates
    float hitDistance;  // Ray distance to hit
    bool frontFacing;   // True if ray hit front face
};

// Material evaluation result
struct MaterialResult {
    vec3 albedo;        // Base color
    vec3 emission;      // Emissive color
    float roughness;    // Surface roughness [0,1]
    float metallic;     // Metallic factor [0,1]
    float specular;     // Specular intensity
    float specularExp;  // Specular exponent (for Blinn-Phong)
    vec3 F0;            // Fresnel reflectance at normal incidence
    float alpha;        // Transparency [0,1]
};

// ============================================================
// RANDOM NUMBER GENERATION
// ============================================================

// PCG hash - better quality than simple hash
uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// 2D hash for better spatial decorrelation
uint hash2D(uvec2 v, uint frame) {
    return pcg_hash(v.x + pcg_hash(v.y + pcg_hash(frame)));
}

// PCG 3D hash - for multidimensional sampling (better decorrelation)
// Source: "Hash Functions for GPU Rendering" (JCGT 2020, Jarzynski & Olano)
uvec3 pcg3d(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// PCG 4D hash - for even more dimensions
uvec4 pcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

// Random float [0,1] from seed (legacy, use pcg3d for better quality)
float random(inout uint seed) {
    seed = pcg_hash(seed);
    return float(seed) / 4294967295.0;
}

// Random vec2 from seed using pcg3d (better decorrelation than sequential random())
vec2 random2D(inout uint seed) {
    uvec3 v = pcg3d(uvec3(seed, seed + 1u, seed + 2u));
    seed = v.z;  // Update seed for next call
    return vec2(v.xy) / 4294967295.0;
}

// Random vec3 from seed using pcg3d (best for 3D sampling)
vec3 random3D(inout uint seed) {
    uvec3 v = pcg3d(uvec3(seed, seed + 1u, seed + 2u));
    seed = v.z;  // Update seed for next call
    return vec3(v) / 4294967295.0;
}

// Random vec4 from seed using pcg4d (for 4D sampling)
vec4 random4D(inout uint seed) {
    uvec4 v = pcg4d(uvec4(seed, seed + 1u, seed + 2u, seed + 3u));
    seed = v.w;  // Update seed for next call
    return vec4(v) / 4294967295.0;
}

// ============================================================
// SAMPLING FUNCTIONS
// ============================================================

// Generate random direction in hemisphere around normal (cosine-weighted)
vec3 sampleHemisphereCosine(vec3 normal, inout uint seed) {
    // Use random2D for better decorrelation between r1 and r2
    vec2 r = random2D(seed);

    float phi = 2.0 * PI * r.x;
    float cosTheta = sqrt(r.y);
    float sinTheta = sqrt(1.0 - r.y);

    // Local tangent space
    vec3 tangent = abs(normal.y) < 0.999
        ? normalize(cross(normal, vec3(0, 1, 0)))
        : normalize(cross(normal, vec3(1, 0, 0)));
    vec3 bitangent = cross(normal, tangent);

    // Transform to world space
    return normalize(tangent * (cos(phi) * sinTheta) +
                     bitangent * (sin(phi) * sinTheta) +
                     normal * cosTheta);
}

// Sample GGX distribution for specular lobes
vec3 sampleGGX(vec3 normal, float roughness, inout uint seed) {
    // Use random2D for better decorrelation
    vec2 r = random2D(seed);

    float a = roughness * roughness;
    float phi = 2.0 * PI * r.x;
    float cosTheta = sqrt((1.0 - r.y) / (1.0 + (a * a - 1.0) * r.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // Spherical to cartesian
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    // Tangent space to world space
    vec3 up = abs(normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

    return normalize(tangent * H.x + bitangent * H.y + normal * H.z);
}

// ============================================================
// NORMAL MAPPING
// ============================================================

// Transform normal from tangent space to world space
vec3 getNormalFromMap(vec3 tangentNormal, vec3 normal, vec3 tangent, vec3 bitangent) {
    // Construct TBN matrix (tangent space to world space)
    mat3 TBN = mat3(tangent, bitangent, normal);
    return normalize(TBN * tangentNormal);
}

// Decode normal map (assumes DXT5nm format or standard RGB)
vec3 decodeNormalMap(vec4 normalSample) {
    // AC uses standard RGB normal maps (not DXT5nm)
    vec3 normal = normalSample.xyz * 2.0 - 1.0;
    return normalize(normal);
}

// ============================================================
// FRESNEL FUNCTIONS
// ============================================================

// Fresnel-Schlick approximation
float fresnelSchlick(float cosTheta, float F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 fresnelSchlickVec3(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// AC's custom fresnel (from fresnelParams)
// NOTE: Not used to multiply Cook-Torrance BRDF (which has its own Schlick fresnel).
// Only used where Blinn-Phong needs an external fresnel term.
float fresnelAC(float cosTheta, float fresnelExp, float fresnelC, float fresnelMaxLevel) {
    float fresnel = fresnelC + (1.0 - fresnelC) * pow(1.0 - cosTheta, fresnelExp);
    return fresnel * fresnelMaxLevel;
}

// ============================================================
// BRDF FUNCTIONS
// ============================================================

// GGX/Trowbridge-Reitz normal distribution function
float distributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / max(denom, EPSILON);
}

// Schlick-GGX geometry function
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / max(denom, EPSILON);
}

// Smith's method for geometry obstruction
float geometrySmith(float NdotV, float NdotL, float roughness) {
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

// Cook-Torrance BRDF (Specular)
vec3 cookTorranceBRDF(vec3 N, vec3 V, vec3 L, float roughness, vec3 F0) {
    vec3 H = normalize(V + L);

    float NdotV = max(dot(N, V), EPSILON);
    float NdotL = max(dot(N, L), EPSILON);
    float NdotH = max(dot(N, H), EPSILON);
    float VdotH = max(dot(V, H), EPSILON);

    // Calculate D, G, F
    float D = distributionGGX(NdotH, roughness);
    float G = geometrySmith(NdotV, NdotL, roughness);
    vec3 F = fresnelSchlickVec3(VdotH, F0);

    // Cook-Torrance specular
    vec3 numerator = D * G * F;
    float denominator = 4.0 * NdotV * NdotL;
    vec3 specular = numerator / max(denominator, EPSILON);

    return specular;
}

// Lambertian diffuse BRDF
vec3 lambertianBRDF(vec3 albedo) {
    return albedo * INV_PI;
}

// Blinn-Phong specular (AC's original model for backward compatibility)
float blinnPhongSpecular(vec3 N, vec3 V, vec3 L, float specularExp) {
    vec3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0);
    return pow(NdotH, specularExp);
}

// ============================================================
// MATERIAL EVALUATION
// ============================================================

// Evaluate detail texture (high frequency overlay)
vec3 applyDetailTexture(vec3 baseColor, vec3 detailColor, float detailBlend) {
    // Detail texture is usually a grayscale overlay
    // Mix between baseColor and baseColor*detail based on blend factor
    return mix(baseColor, baseColor * detailColor, detailBlend * 0.5);
}

// Evaluate txMaps texture (R=Specular, G=Gloss, B=Detail blend)
void applyTxMaps(inout MaterialResult material, vec3 txMapsValue) {
    material.specular *= txMapsValue.r;        // Specular multiplier
    material.specularExp *= txMapsValue.g;     // Gloss multiplier
    // txMapsValue.b is used for detail blend in applyDetailTexture
}

// Convert AC's ksSpecularEXP to roughness
float specularExpToRoughness(float specularExp) {
    // AC uses specular exponent (high = shiny, low = rough)
    // We need roughness (high = rough, low = shiny)
    // Empirical mapping: roughness = sqrt(2.0 / (specularExp + 2.0))
    return sqrt(2.0 / (specularExp + 2.0));
}

// Convert roughness to specular exponent
float roughnessToSpecularExp(float roughness) {
    // Inverse of above
    float r2 = roughness * roughness;
    return (2.0 / r2) - 2.0;
}

// ============================================================
// LIGHTING HELPERS
// ============================================================

// Note: reflect() and refract() are built-in GLSL functions

// Luminance
float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

#endif // COMMON_GLSL
