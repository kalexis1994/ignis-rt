// ============================================================
// pt_replay.glsl — ReSTIR PT deep-prefix replay (RTXPT-style)
//
// Re-traces the camera prefix of a reused reservoir's path from the
// CURRENT pixel's primary surface, using the stored RNG seed to
// regenerate the same BRDF sampling decisions as the original path.
// At the end we get the candidate "shifted receiver" — the surface
// the rcVertex connects from in the new pixel. If that surface's
// material/normal/position doesn't match the reservoir's saved
// source vertex, the shift is rejected → no leak.
//
// Reference: NVIDIA RTXPT PathTracerNEE.hlsli + ignis-ac's port.
//
// Requires: PTReservoir struct + ptSurfaceSimilar / ptMaterialSimilarLuma
// from pt_reservoir.glsl, and `topLevelAS` (set 0 binding 0) declared
// by the including kernel.
// ============================================================

#ifndef PT_REPLAY_GLSL
#define PT_REPLAY_GLSL

struct GeometryMetadata {
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
    uint64_t normalBufferAddress;
    uint64_t uvBufferAddress;
    uint64_t primMaterialBufferAddress;
    uint64_t bitangentBufferAddress;
    uint64_t primYBoundsAddress;
    uint64_t colorBufferAddress;
    uint vertexCount;
    uint indexCount;
};

layout(binding = 3, set = 0, scalar) buffer GeometryMetadataBuffer {
    GeometryMetadata geometries[];
} geometryMetadata;

struct Material {
    uint diffuseTexIndex;
    uint normalTexIndex;
    uint mapsTexIndex;
    uint detailTexIndex;
    uint normalDetailTexIndex;
    float ksAmbient;
    float ksDiffuse;
    float ksSpecular;
    float ksSpecularEXP;
    float emissiveR, emissiveG, emissiveB;
    float fresnelC;
    float fresnelEXP;
    float detailUVMultiplier;
    float detailNormalBlend;
    uint flags;
    float alphaRef;
    uint shaderType;
    float fresnelMaxLevel;
    uint maskTexIndex;
    uint detailRTexIndex;
    uint detailGTexIndex;
    uint detailBTexIndex;
    uint detailATexIndex;
    uint detailNMTexIndex;
    float multR;
    float multG;
    float multB;
    float multA;
    float magicMult;
    float detailNMMult;
    float detailNMMultV;
    float sunSpecular;
    float sunSpecularEXP;
    uint nodeVmHeader;
    uint nodeVmPad[3];
    uvec4 nodeVmCode[128];
};

layout(binding = 4, set = 0, scalar) buffer MaterialBuffer {
    Material materials[];
} materialBuffer;

layout(buffer_reference, scalar) buffer ReplayVertices  { float v[]; };
layout(buffer_reference, scalar) buffer ReplayIndices   { uint  i[]; };
layout(buffer_reference, scalar) buffer ReplayPrimMats  { uint  m[]; };

const uint PT_REPLAY_SHADER_CARPAINT = 11u;
const uint PT_REPLAY_SHADER_MULTIMAP_DAMAGE = 8u;
const uint PT_REPLAY_SHADER_REFLECTION = 3u;

float ptReplayRoughnessFromSpecExp(float specExp, uint shaderType) {
    specExp = clamp(specExp, 1.0, 255.0);
    float roughness = clamp(1.0 - sqrt(specExp / 255.0), 0.04, 1.0);
    if (shaderType == PT_REPLAY_SHADER_CARPAINT ||
        shaderType == PT_REPLAY_SHADER_MULTIMAP_DAMAGE) {
        roughness = clamp(1.0 - (specExp + 400.0) / 255.0, 0.02, 1.0);
    }
    return roughness;
}

void ptApproxReplayMaterial(Material mat, out vec3 albedo,
                            out float roughness, out float metallic,
                            out float specularLevel) {
    float scalar = max(max(mat.ksAmbient, mat.ksDiffuse), mat.ksSpecular);
    scalar = clamp(scalar, 0.04, 0.85);
    albedo = clamp(vec3(mat.multR, mat.multG, mat.multB) * scalar, vec3(0.02), vec3(1.0));
    roughness = ptReplayRoughnessFromSpecExp(mat.ksSpecularEXP, mat.shaderType);
    metallic = (mat.shaderType == PT_REPLAY_SHADER_CARPAINT ||
                mat.shaderType == PT_REPLAY_SHADER_MULTIMAP_DAMAGE ||
                mat.shaderType == PT_REPLAY_SHADER_REFLECTION) ? 1.0 : 0.0;
    specularLevel = clamp(max(mat.ksSpecular, mat.fresnelMaxLevel), 0.0, 1.0);
}

bool ptReconstructReplayHit(uint customIndexRaw, int primitiveId, vec2 bary2,
                            mat4x3 objToWorld, vec3 rayDir,
                            out vec3 worldPos, out vec3 faceNormal,
                            out vec3 albedo, out float roughness,
                            out float metallic, out float specularLevel) {
    uint customIndex = customIndexRaw & 0xFFFFFu;
    if (primitiveId < 0 || customIndex >= geometryMetadata.geometries.length()) {
        return false;
    }

    GeometryMetadata geo = geometryMetadata.geometries[customIndex];
    if (geo.vertexBufferAddress == 0u || geo.indexBufferAddress == 0u) {
        return false;
    }

    uint prim = uint(primitiveId);
    if (prim * 3u + 2u >= geo.indexCount) {
        return false;
    }

    ReplayIndices indices = ReplayIndices(geo.indexBufferAddress);
    uint i0 = indices.i[prim * 3u + 0u];
    uint i1 = indices.i[prim * 3u + 1u];
    uint i2 = indices.i[prim * 3u + 2u];
    if (i0 >= geo.vertexCount || i1 >= geo.vertexCount || i2 >= geo.vertexCount) {
        return false;
    }

    ReplayVertices vertices = ReplayVertices(geo.vertexBufferAddress);
    vec3 p0 = vec3(vertices.v[i0 * 3u + 0u], vertices.v[i0 * 3u + 1u], vertices.v[i0 * 3u + 2u]);
    vec3 p1 = vec3(vertices.v[i1 * 3u + 0u], vertices.v[i1 * 3u + 1u], vertices.v[i1 * 3u + 2u]);
    vec3 p2 = vec3(vertices.v[i2 * 3u + 0u], vertices.v[i2 * 3u + 1u], vertices.v[i2 * 3u + 2u]);

    vec3 bary = vec3(1.0 - bary2.x - bary2.y, bary2.x, bary2.y);
    vec3 localPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    worldPos = objToWorld * vec4(localPos, 1.0);

    vec3 wp0 = objToWorld * vec4(p0, 1.0);
    vec3 wp1 = objToWorld * vec4(p1, 1.0);
    vec3 wp2 = objToWorld * vec4(p2, 1.0);
    faceNormal = cross(wp1 - wp0, wp2 - wp0);
    float nLenSq = dot(faceNormal, faceNormal);
    if (nLenSq < 1e-8) {
        return false;
    }
    faceNormal *= inversesqrt(nLenSq);
    if (dot(faceNormal, -rayDir) < 0.0) {
        faceNormal = -faceNormal;
    }

    uint matIdx = 0u;
    if (geo.primMaterialBufferAddress != 0u) {
        ReplayPrimMats primMats = ReplayPrimMats(geo.primMaterialBufferAddress);
        matIdx = primMats.m[prim];
    }
    Material mat = materialBuffer.materials[matIdx];
    ptApproxReplayMaterial(mat, albedo, roughness, metallic, specularLevel);
    return true;
}

// Replays the prefix of a reused path from the CURRENT pixel's primary
// surface. Returns the surface where the new prefix ends (shiftedReceiver*)
// and whether that surface materially matches the saved source vertex of
// the reservoir. If it doesn't match → the caller must reject the reuse.
//
// For rcBounceDepth ≤ 1, no intermediate hops are needed: the shift goes
// from primary directly to rcVertex, and we return primary as the source.
bool ptReplayDeepPrefix(inout PTReservoir r,
                        vec3 primaryPos, vec3 primaryNormal,
                        vec3 primaryAlbedo,
                        float primaryRoughness,
                        float primaryMetallic,
                        float primarySpecular,
                        out vec3 shiftedReceiverPos,
                        out vec3 shiftedReceiverNormal,
                        out vec3 shiftedReceiverAlbedo,
                        out float shiftedReceiverRoughness,
                        out float shiftedReceiverMetallic,
                        out float shiftedReceiverSpecular,
                        out vec3 shiftedPrefixThroughput) {
    shiftedReceiverPos = primaryPos;
    shiftedReceiverNormal = primaryNormal;
    shiftedReceiverAlbedo = primaryAlbedo;
    shiftedReceiverRoughness = primaryRoughness;
    shiftedReceiverMetallic = primaryMetallic;
    shiftedReceiverSpecular = primarySpecular;
    shiftedPrefixThroughput = r.prefixThroughput;

    if (r.rcBounceDepth <= 1u) {
        // Bounce-1 reconnection: primary → rcVertex directly. No replay
        // needed. Same-source check still validates that the reservoir was
        // born on a primary similar to the new primary.
        return true;
    }

    shiftedPrefixThroughput = primaryAlbedo;
    uint replayRng = r.rngSeed;
    vec3 replayOrigin = primaryPos;
    vec3 replayDir;

    // Generate first scatter direction using the stored RNG (matches the
    // cosine-hemisphere sampling that wf_shade does for diffuse bounces).
    replayRng = replayRng * 747796405u + 2891336453u;
    float u1 = float(replayRng) / 4294967295.0;
    replayRng = replayRng * 747796405u + 2891336453u;
    float u2 = float(replayRng) / 4294967295.0;
    float r_s = sqrt(u1);
    float phi = 6.28318 * u2;
    vec3 localDir = vec3(r_s * cos(phi), r_s * sin(phi), sqrt(max(1.0 - u1, 0.0)));
    vec3 up = abs(primaryNormal.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
    vec3 T = normalize(cross(up, primaryNormal));
    vec3 B = cross(primaryNormal, T);
    replayDir = T * localDir.x + B * localDir.y + primaryNormal * localDir.z;

    for (uint ri = 0u; ri < r.rcBounceDepth - 1u; ri++) {
        rayQueryEXT rpRQ;
        rayQueryInitializeEXT(rpRQ, topLevelAS, gl_RayFlagsOpaqueEXT,
            0xFF, replayOrigin + shiftedReceiverNormal * 0.002,
            0.001, replayDir, 10000.0);
        while (rayQueryProceedEXT(rpRQ)) {}

        if (rayQueryGetIntersectionTypeEXT(rpRQ, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
            return false;
        }

        uint rpCustomIndex = rayQueryGetIntersectionInstanceCustomIndexEXT(rpRQ, true);
        int rpPrimitiveId = rayQueryGetIntersectionPrimitiveIndexEXT(rpRQ, true);
        vec2 rpBary = rayQueryGetIntersectionBarycentricsEXT(rpRQ, true);
        mat4x3 rpOTW = rayQueryGetIntersectionObjectToWorldEXT(rpRQ, true);
        if (!ptReconstructReplayHit(rpCustomIndex, rpPrimitiveId, rpBary, rpOTW,
                                    replayDir, shiftedReceiverPos,
                                    shiftedReceiverNormal,
                                    shiftedReceiverAlbedo,
                                    shiftedReceiverRoughness,
                                    shiftedReceiverMetallic,
                                    shiftedReceiverSpecular)) {
            return false;
        }

        // Throughput accumulates the diffuse albedo along the prefix.
        // Approximation good enough for ReSTIR's target-PDF estimation.
        shiftedPrefixThroughput *= shiftedReceiverAlbedo;

        // Generate next direction with stored RNG (independent stream).
        replayRng = replayRng * 747796405u + 2891336453u;
        u1 = float(replayRng) / 4294967295.0;
        replayRng = replayRng * 747796405u + 2891336453u;
        u2 = float(replayRng) / 4294967295.0;
        r_s = sqrt(u1);
        phi = 6.28318 * u2;
        localDir = vec3(r_s * cos(phi), r_s * sin(phi), sqrt(max(1.0 - u1, 0.0)));
        up = abs(shiftedReceiverNormal.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
        T = normalize(cross(up, shiftedReceiverNormal));
        B = cross(shiftedReceiverNormal, T);
        replayDir = T * localDir.x + B * localDir.y + shiftedReceiverNormal * localDir.z;
        replayOrigin = shiftedReceiverPos;
    }

    // Final check: the surface where the prefix replay ended must materially
    // match the reservoir's saved source vertex. If not, the shift is invalid.
    bool sameSource =
        ptSurfaceSimilar(shiftedReceiverNormal, r.sourceNormal,
                         shiftedReceiverRoughness, r.sourceRoughness,
                         shiftedReceiverPos, r.sourcePos) &&
        ptMaterialSimilarLuma(ptLuminance(shiftedReceiverAlbedo),
                              shiftedReceiverMetallic,
                              shiftedReceiverSpecular,
                              r.sourceAlbedoLuma,
                              r.sourceMetallic,
                              r.sourceSpecularLevel);
    if (!sameSource) {
        return false;
    }
    return true;
}

#endif // PT_REPLAY_GLSL
