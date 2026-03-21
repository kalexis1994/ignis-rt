#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

namespace acpt {
namespace vk {

class Context;
class AccelStructureBuilder;
class Interop;
class TextureManager;

// Shader type enum (matches AC shader names)
enum ACShaderType : uint32_t {
    SHADER_STANDARD = 0,              // ksPerPixel (basic diffuse, no normal map)
    SHADER_NM = 1,                    // ksPerPixelNM (normal mapped)
    SHADER_MULTIMAP = 2,              // ksPerPixelMultiMap (txMaps + reflections)
    SHADER_REFLECTION = 3,            // ksPerPixelReflection (simple fresnel reflections)
    SHADER_AT = 4,                    // ksPerPixelAT (alpha tested, no normal map)
    SHADER_AT_NM = 5,                 // ksPerPixelAT_NM (alpha tested + normal map)
    SHADER_MULTIMAP_NMDETAIL = 6,     // ksPerPixelMultiMap_NMDetail
    SHADER_MULTIMAP_AT = 7,           // ksPerPixelMultiMap_AT
    SHADER_MULTIMAP_DAMAGE = 8,       // ksPerPixelMultiMap_damage_dirt
    SHADER_MULTIMAP_EMISSIVE = 9,     // ksPerPixelMultiMap_emissive
    SHADER_GLASS = 10,                // ksWindscreen, smGlass
    SHADER_CARPAINT = 11,             // smCarPaint
    SHADER_MULTILAYER = 12,           // ksMultilayer, ksMultilayer_fresnel_nm, etc.
    SHADER_TREE = 13,                 // ksTree (alpha-tested foliage)
    SHADER_GRASS = 14,                // ksGrass
    SHADER_TYRES = 15,                // ksTyres
    SHADER_SKINNED = 16,              // ksSkinnedMesh
    SHADER_BRAKE_DISC = 17,           // ksBrakeDisc
    SHADER_SKIDMARK = 18,             // ksSkidMark
    SHADER_ALPHA = 19,                // ksPerPixelAlpha (transparent blend)
    SHADER_TREE_TRUNK = 20,           // ksTreeTrunk (PBR bark, Cook-Torrance GGX)
};

// GPU material struct (matches shader Material struct in raygen.rgen)
// 140 bytes per material, scalar layout
struct GPUMaterial {
    // Texture indices (0xFFFFFFFF = none)
    uint32_t diffuseTexIndex;
    uint32_t normalTexIndex;
    uint32_t mapsTexIndex;          // txMaps: R=specular mask, G=glossiness, B=sun specular
    uint32_t detailTexIndex;        // txDetail: detail albedo

    uint32_t normalDetailTexIndex;  // txNormalDetail: detail normal map (NMDetail shaders)
    float ksAmbient;
    float ksDiffuse;
    float ksSpecular;

    float ksSpecularEXP;
    float emissiveR, emissiveG, emissiveB;

    float fresnelC;
    float fresnelEXP;
    float detailUVMultiplier;       // UV scale for detail texture
    float detailNormalBlend;        // Blend strength for detail normal (0-1)

    uint32_t flags;                 // bit 0: alphaTested, bit 1: useDetail, bit 2: isCarPaint, bit 6: multilayerObjSp
    float alphaRef;                 // Alpha test reference value
    uint32_t shaderType;            // ACShaderType enum
    float fresnelMaxLevel;          // max fresnel reflection intensity

    // === Multilayer fields (48 bytes, only used when shaderType == SHADER_MULTILAYER) ===
    uint32_t maskTexIndex;          // txMask: RGBA channel selection mask
    uint32_t detailRTexIndex;       // txDetailR: detail layer for mask.r
    uint32_t detailGTexIndex;       // txDetailG: detail layer for mask.g
    uint32_t detailBTexIndex;       // txDetailB: detail layer for mask.b

    uint32_t detailATexIndex;       // txDetailA: detail layer for mask.a
    uint32_t detailNMTexIndex;      // txDetailNM: shared detail normal map
    float multR;                    // UV multiplier for detailR (world-space)
    float multG;                    // UV multiplier for detailG

    float multB;                    // UV multiplier for detailB
    float multA;                    // UV multiplier for detailA
    float magicMult;                // intensity multiplier for combined detail
    float detailNMMult;             // UV scale for detail normal map (U component)
    float detailNMMultV;            // UV scale for detail normal map (V component)

    // Sun specular (Maps shaders only — additional specular highlight from sun)
    float sunSpecular;              // sunSpecular intensity (separate from ksSpecular)
    float sunSpecularEXP;           // sunSpecular exponent (separate from ksSpecularEXP)
};

// Camera UBO matching raygen.rgen CameraProperties
struct CameraUBO {
    float viewInverse[16];
    float projInverse[16];
    float view[16];
    float proj[16];
    float viewPrev[16];
    float projPrev[16];
    uint32_t parameters[4];   // x=frameIndex, y=maxBounces, zw=padding
    float jitterData[4];      // xy=jitter, zw=padding
    float sunLight[4];
    float ambientLight[4];
    float skyLight[4];
    float ptParams[8];         // Path tracing tuning: [0]=exposure, [1]=giIntensity, [2]=saturation,
                               // [3]=contrast, [4]=skyReflIntensity, [5]=ambientMax, [6]=sunMinIntensity, [7]=skyBounceIntensity
    float windParams[4];       // [0]=windDirX, [1]=windDirZ, [2]=windSpeed(m/s), [3]=time(seconds)
    float rainParams[4];       // [0]=wetness, [1]=waterLevel, [2]=rainIntensity, [3]=reserved
    int32_t jitterPattern[32]; // 8 x ivec4
    // Point/spot lights (NEE direct sampling)
    uint32_t lightCount;       // number of active lights (0-32)
    uint32_t lightPad[3];
    float lights[512];         // 32 lights × 16 floats: [pos.xyz+range, col.rgb+intensity, dir.xyz+sizeX, tan.xyz+sizeY]
};

// Pick result from GPU raycast
struct PickResult {
    uint32_t customIndex;
    uint32_t primitiveId;
    uint32_t materialId;
    uint32_t valid;  // 1 if pick hit something
};

// RT pipeline using ray tracing pipeline with raygen.rgen
class RTPipeline {
public:
    bool Initialize(Context* context, AccelStructureBuilder* accelBuilder, Interop* interop);
    void Shutdown();

    // Update camera UBO
    void UpdateCamera(const CameraUBO& camera);

    // Record dispatch commands into command buffer
    void RecordDispatch(VkCommandBuffer cmd, uint32_t width, uint32_t height);

    // Update TLAS descriptor after rebuild
    void UpdateTLASDescriptor();

    // Update the output storage image descriptor (e.g. after importing D3D11 shared texture)
    void UpdateStorageImage(VkImageView imageView);

    // Update material buffer (binding 4)
    void UpdateMaterialBuffer(const GPUMaterial* materials, uint32_t count);

    // Update texture array descriptors (binding 5)
    void UpdateTextureDescriptors(TextureManager* texMgr);

    // Pick buffer (binding 17)
    PickResult ReadPickResult();
    void ResetPickBuffer();

    // Cloud shadow map (binding 23)
    void UpdateCloudShadowDescriptor(VkImageView view, VkSampler sampler);

    // Per-instance previous transforms (binding 28)
    void UpdatePrevTransforms(const float* transforms, uint32_t instanceCount);

    // Emissive triangle buffer for MIS (binding 26)
    void UpdateEmissiveTriangleBuffer(const float* data, uint32_t triangleCount);

    // Light tree buffer (binding 27)
    void UpdateLightTreeBuffer(const void* nodes, uint32_t nodeCount);

    // Create full-resolution G-buffer images and update descriptors
    bool CreateGBuffers(uint32_t width, uint32_t height);

    // G-buffer image accessors (for NRD)
    VkImage GetNormalRoughnessImage() const { return normalRoughnessGBuffer_.image; }
    VkImageView GetNormalRoughnessView() const { return normalRoughnessGBuffer_.view; }
    VkImage GetViewDepthImage() const { return viewDepthGBuffer_.image; }
    VkImage GetMotionVectorsImage() const { return motionVectorsGBuffer_.image; }
    VkImage GetDiffuseRadianceImage() const { return diffuseRadianceGBuffer_.image; }
    VkImageView GetDiffuseRadianceView() const { return diffuseRadianceGBuffer_.view; }
    VkImage GetSpecularRadianceImage() const { return specularRadianceGBuffer_.image; }
    VkImageView GetSpecularRadianceView() const { return specularRadianceGBuffer_.view; }
    VkImage GetSpecularAlbedoImage() const { return specularAlbedoGBuffer_.image; }
    VkImageView GetSpecularAlbedoView() const { return specularAlbedoGBuffer_.view; }
    VkImage GetSpecularMVImage() const { return specularMVGBuffer_.image; }
    VkImageView GetSpecularMVView() const { return specularMVGBuffer_.view; }
    VkImage GetAlbedoBufferImage() const { return albedoGBuffer_.image; }
    VkImageView GetAlbedoBufferView() const { return albedoGBuffer_.view; }
    VkImage GetPenumbraImage() const { return penumbraGBuffer_.image; }
    VkImageView GetViewDepthView() const { return viewDepthGBuffer_.view; }
    VkImageView GetMotionVectorsView() const { return motionVectorsGBuffer_.view; }
    VkImage GetDlssDepthImage() const { return dlssDepthGBuffer_.image; }
    VkImageView GetDlssDepthView() const { return dlssDepthGBuffer_.view; }
    VkImage GetReactiveMaskImage() const { return reactiveMaskGBuffer_.image; }
    VkImageView GetReactiveMaskView() const { return reactiveMaskGBuffer_.view; }
    VkImage GetDiffConfidenceImage() const { return diffConfidenceGBuffer_.image; }
    VkDescriptorSetLayout GetDescriptorSetLayout() const { return descriptorSetLayout_; }
    VkDescriptorSet GetDescriptorSet() const { return descriptorSet_; }
    VkImage GetSpecConfidenceImage() const { return specConfidenceGBuffer_.image; }
    bool HasGBuffers() const { return gbuffersCreated_; }

    // Shadow temporal accumulation — swap ping-pong buffers each frame
    void SwapShadowAccumBuffers();
    bool HasShadowAccumBuffers() const { return shadowAccumCreated_; }

    // GI Reservoir (ReSTIR) — swap ping-pong buffers each frame
    void SwapGIReservoirBuffers();
    bool HasGIReservoirBuffers() const { return giReservoirCreated_; }

    // SHARC radiance cache accessors
    bool HasSHARCBuffers() const { return sharcCreated_; }
    VkBuffer GetSHARCWriteBuffer() const { return sharcBuffer_[0]; }
    VkBuffer GetSHARCReadBuffer() const { return sharcBuffer_[1]; }
    static constexpr VkDeviceSize GetSHARCBufferSize() { return SHARC_TABLE_SIZE * SHARC_ENTRY_SIZE; }

private:
    bool CreateDescriptorSetLayout();
    bool CreatePipeline();
    bool CreateDescriptorPool();
    bool CreateDescriptorSet();
    bool CreateSBT();
    bool CreateDummyResources();
    bool LoadShaderModule(const char* path, VkShaderModule* outModule);

    Context* context_ = nullptr;
    AccelStructureBuilder* accelBuilder_ = nullptr;
    Interop* interop_ = nullptr;

    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;

    // Camera UBO
    VkBuffer cameraBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory cameraMemory_ = VK_NULL_HANDLE;

    // SBT
    VkBuffer sbtBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory sbtMemory_ = VK_NULL_HANDLE;
    VkStridedDeviceAddressRegionKHR raygenRegion_{};
    VkStridedDeviceAddressRegionKHR missRegion_{};
    VkStridedDeviceAddressRegionKHR hitRegion_{};
    VkStridedDeviceAddressRegionKHR callableRegion_{};

    // NRD G-buffer images (full-resolution, written by raygen shader)
    struct GBufferImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };
    GBufferImage normalRoughnessGBuffer_;    // binding 6, RGBA16F
    GBufferImage viewDepthGBuffer_;          // binding 7, R32F
    GBufferImage motionVectorsGBuffer_;      // binding 8, RGBA16F
    GBufferImage diffuseRadianceGBuffer_;    // binding 9, RGBA16F
    GBufferImage specularRadianceGBuffer_;   // binding 10, RGBA16F
    GBufferImage specularMVGBuffer_;          // binding 13, RGBA16F — specular motion vectors for RR
    GBufferImage specularAlbedoGBuffer_;     // binding 15, RGBA16F — EnvBRDFApprox for RR
    GBufferImage albedoGBuffer_;             // binding 16, RGBA16F
    GBufferImage penumbraGBuffer_;           // binding 18, R16F — SIGMA shadow input
    GBufferImage dlssDepthGBuffer_;          // binding 22, R32F — NDC depth [0,1] for DLSS
    GBufferImage reactiveMaskGBuffer_;      // binding 29, R8_UNORM — DLSS reactive mask
    GBufferImage diffConfidenceGBuffer_;    // binding 30, R8_UNORM — NRD diffuse confidence
    GBufferImage specConfidenceGBuffer_;    // binding 31, R8_UNORM — NRD specular confidence
    bool gbuffersCreated_ = false;
    uint32_t gbufferWidth_ = 0;
    uint32_t gbufferHeight_ = 0;

    bool CreateGBufferImage(GBufferImage& gb, VkFormat format, uint32_t width, uint32_t height, const char* name);
    void DestroyGBufferImage(GBufferImage& gb);

    // GI Reservoir buffers for ReSTIR (bindings 24-25)
    VkBuffer giReservoirBuffer_[2] = {};   // [0]=current write, [1]=previous read
    VkDeviceMemory giReservoirMemory_[2] = {};
    bool giReservoirCreated_ = false;
    uint32_t giReservoirPixelCount_ = 0;
    static constexpr uint32_t GI_RESERVOIR_VEC4S_PER_PIXEL = 3;  // 3 vec4s = 48 bytes
    bool CreateGIReservoirBuffers(uint32_t width, uint32_t height);
    void DestroyGIReservoirBuffers();

    // SHARC radiance cache (bindings 20-21)
    VkBuffer sharcBuffer_[2] = {};       // [0]=write, [1]=read
    VkDeviceMemory sharcMemory_[2] = {};
    bool sharcCreated_ = false;
    static constexpr uint32_t SHARC_TABLE_SIZE = 4194304;  // 2^22 (doubled for fewer hash collisions)
    static constexpr uint32_t SHARC_ENTRY_SIZE = 32;       // 8 uints = 32 bytes
    static constexpr VkDeviceSize SHARC_BUFFER_SIZE = SHARC_TABLE_SIZE * SHARC_ENTRY_SIZE; // 64 MiB
    bool CreateSHARCBuffers();
    void DestroySHARCBuffers();

    // Shadow temporal accumulation (ping-pong, bindings 18-19)
    GBufferImage shadowAccumImage_[2];  // [0] and [1] alternate each frame
    uint32_t shadowAccumCurrent_ = 0;   // Index of current write buffer
    bool shadowAccumCreated_ = false;
    bool CreateShadowAccumBuffers(uint32_t width, uint32_t height);
    void DestroyShadowAccumBuffers();
    void UpdateShadowAccumDescriptors();

    // Dummy resources for unused bindings
    VkImage dummyImage_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyImageMemory_ = VK_NULL_HANDLE;
    VkImageView dummyImageView_ = VK_NULL_HANDLE;
    // RGBA16F dummy for G-buffer storage image bindings (6,8,9,10,13,15,16)
    VkImage dummyImage16f_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyImageMemory16f_ = VK_NULL_HANDLE;
    VkImageView dummyImageView16f_ = VK_NULL_HANDLE;
    // R16F dummy for viewDepth binding (7)
    VkImage dummyImageR16f_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyImageMemoryR16f_ = VK_NULL_HANDLE;
    VkImageView dummyImageViewR16f_ = VK_NULL_HANDLE;
    // R8 dummy for confidence/reactive mask bindings (29-31)
    VkImage dummyImageR8_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyImageMemoryR8_ = VK_NULL_HANDLE;
    VkImageView dummyImageViewR8_ = VK_NULL_HANDLE;
    VkSampler dummySampler_ = VK_NULL_HANDLE;
    VkBuffer dummyBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyBufferMemory_ = VK_NULL_HANDLE;
    // Geometry metadata buffer (binding 3)
    VkBuffer geometryMetadataBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory geometryMetadataMemory_ = VK_NULL_HANDLE;
    // Material SSBO (binding 4)
    VkBuffer materialBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory materialMemory_ = VK_NULL_HANDLE;
    // Pick buffer SSBO (binding 17)
    VkBuffer pickBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory pickMemory_ = VK_NULL_HANDLE;
    PickResult* pickMappedPtr_ = nullptr;

    // Emissive triangle SSBO (binding 26)
    VkBuffer emissiveTriBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory emissiveTriMemory_ = VK_NULL_HANDLE;

    // Light tree SSBO (binding 27)
    VkBuffer lightTreeBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory lightTreeMemory_ = VK_NULL_HANDLE;
    uint32_t lightTreeNodeCount_ = 0;
    uint32_t emissiveTriCount_ = 0;

    // Previous-frame instance transforms SSBO (binding 28)
    VkBuffer prevTransformsBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory prevTransformsMemory_ = VK_NULL_HANDLE;
    void* prevTransformsMapped_ = nullptr;
    uint32_t prevTransformsCapacity_ = 0;

    // Function pointers
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR_ = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR_ = nullptr;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR_ = nullptr;
    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR_ = nullptr;
};

} // namespace vk
} // namespace acpt
