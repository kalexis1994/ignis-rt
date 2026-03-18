#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include "vk_types.h"

namespace acpt {
namespace vk {

class Context;
class Pipeline;
class Geometry;
struct Mesh;

// Rasterization system - manages all rasterization rendering
class Rasterizer {
public:
    bool Initialize(Context* context, Pipeline* pipeline, Geometry* geometry);
    void Shutdown();

    void SetScene(Mesh* sphere, Mesh* plane);
    void SetDrawCalls(const std::vector<RasterDrawCall>& draws);
    void Render(uint32_t imageIndex, VkCommandBuffer commandBuffer);
    void UpdateCamera(float distance, float angle, float targetY = 0.0f, float pitch = 0.3f);

    // Viewport override for split-screen rendering
    void SetViewportOverride(float x, float y, float w, float h);
    void ClearViewportOverride();

    // Albedo texture for ksTree reference rendering
    void SetAlbedoTexture(VkImageView imageView, VkSampler sampler);
    void ClearAlbedoTexture();

    // Upload RGBA8 pixel data as albedo texture (for procedural leaf textures)
    bool UploadAlbedoFromPixels(const uint8_t* rgba, int width, int height);

    // Upload RGBA8 pixel data as normal map
    bool UploadNormalFromPixels(const uint8_t* rgba, int width, int height);

    // Upload RGBA8 pixel data as specular map
    bool UploadSpecularFromPixels(const uint8_t* rgba, int width, int height);

    // Upload RGBA8 pixel data as bark textures (bindings 5, 6, 7)
    bool UploadBarkAlbedoFromPixels(const uint8_t* rgba, int width, int height);
    bool UploadBarkNormalFromPixels(const uint8_t* rgba, int width, int height);
    bool UploadBarkSpecularFromPixels(const uint8_t* rgba, int width, int height);

    const std::vector<VkFramebuffer>& GetFramebuffers() const { return framebuffers_; }

    // Billboard capture: render current draw calls to an offscreen RGBA8 image
    // using an orthographic projection sized to the tree bounding box.
    // Returns pixel data in outPixels (size*size*4 bytes, RGBA8).
    bool CaptureBillboardTexture(uint32_t size, float treeWidth, float treeHeight,
                                 float treeBaseY, std::vector<uint8_t>& outPixels);

private:
    bool CreateFramebuffers();
    bool CreateDepthResources();
    bool CreateUniformBuffers();
    bool CreateDescriptorPool();
    bool CreateDescriptorSets();
    void UpdateUniformBuffer(uint32_t currentImage);

    // Albedo texture (dummy 1x1 white + optional override)
    bool CreateDummyAlbedoTexture();

    // Shadow mapping
    bool CreateShadowResources();
    void RenderShadowPass(VkCommandBuffer commandBuffer);
    void ComputeLightMatrices(float* outLightView, float* outLightProj, float* outLightViewProj);

    Context* context_ = nullptr;
    Pipeline* pipeline_ = nullptr;
    Geometry* geometry_ = nullptr;

    // Scene
    Mesh* sphereMesh_ = nullptr;
    Mesh* planeMesh_ = nullptr;

    // Dynamic draw call list
    std::vector<RasterDrawCall> drawCalls_;

    // Framebuffers and depth
    std::vector<VkFramebuffer> framebuffers_;
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory_ = VK_NULL_HANDLE;
    VkImageView depthImageView_ = VK_NULL_HANDLE;

    // Uniform buffers (camera)
    std::vector<VkBuffer> uniformBuffers_;
    std::vector<VkDeviceMemory> uniformBuffersMemory_;
    std::vector<void*> uniformBuffersMapped_;

    // Descriptors
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;

    // Camera
    float cameraDistance_ = 5.0f;
    float cameraAngle_ = 0.0f;
    float cameraTargetY_ = 0.0f;
    float cameraPitch_ = 0.3f;

    // Viewport override (for split-screen)
    float viewportX_ = 0, viewportY_ = 0, viewportW_ = 0, viewportH_ = 0;
    bool viewportOverride_ = false;

    // Albedo texture (for ksTree reference)
    VkImage dummyAlbedoImage_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyAlbedoMemory_ = VK_NULL_HANDLE;
    VkImageView dummyAlbedoView_ = VK_NULL_HANDLE;
    VkSampler dummyAlbedoSampler_ = VK_NULL_HANDLE;
    VkImageView activeAlbedoView_ = VK_NULL_HANDLE;
    VkSampler activeAlbedoSampler_ = VK_NULL_HANDLE;

    // Uploaded albedo texture (from UploadAlbedoFromPixels)
    VkImage uploadedAlbedoImage_ = VK_NULL_HANDLE;
    VkDeviceMemory uploadedAlbedoMemory_ = VK_NULL_HANDLE;
    VkImageView uploadedAlbedoView_ = VK_NULL_HANDLE;
    VkSampler uploadedAlbedoSampler_ = VK_NULL_HANDLE;
    void DestroyUploadedAlbedo();

    // Normal map texture (dummy flat normal + optional upload)
    VkImage dummyNormalImage_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyNormalMemory_ = VK_NULL_HANDLE;
    VkImageView dummyNormalView_ = VK_NULL_HANDLE;
    VkSampler dummyNormalSampler_ = VK_NULL_HANDLE;
    VkImage uploadedNormalImage_ = VK_NULL_HANDLE;
    VkDeviceMemory uploadedNormalMemory_ = VK_NULL_HANDLE;
    VkImageView uploadedNormalView_ = VK_NULL_HANDLE;
    VkSampler uploadedNormalSampler_ = VK_NULL_HANDLE;
    VkImageView activeNormalView_ = VK_NULL_HANDLE;
    VkSampler activeNormalSampler_ = VK_NULL_HANDLE;
    void DestroyUploadedNormal();
    bool CreateDummyNormalTexture();
    void SetNormalTexture(VkImageView imageView, VkSampler sampler);

    // Specular map texture (dummy white + optional upload)
    VkImage dummySpecImage_ = VK_NULL_HANDLE;
    VkDeviceMemory dummySpecMemory_ = VK_NULL_HANDLE;
    VkImageView dummySpecView_ = VK_NULL_HANDLE;
    VkSampler dummySpecSampler_ = VK_NULL_HANDLE;
    VkImage uploadedSpecImage_ = VK_NULL_HANDLE;
    VkDeviceMemory uploadedSpecMemory_ = VK_NULL_HANDLE;
    VkImageView uploadedSpecView_ = VK_NULL_HANDLE;
    VkSampler uploadedSpecSampler_ = VK_NULL_HANDLE;
    VkImageView activeSpecView_ = VK_NULL_HANDLE;
    VkSampler activeSpecSampler_ = VK_NULL_HANDLE;
    void DestroyUploadedSpec();
    bool CreateDummySpecTexture();
    void SetSpecTexture(VkImageView imageView, VkSampler sampler);

    // Bark albedo texture (binding 5)
    VkImage dummyBarkAlbedoImage_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyBarkAlbedoMemory_ = VK_NULL_HANDLE;
    VkImageView dummyBarkAlbedoView_ = VK_NULL_HANDLE;
    VkSampler dummyBarkAlbedoSampler_ = VK_NULL_HANDLE;
    VkImage uploadedBarkAlbedoImage_ = VK_NULL_HANDLE;
    VkDeviceMemory uploadedBarkAlbedoMemory_ = VK_NULL_HANDLE;
    VkImageView uploadedBarkAlbedoView_ = VK_NULL_HANDLE;
    VkSampler uploadedBarkAlbedoSampler_ = VK_NULL_HANDLE;
    VkImageView activeBarkAlbedoView_ = VK_NULL_HANDLE;
    VkSampler activeBarkAlbedoSampler_ = VK_NULL_HANDLE;
    void DestroyUploadedBarkAlbedo();
    bool CreateDummyBarkAlbedoTexture();
    void SetBarkAlbedoTexture(VkImageView imageView, VkSampler sampler);

    // Bark normal map texture (binding 6)
    VkImage dummyBarkNormalImage_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyBarkNormalMemory_ = VK_NULL_HANDLE;
    VkImageView dummyBarkNormalView_ = VK_NULL_HANDLE;
    VkSampler dummyBarkNormalSampler_ = VK_NULL_HANDLE;
    VkImage uploadedBarkNormalImage_ = VK_NULL_HANDLE;
    VkDeviceMemory uploadedBarkNormalMemory_ = VK_NULL_HANDLE;
    VkImageView uploadedBarkNormalView_ = VK_NULL_HANDLE;
    VkSampler uploadedBarkNormalSampler_ = VK_NULL_HANDLE;
    VkImageView activeBarkNormalView_ = VK_NULL_HANDLE;
    VkSampler activeBarkNormalSampler_ = VK_NULL_HANDLE;
    void DestroyUploadedBarkNormal();
    bool CreateDummyBarkNormalTexture();
    void SetBarkNormalTexture(VkImageView imageView, VkSampler sampler);

    // Bark specular map texture (binding 7)
    VkImage dummyBarkSpecImage_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyBarkSpecMemory_ = VK_NULL_HANDLE;
    VkImageView dummyBarkSpecView_ = VK_NULL_HANDLE;
    VkSampler dummyBarkSpecSampler_ = VK_NULL_HANDLE;
    VkImage uploadedBarkSpecImage_ = VK_NULL_HANDLE;
    VkDeviceMemory uploadedBarkSpecMemory_ = VK_NULL_HANDLE;
    VkImageView uploadedBarkSpecView_ = VK_NULL_HANDLE;
    VkSampler uploadedBarkSpecSampler_ = VK_NULL_HANDLE;
    VkImageView activeBarkSpecView_ = VK_NULL_HANDLE;
    VkSampler activeBarkSpecSampler_ = VK_NULL_HANDLE;
    void DestroyUploadedBarkSpec();
    bool CreateDummyBarkSpecTexture();
    void SetBarkSpecTexture(VkImageView imageView, VkSampler sampler);

    // Generic texture upload helper (shared by albedo/normal/specular)
    // addressMode: CLAMP_TO_EDGE for leaf textures (UVs in [0,1]),
    //              REPEAT for bark textures (proctree V is unbounded)
    bool UploadTextureFromPixels(const uint8_t* rgba, int width, int height,
        VkImage& outImage, VkDeviceMemory& outMemory,
        VkImageView& outView, VkSampler& outSampler,
        VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // Billboard offscreen capture
    bool CreateBillboardResources(uint32_t size);
    void DestroyBillboardResources();
    uint32_t billboardSize_ = 0;
    VkImage billboardColorImage_ = VK_NULL_HANDLE;
    VkDeviceMemory billboardColorMemory_ = VK_NULL_HANDLE;
    VkImageView billboardColorView_ = VK_NULL_HANDLE;
    VkImage billboardDepthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory billboardDepthMemory_ = VK_NULL_HANDLE;
    VkImageView billboardDepthView_ = VK_NULL_HANDLE;
    VkRenderPass billboardRenderPass_ = VK_NULL_HANDLE;
    VkFramebuffer billboardFramebuffer_ = VK_NULL_HANDLE;
    VkBuffer billboardReadbackBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory billboardReadbackMemory_ = VK_NULL_HANDLE;

    // Shadow mapping
    static const uint32_t SHADOW_MAP_SIZE = 2048;
    VkImage shadowMapImage_ = VK_NULL_HANDLE;
    VkDeviceMemory shadowMapMemory_ = VK_NULL_HANDLE;
    VkImageView shadowMapView_ = VK_NULL_HANDLE;
    VkSampler shadowMapSampler_ = VK_NULL_HANDLE;
    VkRenderPass shadowRenderPass_ = VK_NULL_HANDLE;
    VkFramebuffer shadowFramebuffer_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout shadowDescSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline shadowPipeline_ = VK_NULL_HANDLE;

    // Shadow UBOs (light VP in view/proj slots)
    std::vector<VkBuffer> shadowUniformBuffers_;
    std::vector<VkDeviceMemory> shadowUniformBuffersMemory_;
    std::vector<void*> shadowUniformBuffersMapped_;

    // Shadow descriptor sets
    VkDescriptorPool shadowDescPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> shadowDescSets_;
};

} // namespace vk
} // namespace acpt
