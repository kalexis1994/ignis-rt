#pragma once

#include <windows.h>
#include <vector>
#include <vulkan/vulkan.h>
#include "vk_accel_structure.h"
#include "../../include/dlss_ngx.h"

namespace acpt {
namespace vk {

class Context;
class Pipeline;
class Geometry;
class Rasterizer;
class RTPipeline;
class WavefrontPipeline;
class Interop;
struct Mesh;
struct CameraUBO;

// Main Vulkan renderer - orchestrates all modules
class Renderer {
public:
    bool Initialize(HWND hwnd, uint32_t width, uint32_t height);
    void Shutdown();
    void RenderFrame();

    // RT path: upload geometry and build acceleration structures
    bool InitRT();
    bool BuildAccelStructure(const float* vertices, uint32_t vertexCount,
                             const uint32_t* indices, uint32_t indexCount);
    bool BuildTLAS();
    int BuildBLAS(const float* vertices, uint32_t vertexCount,
                  const uint32_t* indices, uint32_t indexCount);
    bool RefitBLAS(int blasIndex, const float* vertices, uint32_t vertexCount,
                   const uint32_t* indices, uint32_t indexCount);
    bool UploadBLASAttributes(int blasIndex, const float* normals, const float* uvs, uint32_t vertexCount);
    bool UploadBLASPrimitiveMaterials(int blasIndex, const uint32_t* materialIds, uint32_t primitiveCount);
    bool UploadBLASPrimitiveYBounds(int blasIndex, const float* yBounds, uint32_t primitiveCount);
    void UploadMaterialBuffer(const void* materials, uint32_t count);
    void UploadEmissiveTriangles(const float* data, uint32_t triangleCount);
    void UpdateTextureDescriptors(void* texManager);
    void UploadLightTree(const void* nodes, uint32_t nodeCount,
                         const void* emitters, uint32_t emitterCount);
    bool BuildTLASInstanced(const std::vector<vk::TLASInstance>& instances);
    void UpdateCamera(const CameraUBO& camera);
    void RenderFrameRT();

    // Interop
    HANDLE GetInteropNTHandle() const;
    bool ReadbackPixels(void* outData, uint32_t bufferSize);
    void SetDirectInterop(bool enabled) { useDirectInterop_ = enabled; }
    bool ImportD3D11Texture(HANDLE ntHandle, uint32_t width, uint32_t height);

    // GL interop (Vulkan → OpenGL zero-copy)
    bool InitGLInterop();
    void DrawGL(uint32_t w, uint32_t h);
    bool IsRTReady() const { return rtReady_; }
    bool IsRTSupported() const;
    bool HasInterop() const { return interop_ != nullptr; }

    bool IsDLSSActive() const { return dlssActive_; }
    bool IsDLSSRRActive() const { return dlssRRActive_; }
    void GetRenderResolution(uint32_t* w, uint32_t* h) const { *w = renderWidth_; *h = renderHeight_; }
    void GetDisplayResolution(uint32_t* w, uint32_t* h) const { *w = width_; *h = height_; }
    uint32_t GetRenderWidth() const { return width_; }
    uint32_t GetRenderHeight() const { return height_; }
    float GetComputedExposure() const { return computedExposure_; }
    void ResetFrameIndex() { frameIndex_ = 0; }
    int GetActualDLSSQuality() const;

    Context* GetContext() const { return context_; }
    AccelStructureBuilder* GetAccelBuilder() const { return accelBuilder_; }
    Geometry* GetGeometry() const { return geometry_; }
    Rasterizer* GetRasterizer() const { return rasterizer_; }
    void SetExternalCameraControl(bool ext) { externalCameraControl_ = ext; }
    bool IsExternalCameraControl() const { return externalCameraControl_; }

    // Pick result readback
    bool ReadPickResult(uint32_t& outCustomIndex, uint32_t& outPrimitiveId, uint32_t& outMaterialId);

    // ImGui overlay
    bool InitImGui(HWND hwnd, bool forceRasterPath = false);
    void RenderImGuiOverlay(VkCommandBuffer cmd);
    void ShutdownImGui();

private:
    bool CreateCommandBuffers();
    bool CreateSyncObjects();

    Context* context_ = nullptr;
    Pipeline* pipeline_ = nullptr;
    Geometry* geometry_ = nullptr;
    Rasterizer* rasterizer_ = nullptr;

    // RT modules
    AccelStructureBuilder* accelBuilder_ = nullptr;
    RTPipeline* rtPipeline_ = nullptr;
    WavefrontPipeline* wavefrontPipeline_ = nullptr;
    Interop* interop_ = nullptr;
    bool rtReady_ = false;
    bool useDirectInterop_ = false;
    bool glInteropFailed_ = false;

    // Scene meshes (rasterization demo)
    Mesh* sphereMesh_ = nullptr;
    Mesh* planeMesh_ = nullptr;

    // Command buffers
    std::vector<VkCommandBuffer> commandBuffers_;

    // Sync objects
    std::vector<VkSemaphore> imageAvailableSemaphores_;
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<VkFence> inFlightFences_;
    uint32_t currentFrame_ = 0;
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    // Per-instance transforms for motion vectors
    std::vector<float> currInstanceTransforms_;  // 12 floats per instance (3x4 row-major)
    std::vector<float> prevInstanceTransforms_;
    uint32_t instanceTransformCount_ = 0;

    // Camera
    float cameraDistance_ = 5.0f;
    float cameraAngle_ = 0.0f;
    bool externalCameraControl_ = false;

    uint32_t width_ = 0;
    uint32_t height_ = 0;

    // DLSS upscaling
    DLSS_NGX* dlss_ = nullptr;
    bool dlssActive_ = false;
    bool dlssRRActive_ = false;
    bool dlssDebugBypass_ = false;
    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;

    VkImage dlssColorInput_ = VK_NULL_HANDLE;
    VkDeviceMemory dlssColorInputMemory_ = VK_NULL_HANDLE;
    VkImageView dlssColorInputView_ = VK_NULL_HANDLE;

    VkImage dlssHdrOutput_ = VK_NULL_HANDLE;
    VkDeviceMemory dlssHdrOutputMemory_ = VK_NULL_HANDLE;
    VkImageView dlssHdrOutputView_ = VK_NULL_HANDLE;

    float jitterX_ = 0.0f;
    float jitterY_ = 0.0f;
    float prevJitterX_ = 0.0f;
    float prevJitterY_ = 0.0f;

    void ShutdownDLSS();

    // Tonemap compute pipeline (post-DLSS HDR -> LDR)
    VkPipeline tonemapPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout tonemapPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout tonemapDescSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool tonemapDescPool_ = VK_NULL_HANDLE;
    VkDescriptorSet tonemapDescSet_ = VK_NULL_HANDLE;
    VkSampler tonemapSampler_ = VK_NULL_HANDLE;
    bool tonemapReady_ = false;

    bool CreateTonemapPipeline();
    void UpdateTonemapDescriptors();
    void ShutdownTonemap();

    // NRD denoiser
    bool nrdInitialized_ = false;
    uint32_t frameIndex_ = 0;
    float lastView_[16] = {0};
    float lastProj_[16] = {0};
    float lastViewPrev_[16] = {0};
    float lastProjPrev_[16] = {0};

    // Composite compute pipeline
    VkPipeline compositePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout compositePipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout compositeDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool compositeDescriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet compositeDescriptorSet_ = VK_NULL_HANDLE;
    VkSampler compositeSampler_ = VK_NULL_HANDLE;
    bool compositeReady_ = false;

    bool InitNRD();
    bool CreateCompositePipeline();
    void UpdateCompositeDescriptors();
    void ShutdownNRD();

    // Auto-exposure resolve pipeline
    VkPipeline exposureResolvePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout exposureResolvePipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout exposureResolveDescSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool exposureResolveDescPool_ = VK_NULL_HANDLE;
    VkDescriptorSet exposureResolveDescSet_ = VK_NULL_HANDLE;
    AccelBuffer exposureSSBO_;
    AccelBuffer exposureStagingSSBO_;
    float computedExposure_ = 0.55f;
    bool exposureResolveReady_ = false;

    bool CreateExposureResolvePipeline();
    void ShutdownExposureResolve();

    // SHARC resolve compute pipeline
    VkPipeline sharcResolvePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout sharcResolvePipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sharcResolveDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool sharcResolveDescriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet sharcResolveDescriptorSet_ = VK_NULL_HANDLE;
    bool sharcResolveReady_ = false;
    bool CreateSHARCResolvePipeline();
    void UpdateSHARCResolveDescriptors();
    void ShutdownSHARCResolve();

    // ImGui overlay
    VkRenderPass imguiRenderPass_ = VK_NULL_HANDLE;
    VkFramebuffer imguiFramebuffer_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> imguiSwapchainFramebuffers_;
    uint32_t imguiCurrentImageIndex_ = 0;
    VkDescriptorPool imguiDescriptorPool_ = VK_NULL_HANDLE;
    bool imguiReady_ = false;
};

} // namespace vk
} // namespace acpt
