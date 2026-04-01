#pragma once

#include <windows.h>
#include <vector>
#include <string>
#include <cstdint>
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
    // Phased initialization (call one per frame for smooth loading screen)
    // Step 0: Vulkan context (instance, device, swapchain)
    // Step 1: Basic pipelines + geometry
    // Step 2: RT pipeline + interop
    // Step 3: DLSS initialization
    // Step 4: NRD + G-buffers + compose pipelines
    // Returns step name string, or nullptr when all steps complete.
    const char* InitializeStep(HWND hwnd, uint32_t width, uint32_t height);
    int GetInitStep() const { return initStep_; }
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
    bool UploadBLASAttributes(int blasIndex, const float* normals, const float* uvs, uint32_t vertexCount, const float* colors = nullptr);
    bool UploadBLASPrimitiveMaterials(int blasIndex, const uint32_t* materialIds, uint32_t primitiveCount);
    bool UploadBLASPrimitiveYBounds(int blasIndex, const float* yBounds, uint32_t primitiveCount);
    void ClearGeometry();
    void UploadMaterialBuffer(const void* materials, uint32_t count);
    void UploadEmissiveTriangles(const float* data, uint32_t triangleCount);
    void UpdateTextureDescriptors(void* texManager);
    void UploadLightTree(const void* nodes, uint32_t nodeCount,
                         const void* emitters, uint32_t emitterCount);
    bool BuildTLASInstanced(const std::vector<vk::TLASInstance>& instances);
    // Update specific instance transforms in-place (TLAS refit, no full rebuild)
    bool UpdateInstanceTransforms(const uint32_t* indices, const float* transforms, uint32_t count);
    // GPU hair generation: upload parent keys, compute shader generates children + ribbons, builds BLAS
    int GenerateHairGPU(const float* parentKeys, uint32_t nParents,
                        uint32_t keysPerStrand, uint32_t childrenPerParent,
                        const float* emitterVerts, uint32_t nEmitterVerts,
                        const uint32_t* emitterTris, uint32_t nEmitterTris,
                        const float* emitterCDF,
                        float rootRadius, float tipFactor,
                        float clumpNoiseSize, float childRoundness,
                        float childLength, float avgSpacing,
                        float kinkAmplitude, float kinkFrequency,
                        float clumpFactor, float clumpShape,
                        float rough1, float rough1Size,
                        float rough2, float roughEnd,
                        uint32_t childMode,
                        float kinkShape, float kinkFlat, float kinkAmpRandom,
                        bool opaqueHair,
                        float childSizeRandom, bool useParentParticles,
                        bool precomputedStrands,
                        uint32_t blenderSeed,
                        const float* frandTable, uint32_t frandCount);
    void UpdateCamera(const CameraUBO& camera);
    void RenderFrameRT();

    // Interop
    HANDLE GetInteropNTHandle() const;
    bool ReadbackPixels(void* outData, uint32_t bufferSize);
    bool ReadbackHDRPixelsFloat(float* outData, uint32_t pixelCount);
    void SetDirectInterop(bool enabled) { useDirectInterop_ = enabled; }
    bool ImportD3D11Texture(HANDLE ntHandle, uint32_t width, uint32_t height);

    // GL interop (Vulkan → OpenGL zero-copy)
    bool InitGLInterop();
    void DrawGL(uint32_t w, uint32_t h);
    void WaitForReadBuffer();  // wait for the frame that wrote the GL read buffer
    bool IsRTReady() const { return rtReady_; }
    bool IsNircReady() const { return nirc_ != nullptr; }
    bool IsRTSupported() const;
    bool HasInterop() const { return interop_ != nullptr; }

    bool IsDLSSActive() const { return dlssActive_; }
    bool IsDLSSRRActive() const { return dlssRRActive_; }
    bool IsHybridGBufferReady() const { return hybridGBufferRendered_; }
    void GetRenderResolution(uint32_t* w, uint32_t* h) const { *w = renderWidth_; *h = renderHeight_; }
    void GetDisplayResolution(uint32_t* w, uint32_t* h) const { *w = width_; *h = height_; }
    uint32_t GetRenderWidth() const { return width_; }
    uint32_t GetRenderHeight() const { return height_; }
    float GetComputedExposure() const { return computedExposure_; }
    void ResetFrameIndex() { frameIndex_ = 0; }
    int GetActualDLSSQuality() const;

    Context* GetContext() const { return context_; }
    RTPipeline* GetRTPipeline() const { return rtPipeline_; }
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
    void InitRT_Remaining();  // RT pipeline + NRD (called after DLSS init in phased mode)

    Context* context_ = nullptr;
    Pipeline* pipeline_ = nullptr;
    Geometry* geometry_ = nullptr;
    Rasterizer* rasterizer_ = nullptr;

    // RT modules
    AccelStructureBuilder* accelBuilder_ = nullptr;
    RTPipeline* rtPipeline_ = nullptr;
    WavefrontPipeline* wavefrontPipeline_ = nullptr;
    Interop* interop_ = nullptr;
    class NircIntegration* nirc_ = nullptr;
    bool rtReady_ = false;
    bool useDirectInterop_ = false;
    int initStep_ = 0;          // phased init progress
    HWND initHwnd_ = nullptr;   // saved for phased init
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
    std::vector<vk::TLASInstance> cachedTLASInstances_; // full instance cache for partial updates

    // GPU hair compute pipeline
    VkPipeline hairComputePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout hairComputePipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout hairComputeDescSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool hairComputeDescPool_ = VK_NULL_HANDLE;
    VkDescriptorSet hairComputeDescSet_ = VK_NULL_HANDLE;
    bool hairComputeReady_ = false;
    bool CreateHairComputePipeline();
    void DestroyHairComputePipeline();

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
    // AgX 3D LUT
    VkImage agxLutImage_ = VK_NULL_HANDLE;
    VkDeviceMemory agxLutMemory_ = VK_NULL_HANDLE;
    VkImageView agxLutView_ = VK_NULL_HANDLE;
    VkSampler agxLutSampler_ = VK_NULL_HANDLE;
    std::string agxLutPath_;
    uint64_t agxLutStamp_ = 0;
    bool LoadAgXLut();
    void DestroyAgXLut();
    bool ReloadAgXLutIfChanged();

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
    float camWorldPos_[3] = {0};

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

    // Hybrid G-buffer rasterization pipeline
    VkRenderPass hybridRenderPass_ = VK_NULL_HANDLE;
    VkPipeline hybridPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout hybridPipelineLayout_ = VK_NULL_HANDLE;
    VkFramebuffer hybridFramebuffer_ = VK_NULL_HANDLE;
    VkImage hybridPrimIdImage_ = VK_NULL_HANDLE;
    VkDeviceMemory hybridPrimIdMemory_ = VK_NULL_HANDLE;
    VkImageView hybridPrimIdView_ = VK_NULL_HANDLE;
    VkImage hybridInstanceInfoImage_ = VK_NULL_HANDLE;
    VkDeviceMemory hybridInstanceInfoMemory_ = VK_NULL_HANDLE;
    VkImageView hybridInstanceInfoView_ = VK_NULL_HANDLE;
    VkImage hybridDepthImage_ = VK_NULL_HANDLE;       // R32_SFLOAT linear depth output
    VkDeviceMemory hybridDepthMemory_ = VK_NULL_HANDLE;
    VkImageView hybridDepthView_ = VK_NULL_HANDLE;
    VkImage hybridZBuffer_ = VK_NULL_HANDLE;           // D32_SFLOAT z-test only (not read by shader)
    VkDeviceMemory hybridZBufferMemory_ = VK_NULL_HANDLE;
    VkImageView hybridZBufferView_ = VK_NULL_HANDLE;
    bool hybridGBufferReady_ = false;     // pipeline + images created
    bool hybridGBufferRendered_ = false;  // raster pass has run at least once — safe for shader

    struct HybridPushConstants {
        float mvp[16];
        float camPos[4];  // xyz = camera world position, w = pad
        uint32_t instanceIndex;
        uint32_t blasIndex;
    };

    bool CreateHybridGBufferPipeline();
    void RecordHybridGBufferPass(VkCommandBuffer cmd);
    void ShutdownHybridGBuffer();

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

    // Surfel GI resolve compute pipeline
    VkPipeline surfelResolvePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout surfelResolvePipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout surfelResolveDescSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool surfelResolveDescPool_ = VK_NULL_HANDLE;
    VkDescriptorSet surfelResolveDescSet_ = VK_NULL_HANDLE;
    bool surfelResolveReady_ = false;
    bool CreateSurfelResolvePipeline();
    void UpdateSurfelResolveDescriptors();

    // Hair contour detection compute pass
    VkPipeline hairContourPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout hairContourPipelineLayout_ = VK_NULL_HANDLE;
    bool hairContourReady_ = false;
    bool CreateHairContourPipeline();
    void ShutdownHairContour();

    // GPU timestamp profiling
    // Slots: 0=start, 1=after hybrid raster, 2=after RT dispatch, 3=after hair contour,
    //        4=after denoise, 5=after composite, 6=after tonemap, 7=frame end
    VkQueryPool timestampQueryPool_ = VK_NULL_HANDLE;
    static constexpr uint32_t TS_COUNT = 8;
    enum TSSlot { TS_START=0, TS_HYBRID=1, TS_RT=2, TS_HAIR=3, TS_DENOISE=4, TS_COMPOSITE=5, TS_TONEMAP=6, TS_END=7 };
    float gpuStageMs_[8] = {};  // per-stage durations (computed from consecutive timestamps)
    float timestampPeriod_ = 0.0f;
    bool timestampReady_ = false;
    uint8_t tsWritten_ = 0;    // bitmask of slots written this frame
    void InitTimestampQueries();
    void ShutdownTimestampQueries();
    void ReadbackTimestamps();
    void WriteTimestamp(VkCommandBuffer cmd, uint32_t slot);
    void FillMissingTimestamps(VkCommandBuffer cmd);
public:
    // stage: 0=HybridRaster, 1=RTTrace, 2=HairContour, 3=Denoise, 4=Composite, 5=Tonemap, 6=Total
    float GetGpuStageMs(int stage) const { return (stage >= 0 && stage < 7) ? gpuStageMs_[stage] : 0.0f; }
private:

    // ImGui overlay
    VkRenderPass imguiRenderPass_ = VK_NULL_HANDLE;
    VkFramebuffer imguiFramebuffer_[2] = {};
    std::vector<VkFramebuffer> imguiSwapchainFramebuffers_;
    uint32_t imguiCurrentImageIndex_ = 0;
    VkDescriptorPool imguiDescriptorPool_ = VK_NULL_HANDLE;
    bool imguiReady_ = false;
};

} // namespace vk
} // namespace acpt
