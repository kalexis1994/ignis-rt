#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

namespace acpt {
namespace vk {

class Context;
class RTPipeline;

/// Wavefront path tracing pipeline — replaces monolithic raygen with
/// multiple compute kernels for better GPU occupancy.
///
/// Kernels: K0(camera rays) → K1(intersect) → K2(shade+NEE) →
///          K3(shadow intersect) → K4(accumulate) → K5(output)
/// Bounced K1-K4 loop with path compaction between iterations.
class WavefrontPipeline {
public:
    bool Initialize(Context* context, RTPipeline* rtPipeline,
                    uint32_t width, uint32_t height, uint32_t maxBounces);
    void Shutdown();

    /// Record all wavefront dispatches into the command buffer.
    /// Uses the same descriptor set 0 as RTPipeline for scene data.
    void RecordDispatch(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                        VkDescriptorSet sceneDescSet, uint32_t maxBounces,
                        uint32_t spp = 1);

    bool IsReady() const { return ready_; }
    void SetMaterialSort(bool enabled) { materialSort_ = enabled; }

private:
    bool CreateBuffers(uint32_t pixelCount);
    bool CreateDescriptorSet();
    bool CreatePipelines();
    bool LoadComputeShader(const char* path, VkShaderModule* outModule);

    Context* context_ = nullptr;
    RTPipeline* rtPipeline_ = nullptr;
    bool ready_ = false;
    bool materialSort_ = false;
    uint32_t maxPixels_ = 0;
    uint32_t frameIndex_ = 0;

    // Wavefront SSBO buffers (PathState is double-buffered to avoid read/write race)
    VkBuffer pathStateBuffer_[2] = {};                 // PathState[] ping-pong
    VkDeviceMemory pathStateMemory_[2] = {};
    uint32_t pathStateCurrent_ = 0;                    // index of current read buffer
    VkBuffer hitResultBuffer_ = VK_NULL_HANDLE;        // HitResult[]
    VkDeviceMemory hitResultMemory_ = VK_NULL_HANDLE;
    VkBuffer shadowRayBuffer_ = VK_NULL_HANDLE;        // ShadowRay[]
    VkDeviceMemory shadowRayMemory_ = VK_NULL_HANDLE;
    VkBuffer pixelRadianceBuffer_ = VK_NULL_HANDLE;    // PixelRadiance[]
    VkDeviceMemory pixelRadianceMemory_ = VK_NULL_HANDLE;
    VkBuffer primaryGBufBuffer_ = VK_NULL_HANDLE;      // PrimaryGBuffer[]
    VkDeviceMemory primaryGBufMemory_ = VK_NULL_HANDLE;
    VkBuffer countersBuffer_ = VK_NULL_HANDLE;          // WavefrontCounters
    VkDeviceMemory countersMemory_ = VK_NULL_HANDLE;
    VkBuffer indirectDispatchBuffer_ = VK_NULL_HANDLE;  // VkDispatchIndirectCommand[3]
    VkDeviceMemory indirectDispatchMemory_ = VK_NULL_HANDLE;
    VkBuffer sharcStateBuffer_ = VK_NULL_HANDLE;       // SharcState[] per-pixel (persists across bounces)
    VkDeviceMemory sharcStateMemory_ = VK_NULL_HANDLE;

    // Descriptor sets for ping-pong (2 sets, no host updates during recording)
    // Set A: binding 0 = buffer[0] (read), binding 7 = buffer[1] (write)
    // Set B: binding 0 = buffer[1] (read), binding 7 = buffer[0] (write)
    VkDescriptorSetLayout wfDescSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool wfDescPool_ = VK_NULL_HANDLE;
    VkDescriptorSet wfDescSet_[2] = {};  // [0] = A, [1] = B

    // Compute pipelines (K0-K5 + compact)
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;  // shared layout for all kernels
    VkPipeline pipelineK0_ = VK_NULL_HANDLE;  // camera rays
    VkPipeline pipelineK1_ = VK_NULL_HANDLE;  // intersect
    VkPipeline pipelineK2_ = VK_NULL_HANDLE;  // shade + NEE
    VkPipeline pipelineK3_ = VK_NULL_HANDLE;  // shadow intersect
    VkPipeline pipelineK4_ = VK_NULL_HANDLE;  // accumulate
    VkPipeline pipelineK5_ = VK_NULL_HANDLE;  // output
    VkPipeline pipelineCompact_ = VK_NULL_HANDLE;  // prepare indirect + compact
    VkPipeline pipelineSortCount_ = VK_NULL_HANDLE;   // material sort phase 1
    VkPipeline pipelineSortPrefix_ = VK_NULL_HANDLE;  // material sort phase 2
    VkPipeline pipelineSortScatter_ = VK_NULL_HANDLE; // material sort phase 3

    static constexpr uint32_t WORKGROUP_SIZE = 256;
    static constexpr uint32_t MAX_SHADOW_RAYS_PER_PATH = 10; // sun + 8 lights + emissive
};

} // namespace vk
} // namespace acpt
