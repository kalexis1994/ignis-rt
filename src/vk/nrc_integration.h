// nrc_integration.h — NVIDIA Neural Radiance Cache integration for Vulkan
// Wraps the NRC SDK (NRC_Vulkan.dll) for use with the ignis-rt path tracer.

#pragma once

#ifdef IGNIS_HAVE_NRC

#include <vulkan/vulkan.h>
#include <NrcVk.h>

namespace acpt {
namespace vk {

class Context;  // forward

class NrcIntegration {
public:
    bool Initialize(Context* ctx, uint32_t renderWidth, uint32_t renderHeight,
                    uint32_t spp, uint32_t maxBounces,
                    const float sceneMin[3], const float sceneMax[3]);
    void Shutdown();

    // Per-frame calls (must be called in order)
    void BeginFrame(VkCommandBuffer cmd);
    void QueryAndTrain(VkCommandBuffer cmd);
    void Resolve(VkCommandBuffer cmd, VkImageView outputImage);
    void EndFrame(VkQueue queue);

    // Reconfigure (resolution change, scene change)
    void Reconfigure(uint32_t renderWidth, uint32_t renderHeight,
                     const float sceneMin[3], const float sceneMax[3]);

    // Sync settings if user changed bounces/SPP/scene bounds (cheap if unchanged)
    void SyncSettings(int maxBounces, int spp, const float sceneMin[3], const float sceneMax[3]);

    // Populate shader constants (call before path tracing dispatch)
    bool PopulateShaderConstants(NrcConstants& outConstants) const;

    // Get NRC buffers for shader binding
    const nrc::vulkan::Buffers* GetBuffers() const;

    bool IsReady() const { return nrcReady_; }

private:
    Context* context_ = nullptr;
    nrc::vulkan::Context* nrcContext_ = nullptr;
    nrc::ContextSettings contextSettings_ = {};
    nrc::FrameSettings frameSettings_ = {};
    bool nrcReady_ = false;
    uint32_t warmupFrames_ = 0;
};

} // namespace vk
} // namespace acpt

#endif // IGNIS_HAVE_NRC
