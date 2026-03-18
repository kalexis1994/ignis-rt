// NRD stub — compiled when IGNIS_USE_NRD=OFF
// Provides no-op implementations of NRD functions referenced by vk_renderer.cpp

#ifndef ACPT_HAVE_NRD

#include <vulkan/vulkan.h>
#include <cstdint>

namespace acpt {

bool NRD_Vulkan_Initialize(VkPhysicalDevice physDevice, VkDevice device,
                            VkQueue queue, VkCommandPool commandPool,
                            uint32_t width, uint32_t height) {
    return false;
}

void NRD_Vulkan_Denoise(VkCommandBuffer cmd, uint32_t frameIndex,
                         const float* viewMatrix, const float* projMatrix,
                         const float* viewMatrixPrev, const float* projMatrixPrev,
                         float jitterX, float jitterY,
                         float prevJitterX, float prevJitterY,
                         float frameDeltaMs, bool dlssActive) {
    // No-op
}

void NRD_Vulkan_SetSunDirection(float x, float y, float z) {
    // No-op
}

void NRD_Vulkan_Shutdown() {
    // No-op
}

VkImageView NRD_Vulkan_GetDiffuseView() { return VK_NULL_HANDLE; }
VkImageView NRD_Vulkan_GetSpecularView() { return VK_NULL_HANDLE; }
VkImageView NRD_Vulkan_GetShadowView() { return VK_NULL_HANDLE; }

} // namespace acpt

#endif // !ACPT_HAVE_NRD
