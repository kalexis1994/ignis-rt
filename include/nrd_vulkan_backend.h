#pragma once

#ifdef ACPT_HAVE_NRD

#include <vulkan/vulkan.h>
#include "NRD.h"

namespace acpt {

// NRD Vulkan backend - integrates NRD denoiser with Vulkan rendering
// Uses Vulkan native resources (VkImage, VkImageView, VkCommandBuffer)

// Initialize NRD Vulkan backend
// - vkDevice: Vulkan logical device
// - vkPhysicalDevice: Vulkan physical device
// - width, height: render resolution
bool NRD_Vulkan_Init(VkDevice vkDevice, VkPhysicalDevice vkPhysicalDevice,
                     uint32_t width, uint32_t height);

// Shutdown NRD Vulkan backend and release resources
void NRD_Vulkan_Shutdown();

// Load NRD shader pipelines from instance
bool NRD_Vulkan_LoadShadersFromInstance(nrd::Instance* instance);

// Register G-buffer Vulkan images with NRD
// These are the input textures that NRD will denoise
struct NRDVulkanGBuffers {
    VkImage normalRoughness;    // RGBA16F - world-space normal (xyz) + roughness (w)
    VkImage viewDepth;          // R32F - view-space depth
    VkImage motionVectors;      // RG16F - screen-space motion vectors
    VkImage diffuseRadiance;    // RGBA16F - diffuse radiance (rgb) + hit distance (a)
    VkImage specularRadiance;   // RGBA16F - specular radiance (rgb) + hit distance (a)
};

bool NRD_Vulkan_RegisterGBuffers(const NRDVulkanGBuffers& gbuffers);

// Execute NRD denoising dispatches on Vulkan command buffer
// - cmdBuffer: Vulkan command buffer to record commands into
// - dispatchDescs: array of dispatch descriptors from nrd::GetComputeDispatches()
// - dispatchDescNum: number of dispatches
void NRD_Vulkan_Execute(VkCommandBuffer cmdBuffer, const nrd::DispatchDesc* dispatchDescs,
                        uint32_t dispatchDescNum);

// Get denoised output image (composed diffuse + specular)
// Returns VkImage that can be used for display
VkImage NRD_Vulkan_GetDenoisedOutput();

// Check if NRD Vulkan backend is ready
bool NRD_Vulkan_IsReady();

} // namespace acpt

#endif // ACPT_HAVE_NRD
