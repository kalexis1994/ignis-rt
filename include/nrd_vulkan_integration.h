#pragma once

// Enable Win32 platform support for Vulkan
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <vulkan/vulkan.h>
#include <cstdint>

namespace acpt {

// Direct Vulkan + NRD integration (no NRI abstraction layer)
// Based on nvpro-samples/vk_denoise_nrd example

// Initialize NRD with direct Vulkan integration
// Call this after Vulkan device is created
bool NRD_Vulkan_Init(VkPhysicalDevice vkPhysicalDevice,
                     VkDevice vkDevice,
                     VkQueue vkQueue,
                     VkCommandPool vkCommandPool,
                     uint32_t width, uint32_t height);

// Shutdown NRD
void NRD_Vulkan_Shutdown();

// G-buffer VkImages for NRD (application owns these)
struct NRD_GBufferImages {
    VkImage normalRoughness;    // RGBA16F
    VkImage viewDepth;          // R32F
    VkImage motionVectors;      // RG16F
    VkImage diffuseRadiance;    // RGBA16F - input noisy diffuse (irradiance with demodulation)
    VkImage specularRadiance;   // RGBA16F - input noisy specular (irradiance with demodulation)
    VkImage albedoBuffer;       // RGBA16F - material albedo for re-modulation after denoising
    VkImage penumbraBuffer;     // R16F — SIGMA IN_PENUMBRA
    VkImage diffuseConfidence;  // R8_UNORM — NRD confidence for diffuse
    VkImage specularConfidence; // R8_UNORM — NRD confidence for specular
};

// Register application G-buffer images with NRD
bool NRD_Vulkan_SetGBuffers(const NRD_GBufferImages& gbuffers);

// Execute denoising for current frame
// cmdBuffer: Vulkan command buffer to record NRD compute dispatches into
// frameIndex: monotonically increasing frame counter
// view, proj, viewPrev, projPrev: camera matrices (16 floats each, row-major)
// jitterX, jitterY: current frame TAA jitter offset in pixels
// jitterXprev, jitterYprev: previous frame TAA jitter offset in pixels
// frameDeltaMs: time since last frame in milliseconds (for temporal accumulation)
void NRD_Vulkan_Denoise(VkCommandBuffer cmdBuffer, uint32_t frameIndex,
                        const float view[16], const float proj[16],
                        const float viewPrev[16], const float projPrev[16],
                        float jitterX = 0.0f, float jitterY = 0.0f,
                        float jitterXprev = 0.0f, float jitterYprev = 0.0f,
                        float frameDeltaMs = 16.67f,
                        bool dlssActive = false);

// Get denoised output images (VkImageViews for shader access)
// Returns image views for denoised diffuse and specular radiance
void NRD_Vulkan_GetDenoisedOutputs(VkImageView& outDiffuse, VkImageView& outSpecular);

// Get albedo buffer view (for material re-modulation in composite)
VkImageView NRD_Vulkan_GetAlbedoBufferView();

// Get pool image by index (for layout transitions)
VkImage NRD_Vulkan_GetPoolImage(uint32_t poolIndex);

// Get last diffuse pool index used by NRD (DEPRECATED - outputs are now dedicated textures)
uint32_t NRD_Vulkan_GetDiffusePoolIndex();

// Get dedicated output images (for readback / debug)
VkImage NRD_Vulkan_GetOutputDiffuseImage();
VkImage NRD_Vulkan_GetOutputSpecularImage();

// Get raw G-buffer input views (for debug: bypass NRD)
void NRD_Vulkan_GetRawInputViews(VkImageView& outDiffuse, VkImageView& outSpecular);

// Get denoised shadow output (from SIGMA)
void NRD_Vulkan_GetDenoisedShadow(VkImageView& outShadow);
VkImage NRD_Vulkan_GetOutputShadowImage();

// Set sun direction for SIGMA (call before Denoise each frame)
void NRD_Vulkan_SetSunDirection(float x, float y, float z);

// Check if NRD is ready
bool NRD_Vulkan_IsReady();

} // namespace acpt
