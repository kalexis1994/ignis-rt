// Direct Vulkan + NRD integration
// Based on nvpro-samples/vk_denoise_nrd example
#include "nrd_vulkan_integration.h"
#include "ignis_log.h"
#include "ignis_config.h"
#include <vulkan/vulkan.h>

#ifdef ACPT_HAVE_NRD

#include "NRD.h"
#include <vector>
#include <cstring>

namespace acpt {

// External config from vk_renderer.cpp
extern PathTracerConfig g_config;

// NRD state
static nrd::Instance* g_nrdInstance = nullptr;
static VkPhysicalDevice g_vkPhysicalDevice = VK_NULL_HANDLE;
static VkDevice g_vkDevice = VK_NULL_HANDLE;
static VkQueue g_vkQueue = VK_NULL_HANDLE;
static VkCommandPool g_vkCommandPool = VK_NULL_HANDLE;
static bool g_initialized = false;

// Frame dimensions
static uint32_t g_width = 0;
static uint32_t g_height = 0;

// NRD internal textures (permanent and transient pools)
struct NRDTexture {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
};
static std::vector<NRDTexture> g_nrdPermanentPool;
static std::vector<NRDTexture> g_nrdTransientPool;

// Application G-buffers (wrapped for NRD)
static VkImageView g_normalRoughnessView = VK_NULL_HANDLE;
static VkImageView g_viewDepthView = VK_NULL_HANDLE;
static VkImageView g_motionVectorsView = VK_NULL_HANDLE;
static VkImageView g_diffuseRadianceView = VK_NULL_HANDLE;
static VkImageView g_specularRadianceView = VK_NULL_HANDLE;
static VkImageView g_albedoBufferView = VK_NULL_HANDLE;  // For material demodulation

// NRD output textures (dedicated, not from pool)
// These are the actual denoised outputs that NRD writes to
static NRDTexture g_nrdOutputDiffuse;
static NRDTexture g_nrdOutputSpecular;

// SIGMA shadow denoiser
static VkImageView g_penumbraView = VK_NULL_HANDLE;
static NRDTexture g_nrdOutputShadow;      // dedicated SIGMA output (RGBA8_UNORM)
static VkImageView g_denoisedShadowView = VK_NULL_HANDLE;
static float g_sunDirection[3] = {0.4f, 0.85f, 0.35f};

// NRD confidence masks
static VkImageView g_diffConfidenceView = VK_NULL_HANDLE;
static VkImageView g_specConfidenceView = VK_NULL_HANDLE;

// NRD output tracking (for composite shader)
static VkImageView g_denoisedDiffuseView = VK_NULL_HANDLE;
static VkImageView g_denoisedSpecularView = VK_NULL_HANDLE;

// NRD compute pipelines and descriptors
struct NRDPipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout resourcesDescriptorSetLayout = VK_NULL_HANDLE;  // Set 0: textures
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};
static std::vector<NRDPipeline> g_nrdPipelines;

// Global descriptor set for constant buffer (Set 1, shared across all pipelines)
static VkDescriptorSetLayout g_constantBufferDescriptorSetLayout = VK_NULL_HANDLE;
static VkDescriptorPool g_constantBufferDescriptorPool = VK_NULL_HANDLE;
static VkDescriptorSet g_constantBufferDescriptorSet = VK_NULL_HANDLE;

// Constant buffer for NRD
static VkBuffer g_constantBuffer = VK_NULL_HANDLE;
static VkDeviceMemory g_constantBufferMemory = VK_NULL_HANDLE;
static void* g_constantBufferMapped = nullptr;
static uint32_t g_constantBufferSize = 0;

// NRD samplers (immutable, bound in Set 1 alongside constant buffer)
static VkSampler g_nrdSamplerNearest = VK_NULL_HANDLE;  // Sampler::NEAREST_CLAMP
static VkSampler g_nrdSamplerLinear = VK_NULL_HANDLE;   // Sampler::LINEAR_CLAMP

// ========== Helper Functions ==========

// Convert NRD Format to VkFormat
static VkFormat NRDFormatToVulkan(nrd::Format format) {
    switch (format) {
        case nrd::Format::R8_UNORM:           return VK_FORMAT_R8_UNORM;
        case nrd::Format::R8_SNORM:           return VK_FORMAT_R8_SNORM;
        case nrd::Format::R8_UINT:            return VK_FORMAT_R8_UINT;
        case nrd::Format::R8_SINT:            return VK_FORMAT_R8_SINT;
        case nrd::Format::RG8_UNORM:          return VK_FORMAT_R8G8_UNORM;
        case nrd::Format::RG8_SNORM:          return VK_FORMAT_R8G8_SNORM;
        case nrd::Format::RG8_UINT:           return VK_FORMAT_R8G8_UINT;
        case nrd::Format::RG8_SINT:           return VK_FORMAT_R8G8_SINT;
        case nrd::Format::RGBA8_UNORM:        return VK_FORMAT_R8G8B8A8_UNORM;
        case nrd::Format::RGBA8_SNORM:        return VK_FORMAT_R8G8B8A8_SNORM;
        case nrd::Format::RGBA8_UINT:         return VK_FORMAT_R8G8B8A8_UINT;
        case nrd::Format::RGBA8_SINT:         return VK_FORMAT_R8G8B8A8_SINT;
        case nrd::Format::RGBA8_SRGB:         return VK_FORMAT_R8G8B8A8_SRGB;
        case nrd::Format::R16_UNORM:          return VK_FORMAT_R16_UNORM;
        case nrd::Format::R16_SNORM:          return VK_FORMAT_R16_SNORM;
        case nrd::Format::R16_UINT:           return VK_FORMAT_R16_UINT;
        case nrd::Format::R16_SINT:           return VK_FORMAT_R16_SINT;
        case nrd::Format::R16_SFLOAT:         return VK_FORMAT_R16_SFLOAT;
        case nrd::Format::RG16_UNORM:         return VK_FORMAT_R16G16_UNORM;
        case nrd::Format::RG16_SNORM:         return VK_FORMAT_R16G16_SNORM;
        case nrd::Format::RG16_UINT:          return VK_FORMAT_R16G16_UINT;
        case nrd::Format::RG16_SINT:          return VK_FORMAT_R16G16_SINT;
        case nrd::Format::RG16_SFLOAT:        return VK_FORMAT_R16G16_SFLOAT;
        case nrd::Format::RGBA16_UNORM:       return VK_FORMAT_R16G16B16A16_UNORM;
        case nrd::Format::RGBA16_SNORM:       return VK_FORMAT_R16G16B16A16_SNORM;
        case nrd::Format::RGBA16_UINT:        return VK_FORMAT_R16G16B16A16_UINT;
        case nrd::Format::RGBA16_SINT:        return VK_FORMAT_R16G16B16A16_SINT;
        case nrd::Format::RGBA16_SFLOAT:      return VK_FORMAT_R16G16B16A16_SFLOAT;
        case nrd::Format::R32_UINT:           return VK_FORMAT_R32_UINT;
        case nrd::Format::R32_SINT:           return VK_FORMAT_R32_SINT;
        case nrd::Format::R32_SFLOAT:         return VK_FORMAT_R32_SFLOAT;
        case nrd::Format::RG32_UINT:          return VK_FORMAT_R32G32_UINT;
        case nrd::Format::RG32_SINT:          return VK_FORMAT_R32G32_SINT;
        case nrd::Format::RG32_SFLOAT:        return VK_FORMAT_R32G32_SFLOAT;
        case nrd::Format::RGB32_UINT:         return VK_FORMAT_R32G32B32_UINT;
        case nrd::Format::RGB32_SINT:         return VK_FORMAT_R32G32B32_SINT;
        case nrd::Format::RGB32_SFLOAT:       return VK_FORMAT_R32G32B32_SFLOAT;
        case nrd::Format::RGBA32_UINT:        return VK_FORMAT_R32G32B32A32_UINT;
        case nrd::Format::RGBA32_SINT:        return VK_FORMAT_R32G32B32A32_SINT;
        case nrd::Format::RGBA32_SFLOAT:      return VK_FORMAT_R32G32B32A32_SFLOAT;
        case nrd::Format::R10_G10_B10_A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case nrd::Format::R10_G10_B10_A2_UINT:  return VK_FORMAT_A2B10G10R10_UINT_PACK32;
        case nrd::Format::R11_G11_B10_UFLOAT:   return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case nrd::Format::R9_G9_B9_E5_UFLOAT:   return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
        default:
            Log(L"[NRD-Vulkan] WARNING: Unknown NRD format %d, defaulting to RGBA16F\n", (int)format);
            return VK_FORMAT_R16G16B16A16_SFLOAT;
    }
}

// Find memory type index for allocation
static uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(g_vkPhysicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    Log(L"[NRD-Vulkan] ERROR: Failed to find suitable memory type\n");
    return 0;
}

// Create a Vulkan texture for NRD
static bool CreateNRDTexture(const nrd::TextureDesc& desc, uint32_t baseWidth, uint32_t baseHeight, NRDTexture& outTexture) {
    VkFormat vkFormat = NRDFormatToVulkan(desc.format);
    uint32_t width = baseWidth / desc.downsampleFactor;
    uint32_t height = baseHeight / desc.downsampleFactor;

    outTexture.format = vkFormat;
    outTexture.width = width;
    outTexture.height = height;

    // Create VkImage
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = vkFormat;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult result = vkCreateImage(g_vkDevice, &imageInfo, nullptr, &outTexture.image);
    if (result != VK_SUCCESS) {
        Log(L"[NRD-Vulkan] ERROR: Failed to create VkImage (result=%d)\n", result);
        return false;
    }

    // Allocate memory
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(g_vkDevice, outTexture.image, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits,
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    result = vkAllocateMemory(g_vkDevice, &allocInfo, nullptr, &outTexture.memory);
    if (result != VK_SUCCESS) {
        Log(L"[NRD-Vulkan] ERROR: Failed to allocate VkDeviceMemory (result=%d)\n", result);
        vkDestroyImage(g_vkDevice, outTexture.image, nullptr);
        return false;
    }

    result = vkBindImageMemory(g_vkDevice, outTexture.image, outTexture.memory, 0);
    if (result != VK_SUCCESS) {
        Log(L"[NRD-Vulkan] ERROR: Failed to bind image memory (result=%d)\n", result);
        vkFreeMemory(g_vkDevice, outTexture.memory, nullptr);
        vkDestroyImage(g_vkDevice, outTexture.image, nullptr);
        return false;
    }

    // Create VkImageView
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = outTexture.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = vkFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    result = vkCreateImageView(g_vkDevice, &viewInfo, nullptr, &outTexture.view);
    if (result != VK_SUCCESS) {
        Log(L"[NRD-Vulkan] ERROR: Failed to create VkImageView (result=%d)\n", result);
        vkFreeMemory(g_vkDevice, outTexture.memory, nullptr);
        vkDestroyImage(g_vkDevice, outTexture.image, nullptr);
        return false;
    }

    return true;
}

// Destroy NRD texture
static void DestroyNRDTexture(NRDTexture& texture) {
    if (texture.view != VK_NULL_HANDLE) {
        vkDestroyImageView(g_vkDevice, texture.view, nullptr);
        texture.view = VK_NULL_HANDLE;
    }
    if (texture.image != VK_NULL_HANDLE) {
        vkDestroyImage(g_vkDevice, texture.image, nullptr);
        texture.image = VK_NULL_HANDLE;
    }
    if (texture.memory != VK_NULL_HANDLE) {
        vkFreeMemory(g_vkDevice, texture.memory, nullptr);
        texture.memory = VK_NULL_HANDLE;
    }
}

// Transition image layout (simple immediate transition)
static void TransitionImageLayoutImmediate(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
    // Create one-time command buffer
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = g_vkCommandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuffer;
    vkAllocateCommandBuffers(g_vkDevice, &allocInfo, &cmdBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmdBuffer,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmdBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    vkQueueSubmit(g_vkQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(g_vkQueue);

    vkFreeCommandBuffers(g_vkDevice, g_vkCommandPool, 1, &cmdBuffer);
}

// Map NRD ResourceDesc to VkImageView
static VkImageView GetResourceView(const nrd::ResourceDesc& resource) {
    switch (resource.type) {
        // Permanent pool textures
        case nrd::ResourceType::PERMANENT_POOL:
            if (resource.indexInPool < g_nrdPermanentPool.size()) {
                return g_nrdPermanentPool[resource.indexInPool].view;
            }
            break;

        // Transient pool textures
        case nrd::ResourceType::TRANSIENT_POOL:
            if (resource.indexInPool < g_nrdTransientPool.size()) {
                return g_nrdTransientPool[resource.indexInPool].view;
            }
            break;

        // Application G-buffer inputs
        case nrd::ResourceType::IN_NORMAL_ROUGHNESS:
            return g_normalRoughnessView;
        case nrd::ResourceType::IN_VIEWZ:
            return g_viewDepthView;
        case nrd::ResourceType::IN_MV:
            return g_motionVectorsView;
        case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST: {
            static bool loggedDiffInput = false;
            if (!loggedDiffInput) {
                Log(L"[NRD-Debug] IN_DIFF_RADIANCE_HITDIST requested -> returning view %p\n", (void*)g_diffuseRadianceView);
                loggedDiffInput = true;
            }
            return g_diffuseRadianceView;
        }
        case nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST:
            return g_specularRadianceView;
        case nrd::ResourceType::IN_BASECOLOR_METALNESS:
            // Return albedo buffer for base color (we don't use metalness, just albedo)
            return g_albedoBufferView;

        // NRD confidence masks
        case nrd::ResourceType::IN_DIFF_CONFIDENCE:
            return g_diffConfidenceView;
        case nrd::ResourceType::IN_SPEC_CONFIDENCE:
            return g_specConfidenceView;

        // SIGMA shadow inputs/outputs
        case nrd::ResourceType::IN_PENUMBRA:
            return g_penumbraView;
        case nrd::ResourceType::OUT_SHADOW_TRANSLUCENCY:
            return g_nrdOutputShadow.view;  // both output and history input

        // Output textures - return DEDICATED output textures (not from pool!)
        // NRD-Sample creates separate textures for outputs with RGBA16_SFLOAT format
        case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST: {
            static bool logged = false;
            if (!logged) {
                Log(L"[NRD-Debug] OUT_DIFF requested -> returning dedicated view %p (image=%p)\n",
                    g_nrdOutputDiffuse.view, g_nrdOutputDiffuse.image);
                logged = true;
            }
            return g_nrdOutputDiffuse.view;
        }
        case nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST: {
            static bool logged = false;
            if (!logged) {
                Log(L"[NRD-Debug] OUT_SPEC requested -> returning dedicated view %p (image=%p)\n",
                    g_nrdOutputSpecular.view, g_nrdOutputSpecular.image);
                logged = true;
            }
            return g_nrdOutputSpecular.view;
        }

        default:
            Log(L"[NRD-Vulkan] WARNING: Unknown resource type %d\n", (int)resource.type);
            break;
    }

    return VK_NULL_HANDLE;
}

// ========== Public API ==========

bool NRD_Vulkan_Init(VkPhysicalDevice vkPhysicalDevice,
                     VkDevice vkDevice,
                     VkQueue vkQueue,
                     VkCommandPool vkCommandPool,
                     uint32_t width, uint32_t height) {
    Log(L"[NRD] Initializing NRD (%ux%u)...\n", width, height);
    g_vkPhysicalDevice = vkPhysicalDevice;
    g_vkDevice = vkDevice;
    g_vkQueue = vkQueue;
    g_vkCommandPool = vkCommandPool;
    g_width = width;
    g_height = height;

    // Use RELAX_DIFFUSE_SPECULAR + SIGMA_SHADOW
    // ReLAX converges in 1-2 frames (vs ~30 for REBLUR) — critical for constant camera motion
    nrd::DenoiserDesc denoiserDescs[] = {
        {nrd::Identifier(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR), nrd::Denoiser::RELAX_DIFFUSE_SPECULAR},
        {nrd::Identifier(nrd::Denoiser::SIGMA_SHADOW), nrd::Denoiser::SIGMA_SHADOW}
    };

    nrd::InstanceCreationDesc instanceDesc = {};
    instanceDesc.denoisers = denoiserDescs;
    instanceDesc.denoisersNum = 2;

    nrd::Result nrdResult = nrd::CreateInstance(instanceDesc, g_nrdInstance);
    if (nrdResult != nrd::Result::SUCCESS || !g_nrdInstance) {
        Log(L"[NRD-Vulkan] ERROR: Failed to create NRD instance (result=%d)\n", (int)nrdResult);
        return false;
    }

    Log(L"[NRD-Vulkan] NRD instance created successfully\n");

    // Query NRD resource requirements
    const nrd::InstanceDesc* nrdDesc = nullptr;
    try {
        nrdDesc = nrd::GetInstanceDesc(*g_nrdInstance);
    } catch (...) {
        Log(L"[NRD-Vulkan] ERROR: Exception caught when calling GetInstanceDesc!\n");
        nrd::DestroyInstance(*g_nrdInstance);
        g_nrdInstance = nullptr;
        return false;
    }

    if (!nrdDesc) {
        Log(L"[NRD-Vulkan] ERROR: Failed to get NRD instance descriptor (returned NULL)\n");
        nrd::DestroyInstance(*g_nrdInstance);
        g_nrdInstance = nullptr;
        return false;
    }

    Log(L"[NRD-Vulkan] NRD requires:\n");
    Log(L"  - Permanent pool: %u textures\n", nrdDesc->permanentPoolSize);
    Log(L"  - Transient pool: %u textures\n", nrdDesc->transientPoolSize);
    Log(L"  - Pipelines: %u\n", nrdDesc->pipelinesNum);
    Log(L"  - Constant buffer: %u bytes\n", nrdDesc->constantBufferMaxDataSize);
    Log(L"  - Space indices: CB+samplers=%u, resources=%u\n",
        nrdDesc->constantBufferAndSamplersSpaceIndex, nrdDesc->resourcesSpaceIndex);
    Log(L"  - Samplers: %u (base register=%u)\n", nrdDesc->samplersNum, nrdDesc->samplersBaseRegisterIndex);
    Log(L"  - CB register index: %u\n", nrdDesc->constantBufferRegisterIndex);
    Log(L"  - Resources base register: %u\n", nrdDesc->resourcesBaseRegisterIndex);

    // Get library desc for encoding info
    const nrd::LibraryDesc* earlyLibDesc = nrd::GetLibraryDesc();
    Log(L"  - Normal encoding: %u, Roughness encoding: %u\n",
        (uint32_t)earlyLibDesc->normalEncoding, (uint32_t)earlyLibDesc->roughnessEncoding);

    // Create permanent texture pool
    Log(L"[NRD-Vulkan] Permanent pool textures:\n");
    g_nrdPermanentPool.resize(nrdDesc->permanentPoolSize);
    for (uint32_t i = 0; i < nrdDesc->permanentPoolSize; i++) {
        const nrd::TextureDesc& texDesc = nrdDesc->permanentPool[i];
        VkFormat vkFormat = NRDFormatToVulkan(texDesc.format);

        Log(L"  pool[%u]: nrdFormat=%d, vkFormat=%d, size=%ux%u\n",
            i, (int)texDesc.format, (int)vkFormat,
            g_width / texDesc.downsampleFactor, g_height / texDesc.downsampleFactor);

        if (!CreateNRDTexture(texDesc, width, height, g_nrdPermanentPool[i])) {
            Log(L"[NRD-Vulkan] ERROR: Failed to create permanent texture %u\n", i);
            NRD_Vulkan_Shutdown();
            return false;
        }
    }
    Log(L"[NRD-Vulkan] Created %u permanent textures\n", nrdDesc->permanentPoolSize);

    // Transition all permanent pool textures to GENERAL layout
    for (uint32_t i = 0; i < nrdDesc->permanentPoolSize; i++) {
        TransitionImageLayoutImmediate(g_nrdPermanentPool[i].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    }
    Log(L"[NRD-Vulkan] Transitioned permanent pool textures to GENERAL layout\n");

    // Create transient texture pool
    g_nrdTransientPool.resize(nrdDesc->transientPoolSize);
    for (uint32_t i = 0; i < nrdDesc->transientPoolSize; i++) {
        if (!CreateNRDTexture(nrdDesc->transientPool[i], width, height, g_nrdTransientPool[i])) {
            Log(L"[NRD-Vulkan] ERROR: Failed to create transient texture %u\n", i);
            NRD_Vulkan_Shutdown();
            return false;
        }
    }
    Log(L"[NRD-Vulkan] Created %u transient textures\n", nrdDesc->transientPoolSize);

    // Transition all transient pool textures to GENERAL layout
    for (uint32_t i = 0; i < nrdDesc->transientPoolSize; i++) {
        TransitionImageLayoutImmediate(g_nrdTransientPool[i].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    }
    Log(L"[NRD-Vulkan] Transitioned transient pool textures to GENERAL layout\n");

    // Create dedicated output textures (NOT from pool)
    // NRD-Sample creates separate RGBA16F textures for outputs instead of using pool
    Log(L"[NRD-Vulkan] Creating dedicated NRD output textures (RGBA16F)...\n");

    nrd::TextureDesc outputDesc = {};
    outputDesc.format = nrd::Format::RGBA16_SFLOAT;
    outputDesc.downsampleFactor = 1;  // Full resolution

    if (!CreateNRDTexture(outputDesc, width, height, g_nrdOutputDiffuse)) {
        Log(L"[NRD-Vulkan] ERROR: Failed to create output diffuse texture\n");
        NRD_Vulkan_Shutdown();
        return false;
    }
    TransitionImageLayoutImmediate(g_nrdOutputDiffuse.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    Log(L"[NRD-Vulkan] Created OUT_DIFF texture: %ux%u RGBA16F\n", width, height);

    if (!CreateNRDTexture(outputDesc, width, height, g_nrdOutputSpecular)) {
        Log(L"[NRD-Vulkan] ERROR: Failed to create output specular texture\n");
        NRD_Vulkan_Shutdown();
        return false;
    }
    TransitionImageLayoutImmediate(g_nrdOutputSpecular.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    Log(L"[NRD-Vulkan] Created OUT_SPEC texture: %ux%u RGBA16F\n", width, height);

    // SIGMA shadow output — RGBA8_UNORM (R=shadow, GBA=translucency unused)
    // OUT_SHADOW_TRANSLUCENCY is dual-use: NRD reads history AND writes result
    nrd::TextureDesc shadowOutputDesc = {};
    shadowOutputDesc.format = nrd::Format::RGBA8_UNORM;
    shadowOutputDesc.downsampleFactor = 1;
    if (!CreateNRDTexture(shadowOutputDesc, width, height, g_nrdOutputShadow)) {
        Log(L"[NRD-Vulkan] ERROR: Failed to create output shadow texture\n");
        NRD_Vulkan_Shutdown();
        return false;
    }
    TransitionImageLayoutImmediate(g_nrdOutputShadow.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    g_denoisedShadowView = g_nrdOutputShadow.view;
    Log(L"[NRD-Vulkan] Created OUT_SHADOW texture: %ux%u RGBA8\n", width, height);

    // Pre-set output views so composite can bind them immediately
    g_denoisedDiffuseView = g_nrdOutputDiffuse.view;
    g_denoisedSpecularView = g_nrdOutputSpecular.view;

    // Get SPIRV binding offsets (needed for descriptor set layouts)
    const nrd::LibraryDesc* libDesc = nrd::GetLibraryDesc();
    const uint32_t samplerBindingOffset = libDesc->spirvBindingOffsets.samplerOffset;
    const uint32_t constantBufferBindingOffset = libDesc->spirvBindingOffsets.constantBufferOffset;
    const uint32_t texturesBindingOffset = libDesc->spirvBindingOffsets.textureOffset;
    const uint32_t storageTextureAndBufferOffset = libDesc->spirvBindingOffsets.storageTextureAndBufferOffset;

    Log(L"[NRD] SPIRV offsets: sampler=%u, cb=%u, tex=%u, storage=%u\n",
        samplerBindingOffset, constantBufferBindingOffset, texturesBindingOffset, storageTextureAndBufferOffset);

    // Create NRD samplers: NEAREST_CLAMP and LINEAR_CLAMP
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        si.magFilter = VK_FILTER_NEAREST;
        si.minFilter = VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        vkCreateSampler(g_vkDevice, &si, nullptr, &g_nrdSamplerNearest);

        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        vkCreateSampler(g_vkDevice, &si, nullptr, &g_nrdSamplerLinear);

        Log(L"[NRD] Created 2 samplers (nearest=%p, linear=%p)\n", g_nrdSamplerNearest, g_nrdSamplerLinear);
    }

    // Create GLOBAL descriptor set layout for constant buffer + samplers (Set 1)
    // NRD requires 2 immutable samplers (NEAREST_CLAMP, LINEAR_CLAMP) in the same
    // descriptor set as the constant buffer.
    VkSampler immutableSamplers[2] = { g_nrdSamplerNearest, g_nrdSamplerLinear };

    VkDescriptorSetLayoutBinding set1Bindings[3] = {};
    // Sampler 0: NEAREST_CLAMP
    set1Bindings[0].binding = samplerBindingOffset + 0;
    set1Bindings[0].descriptorCount = 1;
    set1Bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    set1Bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    set1Bindings[0].pImmutableSamplers = &immutableSamplers[0];
    // Sampler 1: LINEAR_CLAMP
    set1Bindings[1].binding = samplerBindingOffset + 1;
    set1Bindings[1].descriptorCount = 1;
    set1Bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    set1Bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    set1Bindings[1].pImmutableSamplers = &immutableSamplers[1];
    // Constant buffer
    set1Bindings[2].binding = constantBufferBindingOffset;
    set1Bindings[2].descriptorCount = 1;
    set1Bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    set1Bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo cbLayoutInfo = {};
    cbLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    cbLayoutInfo.bindingCount = 3;
    cbLayoutInfo.pBindings = set1Bindings;

    VkResult res = vkCreateDescriptorSetLayout(g_vkDevice, &cbLayoutInfo, nullptr, &g_constantBufferDescriptorSetLayout);
    if (res != VK_SUCCESS) {
        Log(L"[NRD-Vulkan] ERROR: Failed to create Set 1 descriptor set layout (result=%d)\n", res);
        NRD_Vulkan_Shutdown();
        return false;
    }
    Log(L"[NRD] Created Set 1 layout (constant buffer + 2 immutable samplers)\n");

    // Create compute pipelines from SPIR-V
    g_nrdPipelines.resize(nrdDesc->pipelinesNum);
    for (uint32_t i = 0; i < nrdDesc->pipelinesNum; i++) {
        const nrd::PipelineDesc& pipelineDesc = nrdDesc->pipelines[i];

        // Create shader module from SPIR-V bytecode
        VkShaderModuleCreateInfo shaderInfo = {};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = pipelineDesc.computeShaderSPIRV.size;
        shaderInfo.pCode = reinterpret_cast<const uint32_t*>(pipelineDesc.computeShaderSPIRV.bytecode);

        Log(L"[NRD-Vulkan] Creating shader module %u (SPIR-V size=%u bytes, ptr=%p)\n",
            i, shaderInfo.codeSize, shaderInfo.pCode);

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        VkResult result = vkCreateShaderModule(g_vkDevice, &shaderInfo, nullptr, &shaderModule);
        if (result != VK_SUCCESS) {
            Log(L"[NRD-Vulkan] ERROR: Failed to create shader module %u (result=%d)\n", i, result);
            NRD_Vulkan_Shutdown();
            return false;
        }

        // Create descriptor set layout for this pipeline
        std::vector<VkDescriptorSetLayoutBinding> bindings;

        // Add resource bindings using SPIRV offsets
        // CRITICAL: Must match the binding indices used by NRD shaders
        for (uint32_t r = 0; r < pipelineDesc.resourceRangesNum; r++) {
            const nrd::ResourceRangeDesc& range = pipelineDesc.resourceRanges[r];

            // Calculate base binding index for this range using SPIRV offsets
            bool isStorage = (range.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE);
            uint32_t rangeBaseBindingIndex = isStorage ? storageTextureAndBufferOffset : texturesBindingOffset;

            // Create a separate binding for each descriptor in the range
            for (uint32_t d = 0; d < range.descriptorsNum; d++) {
                VkDescriptorSetLayoutBinding binding = {};
                binding.binding = rangeBaseBindingIndex + d;
                binding.descriptorCount = 1;  // Each binding is a single descriptor
                binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

                if (range.descriptorType == nrd::DescriptorType::TEXTURE) {
                    binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                } else if (range.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE) {
                    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                }

                bindings.push_back(binding);
            }
        }

        // NOTE: Constant buffer is in Set 1 (global), not in Set 0 (resources)
        // So we don't add it to the resources descriptor set layout

        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = (uint32_t)bindings.size();
        layoutInfo.pBindings = bindings.data();

        result = vkCreateDescriptorSetLayout(g_vkDevice, &layoutInfo, nullptr,
                                              &g_nrdPipelines[i].resourcesDescriptorSetLayout);
        if (result != VK_SUCCESS) {
            Log(L"[NRD-Vulkan] ERROR: Failed to create resources descriptor set layout %u (result=%d)\n", i, result);
            vkDestroyShaderModule(g_vkDevice, shaderModule, nullptr);
            NRD_Vulkan_Shutdown();
            return false;
        }

        // Create pipeline layout with 2 descriptor sets:
        // Set 0: Resources (textures) - per-pipeline
        // Set 1: Constant buffer - global (shared across all pipelines)
        VkDescriptorSetLayout setLayouts[2] = {
            g_nrdPipelines[i].resourcesDescriptorSetLayout,  // Set 0
            g_constantBufferDescriptorSetLayout               // Set 1 (will be created later)
        };

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 2;  // CRITICAL: 2 descriptor sets
        pipelineLayoutInfo.pSetLayouts = setLayouts;

        result = vkCreatePipelineLayout(g_vkDevice, &pipelineLayoutInfo, nullptr,
                                        &g_nrdPipelines[i].layout);
        if (result != VK_SUCCESS) {
            Log(L"[NRD-Vulkan] ERROR: Failed to create pipeline layout %u (result=%d)\n", i, result);
            vkDestroyShaderModule(g_vkDevice, shaderModule, nullptr);
            NRD_Vulkan_Shutdown();
            return false;
        }

        // Create compute pipeline
        VkComputePipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = nrdDesc->shaderEntryPoint;
        pipelineInfo.layout = g_nrdPipelines[i].layout;

        result = vkCreateComputePipelines(g_vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                          &g_nrdPipelines[i].pipeline);

        // Clean up shader module (can be destroyed after pipeline creation)
        vkDestroyShaderModule(g_vkDevice, shaderModule, nullptr);

        if (result != VK_SUCCESS) {
            Log(L"[NRD-Vulkan] ERROR: Failed to create compute pipeline %u (result=%d)\n", i, result);
            NRD_Vulkan_Shutdown();
            return false;
        }

        // Create descriptor pool for this pipeline
        std::vector<VkDescriptorPoolSize> poolSizes;
        for (uint32_t r = 0; r < pipelineDesc.resourceRangesNum; r++) {
            const nrd::ResourceRangeDesc& range = pipelineDesc.resourceRanges[r];

            VkDescriptorPoolSize poolSize = {};
            poolSize.descriptorCount = range.descriptorsNum * 16; // Allocate enough for multiple frames

            if (range.descriptorType == nrd::DescriptorType::TEXTURE) {
                poolSize.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            } else if (range.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE) {
                poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            }

            poolSizes.push_back(poolSize);
        }

        if (pipelineDesc.hasConstantData) {
            VkDescriptorPoolSize cbPoolSize = {};
            cbPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            cbPoolSize.descriptorCount = 16; // Multiple frames
            poolSizes.push_back(cbPoolSize);
        }

        if (!poolSizes.empty()) {
            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;  // Allow individual free & reset
            poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
            poolInfo.pPoolSizes = poolSizes.data();
            poolInfo.maxSets = 256; // Increased for many dispatches per frame

            result = vkCreateDescriptorPool(g_vkDevice, &poolInfo, nullptr,
                                            &g_nrdPipelines[i].descriptorPool);
            if (result != VK_SUCCESS) {
                Log(L"[NRD-Vulkan] ERROR: Failed to create descriptor pool %u (result=%d)\n", i, result);
                NRD_Vulkan_Shutdown();
                return false;
            }
        }
    }
    Log(L"[NRD-Vulkan] Created %u compute pipelines with descriptor pools\n", nrdDesc->pipelinesNum);

    // Create constant buffer
    g_constantBufferSize = nrdDesc->constantBufferMaxDataSize;

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = g_constantBufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(g_vkDevice, &bufferInfo, nullptr, &g_constantBuffer);
    if (result != VK_SUCCESS) {
        Log(L"[NRD-Vulkan] ERROR: Failed to create constant buffer (result=%d)\n", result);
        NRD_Vulkan_Shutdown();
        return false;
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(g_vkDevice, g_constantBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    result = vkAllocateMemory(g_vkDevice, &allocInfo, nullptr, &g_constantBufferMemory);
    if (result != VK_SUCCESS) {
        Log(L"[NRD-Vulkan] ERROR: Failed to allocate constant buffer memory (result=%d)\n", result);
        NRD_Vulkan_Shutdown();
        return false;
    }

    result = vkBindBufferMemory(g_vkDevice, g_constantBuffer, g_constantBufferMemory, 0);
    if (result != VK_SUCCESS) {
        Log(L"[NRD-Vulkan] ERROR: Failed to bind constant buffer memory (result=%d)\n", result);
        NRD_Vulkan_Shutdown();
        return false;
    }

    // Map constant buffer memory
    result = vkMapMemory(g_vkDevice, g_constantBufferMemory, 0, g_constantBufferSize, 0, &g_constantBufferMapped);
    if (result != VK_SUCCESS) {
        Log(L"[NRD-Vulkan] ERROR: Failed to map constant buffer memory (result=%d)\n", result);
        NRD_Vulkan_Shutdown();
        return false;
    }

    Log(L"[NRD-Vulkan] Created constant buffer (%u bytes)\n", g_constantBufferSize);

    // Create descriptor pool for constant buffer (Set 1)
    VkDescriptorPoolSize cbPoolSize = {};
    cbPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    cbPoolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo cbPoolInfo = {};
    cbPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    cbPoolInfo.maxSets = 1;
    cbPoolInfo.poolSizeCount = 1;
    cbPoolInfo.pPoolSizes = &cbPoolSize;

    result = vkCreateDescriptorPool(g_vkDevice, &cbPoolInfo, nullptr, &g_constantBufferDescriptorPool);
    if (result != VK_SUCCESS) {
        Log(L"[NRD-Vulkan] ERROR: Failed to create constant buffer descriptor pool (result=%d)\n", result);
        NRD_Vulkan_Shutdown();
        return false;
    }

    // Allocate descriptor set for constant buffer
    VkDescriptorSetAllocateInfo cbAllocInfo = {};
    cbAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    cbAllocInfo.descriptorPool = g_constantBufferDescriptorPool;
    cbAllocInfo.descriptorSetCount = 1;
    cbAllocInfo.pSetLayouts = &g_constantBufferDescriptorSetLayout;

    result = vkAllocateDescriptorSets(g_vkDevice, &cbAllocInfo, &g_constantBufferDescriptorSet);
    if (result != VK_SUCCESS) {
        Log(L"[NRD-Vulkan] ERROR: Failed to allocate constant buffer descriptor set (result=%d)\n", result);
        NRD_Vulkan_Shutdown();
        return false;
    }

    // Bind the constant buffer to the descriptor set (once, it stays bound)
    VkDescriptorBufferInfo cbDescBufferInfo = {};
    cbDescBufferInfo.buffer = g_constantBuffer;
    cbDescBufferInfo.offset = 0;
    cbDescBufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet cbWrite = {};
    cbWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    cbWrite.dstSet = g_constantBufferDescriptorSet;
    cbWrite.dstBinding = constantBufferBindingOffset;
    cbWrite.dstArrayElement = 0;
    cbWrite.descriptorCount = 1;
    cbWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    cbWrite.pBufferInfo = &cbDescBufferInfo;

    vkUpdateDescriptorSets(g_vkDevice, 1, &cbWrite, 0, nullptr);

    Log(L"[NRD-Vulkan] Created global constant buffer descriptor set (Set 1)\n");

    g_initialized = true;
    Log(L"[NRD-Vulkan] Initialization complete!\n");
    return true;
}

void NRD_Vulkan_Shutdown() {
    // Destroy G-buffer image views
    if (g_normalRoughnessView != VK_NULL_HANDLE) {
        vkDestroyImageView(g_vkDevice, g_normalRoughnessView, nullptr);
        g_normalRoughnessView = VK_NULL_HANDLE;
    }
    if (g_viewDepthView != VK_NULL_HANDLE) {
        vkDestroyImageView(g_vkDevice, g_viewDepthView, nullptr);
        g_viewDepthView = VK_NULL_HANDLE;
    }
    if (g_motionVectorsView != VK_NULL_HANDLE) {
        vkDestroyImageView(g_vkDevice, g_motionVectorsView, nullptr);
        g_motionVectorsView = VK_NULL_HANDLE;
    }
    if (g_diffuseRadianceView != VK_NULL_HANDLE) {
        vkDestroyImageView(g_vkDevice, g_diffuseRadianceView, nullptr);
        g_diffuseRadianceView = VK_NULL_HANDLE;
    }
    if (g_specularRadianceView != VK_NULL_HANDLE) {
        vkDestroyImageView(g_vkDevice, g_specularRadianceView, nullptr);
        g_specularRadianceView = VK_NULL_HANDLE;
    }

    // Destroy pipelines and descriptor layouts
    for (auto& pipeline : g_nrdPipelines) {
        if (pipeline.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(g_vkDevice, pipeline.pipeline, nullptr);
        }
        if (pipeline.layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(g_vkDevice, pipeline.layout, nullptr);
        }
        if (pipeline.resourcesDescriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(g_vkDevice, pipeline.resourcesDescriptorSetLayout, nullptr);
        }
        if (pipeline.descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(g_vkDevice, pipeline.descriptorPool, nullptr);
        }
    }
    g_nrdPipelines.clear();

    // Destroy constant buffer
    if (g_constantBufferMapped) {
        vkUnmapMemory(g_vkDevice, g_constantBufferMemory);
        g_constantBufferMapped = nullptr;
    }
    if (g_constantBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(g_vkDevice, g_constantBuffer, nullptr);
        g_constantBuffer = VK_NULL_HANDLE;
    }
    if (g_constantBufferMemory != VK_NULL_HANDLE) {
        vkFreeMemory(g_vkDevice, g_constantBufferMemory, nullptr);
        g_constantBufferMemory = VK_NULL_HANDLE;
    }

    // Destroy permanent texture pool
    for (auto& texture : g_nrdPermanentPool) {
        DestroyNRDTexture(texture);
    }
    g_nrdPermanentPool.clear();

    // Destroy transient texture pool
    for (auto& texture : g_nrdTransientPool) {
        DestroyNRDTexture(texture);
    }
    g_nrdTransientPool.clear();

    // Destroy dedicated output textures
    DestroyNRDTexture(g_nrdOutputDiffuse);
    DestroyNRDTexture(g_nrdOutputSpecular);
    DestroyNRDTexture(g_nrdOutputShadow);
    g_denoisedDiffuseView = VK_NULL_HANDLE;
    g_denoisedSpecularView = VK_NULL_HANDLE;
    g_denoisedShadowView = VK_NULL_HANDLE;

    // Destroy penumbra view
    if (g_penumbraView != VK_NULL_HANDLE) {
        vkDestroyImageView(g_vkDevice, g_penumbraView, nullptr);
        g_penumbraView = VK_NULL_HANDLE;
    }

    // Destroy confidence views
    if (g_diffConfidenceView != VK_NULL_HANDLE) {
        vkDestroyImageView(g_vkDevice, g_diffConfidenceView, nullptr);
        g_diffConfidenceView = VK_NULL_HANDLE;
    }
    if (g_specConfidenceView != VK_NULL_HANDLE) {
        vkDestroyImageView(g_vkDevice, g_specConfidenceView, nullptr);
        g_specConfidenceView = VK_NULL_HANDLE;
    }

    // Destroy samplers
    if (g_nrdSamplerNearest != VK_NULL_HANDLE) {
        vkDestroySampler(g_vkDevice, g_nrdSamplerNearest, nullptr);
        g_nrdSamplerNearest = VK_NULL_HANDLE;
    }
    if (g_nrdSamplerLinear != VK_NULL_HANDLE) {
        vkDestroySampler(g_vkDevice, g_nrdSamplerLinear, nullptr);
        g_nrdSamplerLinear = VK_NULL_HANDLE;
    }

    // Destroy NRD instance
    if (g_nrdInstance) {
        nrd::DestroyInstance(*g_nrdInstance);
        g_nrdInstance = nullptr;
    }

    g_initialized = false;
    Log(L"[NRD-Vulkan] Shutdown complete\n");
}

bool NRD_Vulkan_SetGBuffers(const NRD_GBufferImages& gbuffers) {
    if (!g_initialized) {
        return false;
    }

    // Create image views for application G-buffers
    // These are owned by the application, we just create views for NRD to bind

    // Normal + Roughness (RGBA16F)
    if (gbuffers.normalRoughness != VK_NULL_HANDLE && g_normalRoughnessView == VK_NULL_HANDLE) {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = gbuffers.normalRoughness;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(g_vkDevice, &viewInfo, nullptr, &g_normalRoughnessView);
    }

    // View Depth (R32F - increased precision for large worlds)
    if (gbuffers.viewDepth != VK_NULL_HANDLE && g_viewDepthView == VK_NULL_HANDLE) {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = gbuffers.viewDepth;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R32_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(g_vkDevice, &viewInfo, nullptr, &g_viewDepthView);
    }

    // Motion Vectors (RGBA16F - matching NRD official sample)
    if (gbuffers.motionVectors != VK_NULL_HANDLE && g_motionVectorsView == VK_NULL_HANDLE) {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = gbuffers.motionVectors;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(g_vkDevice, &viewInfo, nullptr, &g_motionVectorsView);
    }

    // Diffuse Radiance (RGBA16F)
    if (gbuffers.diffuseRadiance != VK_NULL_HANDLE && g_diffuseRadianceView == VK_NULL_HANDLE) {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = gbuffers.diffuseRadiance;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(g_vkDevice, &viewInfo, nullptr, &g_diffuseRadianceView);
        Log(L"[NRD-Vulkan] Created view for diffuseRadiance image %p -> view %p\n",
            gbuffers.diffuseRadiance, g_diffuseRadianceView);
    }

    // Specular Radiance (RGBA16F)
    if (gbuffers.specularRadiance != VK_NULL_HANDLE && g_specularRadianceView == VK_NULL_HANDLE) {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = gbuffers.specularRadiance;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(g_vkDevice, &viewInfo, nullptr, &g_specularRadianceView);
        Log(L"[NRD-Vulkan] Created view for specularRadiance image %p -> view %p\n",
            gbuffers.specularRadiance, g_specularRadianceView);
    }

    // Albedo Buffer (RGBA16F) - for material demodulation
    if (gbuffers.albedoBuffer != VK_NULL_HANDLE && g_albedoBufferView == VK_NULL_HANDLE) {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = gbuffers.albedoBuffer;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(g_vkDevice, &viewInfo, nullptr, &g_albedoBufferView);
        Log(L"[NRD-Vulkan] Created view for albedoBuffer image %p -> view %p\n",
            gbuffers.albedoBuffer, g_albedoBufferView);
    }

    // Penumbra (R16F) — SIGMA shadow input
    if (gbuffers.penumbraBuffer != VK_NULL_HANDLE && g_penumbraView == VK_NULL_HANDLE) {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = gbuffers.penumbraBuffer;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R16_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(g_vkDevice, &viewInfo, nullptr, &g_penumbraView);
        Log(L"[NRD-Vulkan] Created view for penumbra image %p -> view %p\n",
            gbuffers.penumbraBuffer, g_penumbraView);
    }

    // Diffuse Confidence (R8_UNORM)
    if (gbuffers.diffuseConfidence != VK_NULL_HANDLE && g_diffConfidenceView == VK_NULL_HANDLE) {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = gbuffers.diffuseConfidence;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(g_vkDevice, &viewInfo, nullptr, &g_diffConfidenceView);
    }

    // Specular Confidence (R8_UNORM)
    if (gbuffers.specularConfidence != VK_NULL_HANDLE && g_specConfidenceView == VK_NULL_HANDLE) {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = gbuffers.specularConfidence;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(g_vkDevice, &viewInfo, nullptr, &g_specConfidenceView);
    }

    Log(L"[NRD-Vulkan] G-buffer images registered\n");
    Log(L"[NRD-Vulkan] Input views: diffuse=%p, specular=%p, albedo=%p, penumbra=%p, diffConf=%p, specConf=%p\n",
        g_diffuseRadianceView, g_specularRadianceView, g_albedoBufferView, g_penumbraView,
        g_diffConfidenceView, g_specConfidenceView);
    return true;
}

void NRD_Vulkan_Denoise(VkCommandBuffer cmdBuffer, uint32_t frameIndex,
                        const float view[16], const float proj[16],
                        const float viewPrev[16], const float projPrev[16],
                        float jitterX, float jitterY,
                        float jitterXprev, float jitterYprev,
                        float frameDeltaMs,
                        bool dlssActive) {
    if (!g_initialized) {
        static uint32_t s_notInitCount = 0;
        if (s_notInitCount < 3) {
            Log(L"[NRD] Denoise called but g_initialized=false!\n");
            s_notInitCount++;
        }
        return;
    }

    // Use dlssActive flag for NRD tuning (high accumulation, conservative antilag).
    // Jitter values may be zero even with DLSS active (hidden jitter approach:
    // jitter is applied in projInverse but not reported to NRD/DLSS).
    const bool hasJitter = (jitterX != 0.0f || jitterY != 0.0f);
    const bool useDlssTuning = dlssActive || hasJitter;

    // Setup NRD common settings
    nrd::CommonSettings commonSettings = {};
    commonSettings.frameIndex = frameIndex;

    // Use CLEAR_AND_RESTART on first frame to initialize pool textures
    // (they contain random GPU memory after UNDEFINED→GENERAL transition).
    // Previously this caused TDR due to 32 dispatches, but the vkCmdUpdateBuffer
    // CB fix ensures each dispatch gets correct parameters now.
    // CLEAR_AND_RESTART only on very first frame (pool initialization).
    // Object movement is handled by anti-lag (detects luminance changes)
    // instead of destructive full history wipe.
    commonSettings.accumulationMode = (frameIndex == 0)
        ? nrd::AccumulationMode::CLEAR_AND_RESTART
        : nrd::AccumulationMode::CONTINUE;

    if (frameIndex < 5) {
        Log(L"[NRD] Denoise frame=%u accumMode=%s maxAccum=%.0f fastAccum=%.0f disocclusion=%.4f dlss=%d\n",
            frameIndex,
            (frameIndex == 0) ? L"CLEAR_AND_RESTART" : L"CONTINUE",
            g_config.nrdMaxAccumFrames, g_config.nrdFastAccumFrames,
            g_config.nrdDisocclusionThreshold, (int)dlssActive);
    }

    commonSettings.enableValidation = false;  // Disabled - requires additional textures we haven't created
    commonSettings.resourceSize[0] = (uint16_t)g_width;
    commonSettings.resourceSize[1] = (uint16_t)g_height;
    commonSettings.resourceSizePrev[0] = (uint16_t)g_width;
    commonSettings.resourceSizePrev[1] = (uint16_t)g_height;
    commonSettings.rectSize[0] = (uint16_t)g_width;
    commonSettings.rectSize[1] = (uint16_t)g_height;
    commonSettings.rectSizePrev[0] = (uint16_t)g_width;
    commonSettings.rectSizePrev[1] = (uint16_t)g_height;
    // Denoising range - pixels with abs(viewZ) > this value will be skipped
    // AC uses large world units, so we need a large range
    commonSettings.denoisingRange = 10000.0f;  // Was 200.0, increased for AC's large world scale

    // Disocclusion threshold — controls how aggressively NRD discards stale history.
    // With DLSS jitter: 0.03 tolerates sub-pixel depth variation from jitter → more history kept
    // Without: 0.01 for sharper shadow edges during motion
    commonSettings.disocclusionThreshold = useDlssTuning ? 0.03f : 0.01f;

    // Dynamic frame time (measured from actual frame delta)
    // NRD expects seconds, not milliseconds
    commonSettings.timeDeltaBetweenFrames = frameDeltaMs * 0.001f;

    // Motion vector configuration — 2.5D screen-space MVs in UV space
    // NRD doc: "mv = IN_MV * motionVectorScale", then "pixelUvPrev = pixelUv + mv.xy"
    // Shader stores MV directly in NRD UV space (y=0 at top), so xy scale = (1, 1).
    // z scale = -1: our RH viewZ is negative, NRD uses abs(viewZ) internally.
    //   Shader stores: mvZ = viewZprev - viewZ (both negative in RH)
    //   NRD computes: viewZprev = abs(viewZ) + mvZ * (-1) = |curr| + |prev| - |curr| = |prev| ✓
    commonSettings.motionVectorScale[0] = 1.0f;
    commonSettings.motionVectorScale[1] = 1.0f;
    commonSettings.motionVectorScale[2] = -1.0f;  // 2.5D enabled, negate for RH viewZ convention
    commonSettings.isMotionVectorInWorldSpace = false;

    // Camera jitter for TAA
    commonSettings.cameraJitter[0] = jitterX;
    commonSettings.cameraJitter[1] = jitterY;
    commonSettings.cameraJitterPrev[0] = jitterXprev;
    commonSettings.cameraJitterPrev[1] = jitterYprev;

    // D3D row-major matrices stored as float[16] are byte-equivalent to
    // SPIRV/GLSL column-major mat4. NRD's "M * v" convention with column-major
    // storage produces the same transform as D3D's "v * M" with row-major storage.
    // No transpose needed — pass the raw bytes directly.
    memcpy(commonSettings.viewToClipMatrix, proj, 16 * sizeof(float));
    memcpy(commonSettings.viewToClipMatrixPrev, projPrev, 16 * sizeof(float));
    memcpy(commonSettings.worldToViewMatrix, view, 16 * sizeof(float));
    memcpy(commonSettings.worldToViewMatrixPrev, viewPrev, 16 * sizeof(float));

    // Don't use basecolor/metalness — our albedo alpha stores 1.0 (would mean fully metallic)
    commonSettings.isBaseColorMetalnessAvailable = false;

    // Enable confidence masks (infrastructure for future dynamic objects)
    commonSettings.isHistoryConfidenceAvailable = (g_diffConfidenceView != VK_NULL_HANDLE);

    // Log matrix info on first frame
    if (frameIndex == 0) {
        Log(L"[NRD] First frame (CLEAR_AND_RESTART, matrices passed as-is):\n");
        const float* m = commonSettings.worldToViewMatrix;
        Log(L"  worldToView row0: [%.4f, %.4f, %.4f, %.4f]\n", m[0], m[1], m[2], m[3]);
        Log(L"  worldToView row3: [%.4f, %.4f, %.4f, %.4f]\n", m[12], m[13], m[14], m[15]);
        const float* p = commonSettings.viewToClipMatrix;
        Log(L"  viewToClip  row0: [%.4f, %.4f, %.4f, %.4f]\n", p[0], p[1], p[2], p[3]);
        Log(L"  viewToClip  row3: [%.4f, %.4f, %.4f, %.4f]\n", p[12], p[13], p[14], p[15]);
    }

    // Setup ReLAX settings — A-trous wavelet denoiser, converges in 1-2 frames
    // Tuned for sharp shadows during camera motion (racing game with constant movement)
    nrd::RelaxSettings relaxSettings = {};

    // Temporal accumulation — higher = more stable but blurrier during motion
    uint32_t maxAccum = (uint32_t)g_config.nrdMaxAccumFrames;
    uint32_t fastAccum = (uint32_t)g_config.nrdFastAccumFrames;
    if (useDlssTuning) {
        if (maxAccum < 30u) maxAccum = 30u;
        if (fastAccum < 6u) fastAccum = 6u;
    }
    relaxSettings.diffuseMaxAccumulatedFrameNum = maxAccum;
    // Specular needs equal or higher accumulation — 1spp reflections are very noisy
    relaxSettings.specularMaxAccumulatedFrameNum = maxAccum;

    // Fast history (responsive to changes)
    relaxSettings.diffuseMaxFastAccumulatedFrameNum = fastAccum;
    relaxSettings.specularMaxFastAccumulatedFrameNum = fastAccum;

    // History fix — spatial reconstruction for freshly disoccluded pixels
    // Lower frameNum = less time in the blurry spatial-only phase
    // Lower stride = smaller spatial filter = less blur during reconstruction
    relaxSettings.historyFixFrameNum = (uint32_t)g_config.nrdHistoryFixFrameNum;
    relaxSettings.historyFixBasePixelStride = 3;
    relaxSettings.historyFixAlternatePixelStride = 6;

    // Pre-pass blur: stabilizes 1spp input before temporal accumulation.
    // Specular needs equal or more blur than diffuse — reflections are noisier.
    float minPrepass = useDlssTuning ? 20.0f : 15.0f;
    relaxSettings.diffusePrepassBlurRadius = fmaxf(g_config.nrdDiffusePrepassBlur, minPrepass);
    relaxSettings.specularPrepassBlurRadius = fmaxf(g_config.nrdSpecularPrepassBlur, minPrepass);

    // A-trous wavelet iterations — 6 for good spatial coverage without over-blur
    relaxSettings.atrousIterationNum = (uint32_t)g_config.nrdAtrousIterations;
    if (relaxSettings.atrousIterationNum < 6u) relaxSettings.atrousIterationNum = 6u;

    // Edge stopping — luminance-based rejection in A-trous passes
    // With DLSS jitter: boost phi by 50% for more spatial smoothing → more stable output
    float dlssPhiBoost = useDlssTuning ? 1.5f : 1.0f;
    relaxSettings.diffusePhiLuminance = g_config.nrdDiffusePhiLuminance * dlssPhiBoost;
    relaxSettings.specularPhiLuminance = g_config.nrdSpecularPhiLuminance * dlssPhiBoost;

    // Rejection thresholds
    relaxSettings.lobeAngleFraction = g_config.nrdLobeAngleFraction;
    relaxSettings.roughnessFraction = g_config.nrdRoughnessFraction;
    relaxSettings.depthThreshold = g_config.nrdDepthThreshold;

    // Hit distance weight
    relaxSettings.minHitDistanceWeight = g_config.nrdMinHitDistanceWeight;

    // Hit distance reconstruction — AREA_3X3 for better edge preservation
    // (was OFF — AREA_3X3 improves shadow/edge quality with minimal cost)
    relaxSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;

    // Temporal clamping sigma — [1; 3], NRD default=2.0
    // Higher = wider clamping = more temporal history preserved = smoother
    relaxSettings.fastHistoryClampingSigmaScale = 2.5f;

    // Antilag — detect actual content changes (object moves, reveals new geometry)
    // and reduce temporal weight accordingly. Tuned to avoid false positives from
    // 1spp noise variance while still catching real disocclusions.
    // Anti-lag: tuned for 1spp path tracing in a Blender viewport.
    // Must tolerate high per-frame noise variance (1spp) while still detecting
    // real content changes (object/shadow movement).
    // Too aggressive → NRD resets specular history every frame → reflections never converge.
    // Too conservative → shadows ghost when objects move.
    relaxSettings.antilagSettings.accelerationAmount = 0.3f;   // moderate temporal weight reduction
    relaxSettings.antilagSettings.spatialSigmaScale = 4.0f;    // spatial: moderate sensitivity
    relaxSettings.antilagSettings.temporalSigmaScale = 2.0f;   // temporal: tolerant of 1spp noise variance
    relaxSettings.antilagSettings.resetAmount = 0.4f;           // partial reset on real changes

    // Anti-firefly (configurable, default on for 1spp)
    relaxSettings.enableAntiFirefly = g_config.nrdAntiFirefly;
    relaxSettings.enableRoughnessEdgeStopping = true;

    // Checkerboard mode
    relaxSettings.checkerboardMode = nrd::CheckerboardMode::OFF;

    // Log ReLAX settings every 300 frames
    static uint32_t s_nrdLogCounter = 0;
    if (s_nrdLogCounter % 300 == 0) {
        Log(L"[NRD] ReLAX Settings: maxAccumD=%u maxAccumS=%u fastAccumD=%u fastAccumS=%u preBlurD=%.0f preBlurS=%.0f atrousIter=%u phiLumD=%.1f phiLumS=%.1f\n",
            relaxSettings.diffuseMaxAccumulatedFrameNum, relaxSettings.specularMaxAccumulatedFrameNum,
            relaxSettings.diffuseMaxFastAccumulatedFrameNum, relaxSettings.specularMaxFastAccumulatedFrameNum,
            relaxSettings.diffusePrepassBlurRadius, relaxSettings.specularPrepassBlurRadius,
            relaxSettings.atrousIterationNum,
            relaxSettings.diffusePhiLuminance, relaxSettings.specularPhiLuminance);
    }
    s_nrdLogCounter++;

    // Set NRD settings for ReLAX denoiser
    nrd::SetCommonSettings(*g_nrdInstance, commonSettings);
    const nrd::Identifier denoiserId = nrd::Identifier(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR);
    nrd::SetDenoiserSettings(*g_nrdInstance, denoiserId, &relaxSettings);

    // SIGMA shadow settings — tuned for stable shadows during camera motion
    nrd::SigmaSettings sigmaSettings = {};
    sigmaSettings.planeDistanceSensitivity = 0.005f;  // edge detection
    sigmaSettings.maxStabilizedFrameNum = 4;           // moderate temporal — keeps shadows responsive to moving objects
    // Normalize and pass sun direction (CRITICAL for directional shadow denoising)
    float len = sqrtf(g_sunDirection[0]*g_sunDirection[0] + g_sunDirection[1]*g_sunDirection[1] + g_sunDirection[2]*g_sunDirection[2]);
    if (len > 1e-6f) {
        sigmaSettings.lightDirection[0] = g_sunDirection[0] / len;
        sigmaSettings.lightDirection[1] = g_sunDirection[1] / len;
        sigmaSettings.lightDirection[2] = g_sunDirection[2] / len;
    }
    const nrd::Identifier sigmaId = nrd::Identifier(nrd::Denoiser::SIGMA_SHADOW);
    nrd::SetDenoiserSettings(*g_nrdInstance, sigmaId, &sigmaSettings);

    // Get compute dispatches for both ReLAX and SIGMA
    nrd::Identifier denoiserIds[] = { denoiserId, sigmaId };
    const nrd::DispatchDesc* dispatchDescs = nullptr;
    uint32_t dispatchCount = 0;
    nrd::GetComputeDispatches(*g_nrdInstance, denoiserIds, 2, dispatchDescs, dispatchCount);

    if (dispatchCount == 0) {
        return; // Nothing to do
    }

    // Log dispatch count for first few frames
    if (frameIndex <= 2) {
        Log(L"[NRD] Frame %u: %u dispatches requested\n", frameIndex, dispatchCount);
    }

    // Reset all descriptor pools at the start of each frame
    // This allows descriptor sets to be reused without explicit freeing
    for (auto& pipeline : g_nrdPipelines) {
        if (pipeline.descriptorPool != VK_NULL_HANDLE) {
            vkResetDescriptorPool(g_vkDevice, pipeline.descriptorPool, 0);
        }
    }
    // (pool reset logging removed to avoid TDR from I/O during dispatch)

    // Get NRD library and instance descriptors
    const nrd::LibraryDesc* libDesc = nrd::GetLibraryDesc();
    const nrd::InstanceDesc* nrdDesc = nrd::GetInstanceDesc(*g_nrdInstance);

    // Get SPIRV binding offsets (CRITICAL for correct descriptor binding)
    const uint32_t texturesBindingOffset = libDesc->spirvBindingOffsets.textureOffset;
    const uint32_t storageTextureAndBufferOffset = libDesc->spirvBindingOffsets.storageTextureAndBufferOffset;

    // Execute each dispatch
    for (uint32_t d = 0; d < dispatchCount; d++) {
        const nrd::DispatchDesc& dispatch = dispatchDescs[d];

        if (dispatch.pipelineIndex >= g_nrdPipelines.size()) {
            Log(L"[NRD-Vulkan] ERROR: Invalid pipeline index %u\n", dispatch.pipelineIndex);
            continue;
        }

        const NRDPipeline& pipeline = g_nrdPipelines[dispatch.pipelineIndex];
        const nrd::PipelineDesc& pipelineDesc = nrdDesc->pipelines[dispatch.pipelineIndex];

        // Allocate descriptor set
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        if (pipeline.descriptorPool != VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = pipeline.descriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &pipeline.resourcesDescriptorSetLayout;

            VkResult result = vkAllocateDescriptorSets(g_vkDevice, &allocInfo, &descriptorSet);
            if (result != VK_SUCCESS) {
                Log(L"[NRD-Vulkan] ERROR: Failed to allocate descriptor set for dispatch %u (result=%d)\n", d, result);
                continue;
            }
        }

        // Build descriptor writes
        std::vector<VkWriteDescriptorSet> descriptorWrites;
        std::vector<VkDescriptorImageInfo> imageInfos;
        VkDescriptorBufferInfo bufferInfo = {};

        // Reserve space to prevent reallocation (which would invalidate pointers in descriptorWrites)
        // Largest dispatch in RELAX_DIFFUSE_SPECULAR has ~26 descriptors
        descriptorWrites.reserve(32);
        imageInfos.reserve(32);

        uint32_t resourceIndex = 0;

        // Log detailed resource info on frame 1 (first temporal frame)
        if (frameIndex == 1) {
            Log(L"[NRD] Frame 1, dispatch %u '%S': %u resources, grid=%ux%u\n",
                d, dispatch.name, dispatch.resourcesNum, dispatch.gridWidth, dispatch.gridHeight);
            for (uint32_t ri = 0; ri < dispatch.resourcesNum; ri++) {
                const nrd::ResourceDesc& res = dispatch.resources[ri];
                const wchar_t* typeName = L"?";
                switch (res.type) {
                    case nrd::ResourceType::PERMANENT_POOL: typeName = L"PERM_POOL"; break;
                    case nrd::ResourceType::TRANSIENT_POOL: typeName = L"TRANS_POOL"; break;
                    case nrd::ResourceType::IN_NORMAL_ROUGHNESS: typeName = L"IN_NORMAL"; break;
                    case nrd::ResourceType::IN_VIEWZ: typeName = L"IN_VIEWZ"; break;
                    case nrd::ResourceType::IN_MV: typeName = L"IN_MV"; break;
                    case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST: typeName = L"IN_DIFF"; break;
                    case nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST: typeName = L"IN_SPEC"; break;
                    case nrd::ResourceType::IN_BASECOLOR_METALNESS: typeName = L"IN_BASECOLOR"; break;
                    case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST: typeName = L"OUT_DIFF"; break;
                    case nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST: typeName = L"OUT_SPEC"; break;
                    case nrd::ResourceType::IN_PENUMBRA: typeName = L"IN_PENUMBRA"; break;
                    case nrd::ResourceType::OUT_SHADOW_TRANSLUCENCY: typeName = L"OUT_SHADOW"; break;
                    default: break;
                }
                Log(L"  [%u] type=%ls descType=%d pool=%u\n",
                    ri, typeName, (int)res.descriptorType, res.indexInPool);
            }
        }

        // Bind texture resources
        for (uint32_t r = 0; r < pipelineDesc.resourceRangesNum; r++) {
            const nrd::ResourceRangeDesc& range = pipelineDesc.resourceRanges[r];

            // Calculate base binding index for this range using SPIRV offsets
            bool isStorage = (range.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE);
            uint32_t rangeBaseBindingIndex = isStorage ? storageTextureAndBufferOffset : texturesBindingOffset;

            for (uint32_t i = 0; i < range.descriptorsNum; i++) {
                if (resourceIndex >= dispatch.resourcesNum) {
                    Log(L"[NRD-Vulkan] WARNING: Not enough resources for dispatch %u\n", d);
                    break;
                }

                const nrd::ResourceDesc& resource = dispatch.resources[resourceIndex];

                // Track output resources (ONLY when used as STORAGE/write, not SAMPLED/read)
                if (isStorage) {
                    if (resource.type == nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST) {
                        g_denoisedDiffuseView = g_nrdOutputDiffuse.view;
                    } else if (resource.type == nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST) {
                        g_denoisedSpecularView = g_nrdOutputSpecular.view;
                    }
                }

                resourceIndex++;
                VkImageView view = GetResourceView(resource);

                if (view == VK_NULL_HANDLE) {
                    Log(L"[NRD-Vulkan] CRITICAL: NULL view for resource type %d in dispatch %u\n",
                        (int)resource.type, d);
                    return;
                }

                VkDescriptorImageInfo imageInfo = {};
                imageInfo.imageView = view;
                // All NRD resources use GENERAL layout:
                // - G-buffer inputs stay in GENERAL (RT wrote them as storage images)
                // - NRD pool textures are in GENERAL (created that way)
                // Using SHADER_READ_ONLY_OPTIMAL for sampled inputs when images are
                // actually in GENERAL causes device lost on NVIDIA.
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                imageInfos.push_back(imageInfo);

                VkWriteDescriptorSet write = {};
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = descriptorSet;
                // CRITICAL: Use SPIRV binding offsets from NRD library descriptor
                // Each descriptor gets: baseOffset + descriptorIndexInRange
                write.dstBinding = rangeBaseBindingIndex + i;
                write.dstArrayElement = 0;  // Always 0 - not using array bindings
                write.descriptorCount = 1;
                write.descriptorType = (range.descriptorType == nrd::DescriptorType::TEXTURE)
                    ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                    : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write.pImageInfo = &imageInfos.back();

                descriptorWrites.push_back(write);
            }
        }

        // Update descriptor set
        if (!descriptorWrites.empty()) {
            vkUpdateDescriptorSets(g_vkDevice, (uint32_t)descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
        }

        // Update constant buffer data using vkCmdUpdateBuffer (inline in command buffer)
        // CRITICAL: Cannot use memcpy to mapped buffer because all dispatches share the same
        // buffer and memcpy happens during recording — by execution time, only the last
        // dispatch's data would remain. vkCmdUpdateBuffer embeds data in the command stream
        // so each dispatch gets its own snapshot.
        if (pipelineDesc.hasConstantData && dispatch.constantBufferData && dispatch.constantBufferDataSize > 0) {
            uint32_t alignedSize = (dispatch.constantBufferDataSize + 3) & ~3u;  // Must be multiple of 4
            if (alignedSize <= g_constantBufferSize) {
                vkCmdUpdateBuffer(cmdBuffer, g_constantBuffer, 0, alignedSize, dispatch.constantBufferData);
            }
        }

        // Barrier: ensure CB update (transfer write) is visible to compute dispatch,
        // AND previous dispatch's shader writes are visible for this dispatch's reads
        {
            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | (d > 0 ? VK_ACCESS_SHADER_WRITE_BIT : 0);
            barrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT;

            VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT |
                                             (d > 0 ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0);

            vkCmdPipelineBarrier(cmdBuffer,
                                  srcStage,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // Bind pipeline
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

        // Bind BOTH descriptor sets:
        // Set 0: Resources (textures) - per-dispatch
        // Set 1: Constant buffer - global (shared)
        if (descriptorSet != VK_NULL_HANDLE) {
            // Bind Set 0 (resources)
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout,
                                     0, 1, &descriptorSet, 0, nullptr);

            // Bind Set 1 (constant buffer - global, shared across all dispatches)
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout,
                                     1, 1, &g_constantBufferDescriptorSet, 0, nullptr);
        }

        // Dispatch compute shader
        vkCmdDispatch(cmdBuffer, dispatch.gridWidth, dispatch.gridHeight, 1);

        // NOTE: Do NOT free descriptor sets - pool handles reuse automatically
        // Calling vkFreeDescriptorSets causes crashes in subsequent frames
    }

    // Log periodically (avoid per-frame disk I/O)
    if (frameIndex <= 2 || frameIndex % 600 == 0) {
        Log(L"[NRD] Frame %u: %u dispatches OK\n", frameIndex, dispatchCount);
    }
}

void NRD_Vulkan_GetDenoisedOutputs(VkImageView& outDiffuse, VkImageView& outSpecular) {
    // Return dedicated output texture views (RGBA16F format, tracked from last NRD execution)
    // Safe because composite executes in same command buffer
    outDiffuse = g_denoisedDiffuseView;
    outSpecular = g_denoisedSpecularView;
}

void NRD_Vulkan_GetRawInputViews(VkImageView& outDiffuse, VkImageView& outSpecular) {
    outDiffuse = g_diffuseRadianceView;
    outSpecular = g_specularRadianceView;
}

VkImageView NRD_Vulkan_GetAlbedoBufferView() {
    // Return albedo buffer view for material re-modulation in composite
    return g_albedoBufferView;
}

VkImage NRD_Vulkan_GetPoolImage(uint32_t poolIndex) {
    if (poolIndex >= g_nrdPermanentPool.size()) {
        return VK_NULL_HANDLE;
    }
    // DEBUG: Log format of pool texture
    static bool logged = false;
    if (!logged && poolIndex == 0) {
        Log(L"[NRD-Debug] Pool[0] format = %d (RGBA16F=%d, RGBA32F=%d)\n",
            g_nrdPermanentPool[poolIndex].format,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_FORMAT_R32G32B32A32_SFLOAT);
        logged = true;
    }
    return g_nrdPermanentPool[poolIndex].image;
}

uint32_t NRD_Vulkan_GetDiffusePoolIndex() {
    // Deprecated: outputs are now dedicated textures, not from pool
    return 0xFFFFu;
}

VkImage NRD_Vulkan_GetOutputDiffuseImage() {
    return g_nrdOutputDiffuse.image;
}

VkImage NRD_Vulkan_GetOutputSpecularImage() {
    return g_nrdOutputSpecular.image;
}

void NRD_Vulkan_SetSunDirection(float x, float y, float z) {
    g_sunDirection[0] = x; g_sunDirection[1] = y; g_sunDirection[2] = z;
}

void NRD_Vulkan_GetDenoisedShadow(VkImageView& outShadow) {
    outShadow = g_denoisedShadowView;
}

VkImage NRD_Vulkan_GetOutputShadowImage() {
    return g_nrdOutputShadow.image;
}

bool NRD_Vulkan_IsReady() {
    return g_initialized && g_nrdInstance != nullptr;
}

} // namespace acpt

#else  // !ACPT_HAVE_NRD

namespace acpt {

bool NRD_Vulkan_Init(VkPhysicalDevice, VkDevice, VkQueue, VkCommandPool, uint32_t, uint32_t) {
    return false;
}

void NRD_Vulkan_Shutdown() {}

bool NRD_Vulkan_SetGBuffers(const NRD_GBufferImages&) {
    return false;
}

void NRD_Vulkan_Denoise(VkCommandBuffer, uint32_t, const float[16], const float[16],
                        const float[16], const float[16],
                        float, float, float, float, float, bool) {}

void NRD_Vulkan_GetDenoisedOutputs(VkImageView& outDiffuse, VkImageView& outSpecular) {
    outDiffuse = VK_NULL_HANDLE;
    outSpecular = VK_NULL_HANDLE;
}

void NRD_Vulkan_GetRawInputViews(VkImageView& outDiffuse, VkImageView& outSpecular) {
    outDiffuse = VK_NULL_HANDLE;
    outSpecular = VK_NULL_HANDLE;
}

uint32_t NRD_Vulkan_GetDiffusePoolIndex() {
    return 0xFFFFu;
}

VkImage NRD_Vulkan_GetOutputDiffuseImage() {
    return VK_NULL_HANDLE;
}

VkImage NRD_Vulkan_GetOutputSpecularImage() {
    return VK_NULL_HANDLE;
}

VkImageView NRD_Vulkan_GetAlbedoBufferView() {
    return VK_NULL_HANDLE;
}

VkImage NRD_Vulkan_GetPoolImage(uint32_t) {
    return VK_NULL_HANDLE;
}

void NRD_Vulkan_SetSunDirection(float, float, float) {}

void NRD_Vulkan_GetDenoisedShadow(VkImageView& outShadow) {
    outShadow = VK_NULL_HANDLE;
}

VkImage NRD_Vulkan_GetOutputShadowImage() {
    return VK_NULL_HANDLE;
}

bool NRD_Vulkan_IsReady() {
    return false;
}

} // namespace acpt

#endif // ACPT_HAVE_NRD
