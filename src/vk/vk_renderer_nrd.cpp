// vk_renderer_nrd.cpp — NRD denoiser bring-up + the composite compute pass
// that combines NRD outputs (diffuse/specular/penumbra) with the RT raw image.
// Extracted from vk_renderer.cpp.

#include "vk_renderer.h"
#include "vk_context.h"
#include "vk_interop.h"
#include "vk_rt_resources.h"
#include "ignis_log.h"
#include "ignis_config.h"
#include "nrd_vulkan_integration.h"
#include <vector>
#include <fstream>
#include <cstring>

namespace acpt {
namespace vk {

bool Renderer::InitNRD() {
    if (nrdInitialized_) return true;
    if (!rtResources_ || !context_) return false;

    // Create G-buffers first (at render resolution — may be < display when DLSS active)
    // RR also needs G-buffers (normals, albedo, depth, MVs)
    if (!rtResources_->CreateGBuffers(renderWidth_, renderHeight_)) {
        Log(L"[VK Renderer] WARNING: Failed to create G-buffers\n");
        return false;
    }

    // When Ray Reconstruction is active, skip NRD and composite — RR replaces them.
    // Still create G-buffers (done above) and tonemap pipeline (RR outputs HDR).
    if (dlssRRActive_) {
        Log(L"[VK Renderer] RR active — skipping NRD init, creating tonemap only\n");

        // Create tonemap pipeline (RR outputs HDR to dlssHdrOutput_, needs tonemap to LDR)
        if (dlssActive_ && dlssHdrOutput_) {
            if (!CreateTonemapPipeline()) {
                Log(L"[VK Renderer] WARNING: Tonemap pipeline creation failed\n");
            }
        }

        // Mark as initialized so RenderFrameRT proceeds
        nrdInitialized_ = false;  // NRD itself is NOT initialized
        return true;
    }

    // Initialize NRD (at render resolution)
    if (!acpt::NRD_Vulkan_Init(context_->GetPhysicalDevice(), context_->GetDevice(),
                                context_->GetGraphicsQueue(), context_->GetCommandPool(),
                                renderWidth_, renderHeight_)) {
        Log(L"[VK Renderer] WARNING: NRD init failed, running without denoiser\n");
        return false;
    }

    // Register G-buffer images
    acpt::NRD_GBufferImages gbuffers;
    gbuffers.normalRoughness = rtResources_->GetNormalRoughnessImage();
    gbuffers.viewDepth = rtResources_->GetViewDepthImage();
    gbuffers.motionVectors = rtResources_->GetMotionVectorsImage();
    gbuffers.diffuseRadiance = rtResources_->GetDiffuseRadianceImage();
    gbuffers.specularRadiance = rtResources_->GetSpecularRadianceImage();
    gbuffers.albedoBuffer = rtResources_->GetAlbedoBufferImage();
    gbuffers.penumbraBuffer = rtResources_->GetPenumbraImage();
    gbuffers.diffuseConfidence = rtResources_->GetDiffConfidenceImage();
    gbuffers.specularConfidence = rtResources_->GetSpecConfidenceImage();

    if (!acpt::NRD_Vulkan_SetGBuffers(gbuffers)) {
        Log(L"[VK Renderer] WARNING: NRD SetGBuffers failed\n");
        acpt::NRD_Vulkan_Shutdown();
        return false;
    }

    nrdInitialized_ = true;
    Log(L"[VK Renderer] NRD initialized successfully\n");

    // Create composite pipeline
    if (!CreateCompositePipeline()) {
        Log(L"[VK Renderer] WARNING: Composite pipeline creation failed\n");
    }

    // Create auto-exposure resolve pipeline (after composite, uses same SSBO)
    if (!CreateExposureResolvePipeline()) {
        Log(L"[VK Renderer] WARNING: Auto-exposure resolve pipeline creation failed (non-fatal)\n");
    }

    // Create tonemap pipeline (post-DLSS HDR → LDR, only when DLSS active)
    if (dlssActive_ && dlssHdrOutput_) {
        if (!CreateTonemapPipeline()) {
            Log(L"[VK Renderer] WARNING: Tonemap pipeline creation failed\n");
        }
    }

    // Create SHARC resolve pipeline
    if (!CreateSHARCResolvePipeline()) {
        Log(L"[VK Renderer] WARNING: SHARC resolve pipeline creation failed (non-fatal)\n");
    }

    // Create Surfel GI resolve pipeline
    if (!CreateSurfelResolvePipeline()) {
        Log(L"[VK Renderer] WARNING: Surfel resolve pipeline creation failed (non-fatal)\n");
    }

    // Create hair contour detection pipeline
    if (!CreateHairContourPipeline()) {
        Log(L"[VK Renderer] WARNING: Hair contour pipeline creation failed (non-fatal)\n");
    }

    return true;
}

bool Renderer::CreateCompositePipeline() {
    VkDevice device = context_->GetDevice();

    // Sampler for NRD denoised textures
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &compositeSampler_) != VK_SUCCESS) {
        return false;
    }

    // Descriptor set layout: 10 bindings matching nrd_composite.comp
    VkDescriptorSetLayoutBinding bindings[10] = {};
    // binding 0: denoised diffuse (sampler2D)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 1: denoised specular (sampler2D)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 2: final output (storage image, read-write for cloud blending)
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 3: raw PT output (sampler2D)
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 4: albedo buffer (sampler2D)
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 5: denoised shadow from SIGMA (sampler2D)
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 6: exposure SSBO (auto-exposure luminance accumulation)
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 7: cloud buffer (volumetric clouds, full-res RGBA16F)
    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 8: cloud depth (first-hit distance, full-res R32F)
    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 9: scene depth (linear view-space Z, R32F)
    bindings[9].binding = 9;
    bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[9].descriptorCount = 1;
    bindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 10;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &compositeDescriptorSetLayout_) != VK_SUCCESS) {
        return false;
    }

    // Push constant: useNRD (uint32_t)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 6 * sizeof(uint32_t);  // mode + tonemapMode + exposure + saturation + contrast + hdrOutput

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &compositeDescriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &compositePipelineLayout_) != VK_SUCCESS) {
        return false;
    }

    // Load compute shader
    std::ifstream file(IgnisResolvePath("shaders/nrd_composite.spv"), std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[VK Renderer] WARNING: nrd_composite.spv not found\n");
        return false;
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> code(fileSize);
    file.seekg(0);
    file.read(code.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = code.size();
    moduleInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return false;
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = compositePipelineLayout_;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &compositePipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create composite pipeline\n");
        return false;
    }

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},  // 8 samplers (diffuse, specular, rawPT, albedo, shadow, clouds, cloudDepth, sceneDepth)
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},          // exposure SSBO
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &compositeDescriptorPool_) != VK_SUCCESS) {
        return false;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = compositeDescriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &compositeDescriptorSetLayout_;
    if (vkAllocateDescriptorSets(device, &allocInfo, &compositeDescriptorSet_) != VK_SUCCESS) {
        return false;
    }

    compositeReady_ = true;
    Log(L"[VK Renderer] Composite pipeline created\n");
    return true;
}

void Renderer::UpdateCompositeDescriptors() {
    if (!compositeReady_ || !nrdInitialized_) return;

    VkDevice device = context_->GetDevice();

    VkImageView denoisedDiffuseView, denoisedSpecularView;
    acpt::NRD_Vulkan_GetDenoisedOutputs(denoisedDiffuseView, denoisedSpecularView);
    VkImageView albedoView = acpt::NRD_Vulkan_GetAlbedoBufferView();
    VkImageView shadowView;
    acpt::NRD_Vulkan_GetDenoisedShadow(shadowView);

    if (!denoisedDiffuseView || !denoisedSpecularView) return;

    // binding 0: denoised diffuse (NRD output textures stay in GENERAL)
    VkDescriptorImageInfo diffuseInfo{};
    diffuseInfo.imageView = denoisedDiffuseView;
    diffuseInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    diffuseInfo.sampler = compositeSampler_;

    // binding 1: denoised specular
    VkDescriptorImageInfo specularInfo{};
    specularInfo.imageView = denoisedSpecularView;
    specularInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    specularInfo.sampler = compositeSampler_;

    // binding 2: final output — when DLSS active, write to intermediate image (render res)
    //            otherwise write directly to interop image (display res)
    VkImageView compositeOutputView = dlssActive_ ? dlssColorInputView_ : interop_->GetSharedImageView();
    VkDescriptorImageInfo outputInfo{};
    outputInfo.imageView = compositeOutputView;
    outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // binding 3: raw PT output — same destination as composite output
    VkDescriptorImageInfo rawPTInfo{};
    rawPTInfo.imageView = compositeOutputView;
    rawPTInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    rawPTInfo.sampler = compositeSampler_;

    // binding 4: albedo buffer
    VkDescriptorImageInfo albedoInfo{};
    albedoInfo.imageView = albedoView;
    albedoInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    albedoInfo.sampler = compositeSampler_;

    // binding 5: denoised shadow (SIGMA output)
    VkDescriptorImageInfo shadowInfo{};
    shadowInfo.imageView = shadowView ? shadowView : denoisedDiffuseView;  // fallback if no shadow
    shadowInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    shadowInfo.sampler = compositeSampler_;

    // binding 6: exposure SSBO
    VkDescriptorBufferInfo exposureBufferInfo{};
    exposureBufferInfo.buffer = exposureSSBO_.buffer;
    exposureBufferInfo.offset = 0;
    exposureBufferInfo.range = VK_WHOLE_SIZE;

    // binding 7: cloud buffer (volumetric clouds — stubbed out, always fallback)
    VkDescriptorImageInfo cloudInfo{};
    cloudInfo.imageView = denoisedDiffuseView;
    cloudInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    cloudInfo.sampler = compositeSampler_;

    // binding 8: cloud depth (stubbed out, always fallback)
    VkDescriptorImageInfo cloudDepthInfo{};
    cloudDepthInfo.imageView = denoisedDiffuseView;
    cloudDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    cloudDepthInfo.sampler = compositeSampler_;

    // binding 9: scene depth (linear view-space Z)
    VkDescriptorImageInfo sceneDepthInfo{};
    sceneDepthInfo.imageView = rtResources_ ? rtResources_->GetViewDepthView() : denoisedDiffuseView;
    sceneDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    sceneDepthInfo.sampler = compositeSampler_;

    VkWriteDescriptorSet writes[10] = {};
    for (int i = 0; i < 10; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = compositeDescriptorSet_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &diffuseInfo;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &specularInfo;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[2].pImageInfo = &outputInfo;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &rawPTInfo;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].pImageInfo = &albedoInfo;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[5].pImageInfo = &shadowInfo;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[6].pBufferInfo = &exposureBufferInfo;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[7].pImageInfo = &cloudInfo;
    writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[8].pImageInfo = &cloudDepthInfo;
    writes[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[9].pImageInfo = &sceneDepthInfo;

    uint32_t writeCount = exposureSSBO_.buffer ? 10u : 6u;
    vkUpdateDescriptorSets(device, writeCount, writes, 0, nullptr);
}

void Renderer::ShutdownNRD() {
    if (!nrdInitialized_) return;
    VkDevice device = context_->GetDevice();

    acpt::NRD_Vulkan_Shutdown();
    nrdInitialized_ = false;

    // Destroy composite pipeline
    if (compositePipeline_) { vkDestroyPipeline(device, compositePipeline_, nullptr); compositePipeline_ = VK_NULL_HANDLE; }
    if (compositePipelineLayout_) { vkDestroyPipelineLayout(device, compositePipelineLayout_, nullptr); compositePipelineLayout_ = VK_NULL_HANDLE; }
    if (compositeDescriptorPool_) { vkDestroyDescriptorPool(device, compositeDescriptorPool_, nullptr); compositeDescriptorPool_ = VK_NULL_HANDLE; }
    if (compositeDescriptorSetLayout_) { vkDestroyDescriptorSetLayout(device, compositeDescriptorSetLayout_, nullptr); compositeDescriptorSetLayout_ = VK_NULL_HANDLE; }
    if (compositeSampler_) { vkDestroySampler(device, compositeSampler_, nullptr); compositeSampler_ = VK_NULL_HANDLE; }
    compositeReady_ = false;

    Log(L"[VK Renderer] NRD shutdown\n");
}

} // namespace vk
} // namespace acpt
