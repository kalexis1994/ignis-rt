// vk_renderer_exposure.cpp — auto-exposure resolve compute pass.
// Reads a luminance accumulator SSBO written by the wavefront output kernel,
// computes an EMA-smoothed exposure, and stages the result for CPU readback
// the next frame. Extracted from vk_renderer.cpp.

#include "vk_renderer.h"
#include "vk_context.h"
#include "vk_accel_structure.h"
#include "ignis_log.h"
#include <vector>
#include <fstream>
#include <cstring>

namespace acpt {
namespace vk {

bool Renderer::CreateExposureResolvePipeline() {
    VkDevice device = context_->GetDevice();

    // Create 12-byte device-local SSBO: { uint luminanceSum, uint pixelCount, float currentExposure }
    exposureSSBO_ = accelBuilder_->CreateAccelBuffer(
        12,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!exposureSSBO_.buffer) {
        Log(L"[VK Renderer] WARNING: Failed to create exposure SSBO\n");
        return false;
    }

    // Create 12-byte host-visible staging buffer for CPU readback
    exposureStagingSSBO_ = accelBuilder_->CreateAccelBuffer(
        12,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!exposureStagingSSBO_.buffer) {
        Log(L"[VK Renderer] WARNING: Failed to create exposure staging buffer\n");
        return false;
    }

    // Initialize SSBO: luminanceSum=0, pixelCount=0, currentExposure=0.55
    {
        struct { uint32_t lumSum; uint32_t pixCount; float exposure; } initData = { 0, 0, 0.55f };
        void* mapped;
        vkMapMemory(device, exposureStagingSSBO_.memory, 0, 12, 0, &mapped);
        memcpy(mapped, &initData, 12);
        vkUnmapMemory(device, exposureStagingSSBO_.memory);

        VkCommandBuffer initCmd = context_->BeginSingleTimeCommands();
        VkBufferCopy copyRegion{};
        copyRegion.size = 12;
        vkCmdCopyBuffer(initCmd, exposureStagingSSBO_.buffer, exposureSSBO_.buffer, 1, &copyRegion);
        context_->EndSingleTimeCommands(initCmd);
    }

    // Descriptor set layout: 1 SSBO binding
    VkDescriptorSetLayoutBinding ssboBinding{};
    ssboBinding.binding = 0;
    ssboBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ssboBinding.descriptorCount = 1;
    ssboBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descLayoutInfo{};
    descLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descLayoutInfo.bindingCount = 1;
    descLayoutInfo.pBindings = &ssboBinding;
    if (vkCreateDescriptorSetLayout(device, &descLayoutInfo, nullptr, &exposureResolveDescSetLayout_) != VK_SUCCESS) {
        return false;
    }

    // Push constant: 4 floats (keyValue, adaptSpeed, minExposure, maxExposure)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 4 * sizeof(float);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &exposureResolveDescSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &exposureResolvePipelineLayout_) != VK_SUCCESS) {
        return false;
    }

    // Load compute shader
    std::ifstream file(IgnisResolvePath("shaders/exposure_resolve.comp.spv"), std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[VK Renderer] WARNING: exposure_resolve.comp.spv not found\n");
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
    pipelineInfo.layout = exposureResolvePipelineLayout_;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &exposureResolvePipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create exposure resolve pipeline\n");
        return false;
    }

    // Descriptor pool + set
    VkDescriptorPoolSize resolvePoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo resolvePoolInfo{};
    resolvePoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    resolvePoolInfo.maxSets = 1;
    resolvePoolInfo.poolSizeCount = 1;
    resolvePoolInfo.pPoolSizes = &resolvePoolSize;
    if (vkCreateDescriptorPool(device, &resolvePoolInfo, nullptr, &exposureResolveDescPool_) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorSetAllocateInfo resolveAllocInfo{};
    resolveAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    resolveAllocInfo.descriptorPool = exposureResolveDescPool_;
    resolveAllocInfo.descriptorSetCount = 1;
    resolveAllocInfo.pSetLayouts = &exposureResolveDescSetLayout_;
    if (vkAllocateDescriptorSets(device, &resolveAllocInfo, &exposureResolveDescSet_) != VK_SUCCESS) {
        return false;
    }

    // Write SSBO descriptor
    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = exposureSSBO_.buffer;
    bufInfo.offset = 0;
    bufInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet ssboWrite{};
    ssboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ssboWrite.dstSet = exposureResolveDescSet_;
    ssboWrite.dstBinding = 0;
    ssboWrite.descriptorCount = 1;
    ssboWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ssboWrite.pBufferInfo = &bufInfo;
    vkUpdateDescriptorSets(device, 1, &ssboWrite, 0, nullptr);

    exposureResolveReady_ = true;
    Log(L"[VK Renderer] Auto-exposure resolve pipeline created\n");
    return true;
}

void Renderer::ShutdownExposureResolve() {
    if (!exposureResolveReady_) return;
    VkDevice device = context_->GetDevice();

    if (exposureResolvePipeline_) { vkDestroyPipeline(device, exposureResolvePipeline_, nullptr); exposureResolvePipeline_ = VK_NULL_HANDLE; }
    if (exposureResolvePipelineLayout_) { vkDestroyPipelineLayout(device, exposureResolvePipelineLayout_, nullptr); exposureResolvePipelineLayout_ = VK_NULL_HANDLE; }
    if (exposureResolveDescPool_) { vkDestroyDescriptorPool(device, exposureResolveDescPool_, nullptr); exposureResolveDescPool_ = VK_NULL_HANDLE; }
    if (exposureResolveDescSetLayout_) { vkDestroyDescriptorSetLayout(device, exposureResolveDescSetLayout_, nullptr); exposureResolveDescSetLayout_ = VK_NULL_HANDLE; }

    if (exposureSSBO_.buffer) accelBuilder_->DestroyAccelBuffer(exposureSSBO_);
    if (exposureStagingSSBO_.buffer) accelBuilder_->DestroyAccelBuffer(exposureStagingSSBO_);

    exposureResolveReady_ = false;
    Log(L"[VK Renderer] Auto-exposure resolve shutdown\n");
}

} // namespace vk
} // namespace acpt
