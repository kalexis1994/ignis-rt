// vk_renderer_sharc.cpp — SHARC (Spatially Hashed Radiance Cache) resolve pass.
// Compute pipeline that turns SHARC's accumulator hash table into the
// resolved cache used by the path tracer. Extracted from vk_renderer.cpp.

#include "vk_renderer.h"
#include "vk_context.h"
#include "vk_rt_resources.h"
#include "ignis_log.h"
#include <vector>
#include <fstream>

namespace acpt {
namespace vk {

bool Renderer::CreateSHARCResolvePipeline() {
    if (!rtResources_ || !rtResources_->HasSHARCBuffers()) return false;

    VkDevice device = context_->GetDevice();

    // Descriptor set layout: 2 SSBOs (write buffer, read buffer)
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &sharcResolveDescriptorSetLayout_) != VK_SUCCESS) {
        return false;
    }

    // Push constants: capacity, frameIndex, accFrameMax, staleMax, radianceScale
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 20;  // 4 uints + 1 float = 20 bytes

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &sharcResolveDescriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &sharcResolvePipelineLayout_) != VK_SUCCESS) {
        return false;
    }

    // Load compute shader
    std::ifstream file(IgnisResolvePath("shaders/sharc_resolve.comp.spv"), std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[VK Renderer] WARNING: sharc_resolve.comp.spv not found\n");
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
    pipelineInfo.layout = sharcResolvePipelineLayout_;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &sharcResolvePipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create SHARC resolve pipeline\n");
        return false;
    }

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &sharcResolveDescriptorPool_) != VK_SUCCESS) {
        return false;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = sharcResolveDescriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &sharcResolveDescriptorSetLayout_;
    if (vkAllocateDescriptorSets(device, &allocInfo, &sharcResolveDescriptorSet_) != VK_SUCCESS) {
        return false;
    }

    UpdateSHARCResolveDescriptors();

    sharcResolveReady_ = true;
    Log(L"[VK Renderer] SHARC resolve pipeline created\n");
    return true;
}

void Renderer::UpdateSHARCResolveDescriptors() {
    if (!rtResources_ || !rtResources_->HasSHARCBuffers()) return;

    VkDevice device = context_->GetDevice();

    VkDescriptorBufferInfo writeInfo{};
    writeInfo.buffer = rtResources_->GetSHARCBuffer(0);  // hashEntries
    writeInfo.offset = 0;
    writeInfo.range = RTResources::SHARC_CAPACITY * 8;

    VkDescriptorBufferInfo readInfo{};
    readInfo.buffer = rtResources_->GetSHARCBuffer(1);  // combined accum+resolved
    readInfo.offset = 0;
    readInfo.range = RTResources::SHARC_CAPACITY * 56;  // accum+resolved+guide

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = sharcResolveDescriptorSet_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &writeInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = sharcResolveDescriptorSet_;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &readInfo;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

void Renderer::ShutdownSHARCResolve() {
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) return;

    if (sharcResolvePipeline_) { vkDestroyPipeline(device, sharcResolvePipeline_, nullptr); sharcResolvePipeline_ = VK_NULL_HANDLE; }
    if (sharcResolvePipelineLayout_) { vkDestroyPipelineLayout(device, sharcResolvePipelineLayout_, nullptr); sharcResolvePipelineLayout_ = VK_NULL_HANDLE; }
    if (sharcResolveDescriptorPool_) { vkDestroyDescriptorPool(device, sharcResolveDescriptorPool_, nullptr); sharcResolveDescriptorPool_ = VK_NULL_HANDLE; }
    if (sharcResolveDescriptorSetLayout_) { vkDestroyDescriptorSetLayout(device, sharcResolveDescriptorSetLayout_, nullptr); sharcResolveDescriptorSetLayout_ = VK_NULL_HANDLE; }
    sharcResolveReady_ = false;
}

} // namespace vk
} // namespace acpt
