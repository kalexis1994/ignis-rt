// vk_renderer_surfel.cpp — Surfel GI cache resolve pass.
// Compute pipeline that resolves the Surfel hash table written by the
// path tracer into per-surfel radiance. Extracted from vk_renderer.cpp.

#include "vk_renderer.h"
#include "vk_context.h"
#include "vk_rt_resources.h"
#include "ignis_log.h"
#include <vector>
#include <fstream>

namespace acpt {
namespace vk {

bool Renderer::CreateSurfelResolvePipeline() {
    VkDevice device = context_->GetDevice();

    // Descriptor set layout: 2 SSBOs (hash entries + data)
    VkDescriptorSetLayoutBinding bindings[2] = {};
    for (int i = 0; i < 2; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &surfelResolveDescSetLayout_) != VK_SUCCESS)
        return false;

    // Pipeline layout with push constants
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 20;  // 4 uints + 1 float
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &surfelResolveDescSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &surfelResolvePipelineLayout_) != VK_SUCCESS)
        return false;

    // Load compute shader
    std::ifstream file(IgnisResolvePath("shaders/surfel_resolve.comp.spv"), std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[VK Renderer] WARNING: surfel_resolve.comp.spv not found\n");
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
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &shaderModule) != VK_SUCCESS)
        return false;

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = surfelResolvePipelineLayout_;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &surfelResolvePipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    if (result != VK_SUCCESS) return false;

    // Descriptor pool + set
    VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &surfelResolveDescPool_) != VK_SUCCESS)
        return false;

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = surfelResolveDescPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &surfelResolveDescSetLayout_;
    if (vkAllocateDescriptorSets(device, &allocInfo, &surfelResolveDescSet_) != VK_SUCCESS)
        return false;

    UpdateSurfelResolveDescriptors();
    surfelResolveReady_ = true;
    Log(L"[VK Renderer] Surfel GI resolve pipeline created\n");
    return true;
}

void Renderer::UpdateSurfelResolveDescriptors() {
    if (!rtResources_ || !rtResources_->HasSurfelBuffers()) return;
    VkDevice device = context_->GetDevice();

    VkDescriptorBufferInfo hashInfo{};
    hashInfo.buffer = rtResources_->GetSurfelBuffer(0);
    hashInfo.offset = 0;
    hashInfo.range = RTResources::SURFEL_CAPACITY * 8;

    VkDescriptorBufferInfo dataInfo{};
    dataInfo.buffer = rtResources_->GetSurfelBuffer(1);
    dataInfo.offset = 0;
    dataInfo.range = RTResources::SURFEL_CAPACITY * 32;

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = surfelResolveDescSet_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &hashInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = surfelResolveDescSet_;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &dataInfo;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

} // namespace vk
} // namespace acpt
