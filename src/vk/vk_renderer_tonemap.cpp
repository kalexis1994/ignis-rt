// vk_renderer_tonemap.cpp — tonemap compute pipeline + AgX 3D LUT management.
// Extracted from vk_renderer.cpp.

#include "vk_renderer.h"
#include "vk_context.h"
#include "vk_interop.h"
#include "ignis_log.h"
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <cstdio>

namespace acpt {
namespace vk {

namespace {
bool ResolveTonemapLutPath(std::string& outPath, uint64_t& outStamp)
{
    const char* lutCandidates[] = {
        "shaders/Runtime_LUT.cube",
        "shaders/AgX_Base_sRGB.cube",
    };

    std::error_code ec;
    for (const char* candidate : lutCandidates) {
        std::string resolved = IgnisResolvePath(candidate);
        if (!std::filesystem::exists(resolved, ec) || ec) {
            ec.clear();
            continue;
        }
        outPath = resolved;
        auto stamp = std::filesystem::last_write_time(resolved, ec);
        outStamp = ec ? 0ull : static_cast<uint64_t>(stamp.time_since_epoch().count());
        return true;
    }

    outPath.clear();
    outStamp = 0;
    return false;
}
}  // namespace

bool Renderer::LoadAgXLut() {
    // Load tonemap 3D LUT (.cube) based on Blender's view_transform.
    // The LUT is set via ignis_set_int("tonemap_lut", id) before create():
    //   0 = AgX (default), 1 = Filmic
    // Try runtime-baked LUT first (from Blender OCIO), fallback to AgX
    std::string lutPath;
    uint64_t lutStamp = 0;
    if (!ResolveTonemapLutPath(lutPath, lutStamp)) {
        Log(L"[VK Renderer] WARNING: No tonemap LUT found\n");
        return false;
    }
    std::ifstream lutFile(lutPath);
    if (!lutFile.is_open()) {
        Log(L"[VK Renderer] WARNING: Failed to open tonemap LUT: %S\n", lutPath.c_str());
        return false;
    }
    Log(L"[VK Renderer] Loaded tonemap LUT: %S\n", lutPath.c_str());

    int lutSize = 0;
    std::vector<float> lutData;
    std::string line;
    while (std::getline(lutFile, line)) {
        // Parse header lines
        if (line.find("LUT_3D_SIZE") != std::string::npos) {
            sscanf(line.c_str(), "LUT_3D_SIZE %d", &lutSize);
            continue;
        }
        if (line.empty() || line[0] == '#' || line[0] == 'T' || line[0] == 'D' || line[0] == 'L') {
            continue;
        }
        float r, g, b;
        if (sscanf(line.c_str(), "%f %f %f", &r, &g, &b) == 3) {
            lutData.push_back(r);
            lutData.push_back(g);
            lutData.push_back(b);
            lutData.push_back(1.0f);  // RGBA padding
        }
    }
    lutFile.close();

    if (lutSize == 0 || lutData.size() != (size_t)lutSize * lutSize * lutSize * 4) {
        Log(L"[VK Renderer] WARNING: AgX LUT parse error (size=%d, data=%zu)\n",
            lutSize, lutData.size());
        return false;
    }

    VkDevice device = context_->GetDevice();
    DestroyAgXLut();

    // Create 3D image
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_3D;
    imgInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    imgInfo.extent = {(uint32_t)lutSize, (uint32_t)lutSize, (uint32_t)lutSize};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(device, &imgInfo, nullptr, &agxLutImage_) != VK_SUCCESS) return false;

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, agxLutImage_, &memReqs);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &agxLutMemory_);
    vkBindImageMemory(device, agxLutImage_, agxLutMemory_, 0);

    // Staging buffer
    VkDeviceSize dataSize = lutData.size() * sizeof(float);
    VkBuffer stagingBuf;
    VkDeviceMemory stagingMem;
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = dataSize;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(device, &bufInfo, nullptr, &stagingBuf);
    VkMemoryRequirements bufReqs;
    vkGetBufferMemoryRequirements(device, stagingBuf, &bufReqs);
    VkMemoryAllocateInfo bufAlloc{};
    bufAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bufAlloc.allocationSize = bufReqs.size;
    bufAlloc.memoryTypeIndex = context_->FindMemoryType(bufReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &bufAlloc, nullptr, &stagingMem);
    vkBindBufferMemory(device, stagingBuf, stagingMem, 0);
    void* mapped;
    vkMapMemory(device, stagingMem, 0, dataSize, 0, &mapped);
    memcpy(mapped, lutData.data(), dataSize);
    vkUnmapMemory(device, stagingMem);

    // Copy to 3D image
    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = agxLutImage_;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {(uint32_t)lutSize, (uint32_t)lutSize, (uint32_t)lutSize};
    vkCmdCopyBufferToImage(cmd, stagingBuf, agxLutImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    context_->EndSingleTimeCommands(cmd);

    vkDestroyBuffer(device, stagingBuf, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    // Image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = agxLutImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device, &viewInfo, nullptr, &agxLutView_);

    // Sampler (trilinear for smooth interpolation)
    VkSamplerCreateInfo sampInfo{};
    sampInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampInfo.magFilter = VK_FILTER_LINEAR;
    sampInfo.minFilter = VK_FILTER_LINEAR;
    sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(device, &sampInfo, nullptr, &agxLutSampler_);

    Log(L"[VK Renderer] AgX 3D LUT loaded: %dx%dx%d (%zu KB)\n",
        lutSize, lutSize, lutSize, dataSize / 1024);
    agxLutPath_ = lutPath;
    agxLutStamp_ = lutStamp;
    return true;
}

bool Renderer::UploadLutData(const float* rgbData, uint32_t lutSize) {
    if (!rgbData || lutSize == 0 || !context_) return false;

    // Convert RGB → RGBA
    size_t totalEntries = (size_t)lutSize * lutSize * lutSize;
    std::vector<float> lutData(totalEntries * 4);
    for (size_t i = 0; i < totalEntries; i++) {
        lutData[i * 4 + 0] = rgbData[i * 3 + 0];
        lutData[i * 4 + 1] = rgbData[i * 3 + 1];
        lutData[i * 4 + 2] = rgbData[i * 3 + 2];
        lutData[i * 4 + 3] = 1.0f;
    }

    VkDevice device = context_->GetDevice();
    DestroyAgXLut();

    // Create 3D image
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_3D;
    imgInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    imgInfo.extent = {lutSize, lutSize, lutSize};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(device, &imgInfo, nullptr, &agxLutImage_) != VK_SUCCESS) return false;

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, agxLutImage_, &memReqs);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &agxLutMemory_);
    vkBindImageMemory(device, agxLutImage_, agxLutMemory_, 0);

    // Staging buffer
    VkDeviceSize dataSize = lutData.size() * sizeof(float);
    VkBuffer stagingBuf;
    VkDeviceMemory stagingMem;
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = dataSize;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(device, &bufInfo, nullptr, &stagingBuf);
    VkMemoryRequirements bufReqs;
    vkGetBufferMemoryRequirements(device, stagingBuf, &bufReqs);
    VkMemoryAllocateInfo bufAlloc{};
    bufAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bufAlloc.allocationSize = bufReqs.size;
    bufAlloc.memoryTypeIndex = context_->FindMemoryType(bufReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &bufAlloc, nullptr, &stagingMem);
    vkBindBufferMemory(device, stagingBuf, stagingMem, 0);
    void* mapped;
    vkMapMemory(device, stagingMem, 0, dataSize, 0, &mapped);
    memcpy(mapped, lutData.data(), dataSize);
    vkUnmapMemory(device, stagingMem);

    // Copy to 3D image
    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = agxLutImage_;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {lutSize, lutSize, lutSize};
    vkCmdCopyBufferToImage(cmd, stagingBuf, agxLutImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    context_->EndSingleTimeCommands(cmd);

    vkDestroyBuffer(device, stagingBuf, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    // Image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = agxLutImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device, &viewInfo, nullptr, &agxLutView_);

    // Sampler
    VkSamplerCreateInfo sampInfo{};
    sampInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampInfo.magFilter = VK_FILTER_LINEAR;
    sampInfo.minFilter = VK_FILTER_LINEAR;
    sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(device, &sampInfo, nullptr, &agxLutSampler_);

    Log(L"[VK Renderer] LUT uploaded in-memory: %dx%dx%d (%zu KB)\n",
        lutSize, lutSize, lutSize, dataSize / 1024);
    agxLutPath_ = "(memory)";
    agxLutStamp_ = 1;  // mark as valid
    return true;
}

void Renderer::DestroyAgXLut() {
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) return;

    if (agxLutSampler_) { vkDestroySampler(device, agxLutSampler_, nullptr); agxLutSampler_ = VK_NULL_HANDLE; }
    if (agxLutView_) { vkDestroyImageView(device, agxLutView_, nullptr); agxLutView_ = VK_NULL_HANDLE; }
    if (agxLutImage_) { vkDestroyImage(device, agxLutImage_, nullptr); agxLutImage_ = VK_NULL_HANDLE; }
    if (agxLutMemory_) { vkFreeMemory(device, agxLutMemory_, nullptr); agxLutMemory_ = VK_NULL_HANDLE; }
    agxLutPath_.clear();
    agxLutStamp_ = 0;
}

bool Renderer::ReloadAgXLutIfChanged() {
    if (agxLutPath_ == "(memory)") return false;  // managed via ignis_upload_lut
    std::string nextPath;
    uint64_t nextStamp = 0;
    if (!ResolveTonemapLutPath(nextPath, nextStamp)) {
        return false;
    }
    if (nextPath == agxLutPath_ && nextStamp == agxLutStamp_) {
        return false;
    }

    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) {
        return false;
    }

    Log(L"[VK Renderer] Tonemap LUT changed on disk, reloading: %S\n", nextPath.c_str());
    vkDeviceWaitIdle(device);
    if (!LoadAgXLut()) {
        Log(L"[VK Renderer] WARNING: Failed to reload tonemap LUT\n");
        return false;
    }
    UpdateTonemapDescriptors();
    return true;
}

bool Renderer::CreateTonemapPipeline() {
    VkDevice device = context_->GetDevice();

    // Sampler for HDR input
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &tonemapSampler_) != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create tonemap sampler\n");
        return false;
    }

    // Load AgX 3D LUT
    if (!LoadAgXLut()) {
        Log(L"[VK Renderer] WARNING: AgX LUT not loaded, falling back to polynomial\n");
    }

    // Descriptor set layout: binding 0 = sampler (HDR input), binding 1 = storage image (LDR output),
    // binding 2 = 3D LUT sampler (AgX color grading)
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &tonemapDescSetLayout_) != VK_SUCCESS) {
        return false;
    }

    // Push constant: tonemapMode (uint) + exposure (float) + saturation (float) + contrast (float) = 16 bytes
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 4 * sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &tonemapDescSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &tonemapPipelineLayout_) != VK_SUCCESS) {
        return false;
    }

    // Load compute shader
    std::ifstream file(IgnisResolvePath("shaders/tonemap.comp.spv"), std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[VK Renderer] WARNING: tonemap.comp.spv not found\n");
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
    pipelineInfo.layout = tonemapPipelineLayout_;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &tonemapPipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create tonemap pipeline\n");
        return false;
    }

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},  // HDR input + AgX LUT
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &tonemapDescPool_) != VK_SUCCESS) {
        return false;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = tonemapDescPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &tonemapDescSetLayout_;
    if (vkAllocateDescriptorSets(device, &allocInfo, &tonemapDescSet_) != VK_SUCCESS) {
        return false;
    }

    tonemapReady_ = true;
    Log(L"[VK Renderer] Tonemap pipeline created (post-DLSS HDR->LDR)\n");
    return true;
}

void Renderer::UpdateTonemapDescriptors() {
    if (!tonemapReady_ || !dlssHdrOutputView_ || !interop_) return;

    VkDevice device = context_->GetDevice();

    // binding 0: DLSS HDR output (sampler)
    VkDescriptorImageInfo hdrInfo{};
    hdrInfo.imageView = dlssHdrOutputView_;
    hdrInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    hdrInfo.sampler = tonemapSampler_;

    // binding 1: interop (storage image, LDR output)
    VkDescriptorImageInfo ldrInfo{};
    ldrInfo.imageView = interop_->GetSharedImageView();
    ldrInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // binding 2: AgX 3D LUT (sampler3D)
    VkDescriptorImageInfo lutInfo{};
    lutInfo.imageView = agxLutView_ ? agxLutView_ : dlssHdrOutputView_; // fallback
    lutInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    lutInfo.sampler = agxLutSampler_ ? agxLutSampler_ : tonemapSampler_;

    VkWriteDescriptorSet writes[3] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = tonemapDescSet_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &hdrInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = tonemapDescSet_;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &ldrInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = tonemapDescSet_;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &lutInfo;

    vkUpdateDescriptorSets(device, agxLutView_ ? 3u : 2u, writes, 0, nullptr);
}

void Renderer::ShutdownTonemap() {
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) return;

    if (tonemapPipeline_) { vkDestroyPipeline(device, tonemapPipeline_, nullptr); tonemapPipeline_ = VK_NULL_HANDLE; }
    if (tonemapPipelineLayout_) { vkDestroyPipelineLayout(device, tonemapPipelineLayout_, nullptr); tonemapPipelineLayout_ = VK_NULL_HANDLE; }
    if (tonemapDescPool_) { vkDestroyDescriptorPool(device, tonemapDescPool_, nullptr); tonemapDescPool_ = VK_NULL_HANDLE; }
    if (tonemapDescSetLayout_) { vkDestroyDescriptorSetLayout(device, tonemapDescSetLayout_, nullptr); tonemapDescSetLayout_ = VK_NULL_HANDLE; }
    if (tonemapSampler_) { vkDestroySampler(device, tonemapSampler_, nullptr); tonemapSampler_ = VK_NULL_HANDLE; }
    DestroyAgXLut();
    tonemapReady_ = false;
}

} // namespace vk
} // namespace acpt
