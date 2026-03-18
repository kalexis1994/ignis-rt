#include "vk_wavefront_pipeline.h"
#include "vk_context.h"
#include "vk_rt_pipeline.h"
#include "ignis_log.h"

#include <fstream>
#include <vector>
#include <cstring>

extern std::string IgnisResolvePath(const char*);

namespace acpt {
namespace vk {

static uint32_t FindMemoryType(VkPhysicalDevice physDevice, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

static bool CreateSSBO(VkDevice device, VkPhysicalDevice physDevice,
                        VkDeviceSize size, VkBuffer* buf, VkDeviceMemory* mem) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    if (vkCreateBuffer(device, &info, nullptr, buf) != VK_SUCCESS) return false;

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, *buf, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = FindMemoryType(physDevice, memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, mem) != VK_SUCCESS) {
        vkDestroyBuffer(device, *buf, nullptr);
        *buf = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(device, *buf, *mem, 0);
    return true;
}

static void DestroySSBO(VkDevice device, VkBuffer& buf, VkDeviceMemory& mem) {
    if (buf) { vkDestroyBuffer(device, buf, nullptr); buf = VK_NULL_HANDLE; }
    if (mem) { vkFreeMemory(device, mem, nullptr); mem = VK_NULL_HANDLE; }
}

bool WavefrontPipeline::LoadComputeShader(const char* path, VkShaderModule* outModule) {
    std::string fullPath = IgnisResolvePath(path);
    std::ifstream file(fullPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        Log(L"[Wavefront] ERROR: Cannot open shader: %S\n", path);
        return false;
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = fileSize;
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

    return vkCreateShaderModule(context_->GetDevice(), &createInfo, nullptr, outModule) == VK_SUCCESS;
}

bool WavefrontPipeline::Initialize(Context* context, RTPipeline* rtPipeline,
                                     uint32_t width, uint32_t height, uint32_t maxBounces) {
    context_ = context;
    rtPipeline_ = rtPipeline;
    maxPixels_ = width * height;

    Log(L"[Wavefront] Initializing wavefront pipeline (%ux%u = %u pixels)...\n",
        width, height, maxPixels_);

    if (!CreateBuffers(maxPixels_)) {
        Log(L"[Wavefront] ERROR: Buffer allocation failed\n");
        return false;
    }

    if (!CreateDescriptorSet()) {
        Log(L"[Wavefront] ERROR: Descriptor set creation failed\n");
        return false;
    }

    if (!CreatePipelines()) {
        Log(L"[Wavefront] ERROR: Pipeline creation failed\n");
        return false;
    }

    ready_ = true;
    Log(L"[Wavefront] Initialized OK (%u max pixels, %.1f MiB buffers)\n",
        maxPixels_,
        (float)(maxPixels_ * (48 + 32 + 48 * MAX_SHADOW_RAYS_PER_PATH + 32 + 64 + 16 + 12)) / (1024.0f * 1024.0f));
    return true;
}

void WavefrontPipeline::Shutdown() {
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) return;

    // Destroy pipelines
    auto destroyPipeline = [&](VkPipeline& p) {
        if (p) { vkDestroyPipeline(device, p, nullptr); p = VK_NULL_HANDLE; }
    };
    destroyPipeline(pipelineK0_);
    destroyPipeline(pipelineK1_);
    destroyPipeline(pipelineK2_);
    destroyPipeline(pipelineK3_);
    destroyPipeline(pipelineK4_);
    destroyPipeline(pipelineK5_);
    destroyPipeline(pipelineCompact_);

    if (pipelineLayout_) { vkDestroyPipelineLayout(device, pipelineLayout_, nullptr); pipelineLayout_ = VK_NULL_HANDLE; }
    if (wfDescPool_) { vkDestroyDescriptorPool(device, wfDescPool_, nullptr); wfDescPool_ = VK_NULL_HANDLE; }
    if (wfDescSetLayout_) { vkDestroyDescriptorSetLayout(device, wfDescSetLayout_, nullptr); wfDescSetLayout_ = VK_NULL_HANDLE; }

    // Destroy buffers
    DestroySSBO(device, pathStateBuffer_, pathStateMemory_);
    DestroySSBO(device, hitResultBuffer_, hitResultMemory_);
    DestroySSBO(device, shadowRayBuffer_, shadowRayMemory_);
    DestroySSBO(device, pixelRadianceBuffer_, pixelRadianceMemory_);
    DestroySSBO(device, primaryGBufBuffer_, primaryGBufMemory_);
    DestroySSBO(device, countersBuffer_, countersMemory_);
    DestroySSBO(device, indirectDispatchBuffer_, indirectDispatchMemory_);

    ready_ = false;
}

bool WavefrontPipeline::CreateBuffers(uint32_t pixelCount) {
    VkDevice device = context_->GetDevice();
    VkPhysicalDevice physDevice = context_->GetPhysicalDevice();

    // PathState: 48 bytes per pixel
    if (!CreateSSBO(device, physDevice, pixelCount * 48, &pathStateBuffer_, &pathStateMemory_)) return false;

    // HitResult: 32 bytes per pixel
    if (!CreateSSBO(device, physDevice, pixelCount * 32, &hitResultBuffer_, &hitResultMemory_)) return false;

    // ShadowRay: 48 bytes, max pixelCount * MAX_SHADOW_RAYS_PER_PATH
    // Limit to 2x pixelCount to save memory (most pixels generate 1-3 shadow rays)
    uint32_t maxShadowRays = pixelCount * 2;
    if (!CreateSSBO(device, physDevice, maxShadowRays * 48, &shadowRayBuffer_, &shadowRayMemory_)) return false;

    // PixelRadiance: 32 bytes per pixel
    if (!CreateSSBO(device, physDevice, pixelCount * 32, &pixelRadianceBuffer_, &pixelRadianceMemory_)) return false;

    // PrimaryGBuffer: 64 bytes per pixel
    if (!CreateSSBO(device, physDevice, pixelCount * 64, &primaryGBufBuffer_, &primaryGBufMemory_)) return false;

    // Counters: 16 bytes
    if (!CreateSSBO(device, physDevice, 16, &countersBuffer_, &countersMemory_)) return false;

    // Indirect dispatch args: 3 dispatches * 12 bytes
    VkBufferCreateInfo indirectInfo{};
    indirectInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    indirectInfo.size = 3 * sizeof(uint32_t) * 3; // 3 VkDispatchIndirectCommand
    indirectInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(device, &indirectInfo, nullptr, &indirectDispatchBuffer_) != VK_SUCCESS) return false;

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, indirectDispatchBuffer_, &memReqs);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = FindMemoryType(physDevice, memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &indirectDispatchMemory_) != VK_SUCCESS) return false;
    vkBindBufferMemory(device, indirectDispatchBuffer_, indirectDispatchMemory_, 0);

    Log(L"[Wavefront] Buffers allocated: PathState=%uMB HitResult=%uMB ShadowRay=%uMB Radiance=%uMB PrimaryGB=%uMB\n",
        (pixelCount * 48) >> 20, (pixelCount * 32) >> 20,
        (maxShadowRays * 48) >> 20, (pixelCount * 32) >> 20, (pixelCount * 64) >> 20);
    return true;
}

bool WavefrontPipeline::CreateDescriptorSet() {
    VkDevice device = context_->GetDevice();

    // Descriptor set 1 layout: 7 SSBOs for wavefront buffers
    VkDescriptorSetLayoutBinding bindings[7] = {};
    for (int i = 0; i < 7; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = (i == 6) ?
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER :  // indirect dispatch is also SSBO
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 7;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &wfDescSetLayout_) != VK_SUCCESS) return false;

    // Pool
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &wfDescPool_) != VK_SUCCESS) return false;

    // Allocate
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = wfDescPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &wfDescSetLayout_;

    if (vkAllocateDescriptorSets(device, &allocInfo, &wfDescSet_) != VK_SUCCESS) return false;

    // Write descriptors
    VkBuffer buffers[7] = {
        pathStateBuffer_, hitResultBuffer_, shadowRayBuffer_,
        pixelRadianceBuffer_, primaryGBufBuffer_, countersBuffer_,
        indirectDispatchBuffer_
    };

    VkDescriptorBufferInfo bufInfos[7] = {};
    VkWriteDescriptorSet writes[7] = {};
    for (int i = 0; i < 7; i++) {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = wfDescSet_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(device, 7, writes, 0, nullptr);
    return true;
}

bool WavefrontPipeline::CreatePipelines() {
    VkDevice device = context_->GetDevice();

    // Pipeline layout: set 0 (scene data from RTPipeline) + set 1 (wavefront buffers)
    // + push constants (width, height, frameIndex, maxBounces, currentBounce)
    VkDescriptorSetLayout setLayouts[2] = {
        rtPipeline_->GetDescriptorSetLayout(),
        wfDescSetLayout_
    };

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 5 * sizeof(uint32_t); // width, height, frameIndex, maxBounces, currentBounce

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 2;
    layoutInfo.pSetLayouts = setLayouts;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
        Log(L"[Wavefront] ERROR: Failed to create pipeline layout\n");
        return false;
    }

    // Create each compute pipeline
    auto createCompute = [&](const char* shaderPath, VkPipeline* pipeline) -> bool {
        VkShaderModule module;
        if (!LoadComputeShader(shaderPath, &module)) return false;

        VkComputePipelineCreateInfo pipeInfo{};
        pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeInfo.stage.module = module;
        pipeInfo.stage.pName = "main";
        pipeInfo.layout = pipelineLayout_;

        VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, pipeline);
        vkDestroyShaderModule(device, module, nullptr);
        return result == VK_SUCCESS;
    };

    // For now, only create K0 (camera rays) — others added in subsequent phases
    if (!createCompute("shaders/wavefront/wf_camera_rays.comp.spv", &pipelineK0_)) {
        Log(L"[Wavefront] WARNING: K0 (camera rays) shader not found — wavefront not ready\n");
        return false;
    }

    Log(L"[Wavefront] Compute pipelines created\n");
    return true;
}

void WavefrontPipeline::RecordDispatch(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                                         VkDescriptorSet sceneDescSet, uint32_t maxBounces) {
    if (!ready_) return;

    // TODO: implement full wavefront dispatch loop
    // Phase 2+: K0 → [K1 → K2 → K3 → K4] × bounces → K5
    Log(L"[Wavefront] RecordDispatch called (not yet functional)\n");
}

} // namespace vk
} // namespace acpt
