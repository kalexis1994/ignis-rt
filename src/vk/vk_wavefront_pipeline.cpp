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
    DestroySSBO(device, pathStateBuffer_[0], pathStateMemory_[0]);
    DestroySSBO(device, pathStateBuffer_[1], pathStateMemory_[1]);
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

    // PathState: 48 bytes per pixel, double-buffered (ping-pong)
    if (!CreateSSBO(device, physDevice, pixelCount * 48, &pathStateBuffer_[0], &pathStateMemory_[0])) return false;
    if (!CreateSSBO(device, physDevice, pixelCount * 48, &pathStateBuffer_[1], &pathStateMemory_[1])) return false;

    // HitResult: 80 bytes per pixel (20 floats: 8 base + 12 objToWorld)
    if (!CreateSSBO(device, physDevice, pixelCount * 80, &hitResultBuffer_, &hitResultMemory_)) return false;

    // ShadowRay: 48 bytes per ray. Budget for sun + lights + emissive per pixel.
    uint32_t maxShadowRays = pixelCount * MAX_SHADOW_RAYS_PER_PATH;
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

    // Descriptor set 1 layout: 8 SSBOs for wavefront buffers
    // binding 0: PathState read (current bounce)
    // binding 1: HitResult
    // binding 2: ShadowRay
    // binding 3: PixelRadiance
    // binding 4: PrimaryGBuffer
    // binding 5: Counters
    // binding 6: IndirectDispatch
    // binding 7: PathState write (next bounce)
    VkDescriptorSetLayoutBinding bindings[8] = {};
    for (int i = 0; i < 8; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 8;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &wfDescSetLayout_) != VK_SUCCESS) return false;

    // Pool
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
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

    // Write descriptors (initial: buffer A=read at binding 0, buffer B=write at binding 7)
    VkBuffer buffers[8] = {
        pathStateBuffer_[0], hitResultBuffer_, shadowRayBuffer_,
        pixelRadianceBuffer_, primaryGBufBuffer_, countersBuffer_,
        indirectDispatchBuffer_, pathStateBuffer_[1]
    };

    VkDescriptorBufferInfo bufInfos[8] = {};
    VkWriteDescriptorSet writes[8] = {};
    for (int i = 0; i < 8; i++) {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = wfDescSet_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }

    vkUpdateDescriptorSets(device, 8, writes, 0, nullptr);
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

    if (!createCompute("shaders/wavefront/wf_camera_rays.comp.spv", &pipelineK0_)) {
        Log(L"[Wavefront] ERROR: K0 (camera rays) shader not found\n");
        return false;
    }
    if (!createCompute("shaders/wavefront/wf_intersect.comp.spv", &pipelineK1_)) {
        Log(L"[Wavefront] ERROR: K1 (intersect) shader not found\n");
        return false;
    }
    if (!createCompute("shaders/wavefront/wf_shade.comp.spv", &pipelineK2_)) {
        Log(L"[Wavefront] ERROR: K2 (shade) shader not found\n");
        return false;
    }
    if (!createCompute("shaders/wavefront/wf_shadow_accumulate.comp.spv", &pipelineK3_)) {
        Log(L"[Wavefront] ERROR: K3 (shadow+accumulate) shader not found\n");
        return false;
    }
    // K4 merged into K3
    pipelineK4_ = VK_NULL_HANDLE;
    if (!createCompute("shaders/wavefront/wf_output.comp.spv", &pipelineK5_)) {
        Log(L"[Wavefront] ERROR: K5 (output) shader not found\n");
        return false;
    }
    if (!createCompute("shaders/wavefront/wf_compact.comp.spv", &pipelineCompact_)) {
        Log(L"[Wavefront] ERROR: Compact shader not found\n");
        return false;
    }

    Log(L"[Wavefront] All compute pipelines created (K0-K5 + compact)\n");
    return true;
}

void WavefrontPipeline::RecordDispatch(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                                         VkDescriptorSet sceneDescSet, uint32_t maxBounces) {
    if (!ready_) return;

    uint32_t totalPixels = width * height;
    uint32_t groupsAll = (totalPixels + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    // Push constants shared by all kernels
    struct {
        uint32_t width, height, frameIndex, maxBounces, currentBounce;
    } push;
    push.width = width;
    push.height = height;
    push.frameIndex = frameIndex_++;
    push.maxBounces = maxBounces;
    push.currentBounce = 0;

    // Bind both descriptor sets (set 0 = scene, set 1 = wavefront)
    VkDescriptorSet sets[2] = { sceneDescSet, wfDescSet_ };

    // Reset ping-pong to buffer 0 for this frame
    pathStateCurrent_ = 0;
    {
        VkDescriptorBufferInfo resetInfos[2] = {};
        VkWriteDescriptorSet resetWrites[2] = {};
        resetInfos[0].buffer = pathStateBuffer_[0]; resetInfos[0].range = VK_WHOLE_SIZE;
        resetWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        resetWrites[0].dstSet = wfDescSet_; resetWrites[0].dstBinding = 0;
        resetWrites[0].descriptorCount = 1; resetWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        resetWrites[0].pBufferInfo = &resetInfos[0];
        resetInfos[1].buffer = pathStateBuffer_[1]; resetInfos[1].range = VK_WHOLE_SIZE;
        resetWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        resetWrites[1].dstSet = wfDescSet_; resetWrites[1].dstBinding = 7;
        resetWrites[1].descriptorCount = 1; resetWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        resetWrites[1].pBufferInfo = &resetInfos[1];
        vkUpdateDescriptorSets(context_->GetDevice(), 2, resetWrites, 0, nullptr);
    }

    // Clear pixel radiance and primary G-buffer
    vkCmdFillBuffer(cmd, pixelRadianceBuffer_, 0, totalPixels * 32, 0);
    vkCmdFillBuffer(cmd, primaryGBufBuffer_, 0, totalPixels * 64, 0);
    // Set viewZ to 1e7 (sky default) — float 1e7 = 0x4B189680
    // Actually, 0 is fine: K5 checks abs(viewZ) < 1e6 for primaryHit

    VkMemoryBarrier clearBarrier{};
    clearBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    clearBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    clearBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &clearBarrier, 0, nullptr, 0, nullptr);

    // K0: Camera ray generation
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineK0_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
        0, 2, sets, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, groupsAll, 1, 1);

    // Barrier: K0 writes → K1 reads
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    // Bounce loop: K1 → K2 → compact(shadow args) → K3 → K4 → compact(swap+active args)
    for (uint32_t bounce = 0; bounce < maxBounces; bounce++) {
        push.currentBounce = bounce;

        // K1: Intersection
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineK1_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
            0, 2, sets, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        if (bounce == 0) {
            vkCmdDispatch(cmd, groupsAll, 1, 1);  // bounce 0: all pixels
        } else {
            vkCmdDispatchIndirect(cmd, indirectDispatchBuffer_, 0);  // bounce 1+: only active rays
        }

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // K2: Shade + NEE + bounce setup
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineK2_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
            0, 2, sets, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        if (bounce == 0) {
            vkCmdDispatch(cmd, groupsAll, 1, 1);
        } else {
            vkCmdDispatchIndirect(cmd, indirectDispatchBuffer_, 0);
        }

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // Compact: write shadow dispatch args + swap counters + write active dispatch args
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineCompact_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
            0, 2, sets, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, 1, 1, 1);

        // Barrier: compact writes indirect args → K3 reads them
        VkMemoryBarrier indirectBarrier{};
        indirectBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        indirectBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        indirectBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &indirectBarrier, 0, nullptr, 0, nullptr);

        // K3: Shadow trace + accumulate (merged — indirect dispatch)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineK3_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
            0, 2, sets, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatchIndirect(cmd, indirectDispatchBuffer_, 3 * sizeof(uint32_t));

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);

        // Reset shadowRayCount for next bounce (AFTER K3 used it)
        // Offset 4 bytes = shadowRayCount field in CountersBuffer
        vkCmdFillBuffer(cmd, countersBuffer_, sizeof(uint32_t), sizeof(uint32_t), 0);

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // Swap PathState ping-pong: next-bounce writes become current-bounce reads
        pathStateCurrent_ = 1 - pathStateCurrent_;
        VkDescriptorBufferInfo swapInfos[2] = {};
        VkWriteDescriptorSet swapWrites[2] = {};
        // binding 0 = read buffer (what was the write buffer)
        swapInfos[0].buffer = pathStateBuffer_[pathStateCurrent_];
        swapInfos[0].range = VK_WHOLE_SIZE;
        swapWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        swapWrites[0].dstSet = wfDescSet_;
        swapWrites[0].dstBinding = 0;
        swapWrites[0].descriptorCount = 1;
        swapWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        swapWrites[0].pBufferInfo = &swapInfos[0];
        // binding 7 = write buffer (what was the read buffer)
        swapInfos[1].buffer = pathStateBuffer_[1 - pathStateCurrent_];
        swapInfos[1].range = VK_WHOLE_SIZE;
        swapWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        swapWrites[1].dstSet = wfDescSet_;
        swapWrites[1].dstBinding = 7;
        swapWrites[1].descriptorCount = 1;
        swapWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        swapWrites[1].pBufferInfo = &swapInfos[1];
        vkUpdateDescriptorSets(context_->GetDevice(), 2, swapWrites, 0, nullptr);
    }

    // K5: Output
    uint32_t groupsX = (width + 7) / 8;
    uint32_t groupsY = (height + 7) / 8;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineK5_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
        0, 2, sets, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
}

} // namespace vk
} // namespace acpt
