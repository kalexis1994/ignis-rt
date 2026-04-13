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
    destroyPipeline(pipelinePTTemporal_);
    destroyPipeline(pipelinePTSpatial_);
    destroyPipeline(pipelinePTFinal_);
    destroyPipeline(pipelineStablePlanes_);

    destroyPipeline(pipelineK2RT_);
    if (pipelineLayoutRT_) { vkDestroyPipelineLayout(device, pipelineLayoutRT_, nullptr); pipelineLayoutRT_ = VK_NULL_HANDLE; }
    if (sbtK2Buffer_) { vkDestroyBuffer(device, sbtK2Buffer_, nullptr); sbtK2Buffer_ = VK_NULL_HANDLE; }
    if (sbtK2Memory_) { vkFreeMemory(device, sbtK2Memory_, nullptr); sbtK2Memory_ = VK_NULL_HANDLE; }
    if (pipelineLayout_) { vkDestroyPipelineLayout(device, pipelineLayout_, nullptr); pipelineLayout_ = VK_NULL_HANDLE; }
    if (wfDescPool_) { vkDestroyDescriptorPool(device, wfDescPool_, nullptr); wfDescPool_ = VK_NULL_HANDLE; }
    if (wfDescSetLayout_) { vkDestroyDescriptorSetLayout(device, wfDescSetLayout_, nullptr); wfDescSetLayout_ = VK_NULL_HANDLE; }

    // Destroy SoA PathState buffers
    for (int i = 0; i < 2; i++) {
        DestroySSBO(device, originDirBuffer_[i], originDirMemory_[i]);
        DestroySSBO(device, pixelRngBuffer_[i], pixelRngMemory_[i]);
        DestroySSBO(device, throughputBuffer_[i], throughputMemory_[i]);
        DestroySSBO(device, flagsBuffer_[i], flagsMemory_[i]);
    }
    DestroySSBO(device, hitResultBuffer_, hitResultMemory_);
    DestroySSBO(device, shadowRayBuffer_, shadowRayMemory_);
    DestroySSBO(device, pixelRadianceBuffer_, pixelRadianceMemory_);
    DestroySSBO(device, primaryGBufBuffer_, primaryGBufMemory_);
    DestroySSBO(device, countersBuffer_, countersMemory_);
    DestroySSBO(device, indirectDispatchBuffer_, indirectDispatchMemory_);
    DestroySSBO(device, sharcStateBuffer_, sharcStateMemory_);
    for (int i = 0; i < 2; i++)
        DestroySSBO(device, ptReservoirBuffer_[i], ptReservoirMemory_[i]);
    DestroySSBO(device, ptPathRecordBuffer_, ptPathRecordMemory_);
    DestroySSBO(device, spHeaderBuffer_, spHeaderMemory_);
    DestroySSBO(device, spDataBuffer_, spDataMemory_);

    ready_ = false;
}

bool WavefrontPipeline::CreateK2RTPipeline() {
    VkDevice device = context_->GetDevice();

    // Load RT function pointers
    vkCreateRayTracingPipelinesKHR_ = (PFN_vkCreateRayTracingPipelinesKHR)
        vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR");
    vkGetRayTracingShaderGroupHandlesKHR_ = (PFN_vkGetRayTracingShaderGroupHandlesKHR)
        vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR");
    vkCmdTraceRaysKHR_ = (PFN_vkCmdTraceRaysKHR)
        vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR");
    vkGetBufferDeviceAddressKHR_ = (PFN_vkGetBufferDeviceAddressKHR)
        vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR");

    if (!vkCreateRayTracingPipelinesKHR_ || !vkGetRayTracingShaderGroupHandlesKHR_ ||
        !vkCmdTraceRaysKHR_ || !vkGetBufferDeviceAddressKHR_) {
        Log(L"[Wavefront] RT function pointers not available\n");
        return false;
    }

    // Load raygen shader module
    VkShaderModule raygenModule = VK_NULL_HANDLE;
    if (!LoadComputeShader("shaders/wavefront/wf_shade.rgen.spv", &raygenModule)) {
        Log(L"[Wavefront] wf_shade.rgen.spv not found (SER disabled)\n");
        return false;
    }

    // Create RT pipeline (raygen only — uses ray queries, no closest hit/miss)
    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stageInfo.module = raygenModule;
    stageInfo.pName = "main";

    VkRayTracingShaderGroupCreateInfoKHR groupInfo{};
    groupInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groupInfo.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groupInfo.generalShader = 0;
    groupInfo.closestHitShader = VK_SHADER_UNUSED_KHR;
    groupInfo.anyHitShader = VK_SHADER_UNUSED_KHR;
    groupInfo.intersectionShader = VK_SHADER_UNUSED_KHR;

    VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineInfo.stageCount = 1;
    pipelineInfo.pStages = &stageInfo;
    pipelineInfo.groupCount = 1;
    pipelineInfo.pGroups = &groupInfo;
    pipelineInfo.maxPipelineRayRecursionDepth = 1;
    pipelineInfo.layout = pipelineLayoutRT_;

    VkResult result = vkCreateRayTracingPipelinesKHR_(device,
        VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipelineK2RT_);

    vkDestroyShaderModule(device, raygenModule, nullptr);

    if (result != VK_SUCCESS) {
        Log(L"[Wavefront] Failed to create K2 RT pipeline (result=%d)\n", result);
        return false;
    }

    // Create SBT (raygen only)
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(context_->GetPhysicalDevice(), &props2);

    uint32_t handleSize = rtProps.shaderGroupHandleSize;
    uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
    uint32_t baseAlignment = rtProps.shaderGroupBaseAlignment;
    uint32_t handleSizeAligned = (handleSize + handleAlignment - 1) & ~(handleAlignment - 1);
    uint32_t sbtSize = baseAlignment;

    std::vector<uint8_t> handleData(handleSize);
    vkGetRayTracingShaderGroupHandlesKHR_(device, pipelineK2RT_, 0, 1, handleSize, handleData.data());

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = sbtSize;
    bufInfo.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufInfo, nullptr, &sbtK2Buffer_) != VK_SUCCESS) return false;

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, sbtK2Buffer_, &memReqs);

    VkMemoryAllocateFlagsInfo allocFlags{};
    allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &allocFlags;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = FindMemoryType(context_->GetPhysicalDevice(), memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &sbtK2Memory_) != VK_SUCCESS) return false;
    vkBindBufferMemory(device, sbtK2Buffer_, sbtK2Memory_, 0);

    void* mapped;
    vkMapMemory(device, sbtK2Memory_, 0, sbtSize, 0, &mapped);
    memset(mapped, 0, sbtSize);
    memcpy(mapped, handleData.data(), handleSize);
    vkUnmapMemory(device, sbtK2Memory_);

    VkBufferDeviceAddressInfo addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = sbtK2Buffer_;
    VkDeviceAddress sbtAddress = vkGetBufferDeviceAddressKHR_(device, &addrInfo);

    sbtK2RaygenRegion_.deviceAddress = sbtAddress;
    sbtK2RaygenRegion_.stride = handleSizeAligned;
    sbtK2RaygenRegion_.size = handleSizeAligned;
    sbtK2MissRegion_ = {};
    sbtK2HitRegion_ = {};
    sbtK2CallableRegion_ = {};

    Log(L"[Wavefront] K2 SBT created (handleSize=%u, aligned=%u)\n", handleSize, handleSizeAligned);
    return true;
}

bool WavefrontPipeline::CreateBuffers(uint32_t pixelCount) {
    VkDevice device = context_->GetDevice();
    VkPhysicalDevice physDevice = context_->GetPhysicalDevice();

    // PathState SoA — 4 field buffers, each double-buffered (48 bytes/path total)
    for (int i = 0; i < 2; i++) {
        if (!CreateSSBO(device, physDevice, pixelCount * 24, &originDirBuffer_[i], &originDirMemory_[i])) return false;     // 6 floats
        if (!CreateSSBO(device, physDevice, pixelCount * 8,  &pixelRngBuffer_[i], &pixelRngMemory_[i])) return false;       // 2 uints
        if (!CreateSSBO(device, physDevice, pixelCount * 12, &throughputBuffer_[i], &throughputMemory_[i])) return false;    // 3 floats
        if (!CreateSSBO(device, physDevice, pixelCount * 4,  &flagsBuffer_[i], &flagsMemory_[i])) return false;             // 1 uint
    }

    // HitResult: 80 bytes per pixel (20 floats: 8 base + 12 objToWorld)
    if (!CreateSSBO(device, physDevice, pixelCount * 80, &hitResultBuffer_, &hitResultMemory_)) return false;

    // ShadowRay: 48 bytes per ray. Budget for sun + lights + emissive per pixel.
    uint32_t maxShadowRays = pixelCount * MAX_SHADOW_RAYS_PER_PATH;
    if (!CreateSSBO(device, physDevice, maxShadowRays * 48, &shadowRayBuffer_, &shadowRayMemory_)) return false;

    // PixelRadiance: 32 bytes per pixel
    if (!CreateSSBO(device, physDevice, pixelCount * 32, &pixelRadianceBuffer_, &pixelRadianceMemory_)) return false;

    // PrimaryGBuffer: 96 bytes per pixel (16 base floats + 8 PSR floats = 24 floats)
    if (!CreateSSBO(device, physDevice, pixelCount * 96, &primaryGBufBuffer_, &primaryGBufMemory_)) return false;

    // ReSTIR PT: reservoirs (128 bytes/pixel, 2x ping-pong) + path records (96 bytes/pixel)
    for (int i = 0; i < 2; i++) {
        if (!CreateSSBO(device, physDevice, pixelCount * 128, &ptReservoirBuffer_[i], &ptReservoirMemory_[i])) return false;
    }
    if (!CreateSSBO(device, physDevice, pixelCount * 96, &ptPathRecordBuffer_, &ptPathRecordMemory_)) return false;

    // Stable Planes: header (16 bytes/pixel) + data (24 vec4s = 384 bytes/pixel)
    if (!CreateSSBO(device, physDevice, pixelCount * 16, &spHeaderBuffer_, &spHeaderMemory_)) return false;
    if (!CreateSSBO(device, physDevice, pixelCount * 384, &spDataBuffer_, &spDataMemory_)) return false;

    // Counters: 64 bytes (16 base + 20 sort bin counts + 20 sort bin offsets + 8 pad)
    if (!CreateSSBO(device, physDevice, 64, &countersBuffer_, &countersMemory_)) return false;

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

    // SharcState: 80 bytes per pixel (20 uints: 4 cacheIndices + 4 vec3 weights + pathLength + pad)
    // Indexed by pixelIndex, persists across bounces within a frame. Cleared per SPP sample.
    if (!CreateSSBO(device, physDevice, pixelCount * 80, &sharcStateBuffer_, &sharcStateMemory_)) return false;

    Log(L"[Wavefront] Buffers allocated: PathState=%uMB HitResult=%uMB ShadowRay=%uMB Radiance=%uMB PrimaryGB=%uMB SharcState=%uMB\n",
        (pixelCount * 48) >> 20, (pixelCount * 32) >> 20,
        (maxShadowRays * 48) >> 20, (pixelCount * 32) >> 20, (pixelCount * 64) >> 20,
        (pixelCount * 80) >> 20);
    return true;
}

bool WavefrontPipeline::CreateDescriptorSet() {
    VkDevice device = context_->GetDevice();

    // Descriptor set 1 layout: 15 SSBOs for wavefront buffers (SoA PathState)
    // binding 0:  originDir READ       binding 7:  originDir WRITE
    // binding 1:  HitResult            binding 8:  SharcState
    // binding 2:  ShadowRay            binding 9:  pixelRng READ
    // binding 3:  PixelRadiance        binding 10: pixelRng WRITE
    // binding 4:  PrimaryGBuffer       binding 11: throughput READ
    // binding 5:  Counters             binding 12: throughput WRITE
    // binding 6:  IndirectDispatch     binding 13: flags READ
    //                                  binding 14: flags WRITE
    // 15 original + 3 ReSTIR PT + 2 Stable Planes
    constexpr int NUM_BINDINGS = 20;
    VkDescriptorSetLayoutBinding bindings[NUM_BINDINGS] = {};
    for (int i = 0; i < NUM_BINDINGS; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = NUM_BINDINGS;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &wfDescSetLayout_) != VK_SUCCESS) return false;

    // Pool — 2 sets × 15 SSBOs = 30 descriptors
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NUM_BINDINGS * 2};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &wfDescPool_) != VK_SUCCESS) return false;

    VkDescriptorSetLayout layouts[2] = { wfDescSetLayout_, wfDescSetLayout_ };
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = wfDescPool_;
    allocInfo.descriptorSetCount = 2;
    allocInfo.pSetLayouts = layouts;

    if (vkAllocateDescriptorSets(device, &allocInfo, wfDescSet_) != VK_SUCCESS) return false;

    // Write descriptors for BOTH sets at init time (never updated during recording)
    // Ping-pong: setIdx=0 reads from [0], writes to [1]; setIdx=1 reads from [1], writes to [0]
    for (int setIdx = 0; setIdx < 2; setIdx++) {
        int r = setIdx;       // read index
        int w = 1 - setIdx;   // write index

        VkBuffer buffers[NUM_BINDINGS] = {
            originDirBuffer_[r],           // 0:  originDir READ
            hitResultBuffer_,              // 1:  HitResult
            shadowRayBuffer_,              // 2:  ShadowRay
            pixelRadianceBuffer_,          // 3:  PixelRadiance
            primaryGBufBuffer_,            // 4:  PrimaryGBuffer
            countersBuffer_,               // 5:  Counters
            indirectDispatchBuffer_,        // 6:  IndirectDispatch
            originDirBuffer_[w],           // 7:  originDir WRITE
            sharcStateBuffer_,             // 8:  SharcState
            pixelRngBuffer_[r],            // 9:  pixelRng READ
            pixelRngBuffer_[w],            // 10: pixelRng WRITE
            throughputBuffer_[r],          // 11: throughput READ
            throughputBuffer_[w],          // 12: throughput WRITE
            flagsBuffer_[r],              // 13: flags READ
            flagsBuffer_[w],              // 14: flags WRITE
            ptReservoirBuffer_[0],        // 15: PT reservoir current write
            ptReservoirBuffer_[1],        // 16: PT reservoir previous read
            ptPathRecordBuffer_,           // 17: PT path record
            spHeaderBuffer_,               // 18: Stable Planes header
            spDataBuffer_,                 // 19: Stable Planes data
        };

        VkDescriptorBufferInfo bufInfos[NUM_BINDINGS] = {};
        VkWriteDescriptorSet writes[NUM_BINDINGS] = {};
        for (int i = 0; i < NUM_BINDINGS; i++) {
            bufInfos[i].buffer = buffers[i];
            bufInfos[i].range = VK_WHOLE_SIZE;

            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = wfDescSet_[setIdx];
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufInfos[i];
        }

        vkUpdateDescriptorSets(device, NUM_BINDINGS, writes, 0, nullptr);
    }

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
    pushRange.size = 7 * sizeof(uint32_t); // width, height, frameIndex, maxBounces, currentBounce, spp, sampleIdx

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

    // ReSTIR PT kernels (optional — skip if shaders not found)
    if (!createCompute("shaders/wavefront/wf_pt_temporal.comp.spv", &pipelinePTTemporal_)) {
        Log(L"[Wavefront] WARNING: PT temporal shader not found (ReSTIR PT disabled)\n");
    }
    if (!createCompute("shaders/wavefront/wf_pt_spatial.comp.spv", &pipelinePTSpatial_)) {
        Log(L"[Wavefront] WARNING: PT spatial shader not found\n");
    }
    if (!createCompute("shaders/wavefront/wf_pt_final.comp.spv", &pipelinePTFinal_)) {
        Log(L"[Wavefront] WARNING: PT final shader not found\n");
    }
    if (!createCompute("shaders/wavefront/wf_stable_planes.comp.spv", &pipelineStablePlanes_)) {
        Log(L"[Wavefront] WARNING: Stable planes shader not found\n");
    }

    Log(L"[Wavefront] All compute pipelines created (K0-K5 + compact + sort + PT)\n");

    // RT pipeline layout for K2 (same descriptor sets, push constants for RAYGEN stage)
    VkPushConstantRange pushRangeRT{};
    pushRangeRT.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    pushRangeRT.offset = 0;
    pushRangeRT.size = 7 * sizeof(uint32_t);

    VkPipelineLayoutCreateInfo layoutInfoRT{};
    layoutInfoRT.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfoRT.setLayoutCount = 2;
    layoutInfoRT.pSetLayouts = setLayouts;
    layoutInfoRT.pushConstantRangeCount = 1;
    layoutInfoRT.pPushConstantRanges = &pushRangeRT;

    if (vkCreatePipelineLayout(device, &layoutInfoRT, nullptr, &pipelineLayoutRT_) != VK_SUCCESS) {
        Log(L"[Wavefront] WARNING: Failed to create RT pipeline layout (SER disabled)\n");
    }

    // Attempt to create RT K2 pipeline for hardware SER
    // Only enable on RTX 40+ (Ada) — RTX 30 has software SER which is slower than material sort
    serAvailable_ = false;
    if (pipelineLayoutRT_ && context_->IsHardwareSERCapable() && CreateK2RTPipeline()) {
        serAvailable_ = true;
        Log(L"[Wavefront] K2 RT pipeline created (hardware SER on RTX %u)\n", context_->GetRTXSeries());
    } else {
        Log(L"[Wavefront] K2 running as compute + material sort (RTX %u)\n", context_->GetRTXSeries());
    }

    return true;
}

void WavefrontPipeline::RecordDispatch(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                                         VkDescriptorSet sceneDescSet, uint32_t maxBounces,
                                         uint32_t spp) {
    if (!ready_) return;

    uint32_t totalPixels = width * height;
    uint32_t groupsAll = (totalPixels + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    if (spp < 1) spp = 1;
    if (spp > 128) spp = 128;

    // Push constants shared by all kernels
    struct {
        uint32_t width, height, frameIndex, maxBounces, currentBounce, spp, sampleIdx;
    } push;
    push.width = width;
    push.height = height;
    push.frameIndex = frameIndex_++;
    push.maxBounces = maxBounces;
    push.currentBounce = 0;
    push.spp = spp;
    push.sampleIdx = 0;

    // Ping-pong: use two pre-allocated descriptor sets (no host updates during recording)
    // Set 0 (A): read=buffer[0], write=buffer[1]
    // Set 1 (B): read=buffer[1], write=buffer[0]
    pathStateCurrent_ = 0;

    // Clear pixel radiance, primary G-buffer, and SharcState (once per frame, before all SPP samples)
    vkCmdFillBuffer(cmd, pixelRadianceBuffer_, 0, totalPixels * 32, 0);
    vkCmdFillBuffer(cmd, primaryGBufBuffer_, 0, totalPixels * 64, 0);
    vkCmdFillBuffer(cmd, sharcStateBuffer_, 0, totalPixels * 80, 0);  // pathLength=0 = initialized

    VkMemoryBarrier clearBarrier{};
    clearBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    clearBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    clearBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &clearBarrier, 0, nullptr, 0, nullptr);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    // ---- SPP loop: each sample generates new camera rays + full bounce chain ----
    for (uint32_t sampleIdx = 0; sampleIdx < spp; sampleIdx++) {
        push.sampleIdx = sampleIdx;

        // K0: Camera ray generation — always writes to buffer[0] (set A: binding 0)
        VkDescriptorSet setsK0[2] = { sceneDescSet, wfDescSet_[0] };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineK0_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
            0, 2, setsK0, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, groupsAll, 1, 1);

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    // Bounce loop: K1 → K2 → compact(shadow args) → K3 → swap
    // Extra iterations for glass: glass bounces don't consume the shader bounce
    // counter but DO consume dispatch loop iterations. Add headroom for glass
    // traversal (enter + exit = 2 extra iterations minimum).
    uint32_t glassHeadroom = 8; // covers multi-layer glass stacks (2 closed meshes = 4-6 iterations)
    uint32_t totalIterations = maxBounces + glassHeadroom;
    for (uint32_t bounce = 0; bounce < totalIterations; bounce++) {
        push.currentBounce = bounce;

        // K1: Intersection — bind current ping-pong descriptor set
        VkDescriptorSet setsBounce[2] = { sceneDescSet, wfDescSet_[pathStateCurrent_] };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineK1_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
            0, 2, setsBounce, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        if (bounce == 0) {
            vkCmdDispatch(cmd, groupsAll, 1, 1);  // bounce 0: all pixels
        } else {
            vkCmdDispatchIndirect(cmd, indirectDispatchBuffer_, 0);  // bounce 1+: only active rays
        }

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // K2: Shade + NEE + bounce setup
        if (serAvailable_) {
            // RT path: raygen shader with hardware SER (reorderThreadNV)
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineK2RT_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayoutRT_,
                0, 2, setsBounce, 0, nullptr);
            vkCmdPushConstants(cmd, pipelineLayoutRT_, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(push), &push);
            vkCmdTraceRaysKHR_(cmd,
                &sbtK2RaygenRegion_, &sbtK2MissRegion_, &sbtK2HitRegion_, &sbtK2CallableRegion_,
                totalPixels, 1, 1);
        } else {
            // Compute fallback (no SER)
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineK2_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
                0, 2, setsBounce, 0, nullptr);
            vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            if (bounce == 0) {
                vkCmdDispatch(cmd, groupsAll, 1, 1);
            } else {
                vkCmdDispatchIndirect(cmd, indirectDispatchBuffer_, 0);
            }
        }

        VkPipelineStageFlags k2DstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkPipelineStageFlags k2SrcStage = serAvailable_
            ? VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
            : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        vkCmdPipelineBarrier(cmd, k2SrcStage, k2DstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // Compact: write shadow dispatch args + swap counters + write active dispatch args
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineCompact_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
            0, 2, setsBounce, 0, nullptr);
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
            0, 2, setsBounce, 0, nullptr);
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

        // Ping-pong swap: K2 wrote to WRITE buffer, next bounce reads from it.
        pathStateCurrent_ = 1 - pathStateCurrent_;
    }

    // Reset ping-pong for next SPP sample
    pathStateCurrent_ = 0;

    } // end SPP loop

    uint32_t groupsX = (width + 7) / 8;
    uint32_t groupsY = (height + 7) / 8;

    // ── Stable Planes BUILD (after bounce loop, before PT and output) ──
    if (pipelineStablePlanes_) {
        VkDescriptorSet setsSP[2] = { sceneDescSet, wfDescSet_[0] };
        struct SPPush { uint32_t width, height, frameIndex, pad; };
        SPPush spPush = { width, height, frameIndex_, 0 };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineStablePlanes_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
            0, 2, setsSP, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(spPush), &spPush);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        VkMemoryBarrier spBarrier{};
        spBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        spBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        spBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &spBarrier, 0, nullptr, 0, nullptr);
    }

    // ── ReSTIR PT passes (after stable planes, before output) ──
    bool ptActive = pipelinePTTemporal_ && pipelinePTSpatial_ && pipelinePTFinal_;
    if (ptActive) {
        VkDescriptorSet setsPT[2] = { sceneDescSet, wfDescSet_[0] };

        // Push constants for PT kernels (width, height, frameIndex, pad)
        struct PTPush { uint32_t width, height, frameIndex, pad; };
        PTPush ptPush = { width, height, frameIndex_, 0 };

        VkMemoryBarrier ptBarrier{};
        ptBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        ptBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ptBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        // PT Temporal: create initial reservoirs + merge with previous frame
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelinePTTemporal_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
            0, 2, setsPT, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ptPush), &ptPush);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &ptBarrier, 0, nullptr, 0, nullptr);

        // PT Spatial: spatial neighbor resampling
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelinePTSpatial_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
            0, 2, setsPT, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ptPush), &ptPush);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &ptBarrier, 0, nullptr, 0, nullptr);

        // PT Final: evaluate resampled path → write to pixel radiance
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelinePTFinal_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
            0, 2, setsPT, 0, nullptr);
        // PT Final uses spp in push constants
        struct PTFinalPush { uint32_t width, height, frameIndex, spp; };
        PTFinalPush ptfPush = { width, height, frameIndex_, spp };
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ptfPush), &ptfPush);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &ptBarrier, 0, nullptr, 0, nullptr);

        // Ping-pong PT reservoir buffers for next frame
        // Swap bindings 15↔16 by updating descriptors
        // (Actually, we swap at descriptor write time — for now, swap the buffer pointers)
        ptReservoirCurrent_ = 1 - ptReservoirCurrent_;
    }

    // K5: Output (after all SPP samples + PT accumulated)
    VkDescriptorSet setsK5[2] = { sceneDescSet, wfDescSet_[0] };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineK5_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
        0, 2, setsK5, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    frameIndex_++;
}

} // namespace vk
} // namespace acpt
