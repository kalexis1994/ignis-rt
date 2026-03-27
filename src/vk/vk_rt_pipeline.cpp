#include "vk_rt_pipeline.h"
#include "vk_context.h"
#include "vk_accel_structure.h"
#include "vk_interop.h"
#include "vk_texture_manager.h"
#include "../../include/ignis_log.h"
#include "../../include/ignis_config.h"
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>

namespace acpt {
    extern PathTracerConfig g_config;
namespace vk {

bool RTPipeline::Initialize(Context* context, AccelStructureBuilder* accelBuilder, Interop* interop) {
    context_ = context;
    accelBuilder_ = accelBuilder;
    interop_ = interop;

    VkDevice device = context_->GetDevice();

    // Load function pointers
    vkCreateRayTracingPipelinesKHR_ = (PFN_vkCreateRayTracingPipelinesKHR)
        vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR");
    vkGetRayTracingShaderGroupHandlesKHR_ = (PFN_vkGetRayTracingShaderGroupHandlesKHR)
        vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR");
    vkCmdTraceRaysKHR_ = (PFN_vkCmdTraceRaysKHR)
        vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR");
    vkGetBufferDeviceAddressKHR_ = (PFN_vkGetBufferDeviceAddressKHR)
        vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR");

    if (!vkCreateRayTracingPipelinesKHR_ || !vkGetRayTracingShaderGroupHandlesKHR_ || !vkCmdTraceRaysKHR_) {
        Log(L"[VK RTPipeline] ERROR: Failed to load RT pipeline function pointers\n");
        return false;
    }

    if (!CreateDummyResources()) return false;

    // Create camera UBO (must be before CreateDescriptorSet which references it)
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = sizeof(CameraUBO);
    bufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufInfo, nullptr, &cameraBuffer_);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, cameraBuffer_, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &cameraMemory_);
    vkBindBufferMemory(device, cameraBuffer_, cameraMemory_, 0);

    // Create pick buffer (16 bytes, HOST_VISIBLE + HOST_COHERENT, persistently mapped)
    {
        VkBufferCreateInfo pickBufInfo{};
        pickBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        pickBufInfo.size = sizeof(PickResult);
        pickBufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        pickBufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &pickBufInfo, nullptr, &pickBuffer_);

        VkMemoryRequirements pickReqs;
        vkGetBufferMemoryRequirements(device, pickBuffer_, &pickReqs);
        VkMemoryAllocateInfo pickAllocInfo{};
        pickAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        pickAllocInfo.allocationSize = pickReqs.size;
        pickAllocInfo.memoryTypeIndex = context_->FindMemoryType(pickReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(device, &pickAllocInfo, nullptr, &pickMemory_);
        vkBindBufferMemory(device, pickBuffer_, pickMemory_, 0);

        // Persistently map
        void* mapped;
        vkMapMemory(device, pickMemory_, 0, sizeof(PickResult), 0, &mapped);
        pickMappedPtr_ = static_cast<PickResult*>(mapped);
        memset(pickMappedPtr_, 0, sizeof(PickResult));
        Log(L"[VK RTPipeline] Pick buffer created (16 bytes, persistently mapped)\n");
    }

    if (!CreateDescriptorSetLayout()) return false;
    if (!CreatePipeline()) return false;
    if (!CreateDescriptorPool()) return false;
    if (!CreateDescriptorSet()) return false;
    if (!CreateSBT()) return false;

    Log(L"[VK RTPipeline] Initialized successfully\n");
    return true;
}

bool RTPipeline::CreateSHARCBuffers() {
    if (sharcCreated_) return true;

    VkDevice device = context_->GetDevice();

    // SHARC uses 2 buffers on bindings 20-21:
    // [0] Hash Entries: uint64 per entry = 8 bytes (binding 20)
    // [1] Combined: accum (uint×4) + resolved (uint×4) + guide bins (uint×6) = 56 bytes/entry
    //   Layout: [accum: cap×4] [resolved: cap×4] [guide: cap×6]
    const VkDeviceSize sizes[3] = {
        SHARC_CAPACITY * 8,   // hash entries (uint64)
        SHARC_CAPACITY * 56,  // combined accum+resolved+guide (14 uints per entry)
        0,                     // unused
    };
    const char* names[3] = { "hashEntries", "accum+resolved+guide", "unused" };
    const int numBuffers = 2;  // only create 2 buffers

    auto vkGetBufferDeviceAddressKHR_ = (PFN_vkGetBufferDeviceAddressKHR)
        vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR");

    for (int i = 0; i < numBuffers; i++) {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = sizes[i];
        bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufInfo, nullptr, &sharcBuffer_[i]) != VK_SUCCESS) {
            Log(L"[VK RTPipeline] ERROR: Failed to create SHARC %S buffer\n", names[i]);
            return false;
        }

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, sharcBuffer_[i], &memReqs);

        VkMemoryAllocateFlagsInfo allocFlags{};
        allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.pNext = &allocFlags;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &sharcMemory_[i]) != VK_SUCCESS) {
            Log(L"[VK RTPipeline] ERROR: Failed to allocate SHARC %S memory\n", names[i]);
            return false;
        }
        vkBindBufferMemory(device, sharcBuffer_[i], sharcMemory_[i], 0);

        // Get device address for shader buffer_reference access
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = sharcBuffer_[i];
        sharcDeviceAddr_[i] = vkGetBufferDeviceAddressKHR_(device, &addrInfo);

        // Zero-fill
        VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
        vkCmdFillBuffer(cmd, sharcBuffer_[i], 0, sizes[i], 0);
        context_->EndSingleTimeCommands(cmd);
    }

    sharcCreated_ = true;

    // Update descriptor set bindings 20-21
    // binding 20 = hashEntries (uint64), binding 21 = combined accum+resolved
    VkDescriptorBufferInfo bufInfos[2] = {};
    bufInfos[0].buffer = sharcBuffer_[0]; bufInfos[0].offset = 0; bufInfos[0].range = sizes[0];
    bufInfos[1].buffer = sharcBuffer_[1]; bufInfos[1].offset = 0; bufInfos[1].range = sizes[1];

    VkWriteDescriptorSet writes[2] = {};
    for (int i = 0; i < 2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptorSet_;
        writes[i].dstBinding = 20 + i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

    VkDeviceSize totalMB = (sizes[0] + sizes[1] + sizes[2]) / (1024 * 1024);
    Log(L"[VK RTPipeline] SHARC buffers created: %u MiB (%u entries), addrs=[%llx, %llx, %llx]\n",
        (uint32_t)totalMB, SHARC_CAPACITY,
        (unsigned long long)sharcDeviceAddr_[0],
        (unsigned long long)sharcDeviceAddr_[1],
        (unsigned long long)sharcDeviceAddr_[2]);
    return true;
}

void RTPipeline::DestroySHARCBuffers() {
    if (!sharcCreated_) return;
    VkDevice device = context_->GetDevice();
    for (int i = 0; i < 2; i++) {
        if (sharcBuffer_[i]) { vkDestroyBuffer(device, sharcBuffer_[i], nullptr); sharcBuffer_[i] = VK_NULL_HANDLE; }
        if (sharcMemory_[i]) { vkFreeMemory(device, sharcMemory_[i], nullptr); sharcMemory_[i] = VK_NULL_HANDLE; }
        sharcDeviceAddr_[i] = 0;
    }
    sharcCreated_ = false;
}

bool RTPipeline::CreateSurfelBuffers() {
    if (surfelCreated_) return true;

    VkDevice device = context_->GetDevice();

    // Surfel uses 2 buffers on bindings 32-33:
    // [0] Hash Entries: uint64 per entry = 8 bytes
    // [1] Combined: accum (uint×4) + resolved (uint×4) = 32 bytes/entry
    const VkDeviceSize sizes[2] = {
        SURFEL_CAPACITY * 8,   // hash entries (uint64)
        SURFEL_CAPACITY * 32,  // combined accum+resolved (8 uints per entry)
    };

    auto vkGetBufferDeviceAddressKHR_ = (PFN_vkGetBufferDeviceAddressKHR)
        vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR");

    for (int i = 0; i < 2; i++) {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = sizes[i];
        bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufInfo, nullptr, &surfelBuffer_[i]) != VK_SUCCESS) {
            Log(L"[VK RTPipeline] ERROR: Failed to create surfel buffer %d\n", i);
            return false;
        }

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, surfelBuffer_[i], &memReqs);

        VkMemoryAllocateFlagsInfo allocFlags{};
        allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.pNext = &allocFlags;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &surfelMemory_[i]) != VK_SUCCESS) {
            Log(L"[VK RTPipeline] ERROR: Failed to allocate surfel memory %d\n", i);
            return false;
        }
        vkBindBufferMemory(device, surfelBuffer_[i], surfelMemory_[i], 0);

        // Zero-fill
        VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
        vkCmdFillBuffer(cmd, surfelBuffer_[i], 0, sizes[i], 0);
        context_->EndSingleTimeCommands(cmd);
    }

    surfelCreated_ = true;

    // Update descriptor set bindings 32-33
    VkDescriptorBufferInfo bufInfos[2] = {};
    bufInfos[0].buffer = surfelBuffer_[0]; bufInfos[0].offset = 0; bufInfos[0].range = sizes[0];
    bufInfos[1].buffer = surfelBuffer_[1]; bufInfos[1].offset = 0; bufInfos[1].range = sizes[1];

    VkWriteDescriptorSet writes[2] = {};
    for (int i = 0; i < 2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptorSet_;
        writes[i].dstBinding = 32 + i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

    VkDeviceSize totalMB = (sizes[0] + sizes[1]) / (1024 * 1024);
    Log(L"[VK RTPipeline] Surfel GI buffers created: %u MiB (%u entries)\n",
        (uint32_t)totalMB, SURFEL_CAPACITY);
    return true;
}

bool RTPipeline::CreateGIReservoirBuffers(uint32_t width, uint32_t height) {
    if (giReservoirCreated_) return true;

    VkDevice device = context_->GetDevice();
    giReservoirPixelCount_ = width * height;
    VkDeviceSize bufSize = (VkDeviceSize)giReservoirPixelCount_ * GI_RESERVOIR_VEC4S_PER_PIXEL * 16; // 3 vec4s × 16 bytes

    for (int i = 0; i < 2; i++) {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = bufSize;
        bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufInfo, nullptr, &giReservoirBuffer_[i]) != VK_SUCCESS) {
            Log(L"[VK RTPipeline] ERROR: Failed to create GI reservoir buffer[%d]\n", i);
            return false;
        }

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, giReservoirBuffer_[i], &memReqs);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &giReservoirMemory_[i]) != VK_SUCCESS) {
            Log(L"[VK RTPipeline] ERROR: Failed to allocate GI reservoir memory[%d]\n", i);
            return false;
        }
        vkBindBufferMemory(device, giReservoirBuffer_[i], giReservoirMemory_[i], 0);

        // Zero-fill
        VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
        vkCmdFillBuffer(cmd, giReservoirBuffer_[i], 0, bufSize, 0);
        context_->EndSingleTimeCommands(cmd);
    }

    giReservoirCreated_ = true;

    // Update descriptor set: binding 24 = write (curr), binding 25 = read (prev)
    VkDescriptorBufferInfo currInfo{};
    currInfo.buffer = giReservoirBuffer_[0];
    currInfo.offset = 0;
    currInfo.range = bufSize;

    VkDescriptorBufferInfo prevInfo{};
    prevInfo.buffer = giReservoirBuffer_[1];
    prevInfo.offset = 0;
    prevInfo.range = bufSize;

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptorSet_;
    writes[0].dstBinding = 24;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &currInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptorSet_;
    writes[1].dstBinding = 25;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &prevInfo;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

    Log(L"[VK RTPipeline] GI reservoir buffers created (2x %.1f MiB, %u pixels)\n",
        (float)bufSize / (1024.0f * 1024.0f), giReservoirPixelCount_);
    return true;
}

void RTPipeline::DestroyGIReservoirBuffers() {
    if (!giReservoirCreated_) return;
    VkDevice device = context_->GetDevice();
    for (int i = 0; i < 2; i++) {
        if (giReservoirBuffer_[i]) { vkDestroyBuffer(device, giReservoirBuffer_[i], nullptr); giReservoirBuffer_[i] = VK_NULL_HANDLE; }
        if (giReservoirMemory_[i]) { vkFreeMemory(device, giReservoirMemory_[i], nullptr); giReservoirMemory_[i] = VK_NULL_HANDLE; }
    }
    giReservoirCreated_ = false;
}

void RTPipeline::SwapGIReservoirBuffers() {
    if (!giReservoirCreated_) return;
    std::swap(giReservoirBuffer_[0], giReservoirBuffer_[1]);
    std::swap(giReservoirMemory_[0], giReservoirMemory_[1]);

    // Re-bind: 24=write(new curr), 25=read(new prev)
    VkDevice device = context_->GetDevice();
    VkDeviceSize bufSize = (VkDeviceSize)giReservoirPixelCount_ * GI_RESERVOIR_VEC4S_PER_PIXEL * 16;

    VkDescriptorBufferInfo currInfo{ giReservoirBuffer_[0], 0, bufSize };
    VkDescriptorBufferInfo prevInfo{ giReservoirBuffer_[1], 0, bufSize };

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptorSet_;
    writes[0].dstBinding = 24;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &currInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptorSet_;
    writes[1].dstBinding = 25;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &prevInfo;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

bool RTPipeline::CreateShadowAccumBuffers(uint32_t width, uint32_t height) {
    if (shadowAccumCreated_) return true;

    for (int i = 0; i < 2; i++) {
        char name[64];
        snprintf(name, sizeof(name), "shadowAccum[%d]", i);
        if (!CreateGBufferImage(shadowAccumImage_[i], VK_FORMAT_R16G16B16A16_SFLOAT, width, height, name)) {
            return false;
        }
    }

    shadowAccumCurrent_ = 0;
    shadowAccumCreated_ = true;
    UpdateShadowAccumDescriptors();
    Log(L"[VK RTPipeline] Shadow accumulation buffers created (%ux%u)\n", width, height);
    return true;
}

void RTPipeline::DestroyShadowAccumBuffers() {
    if (!shadowAccumCreated_) return;
    DestroyGBufferImage(shadowAccumImage_[0]);
    DestroyGBufferImage(shadowAccumImage_[1]);
    shadowAccumCreated_ = false;
}

void RTPipeline::UpdateShadowAccumDescriptors() {
    if (!shadowAccumCreated_ || descriptorSet_ == VK_NULL_HANDLE) return;

    VkDevice device = context_->GetDevice();
    uint32_t writeIdx = shadowAccumCurrent_;
    uint32_t readIdx = 1 - shadowAccumCurrent_;

    // binding 18: current frame write (storage image)
    VkDescriptorImageInfo writeInfo{};
    writeInfo.imageView = shadowAccumImage_[writeIdx].view;
    writeInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w18{};
    w18.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w18.dstSet = descriptorSet_;
    w18.dstBinding = 18;
    w18.descriptorCount = 1;
    w18.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w18.pImageInfo = &writeInfo;

    // binding 19: previous frame read (sampled image)
    VkDescriptorImageInfo readInfo{};
    readInfo.imageView = shadowAccumImage_[readIdx].view;
    readInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    readInfo.sampler = dummySampler_;

    VkWriteDescriptorSet w19{};
    w19.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w19.dstSet = descriptorSet_;
    w19.dstBinding = 19;
    w19.descriptorCount = 1;
    w19.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w19.pImageInfo = &readInfo;

    VkWriteDescriptorSet writes[] = {w18, w19};
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

void RTPipeline::SwapShadowAccumBuffers() {
    if (!shadowAccumCreated_) return;
    shadowAccumCurrent_ = 1 - shadowAccumCurrent_;
    UpdateShadowAccumDescriptors();
}

void RTPipeline::Shutdown() {
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) return;

    vkDeviceWaitIdle(device);

    if (sbtBuffer_) { vkDestroyBuffer(device, sbtBuffer_, nullptr); sbtBuffer_ = VK_NULL_HANDLE; }
    if (sbtMemory_) { vkFreeMemory(device, sbtMemory_, nullptr); sbtMemory_ = VK_NULL_HANDLE; }
    if (cameraBuffer_) { vkDestroyBuffer(device, cameraBuffer_, nullptr); cameraBuffer_ = VK_NULL_HANDLE; }
    if (cameraMemory_) { vkFreeMemory(device, cameraMemory_, nullptr); cameraMemory_ = VK_NULL_HANDLE; }
    if (geometryMetadataBuffer_) { vkDestroyBuffer(device, geometryMetadataBuffer_, nullptr); geometryMetadataBuffer_ = VK_NULL_HANDLE; }
    if (geometryMetadataMemory_) { vkFreeMemory(device, geometryMetadataMemory_, nullptr); geometryMetadataMemory_ = VK_NULL_HANDLE; }
    if (materialBuffer_) { vkDestroyBuffer(device, materialBuffer_, nullptr); materialBuffer_ = VK_NULL_HANDLE; }
    if (materialMemory_) { vkFreeMemory(device, materialMemory_, nullptr); materialMemory_ = VK_NULL_HANDLE; }
    if (pickBuffer_) {
        if (pickMappedPtr_) { vkUnmapMemory(device, pickMemory_); pickMappedPtr_ = nullptr; }
        vkDestroyBuffer(device, pickBuffer_, nullptr); pickBuffer_ = VK_NULL_HANDLE;
    }
    if (pickMemory_) { vkFreeMemory(device, pickMemory_, nullptr); pickMemory_ = VK_NULL_HANDLE; }
    if (prevTransformsBuffer_) {
        if (prevTransformsMapped_) { vkUnmapMemory(device, prevTransformsMemory_); prevTransformsMapped_ = nullptr; }
        vkDestroyBuffer(device, prevTransformsBuffer_, nullptr); prevTransformsBuffer_ = VK_NULL_HANDLE;
    }
    if (prevTransformsMemory_) { vkFreeMemory(device, prevTransformsMemory_, nullptr); prevTransformsMemory_ = VK_NULL_HANDLE; }
    prevTransformsCapacity_ = 0;
    if (emissiveTriBuffer_) { vkDestroyBuffer(device, emissiveTriBuffer_, nullptr); emissiveTriBuffer_ = VK_NULL_HANDLE; }
    if (emissiveTriMemory_) { vkFreeMemory(device, emissiveTriMemory_, nullptr); emissiveTriMemory_ = VK_NULL_HANDLE; }
    emissiveTriCount_ = 0;
    if (pipeline_) { vkDestroyPipeline(device, pipeline_, nullptr); pipeline_ = VK_NULL_HANDLE; }
    if (pipelineLayout_) { vkDestroyPipelineLayout(device, pipelineLayout_, nullptr); pipelineLayout_ = VK_NULL_HANDLE; }
    if (descriptorPool_) { vkDestroyDescriptorPool(device, descriptorPool_, nullptr); descriptorPool_ = VK_NULL_HANDLE; }
    if (descriptorSetLayout_) { vkDestroyDescriptorSetLayout(device, descriptorSetLayout_, nullptr); descriptorSetLayout_ = VK_NULL_HANDLE; }

    // SHARC radiance cache buffers
    DestroySHARCBuffers();

    // GI Reservoir buffers
    DestroyGIReservoirBuffers();

    // Shadow accumulation buffers
    DestroyShadowAccumBuffers();

    // G-buffer images
    if (gbuffersCreated_) {
        DestroyGBufferImage(normalRoughnessGBuffer_);
        DestroyGBufferImage(viewDepthGBuffer_);
        DestroyGBufferImage(motionVectorsGBuffer_);
        DestroyGBufferImage(diffuseRadianceGBuffer_);
        DestroyGBufferImage(specularRadianceGBuffer_);
        DestroyGBufferImage(specularMVGBuffer_);
        DestroyGBufferImage(specularAlbedoGBuffer_);
        DestroyGBufferImage(albedoGBuffer_);
        DestroyGBufferImage(penumbraGBuffer_);
        DestroyGBufferImage(dlssDepthGBuffer_);
        DestroyGBufferImage(reactiveMaskGBuffer_);
        DestroyGBufferImage(diffConfidenceGBuffer_);
        DestroyGBufferImage(specConfidenceGBuffer_);
        gbuffersCreated_ = false;
    }

    // Dummy resources
    if (dummySampler_) { vkDestroySampler(device, dummySampler_, nullptr); dummySampler_ = VK_NULL_HANDLE; }
    if (dummyImageView_) { vkDestroyImageView(device, dummyImageView_, nullptr); dummyImageView_ = VK_NULL_HANDLE; }
    if (dummyImage_) { vkDestroyImage(device, dummyImage_, nullptr); dummyImage_ = VK_NULL_HANDLE; }
    if (dummyImageMemory_) { vkFreeMemory(device, dummyImageMemory_, nullptr); dummyImageMemory_ = VK_NULL_HANDLE; }
    if (dummyImageView16f_) { vkDestroyImageView(device, dummyImageView16f_, nullptr); dummyImageView16f_ = VK_NULL_HANDLE; }
    if (dummyImage16f_) { vkDestroyImage(device, dummyImage16f_, nullptr); dummyImage16f_ = VK_NULL_HANDLE; }
    if (dummyImageMemory16f_) { vkFreeMemory(device, dummyImageMemory16f_, nullptr); dummyImageMemory16f_ = VK_NULL_HANDLE; }
    if (dummyImageViewR16f_) { vkDestroyImageView(device, dummyImageViewR16f_, nullptr); dummyImageViewR16f_ = VK_NULL_HANDLE; }
    if (dummyImageR16f_) { vkDestroyImage(device, dummyImageR16f_, nullptr); dummyImageR16f_ = VK_NULL_HANDLE; }
    if (dummyImageMemoryR16f_) { vkFreeMemory(device, dummyImageMemoryR16f_, nullptr); dummyImageMemoryR16f_ = VK_NULL_HANDLE; }
    if (dummyImageViewR8_) { vkDestroyImageView(device, dummyImageViewR8_, nullptr); dummyImageViewR8_ = VK_NULL_HANDLE; }
    if (dummyImageR8_) { vkDestroyImage(device, dummyImageR8_, nullptr); dummyImageR8_ = VK_NULL_HANDLE; }
    if (dummyImageMemoryR8_) { vkFreeMemory(device, dummyImageMemoryR8_, nullptr); dummyImageMemoryR8_ = VK_NULL_HANDLE; }
    if (dummyBuffer_) { vkDestroyBuffer(device, dummyBuffer_, nullptr); dummyBuffer_ = VK_NULL_HANDLE; }
    if (dummyBufferMemory_) { vkFreeMemory(device, dummyBufferMemory_, nullptr); dummyBufferMemory_ = VK_NULL_HANDLE; }

    Log(L"[VK RTPipeline] Shutdown\n");
}

bool RTPipeline::CreateGBufferImage(GBufferImage& gb, VkFormat format, uint32_t width, uint32_t height, const char* name) {
    VkDevice device = context_->GetDevice();

    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = format;
    imgInfo.extent = {width, height, 1};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &imgInfo, nullptr, &gb.image) != VK_SUCCESS) {
        Log(L"[VK RTPipeline] ERROR: Failed to create G-buffer image: %S\n", name);
        return false;
    }

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, gb.image, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &gb.memory) != VK_SUCCESS) {
        Log(L"[VK RTPipeline] ERROR: Failed to allocate G-buffer memory: %S\n", name);
        return false;
    }
    vkBindImageMemory(device, gb.image, gb.memory, 0);

    // Transition to GENERAL layout
    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = gb.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    context_->EndSingleTimeCommands(cmd);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = gb.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &viewInfo, nullptr, &gb.view) != VK_SUCCESS) {
        Log(L"[VK RTPipeline] ERROR: Failed to create G-buffer image view: %S\n", name);
        return false;
    }

    Log(L"[VK RTPipeline] Created G-buffer: %S (%ux%u)\n", name, width, height);
    return true;
}

void RTPipeline::DestroyGBufferImage(GBufferImage& gb) {
    VkDevice device = context_->GetDevice();
    if (gb.view) { vkDestroyImageView(device, gb.view, nullptr); gb.view = VK_NULL_HANDLE; }
    if (gb.image) { vkDestroyImage(device, gb.image, nullptr); gb.image = VK_NULL_HANDLE; }
    if (gb.memory) { vkFreeMemory(device, gb.memory, nullptr); gb.memory = VK_NULL_HANDLE; }
}

bool RTPipeline::CreateGBuffers(uint32_t width, uint32_t height) {
    if (gbuffersCreated_ && gbufferWidth_ == width && gbufferHeight_ == height) {
        return true;  // Already created at this size
    }

    // Destroy old if resizing
    if (gbuffersCreated_) {
        DestroyGBufferImage(normalRoughnessGBuffer_);
        DestroyGBufferImage(viewDepthGBuffer_);
        DestroyGBufferImage(motionVectorsGBuffer_);
        DestroyGBufferImage(diffuseRadianceGBuffer_);
        DestroyGBufferImage(specularRadianceGBuffer_);
        DestroyGBufferImage(specularMVGBuffer_);
        DestroyGBufferImage(specularAlbedoGBuffer_);
        DestroyGBufferImage(albedoGBuffer_);
        DestroyGBufferImage(penumbraGBuffer_);
        DestroyGBufferImage(dlssDepthGBuffer_);
        DestroyGBufferImage(reactiveMaskGBuffer_);
        DestroyGBufferImage(diffConfidenceGBuffer_);
        DestroyGBufferImage(specConfidenceGBuffer_);
    }

    if (!CreateGBufferImage(normalRoughnessGBuffer_, VK_FORMAT_R16G16B16A16_SFLOAT, width, height, "normalRoughness")) return false;
    if (!CreateGBufferImage(viewDepthGBuffer_, VK_FORMAT_R32_SFLOAT, width, height, "viewDepth")) return false;
    if (!CreateGBufferImage(motionVectorsGBuffer_, VK_FORMAT_R16G16B16A16_SFLOAT, width, height, "motionVectors")) return false;
    if (!CreateGBufferImage(diffuseRadianceGBuffer_, VK_FORMAT_R16G16B16A16_SFLOAT, width, height, "diffuseRadiance")) return false;
    if (!CreateGBufferImage(specularRadianceGBuffer_, VK_FORMAT_R16G16B16A16_SFLOAT, width, height, "specularRadiance")) return false;
    if (!CreateGBufferImage(specularMVGBuffer_, VK_FORMAT_R16G16B16A16_SFLOAT, width, height, "specularMV")) return false;
    if (!CreateGBufferImage(specularAlbedoGBuffer_, VK_FORMAT_R16G16B16A16_SFLOAT, width, height, "specularAlbedo")) return false;
    if (!CreateGBufferImage(albedoGBuffer_, VK_FORMAT_R16G16B16A16_SFLOAT, width, height, "albedoBuffer")) return false;
    if (!CreateGBufferImage(penumbraGBuffer_, VK_FORMAT_R16_SFLOAT, width, height, "penumbra")) return false;
    if (!CreateGBufferImage(dlssDepthGBuffer_, VK_FORMAT_R32_SFLOAT, width, height, "dlssDepth")) return false;
    if (!CreateGBufferImage(reactiveMaskGBuffer_, VK_FORMAT_R8_UNORM, width, height, "reactiveMask")) return false;
    if (!CreateGBufferImage(diffConfidenceGBuffer_, VK_FORMAT_R8_UNORM, width, height, "diffConfidence")) return false;
    if (!CreateGBufferImage(specConfidenceGBuffer_, VK_FORMAT_R8_UNORM, width, height, "specConfidence")) return false;

    gbuffersCreated_ = true;
    gbufferWidth_ = width;
    gbufferHeight_ = height;

    // Create shadow temporal accumulation buffers (same resolution)
    CreateShadowAccumBuffers(width, height);

    // Create SHARC radiance cache buffers (resolution-independent)
    CreateSHARCBuffers();

    // Create Surfel GI cache buffers (experimental)
    CreateSurfelBuffers();

    // Create GI reservoir buffers for ReSTIR (resolution-dependent)
    CreateGIReservoirBuffers(width, height);

    // Update descriptor set with real G-buffer images
    VkDevice device = context_->GetDevice();

    VkDescriptorImageInfo gbufferInfos[13];
    gbufferInfos[0] = {VK_NULL_HANDLE, normalRoughnessGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};
    gbufferInfos[1] = {VK_NULL_HANDLE, viewDepthGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};
    gbufferInfos[2] = {VK_NULL_HANDLE, motionVectorsGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};
    gbufferInfos[3] = {VK_NULL_HANDLE, diffuseRadianceGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};
    gbufferInfos[4] = {VK_NULL_HANDLE, specularRadianceGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};
    gbufferInfos[5] = {VK_NULL_HANDLE, specularMVGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};
    gbufferInfos[6] = {VK_NULL_HANDLE, specularAlbedoGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};
    gbufferInfos[7] = {VK_NULL_HANDLE, albedoGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};
    gbufferInfos[8] = {VK_NULL_HANDLE, penumbraGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};
    gbufferInfos[9] = {VK_NULL_HANDLE, dlssDepthGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};
    gbufferInfos[10] = {VK_NULL_HANDLE, reactiveMaskGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};
    gbufferInfos[11] = {VK_NULL_HANDLE, diffConfidenceGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};
    gbufferInfos[12] = {VK_NULL_HANDLE, specConfidenceGBuffer_.view, VK_IMAGE_LAYOUT_GENERAL};

    int bindings[] = {6, 7, 8, 9, 10, 13, 15, 16, 18, 22, 29, 30, 31};
    std::vector<VkWriteDescriptorSet> writes(13);
    for (int i = 0; i < 13; i++) {
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptorSet_;
        writes[i].dstBinding = bindings[i];
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo = &gbufferInfos[i];
    }

    vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    Log(L"[VK RTPipeline] G-buffer descriptors updated (%ux%u, incl. confidence/reactive masks)\n", width, height);
    return true;
}

bool RTPipeline::CreateDummyResources() {
    VkDevice device = context_->GetDevice();

    // Dummy 1x1 image for unused image bindings
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imgInfo.extent = {1, 1, 1};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &imgInfo, nullptr, &dummyImage_) != VK_SUCCESS) return false;

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, dummyImage_, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &dummyImageMemory_);
    vkBindImageMemory(device, dummyImage_, dummyImageMemory_, 0);

    // Transition to general layout
    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = dummyImage_;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    context_->EndSingleTimeCommands(cmd);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = dummyImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    vkCreateImageView(device, &viewInfo, nullptr, &dummyImageView_);

    // Create RGBA16F dummy image for G-buffer storage bindings (6,8,9,10,13,15,16)
    {
        VkImageCreateInfo img16fInfo = imgInfo;
        img16fInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        vkCreateImage(device, &img16fInfo, nullptr, &dummyImage16f_);
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device, dummyImage16f_, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = context_->FindMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device, &ai, nullptr, &dummyImageMemory16f_);
        vkBindImageMemory(device, dummyImage16f_, dummyImageMemory16f_, 0);

        VkCommandBuffer cmd2 = context_->BeginSingleTimeCommands();
        VkImageMemoryBarrier b = barrier;
        b.image = dummyImage16f_;
        vkCmdPipelineBarrier(cmd2, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
        context_->EndSingleTimeCommands(cmd2);

        VkImageViewCreateInfo vi = viewInfo;
        vi.image = dummyImage16f_;
        vi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        vkCreateImageView(device, &vi, nullptr, &dummyImageView16f_);
    }

    // Create R32F dummy image for viewDepth binding (7)
    {
        VkImageCreateInfo imgR32fInfo = imgInfo;
        imgR32fInfo.format = VK_FORMAT_R32_SFLOAT;
        vkCreateImage(device, &imgR32fInfo, nullptr, &dummyImageR16f_);
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device, dummyImageR16f_, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = context_->FindMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device, &ai, nullptr, &dummyImageMemoryR16f_);
        vkBindImageMemory(device, dummyImageR16f_, dummyImageMemoryR16f_, 0);

        VkCommandBuffer cmd2 = context_->BeginSingleTimeCommands();
        VkImageMemoryBarrier b = barrier;
        b.image = dummyImageR16f_;
        vkCmdPipelineBarrier(cmd2, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
        context_->EndSingleTimeCommands(cmd2);

        VkImageViewCreateInfo vi = viewInfo;
        vi.image = dummyImageR16f_;
        vi.format = VK_FORMAT_R32_SFLOAT;
        vkCreateImageView(device, &vi, nullptr, &dummyImageViewR16f_);
    }

    // Create R8_UNORM dummy image for confidence/reactive mask bindings (29-31)
    {
        VkImageCreateInfo imgR8Info = imgInfo;
        imgR8Info.format = VK_FORMAT_R8_UNORM;
        vkCreateImage(device, &imgR8Info, nullptr, &dummyImageR8_);
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device, dummyImageR8_, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = context_->FindMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device, &ai, nullptr, &dummyImageMemoryR8_);
        vkBindImageMemory(device, dummyImageR8_, dummyImageMemoryR8_, 0);

        VkCommandBuffer cmd2 = context_->BeginSingleTimeCommands();
        VkImageMemoryBarrier b = barrier;
        b.image = dummyImageR8_;
        vkCmdPipelineBarrier(cmd2, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
        context_->EndSingleTimeCommands(cmd2);

        VkImageViewCreateInfo vi = viewInfo;
        vi.image = dummyImageR8_;
        vi.format = VK_FORMAT_R8_UNORM;
        vkCreateImageView(device, &vi, nullptr, &dummyImageViewR8_);
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(device, &samplerInfo, nullptr, &dummySampler_);

    // Dummy buffer for SSBO bindings
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = 256;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufInfo, nullptr, &dummyBuffer_);

    vkGetBufferMemoryRequirements(device, dummyBuffer_, &memReqs);
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &dummyBufferMemory_);
    vkBindBufferMemory(device, dummyBuffer_, dummyBufferMemory_, 0);

    // Zero-fill dummy buffer
    void* mapped;
    vkMapMemory(device, dummyBufferMemory_, 0, 256, 0, &mapped);
    memset(mapped, 0, 256);
    vkUnmapMemory(device, dummyBufferMemory_);

    return true;
}

bool RTPipeline::CreateDescriptorSetLayout() {
    // Bindings matching raygen.rgen (0-31)
    std::vector<VkDescriptorSetLayoutBinding> bindings(32);

    // binding 0: TLAS
    bindings[0] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 1: output image (rgba8)
    bindings[1] = {};
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 2: Camera UBO
    bindings[2] = {};
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 3: GeometryMetadata SSBO
    bindings[3] = {};
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 4: MaterialBuffer SSBO
    bindings[4] = {};
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 5: textures[] (variable count array, max 1024)
    bindings[5] = {};
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[5].descriptorCount = 1024;
    bindings[5].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // bindings 6-10: G-buffer storage images
    for (int i = 6; i <= 10; i++) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;
    }

    // binding 11: sky LUT sampler
    bindings[11] = {};
    bindings[11].binding = 11;
    bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[11].descriptorCount = 1;
    bindings[11].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 12: blue noise sampler
    bindings[12] = {};
    bindings[12].binding = 12;
    bindings[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[12].descriptorCount = 1;
    bindings[12].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 13: cloud low-res storage image
    bindings[13] = {};
    bindings[13].binding = 13;
    bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[13].descriptorCount = 1;
    bindings[13].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 14: cloud history sampler
    bindings[14] = {};
    bindings[14].binding = 14;
    bindings[14].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[14].descriptorCount = 1;
    bindings[14].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 15: cloud reconstructed storage image
    bindings[15] = {};
    bindings[15].binding = 15;
    bindings[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[15].descriptorCount = 1;
    bindings[15].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 16: albedo storage image
    bindings[16] = {};
    bindings[16].binding = 16;
    bindings[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[16].descriptorCount = 1;
    bindings[16].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 17: PickBuffer SSBO
    bindings[17] = {};
    bindings[17].binding = 17;
    bindings[17].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[17].descriptorCount = 1;
    bindings[17].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 18: shadow accumulation current (storage image, write)
    bindings[18] = {};
    bindings[18].binding = 18;
    bindings[18].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[18].descriptorCount = 1;
    bindings[18].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 19: shadow accumulation previous (sampler, read)
    bindings[19] = {};
    bindings[19].binding = 19;
    bindings[19].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[19].descriptorCount = 1;
    bindings[19].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 20: SHARC write buffer (SSBO)
    bindings[20] = {};
    bindings[20].binding = 20;
    bindings[20].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[20].descriptorCount = 1;
    bindings[20].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 21: SHARC read buffer (SSBO)
    bindings[21] = {};
    bindings[21].binding = 21;
    bindings[21].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[21].descriptorCount = 1;
    bindings[21].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 22: DLSS NDC depth (R32F storage image)
    bindings[22] = {};
    bindings[22].binding = 22;
    bindings[22].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[22].descriptorCount = 1;
    bindings[22].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 23: cloud shadow map (R16F sampler)
    bindings[23] = {};
    bindings[23].binding = 23;
    bindings[23].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[23].descriptorCount = 1;
    bindings[23].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // bindings 24-25: GI Reservoir SSBOs (ReSTIR temporal reuse)
    bindings[24] = {};
    bindings[24].binding = 24;
    bindings[24].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[24].descriptorCount = 1;
    bindings[24].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[25] = {};
    bindings[25].binding = 25;
    bindings[25].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[25].descriptorCount = 1;
    bindings[25].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 26: emissive triangles SSBO (MIS)
    bindings[26] = {};
    bindings[26].binding = 26;
    bindings[26].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[26].descriptorCount = 1;
    bindings[26].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 27: Light tree SSBO
    bindings[27] = {};
    bindings[27].binding = 27;
    bindings[27].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[27].descriptorCount = 1;
    bindings[27].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 28: prevTransforms SSBO (dummy for now — needed for dynamic objects)
    bindings[28] = {};
    bindings[28].binding = 28;
    bindings[28].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[28].descriptorCount = 1;
    bindings[28].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 29: reactive mask (R8_UNORM storage image)
    bindings[29] = {};
    bindings[29].binding = 29;
    bindings[29].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[29].descriptorCount = 1;
    bindings[29].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 30: diffuse confidence (R8_UNORM storage image)
    bindings[30] = {};
    bindings[30].binding = 30;
    bindings[30].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[30].descriptorCount = 1;
    bindings[30].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 31: specular confidence (R8_UNORM storage image)
    bindings[31] = {};
    bindings[31].binding = 31;
    bindings[31].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[31].descriptorCount = 1;
    bindings[31].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

    // bindings 32-33: Surfel GI cache (SSBOs)
    bindings.resize(34);
    for (int i = 32; i <= 33; i++) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;
    }

    // Binding flags for partially bound descriptors
    std::vector<VkDescriptorBindingFlags> bindingFlags(34, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flagsInfo.bindingCount = (uint32_t)bindingFlags.size();
    flagsInfo.pBindingFlags = bindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = &flagsInfo;
    // No UPDATE_AFTER_BIND_POOL_BIT — we don't need update-after-bind
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(context_->GetDevice(), &layoutInfo, nullptr, &descriptorSetLayout_) != VK_SUCCESS) {
        Log(L"[VK RTPipeline] ERROR: Failed to create descriptor set layout\n");
        return false;
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;

    if (vkCreatePipelineLayout(context_->GetDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
        Log(L"[VK RTPipeline] ERROR: Failed to create pipeline layout\n");
        return false;
    }

    Log(L"[VK RTPipeline] Descriptor set layout created (24 bindings)\n");
    return true;
}

bool RTPipeline::LoadShaderModule(const char* path, VkShaderModule* outModule) {
    std::string resolved = IgnisResolvePath(path);
    std::ifstream file(resolved, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[VK RTPipeline] ERROR: Failed to open shader: %S\n", resolved.c_str());
        return false;
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> code(fileSize);
    file.seekg(0);
    file.read(code.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    if (vkCreateShaderModule(context_->GetDevice(), &createInfo, nullptr, outModule) != VK_SUCCESS) {
        Log(L"[VK RTPipeline] ERROR: Failed to create shader module: %S\n", path);
        return false;
    }

    Log(L"[VK RTPipeline] Loaded shader: %S (%zu bytes)\n", resolved.c_str(), fileSize);
    return true;
}

bool RTPipeline::CreatePipeline() {
    VkShaderModule raygenModule = VK_NULL_HANDLE;
    const char* shaderPath = (acpt::g_config.shaderMode == 1)
        ? "shaders/raygen_blender.rgen.spv"
        : "shaders/raygen.rgen.spv";
    if (!LoadShaderModule(shaderPath, &raygenModule)) {
        return false;
    }

    // Single shader stage: raygen
    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stageInfo.module = raygenModule;
    stageInfo.pName = "main";

    // Single shader group: raygen
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
    pipelineInfo.layout = pipelineLayout_;

    VkResult result = vkCreateRayTracingPipelinesKHR_(context_->GetDevice(),
        VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_);

    vkDestroyShaderModule(context_->GetDevice(), raygenModule, nullptr);

    if (result != VK_SUCCESS) {
        Log(L"[VK RTPipeline] ERROR: Failed to create RT pipeline (result=%d)\n", result);
        return false;
    }

    Log(L"[VK RTPipeline] RT pipeline created\n");
    return true;
}

bool RTPipeline::CreateSBT() {
    VkDevice device = context_->GetDevice();

    // Query RT pipeline properties for handle size/alignment
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(context_->GetPhysicalDevice(), &props2);

    uint32_t handleSize = rtProps.shaderGroupHandleSize;
    uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
    uint32_t baseAlignment = rtProps.shaderGroupBaseAlignment;

    // Align handle size to alignment
    uint32_t handleSizeAligned = (handleSize + handleAlignment - 1) & ~(handleAlignment - 1);

    // We have 1 group (raygen only)
    uint32_t sbtSize = baseAlignment; // One group aligned to base alignment

    // Get shader group handle
    std::vector<uint8_t> handleData(handleSize);
    vkGetRayTracingShaderGroupHandlesKHR_(device, pipeline_, 0, 1, handleSize, handleData.data());

    // Create SBT buffer
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = sbtSize;
    bufInfo.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufInfo, nullptr, &sbtBuffer_) != VK_SUCCESS) {
        Log(L"[VK RTPipeline] ERROR: Failed to create SBT buffer\n");
        return false;
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, sbtBuffer_, &memReqs);

    VkMemoryAllocateFlagsInfo allocFlags{};
    allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &allocFlags;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &sbtMemory_) != VK_SUCCESS) {
        Log(L"[VK RTPipeline] ERROR: Failed to allocate SBT memory\n");
        return false;
    }
    vkBindBufferMemory(device, sbtBuffer_, sbtMemory_, 0);

    // Write handle to SBT
    void* mapped;
    vkMapMemory(device, sbtMemory_, 0, sbtSize, 0, &mapped);
    memset(mapped, 0, sbtSize);
    memcpy(mapped, handleData.data(), handleSize);
    vkUnmapMemory(device, sbtMemory_);

    // Get SBT device address
    VkBufferDeviceAddressInfo addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = sbtBuffer_;
    VkDeviceAddress sbtAddress = vkGetBufferDeviceAddressKHR_(device, &addrInfo);

    // Set up regions
    raygenRegion_.deviceAddress = sbtAddress;
    raygenRegion_.stride = handleSizeAligned;
    raygenRegion_.size = handleSizeAligned;

    // Empty miss/hit/callable regions
    missRegion_ = {};
    hitRegion_ = {};
    callableRegion_ = {};

    Log(L"[VK RTPipeline] SBT created (handleSize=%u, aligned=%u)\n", handleSize, handleSizeAligned);
    return true;
}

bool RTPipeline::CreateDescriptorPool() {
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 21},   // 13 base + 1 reserved(27) + 3 masks(29-31) + padding
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 11},  // 3 base + 2 SHARC + 1 prevTransforms(28) + 2 GI reservoir(24-25) + 1 emissive(26) + 2 surfel(32-33)
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1034},  // 1024 textures + 10 other samplers
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = 0;  // No UPDATE_AFTER_BIND — feature not enabled in device creation
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 5;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(context_->GetDevice(), &poolInfo, nullptr, &descriptorPool_) != VK_SUCCESS) {
        Log(L"[VK RTPipeline] ERROR: Failed to create descriptor pool\n");
        return false;
    }
    return true;
}

bool RTPipeline::CreateDescriptorSet() {
    // Allocate descriptor set with fixed 1024 texture slots (PARTIALLY_BOUND allows unwritten slots)
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout_;

    if (vkAllocateDescriptorSets(context_->GetDevice(), &allocInfo, &descriptorSet_) != VK_SUCCESS) {
        Log(L"[VK RTPipeline] ERROR: Failed to allocate descriptor set\n");
        return false;
    }

    // Write initial descriptors with dummy resources
    // We'll update TLAS and output image when they become available

    // binding 1: output image (use interop shared image)
    VkDescriptorImageInfo outputImageInfo{};
    outputImageInfo.imageView = interop_->GetSharedImageView();
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // binding 2: camera UBO
    VkDescriptorBufferInfo cameraInfo{};
    cameraInfo.buffer = cameraBuffer_;
    cameraInfo.offset = 0;
    cameraInfo.range = sizeof(CameraUBO);

    // binding 3: geometry metadata (dummy)
    VkDescriptorBufferInfo geomInfo{};
    geomInfo.buffer = dummyBuffer_;
    geomInfo.offset = 0;
    geomInfo.range = 256;

    // binding 4: material buffer (dummy)
    VkDescriptorBufferInfo matInfo{};
    matInfo.buffer = dummyBuffer_;
    matInfo.offset = 0;
    matInfo.range = 256;

    // Dummy image info for samplers (R8G8B8A8)
    VkDescriptorImageInfo dummyImageInfo{};
    dummyImageInfo.imageView = dummyImageView_;
    dummyImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    dummyImageInfo.sampler = dummySampler_;

    // Dummy image info for RGBA16F storage images (G-buffers)
    VkDescriptorImageInfo dummyImage16fInfo{};
    dummyImage16fInfo.imageView = dummyImageView16f_;
    dummyImage16fInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // Dummy image info for R32F storage image (viewDepth)
    VkDescriptorImageInfo dummyImageR16fInfo{};
    dummyImageR16fInfo.imageView = dummyImageViewR16f_;
    dummyImageR16fInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::vector<VkWriteDescriptorSet> writes;

    // binding 1: output image
    VkWriteDescriptorSet write1{};
    write1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write1.dstSet = descriptorSet_;
    write1.dstBinding = 1;
    write1.descriptorCount = 1;
    write1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write1.pImageInfo = &outputImageInfo;
    writes.push_back(write1);

    // binding 2: camera UBO
    VkWriteDescriptorSet write2{};
    write2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write2.dstSet = descriptorSet_;
    write2.dstBinding = 2;
    write2.descriptorCount = 1;
    write2.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write2.pBufferInfo = &cameraInfo;
    writes.push_back(write2);

    // binding 3: geometry SSBO (dummy)
    VkWriteDescriptorSet write3{};
    write3.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write3.dstSet = descriptorSet_;
    write3.dstBinding = 3;
    write3.descriptorCount = 1;
    write3.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write3.pBufferInfo = &geomInfo;
    writes.push_back(write3);

    // binding 4: material SSBO (dummy)
    VkWriteDescriptorSet write4{};
    write4.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write4.dstSet = descriptorSet_;
    write4.dstBinding = 4;
    write4.descriptorCount = 1;
    write4.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write4.pBufferInfo = &matInfo;
    writes.push_back(write4);

    // binding 5: textures (fill all 1024 slots with dummy to prevent UB from bad indices)
    std::vector<VkDescriptorImageInfo> dummyTexInfos(1024, dummyImageInfo);
    VkWriteDescriptorSet write5{};
    write5.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write5.dstSet = descriptorSet_;
    write5.dstBinding = 5;
    write5.descriptorCount = 1024;
    write5.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write5.pImageInfo = dummyTexInfos.data();
    writes.push_back(write5);

    // bindings 6-10: G-buffer storage images (format-matched dummies)
    // 6=normalRoughness(rgba16f), 7=viewDepth(r16f), 8=motionVectors(rgba16f),
    // 9=diffuseRadiance(rgba16f), 10=specularRadiance(rgba16f)
    for (int i = 6; i <= 10; i++) {
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = descriptorSet_;
        w.dstBinding = i;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = (i == 7) ? &dummyImageR16fInfo : &dummyImage16fInfo;
        writes.push_back(w);
    }

    // binding 11: sky LUT (dummy)
    VkWriteDescriptorSet w11{};
    w11.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w11.dstSet = descriptorSet_;
    w11.dstBinding = 11;
    w11.descriptorCount = 1;
    w11.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w11.pImageInfo = &dummyImageInfo;
    writes.push_back(w11);

    // binding 12: blue noise (dummy)
    VkWriteDescriptorSet w12{};
    w12.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w12.dstSet = descriptorSet_;
    w12.dstBinding = 12;
    w12.descriptorCount = 1;
    w12.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w12.pImageInfo = &dummyImageInfo;
    writes.push_back(w12);

    // binding 13: cloud low-res (dummy rgba16f)
    VkWriteDescriptorSet w13{};
    w13.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w13.dstSet = descriptorSet_;
    w13.dstBinding = 13;
    w13.descriptorCount = 1;
    w13.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w13.pImageInfo = &dummyImage16fInfo;
    writes.push_back(w13);

    // binding 14: cloud history (dummy)
    VkWriteDescriptorSet w14{};
    w14.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w14.dstSet = descriptorSet_;
    w14.dstBinding = 14;
    w14.descriptorCount = 1;
    w14.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w14.pImageInfo = &dummyImageInfo;
    writes.push_back(w14);

    // binding 15: cloud reconstructed (dummy rgba16f)
    VkWriteDescriptorSet w15{};
    w15.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w15.dstSet = descriptorSet_;
    w15.dstBinding = 15;
    w15.descriptorCount = 1;
    w15.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w15.pImageInfo = &dummyImage16fInfo;
    writes.push_back(w15);

    // binding 16: albedo (dummy rgba16f)
    VkWriteDescriptorSet w16{};
    w16.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w16.dstSet = descriptorSet_;
    w16.dstBinding = 16;
    w16.descriptorCount = 1;
    w16.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w16.pImageInfo = &dummyImage16fInfo;
    writes.push_back(w16);

    // binding 17: pick buffer SSBO
    VkDescriptorBufferInfo pickBufInfo{};
    pickBufInfo.buffer = pickBuffer_;
    pickBufInfo.offset = 0;
    pickBufInfo.range = sizeof(PickResult);

    VkWriteDescriptorSet w17{};
    w17.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w17.dstSet = descriptorSet_;
    w17.dstBinding = 17;
    w17.descriptorCount = 1;
    w17.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w17.pBufferInfo = &pickBufInfo;
    writes.push_back(w17);

    // binding 18: shadow accumulation current (dummy rgba16f storage)
    VkWriteDescriptorSet w18{};
    w18.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w18.dstSet = descriptorSet_;
    w18.dstBinding = 18;
    w18.descriptorCount = 1;
    w18.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w18.pImageInfo = &dummyImage16fInfo;
    writes.push_back(w18);

    // binding 19: shadow accumulation previous (dummy sampler)
    VkWriteDescriptorSet w19{};
    w19.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w19.dstSet = descriptorSet_;
    w19.dstBinding = 19;
    w19.descriptorCount = 1;
    w19.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w19.pImageInfo = &dummyImageInfo;
    writes.push_back(w19);

    // binding 20: SHARC write buffer (dummy SSBO)
    VkWriteDescriptorSet w20{};
    w20.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w20.dstSet = descriptorSet_;
    w20.dstBinding = 20;
    w20.descriptorCount = 1;
    w20.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w20.pBufferInfo = &geomInfo;  // reuse dummy buffer
    writes.push_back(w20);

    // binding 21: SHARC read buffer (dummy SSBO)
    VkWriteDescriptorSet w21{};
    w21.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w21.dstSet = descriptorSet_;
    w21.dstBinding = 21;
    w21.descriptorCount = 1;
    w21.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w21.pBufferInfo = &geomInfo;  // reuse dummy buffer
    writes.push_back(w21);

    // binding 22: DLSS NDC depth (dummy R32F storage image)
    VkWriteDescriptorSet w22{};
    w22.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w22.dstSet = descriptorSet_;
    w22.dstBinding = 22;
    w22.descriptorCount = 1;
    w22.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w22.pImageInfo = &dummyImageR16fInfo;  // R32F dummy
    writes.push_back(w22);

    // binding 23: cloud shadow map (dummy sampler)
    VkWriteDescriptorSet w23{};
    w23.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w23.dstSet = descriptorSet_;
    w23.dstBinding = 23;
    w23.descriptorCount = 1;
    w23.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w23.pImageInfo = &dummyImageInfo;
    writes.push_back(w23);

    // bindings 24-25: GI Reservoir SSBOs (dummy until CreateGIReservoirBuffers)
    for (int i = 24; i <= 25; i++) {
        VkWriteDescriptorSet wr{};
        wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr.dstSet = descriptorSet_;
        wr.dstBinding = i;
        wr.descriptorCount = 1;
        wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr.pBufferInfo = &geomInfo;  // reuse dummy buffer
        writes.push_back(wr);
    }
    // binding 26: emissive triangles SSBO (dummy)
    {
        VkWriteDescriptorSet wr{};
        wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr.dstSet = descriptorSet_;
        wr.dstBinding = 26;
        wr.descriptorCount = 1;
        wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr.pBufferInfo = &geomInfo;  // reuse dummy buffer
        writes.push_back(wr);
    }

    // binding 27: Light tree SSBO (dummy initially)
    {
        VkWriteDescriptorSet wr{};
        wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr.dstSet = descriptorSet_;
        wr.dstBinding = 27;
        wr.descriptorCount = 1;
        wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr.pBufferInfo = &geomInfo;  // reuse dummy buffer
        writes.push_back(wr);
    }

    // binding 28: prevTransforms SSBO (dummy)
    VkWriteDescriptorSet w28{};
    w28.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w28.dstSet = descriptorSet_;
    w28.dstBinding = 28;
    w28.descriptorCount = 1;
    w28.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w28.pBufferInfo = &geomInfo;  // reuse dummy buffer
    writes.push_back(w28);

    // Dummy image info for R8_UNORM storage images (confidence/reactive masks)
    VkDescriptorImageInfo dummyImageR8Info{};
    dummyImageR8Info.imageView = dummyImageViewR8_;
    dummyImageR8Info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // bindings 29-31: reactive mask, diff confidence, spec confidence (dummy R8 storage images)
    for (int i = 29; i <= 31; i++) {
        VkWriteDescriptorSet wr{};
        wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr.dstSet = descriptorSet_;
        wr.dstBinding = i;
        wr.descriptorCount = 1;
        wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wr.pImageInfo = &dummyImageR8Info;
        writes.push_back(wr);
    }

    // bindings 32-33: Surfel GI cache (dummy SSBOs, replaced by CreateSurfelBuffers)
    for (int i = 32; i <= 33; i++) {
        VkWriteDescriptorSet wr{};
        wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr.dstSet = descriptorSet_;
        wr.dstBinding = i;
        wr.descriptorCount = 1;
        wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr.pBufferInfo = &geomInfo;  // reuse dummy buffer
        writes.push_back(wr);
    }

    vkUpdateDescriptorSets(context_->GetDevice(), (uint32_t)writes.size(), writes.data(), 0, nullptr);

    Log(L"[VK RTPipeline] Descriptor set created with dummy resources\n");
    return true;
}

void RTPipeline::UpdateTLASDescriptor() {
    if (!accelBuilder_->HasTLAS()) return;

    VkDevice device = context_->GetDevice();
    VkAccelerationStructureKHR tlas = accelBuilder_->GetTLAS();

    // Create geometry metadata buffer from BLAS vertex/index addresses
    const auto& blasList = accelBuilder_->GetBLASList();
    if (!blasList.empty()) {
        // Shader struct: 8 x uint64 addresses + 2 x uint32 counts = 72 bytes per entry
        struct GPUGeometryMetadata {
            uint64_t vertexBufferAddress;
            uint64_t indexBufferAddress;
            uint64_t normalBufferAddress;
            uint64_t uvBufferAddress;
            uint64_t primMaterialBufferAddress;
            uint64_t bitangentBufferAddress;
            uint64_t primYBoundsAddress;
            uint64_t colorBufferAddress;
            uint32_t vertexCount;
            uint32_t indexCount;
        };

        std::vector<GPUGeometryMetadata> metaData(blasList.size());
        for (size_t i = 0; i < blasList.size(); i++) {
            metaData[i].vertexBufferAddress = blasList[i].vertexBuf.deviceAddress;
            metaData[i].indexBufferAddress = blasList[i].indexBuf.deviceAddress;
            metaData[i].normalBufferAddress = blasList[i].normalBuf.deviceAddress;
            metaData[i].uvBufferAddress = blasList[i].uvBuf.deviceAddress;
            metaData[i].primMaterialBufferAddress = blasList[i].primMaterialBuf.deviceAddress;
            metaData[i].bitangentBufferAddress = blasList[i].isHair ? 1u : 0u;
            metaData[i].primYBoundsAddress = blasList[i].primYBoundsBuf.deviceAddress;
            metaData[i].colorBufferAddress = blasList[i].colorBuf.deviceAddress;
            metaData[i].vertexCount = blasList[i].vertexCount;
            metaData[i].indexCount = blasList[i].indexCount;
        }

        VkDeviceSize metaSize = metaData.size() * sizeof(GPUGeometryMetadata);

        // Clean up old buffer
        if (geometryMetadataBuffer_) { vkDestroyBuffer(device, geometryMetadataBuffer_, nullptr); }
        if (geometryMetadataMemory_) { vkFreeMemory(device, geometryMetadataMemory_, nullptr); }

        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = metaSize;
        bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &bufInfo, nullptr, &geometryMetadataBuffer_);

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, geometryMetadataBuffer_, &memReqs);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(device, &allocInfo, nullptr, &geometryMetadataMemory_);
        vkBindBufferMemory(device, geometryMetadataBuffer_, geometryMetadataMemory_, 0);

        void* mapped;
        vkMapMemory(device, geometryMetadataMemory_, 0, metaSize, 0, &mapped);
        memcpy(mapped, metaData.data(), metaSize);
        vkUnmapMemory(device, geometryMetadataMemory_);

        static uint32_t s_metaLogCount = 0;
        if (s_metaLogCount++ < 3)
            Log(L"[VK RTPipeline] Geometry metadata buffer created (%zu entries, vtx=0x%llX, idx=0x%llX)\n",
                blasList.size(), metaData[0].vertexBufferAddress, metaData[0].indexBufferAddress);
    }

    // Update descriptors
    std::vector<VkWriteDescriptorSet> writes;

    // binding 0: TLAS
    VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
    asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &tlas;

    VkWriteDescriptorSet write0{};
    write0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write0.pNext = &asWrite;
    write0.dstSet = descriptorSet_;
    write0.dstBinding = 0;
    write0.descriptorCount = 1;
    write0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes.push_back(write0);

    // binding 3: geometry metadata
    if (geometryMetadataBuffer_) {
        VkDescriptorBufferInfo geomMetaInfo{};
        geomMetaInfo.buffer = geometryMetadataBuffer_;
        geomMetaInfo.offset = 0;
        geomMetaInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet write3{};
        write3.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write3.dstSet = descriptorSet_;
        write3.dstBinding = 3;
        write3.descriptorCount = 1;
        write3.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write3.pBufferInfo = &geomMetaInfo;
        writes.push_back(write3);
    }

    vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    static uint32_t s_tlasLogCount = 0;
    if (s_tlasLogCount++ < 3)
        Log(L"[VK RTPipeline] TLAS descriptor updated\n");
}

void RTPipeline::UpdateStorageImage(VkImageView imageView) {
    if (!imageView || descriptorSet_ == VK_NULL_HANDLE) return;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = imageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet_;
    write.dstBinding = 1;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(context_->GetDevice(), 1, &write, 0, nullptr);
    Log(L"[VK RTPipeline] Storage image descriptor updated (D3D11 interop)\n");
}

void RTPipeline::UpdateCloudShadowDescriptor(VkImageView view, VkSampler sampler) {
    if (!view || !sampler || descriptorSet_ == VK_NULL_HANDLE) return;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.sampler = sampler;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet_;
    write.dstBinding = 23;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(context_->GetDevice(), 1, &write, 0, nullptr);
    Log(L"[VK RTPipeline] Cloud shadow map descriptor updated (binding 23)\n");
}

void RTPipeline::UpdateCamera(const CameraUBO& camera) {
    static int logCount = 0;
    if (logCount < 3) {
        Log(L"[VK RTPipeline] UpdateCamera: viewInv col3=(%.2f, %.2f, %.2f), projInv[0]=(%.4f), size=%zu\n",
            camera.viewInverse[12], camera.viewInverse[13], camera.viewInverse[14],
            camera.projInverse[0], sizeof(CameraUBO));
        logCount++;
    }
    void* mapped;
    VkResult r = vkMapMemory(context_->GetDevice(), cameraMemory_, 0, sizeof(CameraUBO), 0, &mapped);
    if (r != VK_SUCCESS) {
        Log(L"[VK RTPipeline] UpdateCamera: vkMapMemory FAILED %d\n", r);
        return;
    }
    memcpy(mapped, &camera, sizeof(CameraUBO));
    vkUnmapMemory(context_->GetDevice(), cameraMemory_);
}

void RTPipeline::UpdateMaterialBuffer(const GPUMaterial* materials, uint32_t count) {
    if (!materials || count == 0) return;

    VkDevice device = context_->GetDevice();
    VkDeviceSize bufSize = count * sizeof(GPUMaterial);
    static bool loggedOnce = false;
    if (!loggedOnce) {
        Log(L"[VK RTPipeline] sizeof(GPUMaterial) = %zu, count=%u, bufSize=%llu\n",
            sizeof(GPUMaterial), count, (unsigned long long)bufSize);
        loggedOnce = true;
    }

    // Recreate if needed
    if (materialBuffer_) { vkDestroyBuffer(device, materialBuffer_, nullptr); }
    if (materialMemory_) { vkFreeMemory(device, materialMemory_, nullptr); }

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = bufSize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufInfo, nullptr, &materialBuffer_);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, materialBuffer_, &memReqs);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &materialMemory_);
    vkBindBufferMemory(device, materialBuffer_, materialMemory_, 0);

    void* mapped;
    vkMapMemory(device, materialMemory_, 0, bufSize, 0, &mapped);
    memcpy(mapped, materials, bufSize);
    vkUnmapMemory(device, materialMemory_);

    // Update descriptor binding 4
    VkDescriptorBufferInfo matInfo{};
    matInfo.buffer = materialBuffer_;
    matInfo.offset = 0;
    matInfo.range = bufSize;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet_;
    write.dstBinding = 4;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &matInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    Log(L"[VK RTPipeline] Material buffer updated: %u materials\n", count);
}

void RTPipeline::UpdateEmissiveTriangleBuffer(const float* data, uint32_t triangleCount) {
    if (!data || triangleCount == 0) {
        emissiveTriCount_ = 0;
        return;
    }

    VkDevice device = context_->GetDevice();
    VkDeviceSize bufSize = triangleCount * 16 * sizeof(float);  // 16 floats per triangle

    // Recreate if needed
    if (emissiveTriBuffer_) { vkDestroyBuffer(device, emissiveTriBuffer_, nullptr); }
    if (emissiveTriMemory_) { vkFreeMemory(device, emissiveTriMemory_, nullptr); }

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = bufSize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufInfo, nullptr, &emissiveTriBuffer_);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, emissiveTriBuffer_, &memReqs);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &emissiveTriMemory_);
    vkBindBufferMemory(device, emissiveTriBuffer_, emissiveTriMemory_, 0);

    void* mapped;
    vkMapMemory(device, emissiveTriMemory_, 0, bufSize, 0, &mapped);
    memcpy(mapped, data, bufSize);
    vkUnmapMemory(device, emissiveTriMemory_);

    emissiveTriCount_ = triangleCount;

    // Update descriptor binding 26
    VkDescriptorBufferInfo emissiveInfo{};
    emissiveInfo.buffer = emissiveTriBuffer_;
    emissiveInfo.offset = 0;
    emissiveInfo.range = bufSize;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet_;
    write.dstBinding = 26;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &emissiveInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    Log(L"[VK RTPipeline] Emissive triangle buffer updated: %u triangles\n", triangleCount);
}

void RTPipeline::UpdateLightTreeBuffer(const void* nodes, uint32_t nodeCount) {
    if (!nodes || nodeCount == 0) {
        lightTreeNodeCount_ = 0;
        return;
    }

    VkDevice device = context_->GetDevice();
    VkDeviceSize bufSize = nodeCount * 32;  // 32 bytes per LightTreeNode (8 floats)

    // Wait, LightTreeNode is 32 bytes but has 12 fields not 8. Let me check:
    // bboxMin[3] + energy + bboxMax[3] + childOrFirst + coneAxis[3] + countAndFlags = 12 floats = 48 bytes
    // Actually sizeof(LightTreeNode) may have padding. Use actual sizeof.
    bufSize = nodeCount * sizeof(float) * 12;  // 12 floats per node = 48 bytes

    if (lightTreeBuffer_) { vkDestroyBuffer(device, lightTreeBuffer_, nullptr); }
    if (lightTreeMemory_) { vkFreeMemory(device, lightTreeMemory_, nullptr); }

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = bufSize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufInfo, nullptr, &lightTreeBuffer_);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, lightTreeBuffer_, &memReqs);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &lightTreeMemory_);
    vkBindBufferMemory(device, lightTreeBuffer_, lightTreeMemory_, 0);

    void* mapped;
    vkMapMemory(device, lightTreeMemory_, 0, bufSize, 0, &mapped);
    memcpy(mapped, nodes, bufSize);
    vkUnmapMemory(device, lightTreeMemory_);

    lightTreeNodeCount_ = nodeCount;

    // Update descriptor binding 27
    VkDescriptorBufferInfo treeInfo{};
    treeInfo.buffer = lightTreeBuffer_;
    treeInfo.offset = 0;
    treeInfo.range = bufSize;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet_;
    write.dstBinding = 27;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &treeInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void RTPipeline::UpdatePrevTransforms(const float* transforms, uint32_t instanceCount) {
    if (!transforms || instanceCount == 0) return;

    VkDevice device = context_->GetDevice();
    VkDeviceSize bufSize = instanceCount * 12 * sizeof(float);  // 48 bytes per instance

    // Recreate buffer if capacity is insufficient
    if (instanceCount > prevTransformsCapacity_) {
        if (prevTransformsBuffer_) {
            if (prevTransformsMapped_) { vkUnmapMemory(device, prevTransformsMemory_); prevTransformsMapped_ = nullptr; }
            vkDestroyBuffer(device, prevTransformsBuffer_, nullptr);
            prevTransformsBuffer_ = VK_NULL_HANDLE;
        }
        if (prevTransformsMemory_) { vkFreeMemory(device, prevTransformsMemory_, nullptr); prevTransformsMemory_ = VK_NULL_HANDLE; }

        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = bufSize;
        bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &bufInfo, nullptr, &prevTransformsBuffer_);

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, prevTransformsBuffer_, &memReqs);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(device, &allocInfo, nullptr, &prevTransformsMemory_);
        vkBindBufferMemory(device, prevTransformsBuffer_, prevTransformsMemory_, 0);

        // Persistently map
        vkMapMemory(device, prevTransformsMemory_, 0, bufSize, 0, &prevTransformsMapped_);
        prevTransformsCapacity_ = instanceCount;
    }

    // Upload transform data
    memcpy(prevTransformsMapped_, transforms, bufSize);

    // Update descriptor set binding 28
    VkDescriptorBufferInfo bufInfoDesc{};
    bufInfoDesc.buffer = prevTransformsBuffer_;
    bufInfoDesc.offset = 0;
    bufInfoDesc.range = bufSize;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet_;
    write.dstBinding = 28;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufInfoDesc;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void RTPipeline::UpdateTextureDescriptors(TextureManager* texMgr) {
    if (!texMgr || texMgr->GetTextureCount() == 0) return;

    VkDevice device = context_->GetDevice();
    uint32_t texCount = texMgr->GetTextureCount();

    // Clamp to descriptor array limit to prevent descriptor overflow
    if (texCount > 1024) {
        Log(L"[VK RTPipeline] WARNING: %u textures exceeds 1024 limit, clamping\n", texCount);
        texCount = 1024;
    }

    // Descriptor set was allocated with 1024 slots and PARTIALLY_BOUND,
    // so we just overwrite binding 5 in place — no pool reset needed
    std::vector<VkDescriptorImageInfo> imageInfos(texCount);
    for (uint32_t i = 0; i < texCount; i++) {
        imageInfos[i].imageView = texMgr->GetImageView(i);
        imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfos[i].sampler = texMgr->GetSampler();
    }

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet_;
    write.dstBinding = 5;
    write.descriptorCount = texCount;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = imageInfos.data();

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    Log(L"[VK RTPipeline] Texture descriptors updated: %u textures\n", texCount);
}

void RTPipeline::RecordDispatch(VkCommandBuffer cmd, uint32_t width, uint32_t height) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);

    vkCmdTraceRaysKHR_(cmd,
        &raygenRegion_,
        &missRegion_,
        &hitRegion_,
        &callableRegion_,
        width, height, 1);
}

PickResult RTPipeline::ReadPickResult() {
    if (!pickMappedPtr_) return {0, 0, 0, 0};
    return *pickMappedPtr_;
}

void RTPipeline::ResetPickBuffer() {
    if (pickMappedPtr_) pickMappedPtr_->valid = 0;
}

} // namespace vk
} // namespace acpt
