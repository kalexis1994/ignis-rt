#include "vk_accel_structure.h"
#include "vk_context.h"
#include "../../include/ignis_log.h"
#include <cstring>

namespace acpt {
namespace vk {

bool AccelStructureBuilder::Initialize(Context* context) {
    context_ = context;
    VkDevice device = context_->GetDevice();

    if (!context_->IsRayQuerySupported()) {
        Log(L"[VK AccelStruct] Ray query not supported, skipping init\n");
        return false;
    }

    // Load function pointers
    vkGetAccelerationStructureBuildSizesKHR_ = (PFN_vkGetAccelerationStructureBuildSizesKHR)
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
    vkCreateAccelerationStructureKHR_ = (PFN_vkCreateAccelerationStructureKHR)
        vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
    vkDestroyAccelerationStructureKHR_ = (PFN_vkDestroyAccelerationStructureKHR)
        vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
    vkCmdBuildAccelerationStructuresKHR_ = (PFN_vkCmdBuildAccelerationStructuresKHR)
        vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
    vkGetAccelerationStructureDeviceAddressKHR_ = (PFN_vkGetAccelerationStructureDeviceAddressKHR)
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
    vkGetBufferDeviceAddressKHR_ = (PFN_vkGetBufferDeviceAddressKHR)
        vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR");
    vkCmdCopyAccelerationStructureKHR_ = (PFN_vkCmdCopyAccelerationStructureKHR)
        vkGetDeviceProcAddr(device, "vkCmdCopyAccelerationStructureKHR");
    vkCmdWriteAccelerationStructuresPropertiesKHR_ = (PFN_vkCmdWriteAccelerationStructuresPropertiesKHR)
        vkGetDeviceProcAddr(device, "vkCmdWriteAccelerationStructuresPropertiesKHR");

    if (!vkGetAccelerationStructureBuildSizesKHR_ || !vkCreateAccelerationStructureKHR_ ||
        !vkCmdBuildAccelerationStructuresKHR_ || !vkGetBufferDeviceAddressKHR_) {
        Log(L"[VK AccelStruct] ERROR: Failed to load RT function pointers\n");
        return false;
    }

    Log(L"[VK AccelStruct] Initialized successfully\n");
    return true;
}

void AccelStructureBuilder::Shutdown() {
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) return;

    // Destroy TLAS
    if (tlas_ != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_) {
        vkDestroyAccelerationStructureKHR_(device, tlas_, nullptr);
        tlas_ = VK_NULL_HANDLE;
    }
    DestroyAccelBuffer(tlasBuffer_);
    DestroyAccelBuffer(instanceBuffer_);
    DestroyAccelBuffer(tlasScratchBuffer_);
    DestroyAccelBuffer(gpuBlasScratchBuffer_);
    gpuBlasScratchSize_ = 0;

    ClearBLAS();
    Log(L"[VK AccelStruct] Shutdown\n");
}

void AccelStructureBuilder::ClearBLAS() {
    if (!context_) return;
    VkDevice device = context_->GetDevice();

    for (auto& blas : blasList_) {
        if (blas.handle != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_) {
            vkDestroyAccelerationStructureKHR_(device, blas.handle, nullptr);
        }
        DestroyAccelBuffer(blas.buffer);
        DestroyAccelBuffer(blas.vertexBuf);
        DestroyAccelBuffer(blas.indexBuf);
        DestroyAccelBuffer(blas.normalBuf);
        DestroyAccelBuffer(blas.uvBuf);
        DestroyAccelBuffer(blas.primMaterialBuf);
        DestroyAccelBuffer(blas.primYBoundsBuf);
    }
    blasList_.clear();

    // Destroy geometry buffers
    for (auto& gb : geometryBuffers_) {
        DestroyAccelBuffer(gb.vertexBuffer);
        DestroyAccelBuffer(gb.indexBuffer);
        DestroyAccelBuffer(gb.normalBuffer);
        DestroyAccelBuffer(gb.uvBuffer);
        DestroyAccelBuffer(gb.tangentBuffer);
    }
    geometryBuffers_.clear();

    Log(L"[VK AccelStruct] Shutdown\n");
}

AccelBuffer AccelStructureBuilder::CreateAccelBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                                      VkMemoryPropertyFlags memProps) {
    AccelBuffer result{};
    result.size = size;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(context_->GetDevice(), &bufferInfo, nullptr, &result.buffer) != VK_SUCCESS) {
        Log(L"[VK AccelStruct] ERROR: Failed to create buffer (size=%llu)\n", size);
        return result;
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(context_->GetDevice(), result.buffer, &memReqs);

    VkMemoryAllocateFlagsInfo allocFlags{};
    allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits, memProps);
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        allocInfo.pNext = &allocFlags;
    }

    if (vkAllocateMemory(context_->GetDevice(), &allocInfo, nullptr, &result.memory) != VK_SUCCESS) {
        Log(L"[VK AccelStruct] ERROR: Failed to allocate buffer memory\n");
        vkDestroyBuffer(context_->GetDevice(), result.buffer, nullptr);
        result.buffer = VK_NULL_HANDLE;
        return result;
    }

    vkBindBufferMemory(context_->GetDevice(), result.buffer, result.memory, 0);

    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        result.deviceAddress = GetBufferDeviceAddress(result.buffer);
    }

    return result;
}

void AccelStructureBuilder::DestroyAccelBuffer(AccelBuffer& buf) {
    if (buf.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(context_->GetDevice(), buf.buffer, nullptr);
    }
    if (buf.memory != VK_NULL_HANDLE) {
        vkFreeMemory(context_->GetDevice(), buf.memory, nullptr);
    }
    buf = {};
}

VkDeviceAddress AccelStructureBuilder::GetBufferDeviceAddress(VkBuffer buffer) {
    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddressKHR_(context_->GetDevice(), &info);
}

int AccelStructureBuilder::BuildBLAS(const float* vertices, uint32_t vertexCount,
                                      const uint32_t* indices, uint32_t indexCount,
                                      bool allowUpdate) {
    VkDevice device = context_->GetDevice();

    // Upload vertex data to GPU buffer
    // STORAGE_BUFFER_BIT added so compute shaders can write displaced vertices (tree wind)
    VkDeviceSize vertexSize = vertexCount * 3 * sizeof(float);
    AccelBuffer vertexBuf = CreateAccelBuffer(vertexSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkDeviceSize indexSize = indexCount * sizeof(uint32_t);
    AccelBuffer indexBuf = CreateAccelBuffer(indexSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!vertexBuf.buffer || !indexBuf.buffer) {
        DestroyAccelBuffer(vertexBuf);
        DestroyAccelBuffer(indexBuf);
        return -1;
    }

    // Stage and copy vertex/index data
    AccelBuffer vertexStaging = CreateAccelBuffer(vertexSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    AccelBuffer indexStaging = CreateAccelBuffer(indexSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* mapped;
    vkMapMemory(device, vertexStaging.memory, 0, vertexSize, 0, &mapped);
    memcpy(mapped, vertices, vertexSize);
    vkUnmapMemory(device, vertexStaging.memory);

    vkMapMemory(device, indexStaging.memory, 0, indexSize, 0, &mapped);
    memcpy(mapped, indices, indexSize);
    vkUnmapMemory(device, indexStaging.memory);

    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    VkBufferCopy copyRegion{};
    copyRegion.size = vertexSize;
    vkCmdCopyBuffer(cmd, vertexStaging.buffer, vertexBuf.buffer, 1, &copyRegion);
    copyRegion.size = indexSize;
    vkCmdCopyBuffer(cmd, indexStaging.buffer, indexBuf.buffer, 1, &copyRegion);

    // Barrier before AS build
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0,
        1, &barrier, 0, nullptr, 0, nullptr);
    context_->EndSingleTimeCommands(cmd);

    DestroyAccelBuffer(vertexStaging);
    DestroyAccelBuffer(indexStaging);

    // Build geometry info
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexBuf.deviceAddress;
    triangles.vertexStride = 3 * sizeof(float);
    triangles.maxVertex = vertexCount - 1;
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexBuf.deviceAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = 0;  // Non-opaque: enables alpha test via candidate intersection
    geometry.geometry.triangles = triangles;

    uint32_t primitiveCount = indexCount / 3;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (allowUpdate) {
        buildInfo.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }
    // DISABLED: compaction suspected of causing mesh mismatch (address invalidation)
    bool canCompact = false; // !allowUpdate && vkCmdCopyAccelerationStructureKHR_ && vkCmdWriteAccelerationStructuresPropertiesKHR_;
    if (canCompact) {
        buildInfo.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    }
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    // Query sizes
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR_(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo);

    // Create AS buffer
    AccelBuffer asBuf = CreateAccelBuffer(sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Create AS
    VkAccelerationStructureCreateInfoKHR asCreateInfo{};
    asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCreateInfo.buffer = asBuf.buffer;
    asCreateInfo.size = sizeInfo.accelerationStructureSize;
    asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    VkAccelerationStructureKHR blas;
    if (vkCreateAccelerationStructureKHR_(device, &asCreateInfo, nullptr, &blas) != VK_SUCCESS) {
        Log(L"[VK AccelStruct] ERROR: Failed to create BLAS\n");
        DestroyAccelBuffer(asBuf);
        DestroyAccelBuffer(vertexBuf);
        DestroyAccelBuffer(indexBuf);
        return -1;
    }

    // Scratch buffer
    AccelBuffer scratchBuf = CreateAccelBuffer(sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Build
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = blas;
    buildInfo.scratchData.deviceAddress = scratchBuf.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    cmd = context_->BeginSingleTimeCommands();
    vkCmdBuildAccelerationStructuresKHR_(cmd, 1, &buildInfo, &pRangeInfo);
    context_->EndSingleTimeCommands(cmd);

    DestroyAccelBuffer(scratchBuf);

    // BVH compaction: query compacted size, copy to smaller buffer
    if (canCompact) {
        VkQueryPoolCreateInfo queryPoolInfo{};
        queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolInfo.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        queryPoolInfo.queryCount = 1;
        VkQueryPool queryPool;
        if (vkCreateQueryPool(device, &queryPoolInfo, nullptr, &queryPool) == VK_SUCCESS) {
            vkResetQueryPool(device, queryPool, 0, 1);

            cmd = context_->BeginSingleTimeCommands();
            vkCmdWriteAccelerationStructuresPropertiesKHR_(cmd,
                1, &blas, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, 0);
            context_->EndSingleTimeCommands(cmd);

            VkDeviceSize compactedSize = 0;
            vkGetQueryPoolResults(device, queryPool, 0, 1, sizeof(VkDeviceSize),
                &compactedSize, sizeof(VkDeviceSize), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            vkDestroyQueryPool(device, queryPool, nullptr);

            if (compactedSize > 0 && compactedSize < sizeInfo.accelerationStructureSize) {
                AccelBuffer compactBuf = CreateAccelBuffer(compactedSize,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

                VkAccelerationStructureCreateInfoKHR compactCreateInfo{};
                compactCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
                compactCreateInfo.buffer = compactBuf.buffer;
                compactCreateInfo.size = compactedSize;
                compactCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

                VkAccelerationStructureKHR compactBlas;
                if (vkCreateAccelerationStructureKHR_(device, &compactCreateInfo, nullptr, &compactBlas) == VK_SUCCESS) {
                    VkCopyAccelerationStructureInfoKHR copyInfo{};
                    copyInfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
                    copyInfo.src = blas;
                    copyInfo.dst = compactBlas;
                    copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;

                    cmd = context_->BeginSingleTimeCommands();
                    vkCmdCopyAccelerationStructureKHR_(cmd, &copyInfo);
                    context_->EndSingleTimeCommands(cmd);

                    // Replace original with compacted version
                    vkDestroyAccelerationStructureKHR_(device, blas, nullptr);
                    DestroyAccelBuffer(asBuf);
                    blas = compactBlas;
                    asBuf = compactBuf;
                } else {
                    DestroyAccelBuffer(compactBuf);
                }
            }
        }
    }

    // Get device address
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = blas;
    VkDeviceAddress blasAddr = vkGetAccelerationStructureDeviceAddressKHR_(device, &addressInfo);

    // Store
    BLAS blasEntry{};
    blasEntry.handle = blas;
    blasEntry.buffer = asBuf;
    blasEntry.deviceAddress = blasAddr;
    blasEntry.vertexCount = vertexCount;
    blasEntry.indexCount = indexCount;
    blasEntry.built = true;
    // Compute Y bounding box from vertex positions (float3, stride=3)
    if (vertexCount > 0) {
        float yMin = vertices[1]; // first vertex Y
        float yMax = vertices[1];
        for (uint32_t v = 1; v < vertexCount; v++) {
            float y = vertices[v * 3 + 1];
            if (y < yMin) yMin = y;
            if (y > yMax) yMax = y;
        }
        blasEntry.minY = yMin;
        blasEntry.maxY = yMax;
    }
    // Keep vertex/index buffers alive for shader attribute access
    blasEntry.vertexBuf = vertexBuf;
    blasEntry.indexBuf = indexBuf;

    int index = (int)blasList_.size();
    blasList_.push_back(blasEntry);

    Log(L"[VK AccelStruct] BLAS[%d] built: %u verts, %u tris\n", index, vertexCount, primitiveCount);

    return index;
}

bool AccelStructureBuilder::RefitBLAS(int blasIndex, const float* vertices, uint32_t vertexCount,
                                       const uint32_t* indices, uint32_t indexCount) {
    if (blasIndex < 0 || blasIndex >= (int)blasList_.size()) return false;
    VkDevice device = context_->GetDevice();

    // Destroy old BLAS resources
    auto& oldBlas = blasList_[blasIndex];
    if (oldBlas.handle != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_) {
        vkDestroyAccelerationStructureKHR_(device, oldBlas.handle, nullptr);
    }
    DestroyAccelBuffer(oldBlas.buffer);
    DestroyAccelBuffer(oldBlas.vertexBuf);
    DestroyAccelBuffer(oldBlas.indexBuf);
    oldBlas.handle = VK_NULL_HANDLE;
    oldBlas.built = false;

    // Upload new vertex/index data
    VkDeviceSize vertexSize = vertexCount * 3 * sizeof(float);
    VkDeviceSize indexSize = indexCount * sizeof(uint32_t);

    AccelBuffer vertexBuf = CreateAccelBuffer(vertexSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    AccelBuffer indexBuf = CreateAccelBuffer(indexSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!vertexBuf.buffer || !indexBuf.buffer) {
        DestroyAccelBuffer(vertexBuf);
        DestroyAccelBuffer(indexBuf);
        return false;
    }

    AccelBuffer vertexStaging = CreateAccelBuffer(vertexSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    AccelBuffer indexStaging = CreateAccelBuffer(indexSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* mapped;
    vkMapMemory(device, vertexStaging.memory, 0, vertexSize, 0, &mapped);
    memcpy(mapped, vertices, vertexSize);
    vkUnmapMemory(device, vertexStaging.memory);
    vkMapMemory(device, indexStaging.memory, 0, indexSize, 0, &mapped);
    memcpy(mapped, indices, indexSize);
    vkUnmapMemory(device, indexStaging.memory);

    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    VkBufferCopy copyRegion{};
    copyRegion.size = vertexSize;
    vkCmdCopyBuffer(cmd, vertexStaging.buffer, vertexBuf.buffer, 1, &copyRegion);
    copyRegion.size = indexSize;
    vkCmdCopyBuffer(cmd, indexStaging.buffer, indexBuf.buffer, 1, &copyRegion);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0,
        1, &barrier, 0, nullptr, 0, nullptr);
    context_->EndSingleTimeCommands(cmd);

    DestroyAccelBuffer(vertexStaging);
    DestroyAccelBuffer(indexStaging);

    // Build geometry info
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexBuf.deviceAddress;
    triangles.vertexStride = 3 * sizeof(float);
    triangles.maxVertex = vertexCount - 1;
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexBuf.deviceAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = 0;  // Non-opaque: enables alpha test via candidate intersection
    geometry.geometry.triangles = triangles;

    uint32_t primitiveCount = indexCount / 3;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR_(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo);

    AccelBuffer asBuf = CreateAccelBuffer(sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkAccelerationStructureCreateInfoKHR asCreateInfo{};
    asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCreateInfo.buffer = asBuf.buffer;
    asCreateInfo.size = sizeInfo.accelerationStructureSize;
    asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    VkAccelerationStructureKHR blas;
    if (vkCreateAccelerationStructureKHR_(device, &asCreateInfo, nullptr, &blas) != VK_SUCCESS) {
        DestroyAccelBuffer(asBuf);
        DestroyAccelBuffer(vertexBuf);
        DestroyAccelBuffer(indexBuf);
        return false;
    }

    AccelBuffer scratchBuf = CreateAccelBuffer(sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = blas;
    buildInfo.scratchData.deviceAddress = scratchBuf.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    cmd = context_->BeginSingleTimeCommands();
    vkCmdBuildAccelerationStructuresKHR_(cmd, 1, &buildInfo, &pRangeInfo);
    context_->EndSingleTimeCommands(cmd);

    DestroyAccelBuffer(scratchBuf);

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = blas;
    VkDeviceAddress blasAddr = vkGetAccelerationStructureDeviceAddressKHR_(device, &addressInfo);

    // Replace in-place (no vector shift)
    oldBlas.handle = blas;
    oldBlas.buffer = asBuf;
    oldBlas.deviceAddress = blasAddr;
    oldBlas.vertexCount = vertexCount;
    oldBlas.indexCount = indexCount;
    oldBlas.built = true;
    oldBlas.vertexBuf = vertexBuf;
    oldBlas.indexBuf = indexBuf;

    return true;
}

bool AccelStructureBuilder::UploadBLASAttributes(int blasIndex, const float* normals, const float* uvs, uint32_t vertexCount) {
    if (blasIndex < 0 || blasIndex >= (int)blasList_.size()) return false;
    if (!normals && !uvs) return false;
    if (vertexCount == 0) return false;

    auto& blas = blasList_[blasIndex];

    auto uploadBuffer = [&](const float* data, VkDeviceSize size) -> AccelBuffer {
        if (!data || size == 0) return {};
        AccelBuffer staging = CreateAccelBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        AccelBuffer gpu = CreateAccelBuffer(size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!staging.buffer || !gpu.buffer) {
            DestroyAccelBuffer(staging);
            DestroyAccelBuffer(gpu);
            return {};
        }

        void* mapped;
        vkMapMemory(context_->GetDevice(), staging.memory, 0, size, 0, &mapped);
        memcpy(mapped, data, size);
        vkUnmapMemory(context_->GetDevice(), staging.memory);

        VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
        VkBufferCopy copy{};
        copy.size = size;
        vkCmdCopyBuffer(cmd, staging.buffer, gpu.buffer, 1, &copy);
        context_->EndSingleTimeCommands(cmd);

        DestroyAccelBuffer(staging);
        return gpu;
    };

    if (normals) {
        DestroyAccelBuffer(blas.normalBuf);
        blas.normalBuf = uploadBuffer(normals, vertexCount * 3 * sizeof(float));
    }
    if (uvs) {
        DestroyAccelBuffer(blas.uvBuf);
        blas.uvBuf = uploadBuffer(uvs, vertexCount * 2 * sizeof(float));
    }

    Log(L"[VK AccelStruct] BLAS[%d] attributes uploaded: normals=%s uvs=%s (%u verts)\n",
        blasIndex, normals ? L"yes" : L"no", uvs ? L"yes" : L"no", vertexCount);
    return true;
}

int AccelStructureBuilder::UploadGeometryBuffers(const float* vertices, uint32_t vertexCount,
                                                   const float* normals,
                                                   const float* uvs,
                                                   const float* tangents,
                                                   const uint32_t* indices, uint32_t indexCount,
                                                   uint32_t materialId) {
    GeometryBuffers gb{};
    gb.vertexCount = vertexCount;
    gb.indexCount = indexCount;
    gb.materialId = materialId;

    auto uploadBuffer = [&](const void* data, VkDeviceSize size, VkBufferUsageFlags extraUsage) -> AccelBuffer {
        if (!data || size == 0) return {};
        AccelBuffer staging = CreateAccelBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        AccelBuffer gpu = CreateAccelBuffer(size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | extraUsage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        void* mapped;
        vkMapMemory(context_->GetDevice(), staging.memory, 0, size, 0, &mapped);
        memcpy(mapped, data, size);
        vkUnmapMemory(context_->GetDevice(), staging.memory);

        VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
        VkBufferCopy copy{};
        copy.size = size;
        vkCmdCopyBuffer(cmd, staging.buffer, gpu.buffer, 1, &copy);
        context_->EndSingleTimeCommands(cmd);

        DestroyAccelBuffer(staging);
        return gpu;
    };

    gb.vertexBuffer = uploadBuffer(vertices, vertexCount * 3 * sizeof(float), 0);
    gb.indexBuffer = uploadBuffer(indices, indexCount * sizeof(uint32_t), 0);
    gb.normalBuffer = uploadBuffer(normals, normals ? vertexCount * 3 * sizeof(float) : 0, 0);
    gb.uvBuffer = uploadBuffer(uvs, uvs ? vertexCount * 2 * sizeof(float) : 0, 0);
    gb.tangentBuffer = uploadBuffer(tangents, tangents ? vertexCount * 4 * sizeof(float) : 0, 0);

    int index = (int)geometryBuffers_.size();
    geometryBuffers_.push_back(gb);

    Log(L"[VK AccelStruct] GeometryBuffers[%d] uploaded: %u verts, %u indices, mat=%u\n",
        index, vertexCount, indexCount, materialId);
    return index;
}

bool AccelStructureBuilder::UploadBLASPrimitiveMaterials(int blasIndex, const uint32_t* materialIds, uint32_t primitiveCount) {
    if (blasIndex < 0 || blasIndex >= (int)blasList_.size()) return false;
    if (!materialIds || primitiveCount == 0) return false;

    auto& blas = blasList_[blasIndex];
    VkDeviceSize dataSize = primitiveCount * sizeof(uint32_t);

    // Staging buffer
    AccelBuffer staging = CreateAccelBuffer(dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // GPU buffer with device address
    AccelBuffer gpu = CreateAccelBuffer(dataSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!staging.buffer || !gpu.buffer) {
        DestroyAccelBuffer(staging);
        DestroyAccelBuffer(gpu);
        return false;
    }

    void* mapped;
    vkMapMemory(context_->GetDevice(), staging.memory, 0, dataSize, 0, &mapped);
    memcpy(mapped, materialIds, dataSize);
    vkUnmapMemory(context_->GetDevice(), staging.memory);

    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    VkBufferCopy copy{};
    copy.size = dataSize;
    vkCmdCopyBuffer(cmd, staging.buffer, gpu.buffer, 1, &copy);
    context_->EndSingleTimeCommands(cmd);

    DestroyAccelBuffer(staging);

    DestroyAccelBuffer(blas.primMaterialBuf);
    blas.primMaterialBuf = gpu;

    Log(L"[VK AccelStruct] BLAS[%d] primitive materials uploaded (%u triangles)\n", blasIndex, primitiveCount);
    return true;
}

bool AccelStructureBuilder::UploadBLASPrimitiveYBounds(int blasIndex, const float* yBounds, uint32_t primitiveCount) {
    if (blasIndex < 0 || blasIndex >= (int)blasList_.size()) return false;
    if (!yBounds || primitiveCount == 0) return false;

    auto& blas = blasList_[blasIndex];
    VkDeviceSize dataSize = primitiveCount * 2 * sizeof(float);

    // Staging buffer
    AccelBuffer staging = CreateAccelBuffer(dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // GPU buffer with device address
    AccelBuffer gpu = CreateAccelBuffer(dataSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!staging.buffer || !gpu.buffer) {
        DestroyAccelBuffer(staging);
        DestroyAccelBuffer(gpu);
        return false;
    }

    void* mapped;
    vkMapMemory(context_->GetDevice(), staging.memory, 0, dataSize, 0, &mapped);
    memcpy(mapped, yBounds, dataSize);
    vkUnmapMemory(context_->GetDevice(), staging.memory);

    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    VkBufferCopy copy{};
    copy.size = dataSize;
    vkCmdCopyBuffer(cmd, staging.buffer, gpu.buffer, 1, &copy);
    context_->EndSingleTimeCommands(cmd);

    DestroyAccelBuffer(staging);

    DestroyAccelBuffer(blas.primYBoundsBuf);
    blas.primYBoundsBuf = gpu;

    Log(L"[VK AccelStruct] BLAS[%d] Y bounds uploaded (%u triangles)\n", blasIndex, primitiveCount);
    return true;
}

int AccelStructureBuilder::BuildBLASFromGPUBuffers(VkDeviceAddress vertexAddr, uint32_t vertexCount,
                                                     VkDeviceAddress indexAddr, uint32_t indexCount,
                                                     int reuseBlasIndex,
                                                     VkCommandBuffer externalCmd,
                                                     bool opaqueGeometry) {
    VkDevice device = context_->GetDevice();

    // Build geometry info
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexAddr;
    triangles.vertexStride = 3 * sizeof(float);
    triangles.maxVertex = vertexCount > 0 ? vertexCount - 1 : 0;
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexAddr;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = opaqueGeometry ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
    geometry.geometry.triangles = triangles;

    uint32_t primitiveCount = indexCount / 3;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    // Query sizes
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR_(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo);

    int blasIndex = reuseBlasIndex;
    bool reusing = (blasIndex >= 0 && blasIndex < (int)blasList_.size());

    // Fast path: reuse existing AS handle + buffer if large enough (zero allocations)
    if (reusing) {
        auto& oldBlas = blasList_[blasIndex];
        if (oldBlas.handle != VK_NULL_HANDLE && oldBlas.buffer.size >= sizeInfo.accelerationStructureSize) {
            // Reuse existing AS — just record the build command, no CPU allocs
            if (sizeInfo.buildScratchSize > gpuBlasScratchSize_) {
                DestroyAccelBuffer(gpuBlasScratchBuffer_);
                gpuBlasScratchBuffer_ = CreateAccelBuffer(sizeInfo.buildScratchSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                gpuBlasScratchSize_ = sizeInfo.buildScratchSize;
            }

            buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            buildInfo.dstAccelerationStructure = oldBlas.handle;
            buildInfo.scratchData.deviceAddress = gpuBlasScratchBuffer_.deviceAddress;

            VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
            rangeInfo.primitiveCount = primitiveCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

            if (externalCmd != VK_NULL_HANDLE) {
                vkCmdBuildAccelerationStructuresKHR_(externalCmd, 1, &buildInfo, &pRangeInfo);
            } else {
                VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
                vkCmdBuildAccelerationStructuresKHR_(cmd, 1, &buildInfo, &pRangeInfo);
                context_->EndSingleTimeCommands(cmd);
            }

            oldBlas.vertexCount = vertexCount;
            oldBlas.indexCount = indexCount;
            oldBlas.built = true;

            return blasIndex;
        }

        // Buffer too small — destroy and reallocate below
        if (oldBlas.handle != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_) {
            vkDestroyAccelerationStructureKHR_(device, oldBlas.handle, nullptr);
            oldBlas.handle = VK_NULL_HANDLE;
        }
        DestroyAccelBuffer(oldBlas.buffer);
        oldBlas.vertexBuf = {};
        oldBlas.indexBuf = {};
        oldBlas.built = false;
    }

    // Slow path: allocate new AS buffer + handle (first build or size increase)
    AccelBuffer asBuf = CreateAccelBuffer(sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!asBuf.buffer) {
        Log(L"[VK AccelStruct] ERROR: Failed to create AS buffer for GPU BLAS\n");
        return -1;
    }

    VkAccelerationStructureCreateInfoKHR asCreateInfo{};
    asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCreateInfo.buffer = asBuf.buffer;
    asCreateInfo.size = sizeInfo.accelerationStructureSize;
    asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    VkAccelerationStructureKHR blas;
    if (vkCreateAccelerationStructureKHR_(device, &asCreateInfo, nullptr, &blas) != VK_SUCCESS) {
        Log(L"[VK AccelStruct] ERROR: Failed to create BLAS from GPU buffers\n");
        DestroyAccelBuffer(asBuf);
        return -1;
    }

    // Persistent scratch buffer: grow if needed
    if (sizeInfo.buildScratchSize > gpuBlasScratchSize_) {
        DestroyAccelBuffer(gpuBlasScratchBuffer_);
        gpuBlasScratchBuffer_ = CreateAccelBuffer(sizeInfo.buildScratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        gpuBlasScratchSize_ = sizeInfo.buildScratchSize;
    }

    // Build
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = blas;
    buildInfo.scratchData.deviceAddress = gpuBlasScratchBuffer_.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    if (externalCmd != VK_NULL_HANDLE) {
        vkCmdBuildAccelerationStructuresKHR_(externalCmd, 1, &buildInfo, &pRangeInfo);
    } else {
        VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
        vkCmdBuildAccelerationStructuresKHR_(cmd, 1, &buildInfo, &pRangeInfo);
        context_->EndSingleTimeCommands(cmd);
    }

    // Get device address
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = blas;
    VkDeviceAddress blasAddr = vkGetAccelerationStructureDeviceAddressKHR_(device, &addressInfo);

    // Store
    if (reusing) {
        auto& entry = blasList_[blasIndex];
        entry.handle = blas;
        entry.buffer = asBuf;
        entry.deviceAddress = blasAddr;
        entry.vertexCount = vertexCount;
        entry.indexCount = indexCount;
        entry.built = true;
    } else {
        BLAS blasEntry{};
        blasEntry.handle = blas;
        blasEntry.buffer = asBuf;
        blasEntry.deviceAddress = blasAddr;
        blasEntry.vertexCount = vertexCount;
        blasEntry.indexCount = indexCount;
        blasEntry.built = true;
        blasIndex = (int)blasList_.size();
        blasList_.push_back(blasEntry);
    }

    Log(L"[VK AccelStruct] BLAS[%d] built from GPU buffers: %u verts, %u tris%s%s\n",
        blasIndex, vertexCount, primitiveCount,
        opaqueGeometry ? L" (opaque)" : L"",
        externalCmd != VK_NULL_HANDLE ? L" (in-cmd)" : L" (single-time)");

    return blasIndex;
}

bool AccelStructureBuilder::BuildTLAS() {
    if (blasList_.empty()) {
        Log(L"[VK AccelStruct] No BLAS to build TLAS from\n");
        return false;
    }

    VkDevice device = context_->GetDevice();

    // Destroy old TLAS
    if (tlas_ != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_) {
        vkDestroyAccelerationStructureKHR_(device, tlas_, nullptr);
        tlas_ = VK_NULL_HANDLE;
    }
    DestroyAccelBuffer(tlasBuffer_);
    DestroyAccelBuffer(instanceBuffer_);

    // Build instance data
    std::vector<VkAccelerationStructureInstanceKHR> instances(blasList_.size());
    for (size_t i = 0; i < blasList_.size(); i++) {
        VkAccelerationStructureInstanceKHR& inst = instances[i];
        memset(&inst, 0, sizeof(inst));
        // Identity transform
        inst.transform.matrix[0][0] = 1.0f;
        inst.transform.matrix[1][1] = 1.0f;
        inst.transform.matrix[2][2] = 1.0f;
        inst.instanceCustomIndex = (uint32_t)i;
        inst.mask = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = 0;  // Allow backface culling via ray flags
        inst.accelerationStructureReference = blasList_[i].deviceAddress;
    }

    // Upload instance buffer
    VkDeviceSize instanceSize = instances.size() * sizeof(VkAccelerationStructureInstanceKHR);
    AccelBuffer instanceStaging = CreateAccelBuffer(instanceSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    instanceBuffer_ = CreateAccelBuffer(instanceSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    void* mapped;
    vkMapMemory(device, instanceStaging.memory, 0, instanceSize, 0, &mapped);
    memcpy(mapped, instances.data(), instanceSize);
    vkUnmapMemory(device, instanceStaging.memory);

    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    VkBufferCopy copy{};
    copy.size = instanceSize;
    vkCmdCopyBuffer(cmd, instanceStaging.buffer, instanceBuffer_.buffer, 1, &copy);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0,
        1, &barrier, 0, nullptr, 0, nullptr);
    context_->EndSingleTimeCommands(cmd);

    DestroyAccelBuffer(instanceStaging);

    // Geometry info for TLAS
    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.data.deviceAddress = instanceBuffer_.deviceAddress;

    VkAccelerationStructureGeometryKHR tlasGeometry{};
    tlasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeometry.geometry.instances = instancesData;

    uint32_t instanceCount = (uint32_t)instances.size();

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &tlasGeometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR_(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &instanceCount, &sizeInfo);

    // Create TLAS buffer
    tlasBuffer_ = CreateAccelBuffer(sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkAccelerationStructureCreateInfoKHR tlasCreateInfo{};
    tlasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    tlasCreateInfo.buffer = tlasBuffer_.buffer;
    tlasCreateInfo.size = sizeInfo.accelerationStructureSize;
    tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    if (vkCreateAccelerationStructureKHR_(device, &tlasCreateInfo, nullptr, &tlas_) != VK_SUCCESS) {
        Log(L"[VK AccelStruct] ERROR: Failed to create TLAS\n");
        return false;
    }

    // Scratch
    AccelBuffer scratch = CreateAccelBuffer(sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = tlas_;
    buildInfo.scratchData.deviceAddress = scratch.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    cmd = context_->BeginSingleTimeCommands();
    vkCmdBuildAccelerationStructuresKHR_(cmd, 1, &buildInfo, &pRangeInfo);
    context_->EndSingleTimeCommands(cmd);

    DestroyAccelBuffer(scratch);

    Log(L"[VK AccelStruct] TLAS built with %u instances\n", instanceCount);
    return true;
}

bool AccelStructureBuilder::BuildTLAS(const std::vector<TLASInstance>& instances) {
    if (instances.empty()) {
        Log(L"[VK AccelStruct] No instances for TLAS\n");
        return false;
    }

    VkDevice device = context_->GetDevice();
    uint32_t instanceCount = (uint32_t)instances.size();
    bool canReuse = (tlas_ != VK_NULL_HANDLE && tlasInstanceCount_ == instanceCount);

    // Build VkAccelerationStructureInstanceKHR array
    std::vector<VkAccelerationStructureInstanceKHR> vkInstances(instanceCount);
    for (uint32_t i = 0; i < instanceCount; i++) {
        const auto& src = instances[i];
        auto& dst = vkInstances[i];
        memset(&dst, 0, sizeof(dst));

        // Copy 3x4 row-major transform
        memcpy(&dst.transform, src.transform, sizeof(float) * 12);

        dst.instanceCustomIndex = src.customIndex;
        dst.mask = src.mask;
        dst.instanceShaderBindingTableRecordOffset = 0;
        dst.flags = 0;  // Allow backface culling via ray flags

        if (src.blasIndex >= 0 && src.blasIndex < (int)blasList_.size()) {
            dst.accelerationStructureReference = blasList_[src.blasIndex].deviceAddress;
        }
    }

    VkDeviceSize instanceSize = instanceCount * sizeof(VkAccelerationStructureInstanceKHR);

    // Reuse instance buffer if same count, otherwise recreate
    if (!canReuse || instanceBuffer_.buffer == VK_NULL_HANDLE) {
        DestroyAccelBuffer(instanceBuffer_);
        instanceBuffer_ = CreateAccelBuffer(instanceSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    // Upload instance data via staging
    AccelBuffer instanceStaging = CreateAccelBuffer(instanceSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* mapped;
    vkMapMemory(device, instanceStaging.memory, 0, instanceSize, 0, &mapped);
    memcpy(mapped, vkInstances.data(), instanceSize);
    vkUnmapMemory(device, instanceStaging.memory);

    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    VkBufferCopy copy{};
    copy.size = instanceSize;
    vkCmdCopyBuffer(cmd, instanceStaging.buffer, instanceBuffer_.buffer, 1, &copy);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0,
        1, &barrier, 0, nullptr, 0, nullptr);
    context_->EndSingleTimeCommands(cmd);

    DestroyAccelBuffer(instanceStaging);

    // TLAS geometry info
    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.data.deviceAddress = instanceBuffer_.deviceAddress;

    VkAccelerationStructureGeometryKHR tlasGeometry{};
    tlasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeometry.geometry.instances = instancesData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &tlasGeometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR_(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &instanceCount, &sizeInfo);

    // Create or reuse TLAS
    if (!canReuse) {
        if (tlas_ != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_) {
            vkDestroyAccelerationStructureKHR_(device, tlas_, nullptr);
            tlas_ = VK_NULL_HANDLE;
        }
        DestroyAccelBuffer(tlasBuffer_);
        DestroyAccelBuffer(tlasScratchBuffer_);

        tlasBuffer_ = CreateAccelBuffer(sizeInfo.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkAccelerationStructureCreateInfoKHR tlasCreateInfo{};
        tlasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        tlasCreateInfo.buffer = tlasBuffer_.buffer;
        tlasCreateInfo.size = sizeInfo.accelerationStructureSize;
        tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

        if (vkCreateAccelerationStructureKHR_(device, &tlasCreateInfo, nullptr, &tlas_) != VK_SUCCESS) {
            Log(L"[VK AccelStruct] ERROR: Failed to create TLAS (instanced)\n");
            return false;
        }

        tlasScratchBuffer_ = CreateAccelBuffer(sizeInfo.buildScratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        tlasInstanceCount_ = instanceCount;
    }

    // Build TLAS
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = tlas_;
    buildInfo.scratchData.deviceAddress = tlasScratchBuffer_.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    cmd = context_->BeginSingleTimeCommands();
    vkCmdBuildAccelerationStructuresKHR_(cmd, 1, &buildInfo, &pRangeInfo);
    context_->EndSingleTimeCommands(cmd);

    return true;
}

} // namespace vk
} // namespace acpt
