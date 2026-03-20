#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

namespace acpt {
namespace vk {

class Context;

struct AccelBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress deviceAddress = 0;
    VkDeviceSize size = 0;
};

struct BLAS {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    AccelBuffer buffer;
    VkDeviceAddress deviceAddress = 0;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    bool built = false;
    // Keep vertex/index buffers alive for shader attribute access
    AccelBuffer vertexBuf;
    AccelBuffer indexBuf;
    AccelBuffer normalBuf;  // float3 normals (SHADER_DEVICE_ADDRESS)
    AccelBuffer uvBuf;       // float2 UVs (SHADER_DEVICE_ADDRESS)
    AccelBuffer primMaterialBuf; // uint32 per primitive: materialId (SHADER_DEVICE_ADDRESS)
    AccelBuffer primYBoundsBuf; // float2 per primitive: (nodeMinY, nodeMaxY) for canopy AO
    float minY = 0.0f;  // Bounding box min Y (for canopy AO gradient)
    float maxY = 0.0f;  // Bounding box max Y
};

struct GeometryBuffers {
    AccelBuffer vertexBuffer;   // float3 positions
    AccelBuffer indexBuffer;    // uint32 indices
    AccelBuffer normalBuffer;   // float3 normals
    AccelBuffer uvBuffer;       // float2 UVs
    AccelBuffer tangentBuffer;  // float4 tangents
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    uint32_t materialId = 0;
};

// Instance description for custom TLAS builds
struct TLASInstance {
    int blasIndex;
    float transform[12]; // VkTransformMatrixKHR: 3x4 row-major
    uint32_t customIndex;
    uint32_t mask;
};

// Acceleration structure builder for Vulkan RT
class AccelStructureBuilder {
public:
    bool Initialize(Context* context);
    void Shutdown();
    void ClearBLAS();  // Destroy all BLAS for full scene reload

    // Build BLAS from vertex/index data
    // Returns BLAS index (-1 on failure)
    int BuildBLAS(const float* vertices, uint32_t vertexCount,
                  const uint32_t* indices, uint32_t indexCount,
                  bool allowUpdate = false);

    // Refit (update) an existing BLAS in-place (for dynamic geometry)
    bool RefitBLAS(int blasIndex, const float* vertices, uint32_t vertexCount,
                   const uint32_t* indices, uint32_t indexCount);

    // Build TLAS from all BLAS instances (identity transform)
    bool BuildTLAS();

    // Build TLAS with custom per-instance transforms (for static/dynamic split)
    bool BuildTLAS(const std::vector<TLASInstance>& instances);

    // Upload normal/UV attribute buffers directly into a BLAS entry
    bool UploadBLASAttributes(int blasIndex, const float* normals, const float* uvs, uint32_t vertexCount);

    // Upload per-primitive material IDs for a BLAS
    bool UploadBLASPrimitiveMaterials(int blasIndex, const uint32_t* materialIds, uint32_t primitiveCount);

    // Upload per-primitive Y bounds (nodeMinY, nodeMaxY) for a BLAS
    bool UploadBLASPrimitiveYBounds(int blasIndex, const float* yBounds, uint32_t primitiveCount);

    // Build BLAS from buffers already on GPU (no staging/upload needed)
    // If externalCmd != VK_NULL_HANDLE, records into that command buffer (no stall)
    // If externalCmd == VK_NULL_HANDLE, uses BeginSingleTimeCommands (fallback with stall)
    // If updateMode == true, performs in-place BLAS update (much faster, requires same topology)
    // Returns BLAS index on success, -1 on failure
    // The vertex/index buffers are NOT owned by the BLAS — caller manages their lifetime
    int BuildBLASFromGPUBuffers(VkDeviceAddress vertexAddr, uint32_t vertexCount,
                                 VkDeviceAddress indexAddr, uint32_t indexCount,
                                 int reuseBlasIndex = -1,
                                 VkCommandBuffer externalCmd = VK_NULL_HANDLE,
                                 bool opaqueGeometry = false);

    // Upload geometry attribute buffers (normals, UVs, tangents) for shader access
    int UploadGeometryBuffers(const float* vertices, uint32_t vertexCount,
                              const float* normals,
                              const float* uvs,
                              const float* tangents,
                              const uint32_t* indices, uint32_t indexCount,
                              uint32_t materialId);

    // Getters
    VkAccelerationStructureKHR GetTLAS() const { return tlas_; }
    bool HasTLAS() const { return tlas_ != VK_NULL_HANDLE; }
    const std::vector<GeometryBuffers>& GetGeometryBuffers() const { return geometryBuffers_; }
    const std::vector<BLAS>& GetBLASList() const { return blasList_; }
    size_t GetBLASCount() const { return blasList_.size(); }

    // Buffer management (public for renderer's grass compute pipeline)
    AccelBuffer CreateAccelBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                   VkMemoryPropertyFlags memProps);
    void DestroyAccelBuffer(AccelBuffer& buf);

private:
    VkDeviceAddress GetBufferDeviceAddress(VkBuffer buffer);

    Context* context_ = nullptr;

    // Function pointers (loaded dynamically)
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR_ = nullptr;
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR_ = nullptr;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR_ = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR_ = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR_ = nullptr;
    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR_ = nullptr;
    PFN_vkCmdCopyAccelerationStructureKHR vkCmdCopyAccelerationStructureKHR_ = nullptr;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR vkCmdWriteAccelerationStructuresPropertiesKHR_ = nullptr;

    std::vector<BLAS> blasList_;
    std::vector<GeometryBuffers> geometryBuffers_;

    // TLAS
    VkAccelerationStructureKHR tlas_ = VK_NULL_HANDLE;
    AccelBuffer tlasBuffer_;
    AccelBuffer instanceBuffer_;
    AccelBuffer tlasScratchBuffer_;
    uint32_t tlasInstanceCount_ = 0; // for reuse check

    // Persistent scratch buffer for GPU-buffer BLAS builds (avoids reallocation)
    AccelBuffer gpuBlasScratchBuffer_;
    VkDeviceSize gpuBlasScratchSize_ = 0;
};

} // namespace vk
} // namespace acpt
