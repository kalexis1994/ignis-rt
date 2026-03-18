#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include "vk_types.h"

namespace acpt {
namespace vk {

class Context;

// Geometry management - creates and manages vertex/index buffers
class Geometry {
public:
    bool Initialize(Context* context);
    void Shutdown();

    // Geometry generation
    static Mesh CreateSphere(float radius, uint32_t segments, uint32_t rings);
    static Mesh CreatePlane(float size);

    // Buffer management
    bool UploadMesh(Mesh& mesh);
    void DestroyMesh(Mesh& mesh);

private:
    bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                     VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

    Context* context_ = nullptr;
};

} // namespace vk
} // namespace acpt
