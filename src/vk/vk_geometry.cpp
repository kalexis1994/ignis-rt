#include "vk_geometry.h"
#include "vk_context.h"
#include "../../include/ignis_log.h"
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace acpt {
namespace vk {

bool Geometry::Initialize(Context* context) {
    context_ = context;
    Log(L"[VK Geometry] Initialized\n");
    return true;
}

void Geometry::Shutdown() {
    Log(L"[VK Geometry] Shutdown\n");
}

Mesh Geometry::CreateSphere(float radius, uint32_t segments, uint32_t rings) {
    Mesh mesh;
    
    // Generate vertices
    for (uint32_t ring = 0; ring <= rings; ring++) {
        float phi = M_PI * float(ring) / float(rings);
        float sinPhi = sinf(phi);
        float cosPhi = cosf(phi);
        
        for (uint32_t seg = 0; seg <= segments; seg++) {
            float theta = 2.0f * M_PI * float(seg) / float(segments);
            float sinTheta = sinf(theta);
            float cosTheta = cosf(theta);
            
            Vertex vertex;
            vertex.position[0] = radius * sinPhi * cosTheta;
            vertex.position[1] = radius * cosPhi;
            vertex.position[2] = radius * sinPhi * sinTheta;
            
            // Normal (normalized position for sphere)
            vertex.normal[0] = sinPhi * cosTheta;
            vertex.normal[1] = cosPhi;
            vertex.normal[2] = sinPhi * sinTheta;
            
            // UV coordinates
            vertex.uv[0] = float(seg) / float(segments);
            vertex.uv[1] = float(ring) / float(rings);
            
            mesh.vertices.push_back(vertex);
        }
    }
    
    // Generate indices
    for (uint32_t ring = 0; ring < rings; ring++) {
        for (uint32_t seg = 0; seg < segments; seg++) {
            uint32_t current = ring * (segments + 1) + seg;
            uint32_t next = current + segments + 1;
            
            mesh.indices.push_back(current);
            mesh.indices.push_back(next);
            mesh.indices.push_back(current + 1);
            
            mesh.indices.push_back(current + 1);
            mesh.indices.push_back(next);
            mesh.indices.push_back(next + 1);
        }
    }
    
    return mesh;
}

Mesh Geometry::CreatePlane(float size) {
    Mesh mesh;
    
    // 4 vertices for a quad
    Vertex v0, v1, v2, v3;
    
    // Positions
    v0.position[0] = -size; v0.position[1] = 0.0f; v0.position[2] = -size;
    v1.position[0] =  size; v1.position[1] = 0.0f; v1.position[2] = -size;
    v2.position[0] =  size; v2.position[1] = 0.0f; v2.position[2] =  size;
    v3.position[0] = -size; v3.position[1] = 0.0f; v3.position[2] =  size;
    
    // Normals (pointing up)
    v0.normal[0] = v1.normal[0] = v2.normal[0] = v3.normal[0] = 0.0f;
    v0.normal[1] = v1.normal[1] = v2.normal[1] = v3.normal[1] = 1.0f;
    v0.normal[2] = v1.normal[2] = v2.normal[2] = v3.normal[2] = 0.0f;
    
    // UVs
    v0.uv[0] = 0.0f; v0.uv[1] = 0.0f;
    v1.uv[0] = 1.0f; v1.uv[1] = 0.0f;
    v2.uv[0] = 1.0f; v2.uv[1] = 1.0f;
    v3.uv[0] = 0.0f; v3.uv[1] = 1.0f;
    
    mesh.vertices = {v0, v1, v2, v3};
    mesh.indices = {0, 1, 2, 2, 3, 0};
    
    return mesh;
}

bool Geometry::UploadMesh(Mesh& mesh) {
    VkDeviceSize vertexBufferSize = sizeof(Vertex) * mesh.vertices.size();
    VkDeviceSize indexBufferSize = sizeof(uint32_t) * mesh.indices.size();
    
    // Create staging buffers
    VkBuffer vertexStagingBuffer, indexStagingBuffer;
    VkDeviceMemory vertexStagingMemory, indexStagingMemory;
    
    if (!CreateBuffer(vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexStagingBuffer, vertexStagingMemory)) {
        return false;
    }
    
    if (!CreateBuffer(indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     indexStagingBuffer, indexStagingMemory)) {
        vkDestroyBuffer(context_->GetDevice(), vertexStagingBuffer, nullptr);
        vkFreeMemory(context_->GetDevice(), vertexStagingMemory, nullptr);
        return false;
    }
    
    // Copy vertex data
    void* data;
    vkMapMemory(context_->GetDevice(), vertexStagingMemory, 0, vertexBufferSize, 0, &data);
    memcpy(data, mesh.vertices.data(), vertexBufferSize);
    vkUnmapMemory(context_->GetDevice(), vertexStagingMemory);
    
    // Copy index data
    vkMapMemory(context_->GetDevice(), indexStagingMemory, 0, indexBufferSize, 0, &data);
    memcpy(data, mesh.indices.data(), indexBufferSize);
    vkUnmapMemory(context_->GetDevice(), indexStagingMemory);
    
    // Create device local buffers
    if (!CreateBuffer(vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mesh.vertexBuffer, mesh.vertexBufferMemory)) {
        vkDestroyBuffer(context_->GetDevice(), vertexStagingBuffer, nullptr);
        vkFreeMemory(context_->GetDevice(), vertexStagingMemory, nullptr);
        vkDestroyBuffer(context_->GetDevice(), indexStagingBuffer, nullptr);
        vkFreeMemory(context_->GetDevice(), indexStagingMemory, nullptr);
        return false;
    }
    
    if (!CreateBuffer(indexBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mesh.indexBuffer, mesh.indexBufferMemory)) {
        vkDestroyBuffer(context_->GetDevice(), mesh.vertexBuffer, nullptr);
        vkFreeMemory(context_->GetDevice(), mesh.vertexBufferMemory, nullptr);
        vkDestroyBuffer(context_->GetDevice(), vertexStagingBuffer, nullptr);
        vkFreeMemory(context_->GetDevice(), vertexStagingMemory, nullptr);
        vkDestroyBuffer(context_->GetDevice(), indexStagingBuffer, nullptr);
        vkFreeMemory(context_->GetDevice(), indexStagingMemory, nullptr);
        return false;
    }
    
    // Copy buffers
    CopyBuffer(vertexStagingBuffer, mesh.vertexBuffer, vertexBufferSize);
    CopyBuffer(indexStagingBuffer, mesh.indexBuffer, indexBufferSize);
    
    // Cleanup staging buffers
    vkDestroyBuffer(context_->GetDevice(), vertexStagingBuffer, nullptr);
    vkFreeMemory(context_->GetDevice(), vertexStagingMemory, nullptr);
    vkDestroyBuffer(context_->GetDevice(), indexStagingBuffer, nullptr);
    vkFreeMemory(context_->GetDevice(), indexStagingMemory, nullptr);
    
    return true;
}

void Geometry::DestroyMesh(Mesh& mesh) {
    // Wait for GPU to finish using these buffers before destroying
    if (mesh.vertexBuffer != VK_NULL_HANDLE || mesh.indexBuffer != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(context_->GetDevice());
    }
    if (mesh.vertexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(context_->GetDevice(), mesh.vertexBuffer, nullptr);
        vkFreeMemory(context_->GetDevice(), mesh.vertexBufferMemory, nullptr);
        mesh.vertexBuffer = VK_NULL_HANDLE;
        mesh.vertexBufferMemory = VK_NULL_HANDLE;
    }
    if (mesh.indexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(context_->GetDevice(), mesh.indexBuffer, nullptr);
        vkFreeMemory(context_->GetDevice(), mesh.indexBufferMemory, nullptr);
        mesh.indexBuffer = VK_NULL_HANDLE;
        mesh.indexBufferMemory = VK_NULL_HANDLE;
    }
}

bool Geometry::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                            VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(context_->GetDevice(), &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        Log(L"[VK Geometry] ERROR: Failed to create buffer\n");
        return false;
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(context_->GetDevice(), buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(context_->GetDevice(), &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        Log(L"[VK Geometry] ERROR: Failed to allocate buffer memory\n");
        return false;
    }

    vkBindBufferMemory(context_->GetDevice(), buffer, bufferMemory, 0);
    return true;
}

void Geometry::CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = context_->GetCommandPool();
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(context_->GetDevice(), &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(context_->GetGraphicsQueue());

    vkFreeCommandBuffers(context_->GetDevice(), context_->GetCommandPool(), 1, &commandBuffer);
}

} // namespace vk
} // namespace acpt
