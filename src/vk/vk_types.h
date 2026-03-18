#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <array>

namespace acpt {
namespace vk {

// Vertex structure for basic geometry
struct Vertex {
    float position[3];
    float normal[3];
    float uv[2];

    static VkVertexInputBindingDescription GetBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> GetAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        // Position
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, position);

        // Normal
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, normal);

        // UV
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, uv);

        return attributeDescriptions;
    }
};

// Uniform buffer object for view/projection matrices
struct UniformBufferObject {
    float view[16];
    float projection[16];
    float lightDir[4];     // xyz = direction, w = unused
    float cameraPos[4];    // xyz = world-space camera position, w = unused
    float lightViewProj[16]; // light view-projection for shadow mapping
};

// Push constants for per-draw-call data
struct RasterPushConstants {
    float model[16];
    float color[4]; // RGBA
};

// Mesh data
struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;
};

// A single draw call for the rasterizer
struct RasterDrawCall {
    Mesh* mesh;
    float modelMatrix[16];
    float color[4]; // RGBA
};

// Swapchain support details
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

} // namespace vk
} // namespace acpt
