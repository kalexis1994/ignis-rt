#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include "../../include/ignis_texture.h"

namespace acpt {

namespace vk {

class Context;

class TextureManager {
public:
    bool Init(Context* context);
    void Shutdown();

    // Load a KN5 texture (DDS BC-compressed) -> VkImage + VkImageView
    // Returns global texture index, or -1 on failure
    int AddTexture(const KN5Texture& kn5Tex);

    // Upload all pending textures to GPU (batch, call after all AddTexture)
    bool UploadAll();

    // Getters for descriptor writes
    uint32_t GetTextureCount() const { return (uint32_t)textures_.size(); }
    VkImageView GetImageView(int index) const;
    VkSampler GetSampler() const { return sampler_; }

private:
    struct GPUTexture {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        int width = 0, height = 0, mipLevels = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkDeviceSize stagingSize = 0;
        bool uploaded = false;
        bool generateMips = false;  // true for uncompressed textures that need GPU mip generation
    };

    static VkFormat DXGIFormatToVulkan(unsigned int dxgiFormat);
    static void CalculateMipOffsetAndSize(int width, int height, int mipLevel, VkFormat format,
                                          size_t& outOffset, size_t& outSize, int& outMipWidth, int& outMipHeight);

    std::vector<GPUTexture> textures_;
    VkSampler sampler_ = VK_NULL_HANDLE;
    Context* context_ = nullptr;
};

} // namespace vk
} // namespace acpt
