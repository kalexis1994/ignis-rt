#pragma once

#define NOMINMAX
#include <windows.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <cstdint>

namespace acpt {
namespace vk {

class Context;

// D3D11 ↔ Vulkan interop via NT handle shared memory
class Interop {
public:
    bool Initialize(Context* context, uint32_t width, uint32_t height);
    void Shutdown();

    // Get the Vulkan image that the RT pipeline writes to
    VkImage GetSharedImage() const { return sharedImage_; }
    VkImageView GetSharedImageView() const { return sharedImageView_; }

    // Get NT handle for D3D11 to open via OpenSharedResource1()
    HANDLE GetNTHandle() const { return ntHandle_; }

    // Transition shared image for RT write or D3D11 read
    void TransitionForRTWrite(VkCommandBuffer cmd);
    void TransitionForExternalRead(VkCommandBuffer cmd);

    // GL interop: import Vulkan shared memory as OpenGL texture, draw to current GL context
    bool InitGLInterop();
    void DrawGL(uint32_t viewportW, uint32_t viewportH);
    void ShutdownGL();
    bool IsGLInteropReady() const { return glInteropReady_; }

    // Import a D3D11 shared texture (NT handle) as the RT output image (zero-copy interop)
    // Replaces the internally-created shared image with the D3D11-owned one
    bool ImportD3D11Texture(HANDLE ntHandle, uint32_t width, uint32_t height);

    // CPU readback: copy rendered image to CPU buffer (RGBA8, row-major)
    bool ReadbackPixels(void* outData, uint32_t bufferSize);

    // Record image→buffer copy into an existing command buffer (no separate submit)
    void RecordReadbackCopy(VkCommandBuffer cmd);

    // After queue submit + fence wait, copy from persistent mapped buffer to outData
    bool CopyReadbackResult(void* outData, uint32_t bufferSize);

    uint32_t GetWidth() const { return width_; }
    uint32_t GetHeight() const { return height_; }

private:
    bool EnsureReadbackBuffer();

    Context* context_ = nullptr;
    uint32_t width_ = 0;
    uint32_t height_ = 0;

    VkImage sharedImage_ = VK_NULL_HANDLE;
    VkDeviceMemory sharedMemory_ = VK_NULL_HANDLE;
    VkImageView sharedImageView_ = VK_NULL_HANDLE;
    HANDLE ntHandle_ = nullptr;

    // Double-buffered readback staging (persistent mapped)
    // Write to readbackCurrent_, read from readbackCurrent_^1
    VkBuffer readbackBuffer_[2] = {};
    VkDeviceMemory readbackMemory_[2] = {};
    void* readbackMapped_[2] = {};
    uint32_t readbackCurrent_ = 0;  // index of buffer being written this frame

    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR_ = nullptr;

    // GL interop state (use raw unsigned int to avoid GL header dependency)
    unsigned int glMemoryObject_ = 0;
    unsigned int glTexture_ = 0;
    unsigned int glShaderProgram_ = 0;
    unsigned int glVAO_ = 0;
    bool glInteropReady_ = false;
    uint64_t allocationSize_ = 0;   // Vulkan memory allocation size (for GL import)
};

} // namespace vk
} // namespace acpt
