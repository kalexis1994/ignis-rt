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

    // Get the Vulkan image that the RT pipeline writes to (current write buffer)
    VkImage GetSharedImage() const { return sharedImage_[writeIdx_]; }
    VkImageView GetSharedImageView() const { return sharedImageView_[writeIdx_]; }
    VkImageView GetSharedImageView(uint32_t idx) const { return sharedImageView_[idx]; }

    // Get NT handle for D3D11 to open via OpenSharedResource1()
    HANDLE GetNTHandle() const { return ntHandle_[writeIdx_]; }

    // Double-buffer swap: call at end of frame to prevent tearing
    void SwapBuffers();
    uint32_t GetWriteIdx() const { return writeIdx_; }
    uint32_t GetReadIdx() const { return 1 - writeIdx_; }

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
    // submitFence = the fence that will be signaled when this command buffer completes
    void RecordReadbackCopy(VkCommandBuffer cmd, VkFence submitFence = VK_NULL_HANDLE);

    // Copy from completed readback buffer to outData (waits on fence if needed)
    bool CopyReadbackResult(void* outData, uint32_t bufferSize, VkDevice device = VK_NULL_HANDLE);

    uint32_t GetWidth() const { return width_; }
    uint32_t GetHeight() const { return height_; }

private:
    bool EnsureReadbackBuffer();

    Context* context_ = nullptr;
    uint32_t width_ = 0;
    uint32_t height_ = 0;

    bool CreateSharedImageSlot(int idx);  // helper: create one shared image + memory + view + NT handle

    // Double-buffered shared images (ping-pong for tear-free display)
    VkImage sharedImage_[2] = {};
    VkDeviceMemory sharedMemory_[2] = {};
    VkImageView sharedImageView_[2] = {};
    HANDLE ntHandle_[2] = {};
    uint64_t allocationSize_[2] = {};   // Vulkan memory allocation size (for GL import)
    uint32_t writeIdx_ = 0;            // index currently being written by RT

    // Double-buffered readback staging (persistent mapped)
    VkBuffer readbackBuffer_[2] = {};
    VkDeviceMemory readbackMemory_[2] = {};
    void* readbackMapped_[2] = {};
    VkFence readbackFence_[2] = {};   // fence protecting each buffer's GPU copy
    uint32_t readbackWriteIdx_ = 0;   // index being written THIS frame
    bool readbackFirstFrame_ = true;  // skip read on first frame

    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR_ = nullptr;

    // GL interop state (use raw unsigned int to avoid GL header dependency)
    unsigned int glMemoryObject_[2] = {};   // double-buffered GL memory objects
    unsigned int glTexture_[2] = {};        // double-buffered GL textures
    unsigned int glShaderProgram_ = 0;
    unsigned int glVAO_ = 0;
    bool glInteropReady_ = false;
};

} // namespace vk
} // namespace acpt
