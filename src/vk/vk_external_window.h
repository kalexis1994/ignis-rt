#pragma once

#define NOMINMAX
#include <windows.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <cstdint>
#include <vector>

namespace acpt {
namespace vk {

class Context;

// Standalone Vulkan-rendered window — presents the tonemapped output
// independently from Blender's viewport, bypassing the GL interop path.
class ExternalWindow {
public:
    bool Init(Context* ctx, uint32_t width, uint32_t height);
    void Shutdown();

    // Record blit from source image into the acquired swapchain image.
    // Call inside an active command buffer, after tonemap writes the LDR output.
    // Returns false if acquire failed (window minimized, etc.) — caller should skip present.
    bool RecordBlit(VkCommandBuffer cmd, VkImage srcImage, uint32_t srcWidth, uint32_t srcHeight);

    // Present the swapchain image to screen. Call after vkQueueSubmit.
    void Present(VkQueue queue);

    // Process Win32 messages (non-blocking). Call once per frame.
    void PumpMessages();

    bool IsActive() const { return active_; }
    bool NeedsResize() const { return needsResize_; }
    void Resize(uint32_t width, uint32_t height);
    void HandleDeferredResize();  // call from main thread only
    void SetTitle(const wchar_t* title);

    uint32_t GetWidth() const { return swapExtent_.width; }
    uint32_t GetHeight() const { return swapExtent_.height; }

private:
    bool CreateSwapchain(uint32_t width, uint32_t height);
    void DestroySwapchain();
    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);

    Context* ctx_ = nullptr;
    HWND hwnd_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapImages_;
    VkFormat swapFormat_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D swapExtent_ = {0, 0};
    uint32_t imageIndex_ = 0;

    VkSemaphore imageAvailable_ = VK_NULL_HANDLE;
    VkSemaphore blitFinished_ = VK_NULL_HANDLE;

    bool active_ = false;
    bool acquired_ = false;  // true after successful AcquireNextImage
    bool needsResize_ = false;
    uint32_t pendingWidth_ = 0, pendingHeight_ = 0;

    // FPS tracking for window title
    double prevTime_ = 0.0;
    float fpsEma_ = 0.0f;
    int titleUpdateCounter_ = 0;
};

}  // namespace vk
}  // namespace acpt
