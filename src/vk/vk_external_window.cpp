#include "vk_external_window.h"
#include "vk_context.h"
#include "../../include/ignis_log.h"

#include <algorithm>

namespace acpt {
namespace vk {

static bool s_windowClassRegistered = false;
static const wchar_t* WINDOW_CLASS = L"IgnisRTExternalWindow";

LRESULT CALLBACK ExternalWindow::WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    auto* self = reinterpret_cast<ExternalWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

    switch (msg) {
    case WM_SIZE:
        if (self && wp != SIZE_MINIMIZED) {
            self->needsResize_ = true;
            self->pendingWidth_ = LOWORD(lp);
            self->pendingHeight_ = HIWORD(lp);
        }
        return 0;
    case WM_CLOSE:
        if (self) self->active_ = false;
        ShowWindow(hwnd, SW_HIDE);
        return 0;
    case WM_DESTROY:
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

bool ExternalWindow::Init(Context* ctx, uint32_t width, uint32_t height) {
    ctx_ = ctx;
    HINSTANCE hInst = GetModuleHandle(nullptr);

    if (!s_windowClassRegistered) {
        WNDCLASSW wc{};
        wc.lpfnWndProc = WndProc;
        wc.hInstance = hInst;
        wc.lpszClassName = WINDOW_CLASS;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
        if (!RegisterClassW(&wc)) {
            Log(L"[ExternalWindow] Failed to register window class\n");
            return false;
        }
        s_windowClassRegistered = true;
    }

    // Create visible, resizable window
    DWORD style = WS_OVERLAPPEDWINDOW;
    RECT rect = {0, 0, (LONG)width, (LONG)height};
    AdjustWindowRect(&rect, style, FALSE);

    hwnd_ = CreateWindowExW(
        0, WINDOW_CLASS, L"Ignis RT — External Viewer",
        style, CW_USEDEFAULT, CW_USEDEFAULT,
        rect.right - rect.left, rect.bottom - rect.top,
        nullptr, nullptr, hInst, nullptr
    );
    if (!hwnd_) {
        Log(L"[ExternalWindow] Failed to create window\n");
        return false;
    }
    SetWindowLongPtrW(hwnd_, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

    // Create Vulkan surface
    VkWin32SurfaceCreateInfoKHR surfInfo{};
    surfInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surfInfo.hinstance = hInst;
    surfInfo.hwnd = hwnd_;
    if (vkCreateWin32SurfaceKHR(ctx_->GetInstance(), &surfInfo, nullptr, &surface_) != VK_SUCCESS) {
        Log(L"[ExternalWindow] Failed to create Vulkan surface\n");
        DestroyWindow(hwnd_); hwnd_ = nullptr;
        return false;
    }

    // Verify present support on the graphics queue family
    VkBool32 presentSupport = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(ctx_->GetPhysicalDevice(),
        ctx_->GetGraphicsQueueFamily(), surface_, &presentSupport);
    if (!presentSupport) {
        Log(L"[ExternalWindow] Graphics queue does not support present to this surface\n");
        vkDestroySurfaceKHR(ctx_->GetInstance(), surface_, nullptr); surface_ = VK_NULL_HANDLE;
        DestroyWindow(hwnd_); hwnd_ = nullptr;
        return false;
    }

    // Create swapchain
    if (!CreateSwapchain(width, height)) {
        vkDestroySurfaceKHR(ctx_->GetInstance(), surface_, nullptr); surface_ = VK_NULL_HANDLE;
        DestroyWindow(hwnd_); hwnd_ = nullptr;
        return false;
    }

    // Semaphores
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkDevice dev = ctx_->GetDevice();
    vkCreateSemaphore(dev, &semInfo, nullptr, &imageAvailable_);
    vkCreateSemaphore(dev, &semInfo, nullptr, &blitFinished_);

    ShowWindow(hwnd_, SW_SHOW);
    UpdateWindow(hwnd_);
    active_ = true;

    Log(L"[ExternalWindow] Created %dx%d\n", width, height);
    return true;
}

void ExternalWindow::Shutdown() {
    VkDevice dev = ctx_ ? ctx_->GetDevice() : VK_NULL_HANDLE;
    if (dev) {
        vkDeviceWaitIdle(dev);
        if (imageAvailable_) { vkDestroySemaphore(dev, imageAvailable_, nullptr); imageAvailable_ = VK_NULL_HANDLE; }
        if (blitFinished_)   { vkDestroySemaphore(dev, blitFinished_, nullptr);   blitFinished_ = VK_NULL_HANDLE; }
        DestroySwapchain();
        if (surface_) { vkDestroySurfaceKHR(ctx_->GetInstance(), surface_, nullptr); surface_ = VK_NULL_HANDLE; }
    }
    if (hwnd_) { DestroyWindow(hwnd_); hwnd_ = nullptr; }
    active_ = false;
    Log(L"[ExternalWindow] Shutdown\n");
}

bool ExternalWindow::CreateSwapchain(uint32_t width, uint32_t height) {
    VkDevice dev = ctx_->GetDevice();
    VkPhysicalDevice phys = ctx_->GetPhysicalDevice();

    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface_, &caps);

    // Clamp extent to surface capabilities
    swapExtent_.width  = std::clamp(width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
    swapExtent_.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    if (swapExtent_.width == 0 || swapExtent_.height == 0) return false;

    // Pick format (prefer B8G8R8A8_UNORM or SRGB)
    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface_, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface_, &fmtCount, formats.data());
    swapFormat_ = formats[0].format;
    VkColorSpaceKHR colorSpace = formats[0].colorSpace;
    for (auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_B8G8R8A8_SRGB) {
            swapFormat_ = f.format;
            colorSpace = f.colorSpace;
            break;
        }
    }

    // Pick present mode (prefer mailbox for lowest latency, then immediate, then FIFO)
    uint32_t modeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface_, &modeCount, nullptr);
    std::vector<VkPresentModeKHR> modes(modeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface_, &modeCount, modes.data());
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    for (auto m : modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) { presentMode = m; break; }
        if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) presentMode = m;
    }

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) imageCount = std::min(imageCount, caps.maxImageCount);

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = surface_;
    ci.minImageCount = imageCount;
    ci.imageFormat = swapFormat_;
    ci.imageColorSpace = colorSpace;
    ci.imageExtent = swapExtent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = presentMode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(dev, &ci, nullptr, &swapchain_) != VK_SUCCESS) {
        Log(L"[ExternalWindow] Failed to create swapchain\n");
        return false;
    }

    uint32_t count = 0;
    vkGetSwapchainImagesKHR(dev, swapchain_, &count, nullptr);
    swapImages_.resize(count);
    vkGetSwapchainImagesKHR(dev, swapchain_, &count, swapImages_.data());

    needsResize_ = false;
    Log(L"[ExternalWindow] Swapchain created: %dx%d, %d images, mode=%d\n",
        swapExtent_.width, swapExtent_.height, count, (int)presentMode);
    return true;
}

void ExternalWindow::DestroySwapchain() {
    VkDevice dev = ctx_ ? ctx_->GetDevice() : VK_NULL_HANDLE;
    if (!dev) return;
    if (swapchain_) {
        vkDestroySwapchainKHR(dev, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
    swapImages_.clear();
}

void ExternalWindow::Resize(uint32_t width, uint32_t height) {
    VkDevice dev = ctx_ ? ctx_->GetDevice() : VK_NULL_HANDLE;
    if (!dev || !surface_) return;
    vkDeviceWaitIdle(dev);
    DestroySwapchain();
    CreateSwapchain(width, height);
    needsResize_ = false;
}

bool ExternalWindow::RecordBlit(VkCommandBuffer cmd, VkImage srcImage, uint32_t srcWidth, uint32_t srcHeight) {
    if (!active_ || !swapchain_ || swapExtent_.width == 0) return false;

    // Handle pending resize
    if (needsResize_ && pendingWidth_ > 0 && pendingHeight_ > 0) {
        // Can't resize inside a command buffer — mark for next frame
        return false;
    }

    // Acquire next swapchain image (non-blocking)
    VkResult acqResult = vkAcquireNextImageKHR(
        ctx_->GetDevice(), swapchain_, 0 /*timeout=0*/,
        imageAvailable_, VK_NULL_HANDLE, &imageIndex_);

    if (acqResult == VK_ERROR_OUT_OF_DATE_KHR || acqResult == VK_SUBOPTIMAL_KHR) {
        needsResize_ = true;
        acquired_ = false;
        return false;
    }
    if (acqResult != VK_SUCCESS) {
        acquired_ = false;
        return false;
    }
    acquired_ = true;

    VkImage dstImage = swapImages_[imageIndex_];

    // Transition swapchain image: UNDEFINED → TRANSFER_DST
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.image = dstImage;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Transition source: GENERAL → TRANSFER_SRC
    VkImageMemoryBarrier srcBarrier = barrier;
    srcBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    srcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    srcBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    srcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    srcBarrier.image = srcImage;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &srcBarrier);

    // Blit with scaling (src may be different resolution than window)
    VkImageBlit region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffsets[0] = {0, 0, 0};
    region.srcOffsets[1] = {(int32_t)srcWidth, (int32_t)srcHeight, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstOffsets[0] = {0, 0, 0};
    region.dstOffsets[1] = {(int32_t)swapExtent_.width, (int32_t)swapExtent_.height, 1};
    vkCmdBlitImage(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &region, VK_FILTER_LINEAR);

    // Transition source back: TRANSFER_SRC → GENERAL
    srcBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    srcBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    srcBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    srcBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &srcBarrier);

    // Transition swapchain image: TRANSFER_DST → PRESENT_SRC
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.image = dstImage;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    return true;
}

void ExternalWindow::Present(VkQueue queue) {
    if (!acquired_ || !swapchain_) return;
    acquired_ = false;

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_;
    presentInfo.pImageIndices = &imageIndex_;

    VkResult result = vkQueuePresentKHR(queue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        needsResize_ = true;
    }

    // Update window title with FPS (every 10 frames to avoid overhead)
    titleUpdateCounter_++;
    if (hwnd_ && (titleUpdateCounter_ % 10) == 0) {
        LARGE_INTEGER freq, now;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&now);
        double t = (double)now.QuadPart / (double)freq.QuadPart;
        if (prevTime_ > 0.0) {
            float dt = (float)(t - prevTime_);
            float instantFps = (dt > 0.0f) ? 10.0f / dt : 0.0f;  // 10 frames / elapsed
            fpsEma_ = fpsEma_ > 0.0f ? fpsEma_ * 0.7f + instantFps * 0.3f : instantFps;
            float ms = fpsEma_ > 0.0f ? 1000.0f / fpsEma_ : 0.0f;
            wchar_t title[128];
            swprintf_s(title, L"Ignis RT \u2014 %.1f FPS | %.1fms | %ux%u",
                       fpsEma_, ms, swapExtent_.width, swapExtent_.height);
            SetWindowTextW(hwnd_, title);
        }
        prevTime_ = t;
    }
}

void ExternalWindow::PumpMessages() {
    if (!hwnd_) return;
    MSG msg;
    while (PeekMessageW(&msg, hwnd_, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    // Deferred resize — only safe when called from main thread (not render thread)
    if (needsResize_ && pendingWidth_ > 0 && pendingHeight_ > 0) {
        // Don't resize here if render thread is active — let main thread handle it
        // The needsResize_ flag stays set, and RecordBlit will skip frames
    }
}

void ExternalWindow::HandleDeferredResize() {
    if (needsResize_ && pendingWidth_ > 0 && pendingHeight_ > 0) {
        Resize(pendingWidth_, pendingHeight_);
        pendingWidth_ = 0;
        pendingHeight_ = 0;
    }
}

void ExternalWindow::SetTitle(const wchar_t* title) {
    if (hwnd_) SetWindowTextW(hwnd_, title);
}

}  // namespace vk
}  // namespace acpt
