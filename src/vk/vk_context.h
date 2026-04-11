#pragma once

#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#include <windows.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <vector>
#include <set>  // For std::set in CreateLogicalDevice
#include "vk_types.h"

namespace acpt {
namespace vk {

// Vulkan context - manages instance, device, and swapchain
class Context {
public:
    bool Initialize(HWND hwnd, uint32_t width, uint32_t height);
    void Shutdown();

    // Getters
    VkInstance GetInstance() const { return instance_; }
    VkPhysicalDevice GetPhysicalDevice() const { return physicalDevice_; }
    VkDevice GetDevice() const { return device_; }
    VkQueue GetGraphicsQueue() const { return graphicsQueue_; }
    VkQueue GetPresentQueue() const { return presentQueue_; }
    VkSurfaceKHR GetSurface() const { return surface_; }
    VkSwapchainKHR GetSwapchain() const { return swapchain_; }
    VkFormat GetSwapchainFormat() const { return swapchainImageFormat_; }
    VkExtent2D GetSwapchainExtent() const { return swapchainExtent_; }
    const std::vector<VkImageView>& GetSwapchainImageViews() const { return swapchainImageViews_; }
    const std::vector<VkImage>& GetSwapchainImages() const { return swapchainImages_; }
    VkCommandPool GetCommandPool() const { return commandPool_; }
    uint32_t GetGraphicsQueueFamily() const { return graphicsQueueFamily_; }

    // Ray tracing support
    bool IsRayQuerySupported() const { return rayQuerySupported_; }

    // GPU generation detection (matches frame-gen branch pattern)
    // RTX series: 2000=Turing, 3000=Ampere, 4000=Ada, 5000=Blackwell
    uint32_t GetRTXSeries() const { return rtxSeries_; }
    bool IsHardwareSERCapable() const { return rtxSeries_ >= 4000; }

    uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;

    // One-shot command buffer helpers
    VkCommandBuffer BeginSingleTimeCommands();
    void EndSingleTimeCommands(VkCommandBuffer commandBuffer);

private:
    bool CreateInstance();
    bool PickPhysicalDevice();
    bool CreateLogicalDevice();
    bool CreateSurface(HWND hwnd);
    bool CreateSwapchain(uint32_t width, uint32_t height);
    bool CreateImageViews();
    bool CreateCommandPool();
    bool CheckRTExtensionSupport(VkPhysicalDevice device) const;

    SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device) const;
    VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) const;
    VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) const;
    VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height) const;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchainImageFormat_;
    VkExtent2D swapchainExtent_;
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;

    uint32_t graphicsQueueFamily_ = 0;
    uint32_t presentQueueFamily_ = 0;
    bool rayQuerySupported_ = false;
    uint32_t rtxSeries_ = 0;  // 2000=Turing, 3000=Ampere, 4000=Ada, 5000=Blackwell
};

} // namespace vk
} // namespace acpt
