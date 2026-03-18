// ImGui stub implementations — empty functions for standalone DLL build.
// When integrating with a real ImGui backend, replace this file.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <vulkan/vulkan.h>
#include <cstdint>

bool ImGui_Init(HWND hwnd, VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                VkQueue queue, uint32_t queueFamily, VkRenderPass renderPass,
                VkDescriptorPool descriptorPool) {
    // No-op stub
    return true;
}

void ImGui_NewFrame() {
    // No-op stub
}

void ImGui_Render(VkCommandBuffer cmd) {
    // No-op stub
}

void ImGui_Shutdown() {
    // No-op stub
}

bool ImGui_WantCaptureMouse() {
    return false;
}
