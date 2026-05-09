// vk_renderer_imgui.cpp — ImGui overlay bring-up + per-frame render pass.
// Extracted from vk_renderer.cpp.

#include "vk_renderer.h"
#include "vk_context.h"
#include "vk_interop.h"
#include "ignis_log.h"
#include <vector>

// ImGui stub declarations (implemented in imgui_stub.cpp)
extern bool ImGui_Init(HWND hwnd, VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                       VkQueue queue, uint32_t queueFamily, VkRenderPass renderPass,
                       VkDescriptorPool descriptorPool);
extern void ImGui_Render(VkCommandBuffer cmd);
extern void ImGui_Shutdown();

namespace acpt {
namespace vk {

bool Renderer::InitImGui(HWND hwnd, bool forceRasterPath) {
    VkDevice device = context_->GetDevice();

    // Create descriptor pool for ImGui (256 sets for texture thumbnails in material inspector)
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256 },
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 256;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &imguiDescriptorPool_) != VK_SUCCESS) {
        Log(L"[VK Renderer] Failed to create ImGui descriptor pool\n");
        return false;
    }

    // Determine format and layouts based on whether we have RT interop or rasterizer
    bool useRTPath = (interop_ != nullptr) && !forceRasterPath;

    VkAttachmentDescription colorAttachment{};
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;      // Preserve previous output
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

    if (useRTPath) {
        colorAttachment.format = VK_FORMAT_R8G8B8A8_UNORM;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        dep.srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    } else {
        colorAttachment.format = context_->GetSwapchainFormat();
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    }

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &colorAttachment;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dep;

    if (vkCreateRenderPass(device, &rpInfo, nullptr, &imguiRenderPass_) != VK_SUCCESS) {
        Log(L"[VK Renderer] Failed to create ImGui render pass\n");
        return false;
    }

    if (useRTPath) {
        // RT path: double-buffered framebuffers (one per interop slot)
        for (int i = 0; i < 2; i++) {
            VkImageView imageView = interop_->GetSharedImageView(i);
            if (!imageView) continue;  // D3D11 import may only have slot 0

            VkFramebufferCreateInfo fbInfo{};
            fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fbInfo.renderPass = imguiRenderPass_;
            fbInfo.attachmentCount = 1;
            fbInfo.pAttachments = &imageView;
            fbInfo.width = width_;
            fbInfo.height = height_;
            fbInfo.layers = 1;

            if (vkCreateFramebuffer(device, &fbInfo, nullptr, &imguiFramebuffer_[i]) != VK_SUCCESS) {
                Log(L"[VK Renderer] Failed to create ImGui framebuffer [%d]\n", i);
                return false;
            }
        }
    } else {
        // Rasterizer path: per-swapchain-image framebuffers
        const auto& swapViews = context_->GetSwapchainImageViews();
        imguiSwapchainFramebuffers_.resize(swapViews.size());
        for (size_t i = 0; i < swapViews.size(); i++) {
            VkImageView iv = swapViews[i];
            VkFramebufferCreateInfo fbInfo{};
            fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fbInfo.renderPass = imguiRenderPass_;
            fbInfo.attachmentCount = 1;
            fbInfo.pAttachments = &iv;
            fbInfo.width = context_->GetSwapchainExtent().width;
            fbInfo.height = context_->GetSwapchainExtent().height;
            fbInfo.layers = 1;

            if (vkCreateFramebuffer(device, &fbInfo, nullptr, &imguiSwapchainFramebuffers_[i]) != VK_SUCCESS) {
                Log(L"[VK Renderer] Failed to create ImGui swapchain framebuffer %zu\n", i);
                return false;
            }
        }
    }

    // Initialize ImGui with Vulkan backend
    if (!ImGui_Init(hwnd,
                    context_->GetInstance(),
                    context_->GetPhysicalDevice(),
                    device,
                    context_->GetGraphicsQueue(),
                    context_->GetGraphicsQueueFamily(),
                    imguiRenderPass_,
                    imguiDescriptorPool_)) {
        Log(L"[VK Renderer] Failed to initialize ImGui\n");
        return false;
    }

    imguiReady_ = true;
    externalCameraControl_ = true; // Tree editor controls the camera
    Log(L"[VK Renderer] ImGui overlay initialized (%ux%u, %s path)\n",
        width_, height_, useRTPath ? L"RT" : L"raster");
    return true;
}

void Renderer::RenderImGuiOverlay(VkCommandBuffer cmd) {
    if (!imguiReady_) return;

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = imguiRenderPass_;

    uint32_t writeIdx = interop_ ? interop_->GetWriteIdx() : 0;
    if (imguiFramebuffer_[writeIdx] != VK_NULL_HANDLE) {
        // RT path — use framebuffer matching current write image
        rpBegin.framebuffer = imguiFramebuffer_[writeIdx];
        rpBegin.renderArea.extent = {width_, height_};
    } else if (!imguiSwapchainFramebuffers_.empty()) {
        // Raster path: use current frame's swapchain framebuffer
        rpBegin.framebuffer = imguiSwapchainFramebuffers_[imguiCurrentImageIndex_];
        rpBegin.renderArea.extent = context_->GetSwapchainExtent();
    } else {
        return;
    }

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_Render(cmd);
    vkCmdEndRenderPass(cmd);
}








void Renderer::ShutdownImGui() {
    if (!imguiReady_) return;
    VkDevice device = context_->GetDevice();

    ImGui_Shutdown();

    for (int i = 0; i < 2; i++) {
        if (imguiFramebuffer_[i]) vkDestroyFramebuffer(device, imguiFramebuffer_[i], nullptr);
        imguiFramebuffer_[i] = VK_NULL_HANDLE;
    }
    for (auto fb : imguiSwapchainFramebuffers_) {
        if (fb) vkDestroyFramebuffer(device, fb, nullptr);
    }
    imguiSwapchainFramebuffers_.clear();
    if (imguiRenderPass_) vkDestroyRenderPass(device, imguiRenderPass_, nullptr);
    if (imguiDescriptorPool_) vkDestroyDescriptorPool(device, imguiDescriptorPool_, nullptr);
    imguiReady_ = false;
    Log(L"[VK Renderer] ImGui overlay shutdown\n");
}

} // namespace vk
} // namespace acpt
