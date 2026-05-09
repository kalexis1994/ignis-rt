// vk_renderer_dlss.cpp — DLSS shutdown + quality-mode query.
// The init/Evaluate paths are still inline in vk_renderer.cpp (Initialize,
// InitRT_Remaining, RenderFrameRT) because they are tightly interleaved
// with non-DLSS code. Extracted from vk_renderer.cpp.

#include "vk_renderer.h"
#include "vk_context.h"
#include "ignis_log.h"

namespace acpt {
namespace vk {

void Renderer::ShutdownDLSS() {
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;

    // Shutdown tonemap pipeline (depends on DLSS HDR output)
    ShutdownTonemap();

    if (dlss_) {
        dlss_->Shutdown();
        delete dlss_;
        dlss_ = nullptr;
    }

    if (device != VK_NULL_HANDLE) {
        if (dlssColorInputView_) { vkDestroyImageView(device, dlssColorInputView_, nullptr); dlssColorInputView_ = VK_NULL_HANDLE; }
        if (dlssColorInput_) { vkDestroyImage(device, dlssColorInput_, nullptr); dlssColorInput_ = VK_NULL_HANDLE; }
        if (dlssColorInputMemory_) { vkFreeMemory(device, dlssColorInputMemory_, nullptr); dlssColorInputMemory_ = VK_NULL_HANDLE; }

        if (dlssHdrOutputView_) { vkDestroyImageView(device, dlssHdrOutputView_, nullptr); dlssHdrOutputView_ = VK_NULL_HANDLE; }
        if (dlssHdrOutput_) { vkDestroyImage(device, dlssHdrOutput_, nullptr); dlssHdrOutput_ = VK_NULL_HANDLE; }
        if (dlssHdrOutputMemory_) { vkFreeMemory(device, dlssHdrOutputMemory_, nullptr); dlssHdrOutputMemory_ = VK_NULL_HANDLE; }
    }

    dlssActive_ = false;
    dlssRRActive_ = false;
    Log(L"[VK Renderer] DLSS shutdown\n");
}

int Renderer::GetActualDLSSQuality() const {
    if (dlss_) return static_cast<int>(dlss_->GetQualityMode());
    return 0;
}

} // namespace vk
} // namespace acpt
