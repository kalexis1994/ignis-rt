// vk_renderer_timestamps.cpp — GPU stage profiling via VkQueryPool timestamps.
// Slots: 0=start, 1=hybrid, 2=RT, 3=hair, 4=denoise, 5=composite, 6=tonemap, 7=end.
// Extracted from vk_renderer.cpp.

#include "vk_renderer.h"
#include "vk_context.h"
#include "ignis_log.h"

namespace acpt {
namespace vk {

// ── GPU Timestamp Profiling ──────────────────────────────────────────

void Renderer::InitTimestampQueries() {
    if (timestampReady_ || !context_) return;
    VkDevice device = context_->GetDevice();

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(context_->GetPhysicalDevice(), &props);
    timestampPeriod_ = props.limits.timestampPeriod;
    if (timestampPeriod_ == 0.0f) {
        Log(L"[GPU Prof] Timestamps not supported on this device\n");
        return;
    }

    VkQueryPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = TS_COUNT;
    if (vkCreateQueryPool(device, &ci, nullptr, &timestampQueryPool_) != VK_SUCCESS) {
        Log(L"[GPU Prof] Failed to create timestamp query pool\n");
        return;
    }
    timestampReady_ = true;
    Log(L"[GPU Prof] Timestamp queries ready (period=%.2f ns)\n", timestampPeriod_);
}

void Renderer::ShutdownTimestampQueries() {
    if (timestampQueryPool_ && context_) {
        vkDestroyQueryPool(context_->GetDevice(), timestampQueryPool_, nullptr);
        timestampQueryPool_ = VK_NULL_HANDLE;
    }
    timestampReady_ = false;
}

void Renderer::WriteTimestamp(VkCommandBuffer cmd, uint32_t slot) {
    if (timestampReady_ && slot < TS_COUNT && !(tsWritten_ & (1u << slot))) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueryPool_, slot);
        tsWritten_ |= (1u << slot);
    }
}

void Renderer::FillMissingTimestamps(VkCommandBuffer cmd) {
    if (!timestampReady_) return;
    for (uint32_t i = 0; i < TS_COUNT; i++) {
        if (!(tsWritten_ & (1u << i)))
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueryPool_, i);
    }
}

void Renderer::ReadbackTimestamps() {
    if (!timestampReady_ || !timestampQueryPool_) return;
    uint64_t ts[TS_COUNT] = {};
    // WAIT_BIT is safe here — fence wait already completed, so results are ready.
    VkResult r = vkGetQueryPoolResults(context_->GetDevice(), timestampQueryPool_,
        0, TS_COUNT, sizeof(ts), ts, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r != VK_SUCCESS) return;

    float nsToMs = timestampPeriod_ / 1e6f;
    auto delta = [&](int a, int b) -> float {
        return (ts[b] >= ts[a]) ? (float)(ts[b] - ts[a]) * nsToMs : 0.0f;
    };
    // 0=HybridRaster, 1=RTTrace, 2=HairContour, 3=Denoise, 4=Composite, 5=Tonemap, 6=Total
    gpuStageMs_[0] = delta(TS_START, TS_HYBRID);     // Hybrid Raster
    gpuStageMs_[1] = delta(TS_HYBRID, TS_RT);         // RT Trace
    gpuStageMs_[2] = delta(TS_RT, TS_HAIR);           // Hair Contour
    gpuStageMs_[3] = delta(TS_HAIR, TS_DENOISE);      // DLSS RR
    gpuStageMs_[4] = delta(TS_DENOISE, TS_COMPOSITE); // (unused — composite removed)
    gpuStageMs_[5] = delta(TS_COMPOSITE, TS_TONEMAP);  // Tonemap/DLSS SR
    gpuStageMs_[6] = delta(TS_START, TS_TONEMAP);      // Total
}

} // namespace vk
} // namespace acpt
