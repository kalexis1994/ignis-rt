// Streamline Frame Generation stub — compiled when IGNIS_USE_STREAMLINE=OFF
// Provides a minimal SLFrameGen class that reports unsupported

#ifndef ACPT_HAVE_STREAMLINE

#include "sl_frame_gen.h"

namespace acpt {

SLFrameGen::SLFrameGen()
    : available_(false), active_(false), reflexReady_(false),
      gpuCap_(FrameGenGPUCap::Unsupported), mode_(FrameGenMode::Off),
      framesToGenerate_(0), maxFramesToGenerate_(0),
      displayWidth_(0), displayHeight_(0), viewportId_(0),
      instance_(VK_NULL_HANDLE), physicalDevice_(VK_NULL_HANDLE),
      device_(VK_NULL_HANDLE), slInterposerLib_(nullptr) {}

SLFrameGen::~SLFrameGen() {}

bool SLFrameGen::Initialize(VkInstance, VkPhysicalDevice, VkDevice,
                             VkQueue, uint32_t, uint32_t, uint32_t) {
    return false;
}

void SLFrameGen::Shutdown() {}

bool SLFrameGen::SetOptions(FrameGenMode, uint32_t) {
    return false;
}

void SLFrameGen::TagResources(VkCommandBuffer, uint32_t,
                               VkImage, VkImageView,
                               VkImage, VkImageView,
                               VkImage, VkImageView,
                               VkImage, VkImageView) {
    // No-op
}

void SLFrameGen::SetReflexMarker(ReflexMarker, uint64_t) {
    // No-op
}

void SLFrameGen::ReflexSleep() {
    // No-op
}

} // namespace acpt

#endif // !ACPT_HAVE_STREAMLINE
