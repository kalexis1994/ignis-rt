#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

namespace acpt {

/// GPU generation capability for Frame Generation
enum class FrameGenGPUCap {
    Unsupported = 0,   ///< No FG support (GTX, RTX 20xx, RTX 30xx)
    SingleFrame = 1,   ///< 1 generated frame (RTX 40xx — DLSS 3)
    MultiFrame  = 2    ///< Up to 3 generated frames (RTX 50xx — DLSS 4)
};

/// Frame Generation mode
enum class FrameGenMode {
    Off  = 0,
    On   = 1,   ///< Fixed frame count
    Auto = 2    ///< Dynamic MFG — adapts to display refresh rate (DLSS 4.5)
};

/// Reflex marker types (for frame pacing + latency reduction)
enum class ReflexMarker {
    SimulationStart,
    SimulationEnd,
    RenderStart,
    RenderEnd,
    PresentStart,
    PresentEnd
};

/// Streamline Frame Generation + Reflex integration
/// Requires Streamline SDK (sl.interposer.dll + sl.dlss_g.dll + sl.reflex.dll)
/// Operates at the swapchain/present level — orthogonal to NGX SR/RR
class SLFrameGen {
public:
    SLFrameGen();
    ~SLFrameGen();

    /// Initialize Streamline with manual Vulkan hooking
    /// Call BEFORE vkCreateDevice (Streamline needs to intercept device creation)
    /// Returns false if Streamline SDK not available or GPU unsupported
    bool Initialize(
        VkInstance instance,
        VkPhysicalDevice physicalDevice,
        VkDevice device,
        VkQueue graphicsQueue,
        uint32_t graphicsQueueFamily,
        uint32_t displayWidth,
        uint32_t displayHeight
    );

    /// Shutdown and cleanup all Streamline features
    void Shutdown();

    /// Query GPU capability (can be called after Initialize)
    FrameGenGPUCap GetGPUCapability() const { return gpuCap_; }

    /// Get maximum frames this GPU can generate (0=unsupported, 1=RTX40, 3=RTX50)
    uint32_t GetMaxFramesToGenerate() const { return maxFramesToGenerate_; }

    /// Set Frame Generation mode and frame count
    /// framesToGenerate: 1 (RTX40+), 2-3 (RTX50 only)
    /// Returns false if unsupported or invalid count for this GPU
    bool SetOptions(FrameGenMode mode, uint32_t framesToGenerate = 1);

    /// Tag resources for Frame Generation (call each frame before present)
    /// hudlessColor: final rendered frame WITHOUT UI overlays (backbuffer resolution)
    /// uiColor:      UI-only pixels with premultiplied alpha (backbuffer resolution, can be VK_NULL_HANDLE)
    /// depth:        depth buffer used for motion vectors
    /// motionVectors: full-screen motion vectors (camera + object motion)
    /// All images must be at display resolution
    void TagResources(
        VkCommandBuffer cmd,
        uint32_t frameIndex,
        VkImage hudlessColor,    VkImageView hudlessColorView,
        VkImage depth,           VkImageView depthView,
        VkImage motionVectors,   VkImageView motionVectorsView,
        VkImage uiColor = VK_NULL_HANDLE, VkImageView uiColorView = VK_NULL_HANDLE
    );

    /// Place Reflex marker (must be called every frame for frame pacing)
    void SetReflexMarker(ReflexMarker marker, uint64_t frameId);

    /// Call Reflex sleep (mandatory — call once per frame in simulation loop)
    void ReflexSleep();

    /// Check if Frame Generation is currently active
    bool IsActive() const { return active_; }

    /// Check if Reflex is initialized
    bool IsReflexReady() const { return reflexReady_; }

    /// Check if the Streamline SDK was loaded successfully
    bool IsAvailable() const { return available_; }

    /// Get current mode
    FrameGenMode GetMode() const { return mode_; }

    /// Get current frames-to-generate setting
    uint32_t GetFramesToGenerate() const { return framesToGenerate_; }

private:
    bool available_;         // Streamline SDK loaded
    bool active_;            // Frame Generation currently running
    bool reflexReady_;       // Reflex initialized

    FrameGenGPUCap gpuCap_;
    FrameGenMode mode_;
    uint32_t framesToGenerate_;
    uint32_t maxFramesToGenerate_;

    uint32_t displayWidth_;
    uint32_t displayHeight_;
    uint32_t viewportId_;

    VkInstance instance_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;

    // Streamline internals (opaque — avoid leaking sl.h into public header)
    void* slInterposerLib_;  // HMODULE for sl.interposer.dll
};

} // namespace acpt
