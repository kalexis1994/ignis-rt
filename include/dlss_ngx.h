#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

namespace acpt {

/// DLSS Quality Mode presets (matches NGX DLSSMode)
enum class DLSSQualityMode {
    Off = 0,
    MaxPerformance = 1,  ///< Ultra Performance (3.0x upscaling)
    Balanced = 2,        ///< Balanced (1.7x upscaling)
    MaxQuality = 3,      ///< Quality (1.5x upscaling)
    UltraQuality = 4,    ///< Ultra Quality / DLAA (1.0x or native)
    Auto = 5             ///< Let DLSS choose automatically
};

/// DLSS integration using NVIDIA NGX SDK (direct, no Streamline)
/// Compatible with INTEROP mode (no swapchain required)
class DLSS_NGX {
public:
    DLSS_NGX();
    ~DLSS_NGX();

    /// Initialize NGX and DLSS
    /// Must be called AFTER Vulkan device is created
    bool Initialize(
        VkInstance instance,
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VkCommandPool commandPool,
        VkQueue queue,
        uint32_t displayWidth,
        uint32_t displayHeight,
        DLSSQualityMode qualityMode = DLSSQualityMode::Balanced
    );

    /// Shutdown and cleanup
    void Shutdown();

    /// Dispatch DLSS upscaling
    void Evaluate(
        VkCommandBuffer cmdBuf,
        VkImage colorInputImage,         // Render resolution color input (VkImage)
        VkImageView colorInputView,      // Render resolution color input (VkImageView)
        VkImage depthInputImage,         // Depth buffer (VkImage)
        VkImageView depthInputView,      // Depth buffer (VkImageView)
        VkImage motionVectorsImage,      // Motion vectors (VkImage)
        VkImageView motionVectorsView,   // Motion vectors (VkImageView)
        VkImage outputImage,             // Display resolution output (VkImage)
        VkImageView outputView,          // Display resolution output (VkImageView)
        VkFormat colorFormat,            // Color buffer format
        VkFormat depthFormat,            // Depth buffer format
        VkFormat motionFormat,           // Motion vectors format
        float jitterX,                   // Jitter offset X
        float jitterY,                   // Jitter offset Y
        float deltaTime,                 // Frame time in seconds
        float sharpness = 0.0f,          // Sharpening amount (0.0-1.0)
        bool reset = false,              // Reset history
        VkImage reactiveMaskImage = VK_NULL_HANDLE,   // Optional reactive mask
        VkImageView reactiveMaskView = VK_NULL_HANDLE  // Optional reactive mask view
    );

    /// Get the recommended render resolution for a quality mode
    static void GetRenderResolution(
        uint32_t displayWidth,
        uint32_t displayHeight,
        DLSSQualityMode qualityMode,
        uint32_t* outRenderWidth,
        uint32_t* outRenderHeight
    );

    /// Get current render resolution
    void GetCurrentRenderResolution(uint32_t* outWidth, uint32_t* outHeight) const {
        *outWidth = m_renderWidth;
        *outHeight = m_renderHeight;
    }

    /// Check if DLSS is initialized and ready
    bool IsInitialized() const { return m_initialized; }

    /// Check if DLSS is supported on this GPU
    bool IsSupported() const { return m_dlssSupported; }

    /// Set DLSS quality mode (requires re-initialization)
    void SetQualityMode(DLSSQualityMode mode) { m_qualityMode = mode; }

private:
    bool m_initialized;
    bool m_dlssSupported;
    DLSSQualityMode m_qualityMode;
    uint32_t m_displayWidth;
    uint32_t m_displayHeight;
    uint32_t m_renderWidth;
    uint32_t m_renderHeight;
    uint32_t m_frameIndex;

    VkInstance m_instance;
    VkDevice m_device;
    VkPhysicalDevice m_physicalDevice;

    // NGX handles
    void* m_ngxParameters;  // NVSDK_NGX_Parameter*
    void* m_ngxFeature;     // NVSDK_NGX_Handle*
};

} // namespace acpt
