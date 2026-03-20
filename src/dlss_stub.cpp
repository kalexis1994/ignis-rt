// DLSS stub — compiled when IGNIS_USE_DLSS=OFF
// Provides a minimal DLSS_NGX class that reports unsupported

#ifndef ACPT_HAVE_NGX

#include "dlss_ngx.h"

namespace acpt {

DLSS_NGX::DLSS_NGX()
    : m_initialized(false), m_dlssSupported(false), m_rrSupported(false),
      m_qualityMode(DLSSQualityMode::Off), m_activeMode(DLSSMode::Off),
      m_displayWidth(0), m_displayHeight(0),
      m_renderWidth(0), m_renderHeight(0), m_frameIndex(0),
      m_instance(VK_NULL_HANDLE), m_device(VK_NULL_HANDLE),
      m_physicalDevice(VK_NULL_HANDLE),
      m_commandPool(VK_NULL_HANDLE), m_queue(VK_NULL_HANDLE),
      m_ngxParameters(nullptr), m_ngxFeature(nullptr), m_ngxFeatureRR(nullptr) {}

DLSS_NGX::~DLSS_NGX() {}

bool DLSS_NGX::Initialize(VkInstance instance, VkDevice device,
                           VkPhysicalDevice physDevice,
                           VkCommandPool commandPool, VkQueue queue,
                           uint32_t displayWidth, uint32_t displayHeight,
                           DLSSQualityMode quality) {
    return false;
}

bool DLSS_NGX::InitializeRR() {
    return false;
}

void DLSS_NGX::GetRenderResolution(uint32_t displayWidth, uint32_t displayHeight,
                                     DLSSQualityMode qualityMode,
                                     uint32_t* outRenderWidth, uint32_t* outRenderHeight) {
    if (outRenderWidth) *outRenderWidth = displayWidth;
    if (outRenderHeight) *outRenderHeight = displayHeight;
}

void DLSS_NGX::Evaluate(VkCommandBuffer cmd,
                          VkImage colorInput, VkImageView colorInputView,
                          VkImage depthInput, VkImageView depthInputView,
                          VkImage mvInput, VkImageView mvInputView,
                          VkImage output, VkImageView outputView,
                          VkFormat colorFormat, VkFormat depthFormat, VkFormat mvFormat,
                          float jitterX, float jitterY,
                          float deltaTime, float sharpness, bool reset,
                          VkImage reactiveMaskImage, VkImageView reactiveMaskView) {
    // No-op
}

void DLSS_NGX::EvaluateRR(VkCommandBuffer cmd,
                            VkImage colorInput, VkImageView colorInputView,
                            VkImage output, VkImageView outputView,
                            VkImage depth, VkImageView depthView,
                            VkImage mv, VkImageView mvView,
                            VkImage normals, VkImageView normalsView,
                            VkImage albedo, VkImageView albedoView,
                            float jitterX, float jitterY,
                            float deltaTime,
                            const float* viewMatrix, const float* projMatrix,
                            VkImage specAlb, VkImageView specAlbV,
                            VkImage specMV, VkImageView specMVV,
                            VkImage diffHD, VkImageView diffHDV,
                            VkImage specHD, VkImageView specHDV,
                            bool reset,
                            VkImage reactiveMask, VkImageView reactiveMaskV) {
    // No-op
}

void DLSS_NGX::Shutdown() {}

} // namespace acpt

#endif // !ACPT_HAVE_NGX
