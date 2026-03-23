#include "dlss_ngx.h"
#include "ignis_log.h"
#include <string>

#ifdef ACPT_HAVE_NGX
#include "ngx/nvsdk_ngx.h"
#include "ngx/nvsdk_ngx_vk.h"
#include "ngx/nvsdk_ngx_defs.h"
#include "ngx/nvsdk_ngx_helpers.h"
#include "ngx/nvsdk_ngx_helpers_vk.h"
#include "ngx/nvsdk_ngx_defs_dlssd.h"
#include "ngx/nvsdk_ngx_helpers_dlssd_vk.h"
#endif

using namespace acpt;

DLSS_NGX::DLSS_NGX()
    : m_initialized(false)
    , m_dlssSupported(false)
    , m_rrSupported(false)
    , m_qualityMode(DLSSQualityMode::Balanced)
    , m_activeMode(DLSSMode::Off)
    , m_displayWidth(0)
    , m_displayHeight(0)
    , m_renderWidth(0)
    , m_renderHeight(0)
    , m_frameIndex(0)
    , m_instance(VK_NULL_HANDLE)
    , m_device(VK_NULL_HANDLE)
    , m_physicalDevice(VK_NULL_HANDLE)
    , m_commandPool(VK_NULL_HANDLE)
    , m_queue(VK_NULL_HANDLE)
    , m_ngxParameters(nullptr)
    , m_ngxFeature(nullptr)
    , m_ngxFeatureRR(nullptr)
{
}

DLSS_NGX::~DLSS_NGX() {
    Shutdown();
}

void DLSS_NGX::GetRenderResolution(
    uint32_t displayWidth,
    uint32_t displayHeight,
    DLSSQualityMode qualityMode,
    uint32_t* outRenderWidth,
    uint32_t* outRenderHeight
) {
#ifdef ACPT_HAVE_NGX
    // Map our enum to NGX PerfQuality
    NVSDK_NGX_PerfQuality_Value ngxQuality;
    switch (qualityMode) {
        case DLSSQualityMode::UltraPerformance: ngxQuality = NVSDK_NGX_PerfQuality_Value_UltraPerformance; break;
        case DLSSQualityMode::MaxPerformance:   ngxQuality = NVSDK_NGX_PerfQuality_Value_MaxPerf; break;
        case DLSSQualityMode::Balanced:         ngxQuality = NVSDK_NGX_PerfQuality_Value_Balanced; break;
        case DLSSQualityMode::MaxQuality:       ngxQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality; break;
        case DLSSQualityMode::UltraQuality:     ngxQuality = NVSDK_NGX_PerfQuality_Value_UltraQuality; break;
        case DLSSQualityMode::DLAA:             ngxQuality = NVSDK_NGX_PerfQuality_Value_DLAA; break;
        default:                                ngxQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality; break;
    }

    // Query NGX for the optimal render resolution (GPU-specific)
    NVSDK_NGX_Parameter* params = nullptr;
    if (NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&params)) && params) {
        unsigned int optW = 0, optH = 0, maxW = 0, maxH = 0, minW = 0, minH = 0;
        float sharpness = 0.0f;
        NVSDK_NGX_Result res = NGX_DLSS_GET_OPTIMAL_SETTINGS(
            params, displayWidth, displayHeight, ngxQuality,
            &optW, &optH, &maxW, &maxH, &minW, &minH, &sharpness);
        if (NVSDK_NGX_SUCCEED(res) && optW > 0 && optH > 0) {
            *outRenderWidth = optW;
            *outRenderHeight = optH;
            Log(L"[DLSS-NGX] Optimal settings: %ux%u (min %ux%u, max %ux%u) for quality %d\n",
                optW, optH, minW, minH, maxW, maxH, (int)qualityMode);
            NVSDK_NGX_VULKAN_DestroyParameters(params);
            return;
        }
        // optW==0 means this quality mode is not supported
        if (optW == 0) {
            Log(L"[DLSS-NGX] Quality mode %d not supported by GPU (optW=0), falling back\n", (int)qualityMode);
        }
        NVSDK_NGX_VULKAN_DestroyParameters(params);
    }
#endif

    // Fallback: hardcoded ratios
    float scalingRatio = 1.0f;
    switch (qualityMode) {
        case DLSSQualityMode::UltraPerformance: scalingRatio = 3.0f; break;
        case DLSSQualityMode::MaxPerformance:   scalingRatio = 2.0f; break;
        case DLSSQualityMode::Balanced:         scalingRatio = 1.7f; break;
        case DLSSQualityMode::MaxQuality:       scalingRatio = 1.5f; break;
        case DLSSQualityMode::UltraQuality:     scalingRatio = 1.3f; break;
        case DLSSQualityMode::DLAA:             scalingRatio = 1.0f; break;
        default:                                scalingRatio = 1.5f; break;
    }

    *outRenderWidth = static_cast<uint32_t>(displayWidth / scalingRatio);
    *outRenderHeight = static_cast<uint32_t>(displayHeight / scalingRatio);
    *outRenderWidth = (*outRenderWidth + 1) & ~1;
    *outRenderHeight = (*outRenderHeight + 1) & ~1;
    if (*outRenderWidth > displayWidth) *outRenderWidth = displayWidth;
    if (*outRenderHeight > displayHeight) *outRenderHeight = displayHeight;
}

bool DLSS_NGX::Initialize(
    VkInstance instance,
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkCommandPool commandPool,
    VkQueue queue,
    uint32_t displayWidth,
    uint32_t displayHeight,
    DLSSQualityMode qualityMode
) {
#ifndef ACPT_HAVE_NGX
    Log(L"[DLSS-NGX] ERROR: NGX SDK not available (ACPT_HAVE_NGX not defined)\n");
    return false;
#else
    if (m_initialized) {
        Log(L"[DLSS-NGX] WARNING: Already initialized, shutting down first\n");
        Shutdown();
    }

    m_instance = instance;
    m_device = device;
    m_physicalDevice = physicalDevice;
    m_commandPool = commandPool;
    m_queue = queue;
    m_displayWidth = displayWidth;
    m_displayHeight = displayHeight;
    m_qualityMode = qualityMode;

    // Use proper render resolution for upscaling
    GetRenderResolution(displayWidth, displayHeight, qualityMode, &m_renderWidth, &m_renderHeight);

    Log(L"[DLSS-NGX] Init: %ux%u -> %ux%u, quality=%d\n",
        m_renderWidth, m_renderHeight, m_displayWidth, m_displayHeight, static_cast<int>(m_qualityMode));

    // Validate Vulkan handles
    if (m_device == VK_NULL_HANDLE) {
        Log(L"[DLSS-NGX] ERROR: VkDevice is VK_NULL_HANDLE\n");
        return false;
    }
    if (m_physicalDevice == VK_NULL_HANDLE) {
        Log(L"[DLSS-NGX] ERROR: VkPhysicalDevice is VK_NULL_HANDLE\n");
        return false;
    }

    // Initialize NGX

    wchar_t appDataPath[MAX_PATH];
    GetModuleFileNameW(NULL, appDataPath, MAX_PATH);
    std::wstring appDataPathStr(appDataPath);
    size_t lastSlash = appDataPathStr.find_last_of(L"\\\\/");
    if (lastSlash != std::wstring::npos) {
        appDataPathStr = appDataPathStr.substr(0, lastSlash);  // Get app root directory
    }

    NVSDK_NGX_Result ngxResult = NVSDK_NGX_VULKAN_Init(
        0x1337BEEF,  // Application ID (can be any unique number for your app)
        appDataPathStr.c_str(),  // Application data path
        m_instance,
        m_physicalDevice,
        m_device,
        vkGetInstanceProcAddr,
        vkGetDeviceProcAddr,
        nullptr,  // FeatureCommonInfo (optional)
        NVSDK_NGX_Version_API
    );

    if (ngxResult != NVSDK_NGX_Result_Success) {
        Log(L"[DLSS-NGX] ERROR: NVSDK_NGX_VULKAN_Init failed with error code %d\n", ngxResult);
        return false;
    }

    // Query NGX for optimal render resolution (GPU-specific).
    // If the requested mode is not supported, fall back to the next lower mode.
    {
        NVSDK_NGX_Parameter* capParams = nullptr;
        if (NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&capParams)) && capParams) {
            // Ordered from highest to lowest quality for fallback
            struct { DLSSQualityMode mode; NVSDK_NGX_PerfQuality_Value ngx; const wchar_t* name; } modes[] = {
                { DLSSQualityMode::DLAA,             NVSDK_NGX_PerfQuality_Value_DLAA,              L"DLAA" },
                { DLSSQualityMode::UltraQuality,     NVSDK_NGX_PerfQuality_Value_UltraQuality,      L"Ultra Quality" },
                { DLSSQualityMode::MaxQuality,       NVSDK_NGX_PerfQuality_Value_MaxQuality,        L"Quality" },
                { DLSSQualityMode::Balanced,         NVSDK_NGX_PerfQuality_Value_Balanced,           L"Balanced" },
                { DLSSQualityMode::MaxPerformance,   NVSDK_NGX_PerfQuality_Value_MaxPerf,            L"Performance" },
                { DLSSQualityMode::UltraPerformance, NVSDK_NGX_PerfQuality_Value_UltraPerformance,   L"Ultra Performance" },
            };

            // Find requested mode, then try fallbacks
            bool found = false;
            bool startTrying = false;
            for (auto& m : modes) {
                if (m.mode == qualityMode) startTrying = true;
                if (!startTrying) continue;

                unsigned int optW = 0, optH = 0, maxW = 0, maxH = 0, minW = 0, minH = 0;
                float sharpness = 0.0f;
                NVSDK_NGX_Result optRes = NGX_DLSS_GET_OPTIMAL_SETTINGS(
                    capParams, displayWidth, displayHeight, m.ngx,
                    &optW, &optH, &maxW, &maxH, &minW, &minH, &sharpness);
                if (NVSDK_NGX_SUCCEED(optRes) && optW > 0 && optH > 0) {
                    m_renderWidth = optW;
                    m_renderHeight = optH;
                    if (m.mode != qualityMode) {
                        Log(L"[DLSS-NGX] %ls not supported, falling back to %ls\n",
                            modes[0].name, m.name);  // approximate — just log the fallback
                    }
                    Log(L"[DLSS-NGX] NGX optimal: %ux%u (min %ux%u, max %ux%u) mode=%ls\n",
                        optW, optH, minW, minH, maxW, maxH, m.name);
                    m_qualityMode = m.mode;
                    found = true;
                    break;
                }
            }
            if (!found) {
                Log(L"[DLSS-NGX] No supported quality mode found, using fallback ratio\n");
            }
            NVSDK_NGX_VULKAN_DestroyParameters(capParams);
        }
    }

    Log(L"[DLSS-NGX] Final render resolution: %ux%u -> %ux%u\n",
        m_renderWidth, m_renderHeight, m_displayWidth, m_displayHeight);

    // Allocate NGX parameters
    ngxResult = NVSDK_NGX_VULKAN_AllocateParameters((NVSDK_NGX_Parameter**)&m_ngxParameters);
    if ((ngxResult != NVSDK_NGX_Result_Success)) {
        Log(L"[DLSS-NGX] ERROR: Failed to allocate NGX parameters (error %d)\n", ngxResult);
        NVSDK_NGX_VULKAN_Shutdown1(m_device);
        return false;
    }

    // Check DLSS support
    NVSDK_NGX_Parameter* capabilityParams = nullptr;
    ngxResult = NVSDK_NGX_VULKAN_GetCapabilityParameters(&capabilityParams);
    if ((ngxResult == NVSDK_NGX_Result_Success) && capabilityParams) {
        int dlssAvailable = 0;
        NVSDK_NGX_Result getResult = capabilityParams->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvailable);
        if ((getResult == NVSDK_NGX_Result_Success) && dlssAvailable) {
            m_dlssSupported = true;
        } else {
            Log(L"[DLSS-NGX] DLSS not supported on this GPU\n");
            NVSDK_NGX_VULKAN_DestroyParameters((NVSDK_NGX_Parameter*)m_ngxParameters);
            NVSDK_NGX_VULKAN_Shutdown1(m_device);
            m_initialized = true;
            m_dlssSupported = false;
            return true;  // Not a fatal error
        }
    } else {
        Log(L"[DLSS-NGX] ERROR: GetCapabilityParameters failed (result=%d)\n", ngxResult);
    }

    // Create DLSS feature (requires command buffer)

    // Create temporary command buffer for DLSS feature creation
    VkCommandBufferAllocateInfo cmdBufAllocInfo = {};
    cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocInfo.commandPool = commandPool;
    cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocInfo.commandBufferCount = 1;

    VkCommandBuffer tempCmdBuf = VK_NULL_HANDLE;
    VkResult vkResult = vkAllocateCommandBuffers(m_device, &cmdBufAllocInfo, &tempCmdBuf);
    if (vkResult != VK_SUCCESS) {
        Log(L"[DLSS-NGX] ERROR: Failed to allocate temporary command buffer (result=%d)\n", vkResult);
        NVSDK_NGX_VULKAN_DestroyParameters((NVSDK_NGX_Parameter*)m_ngxParameters);
        NVSDK_NGX_VULKAN_Shutdown1(m_device);
        return false;
    }

    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResult = vkBeginCommandBuffer(tempCmdBuf, &beginInfo);
    if (vkResult != VK_SUCCESS) {
        Log(L"[DLSS-NGX] ERROR: Failed to begin command buffer (result=%d)\n", vkResult);
        vkFreeCommandBuffers(m_device, commandPool, 1, &tempCmdBuf);
        NVSDK_NGX_VULKAN_DestroyParameters((NVSDK_NGX_Parameter*)m_ngxParameters);
        NVSDK_NGX_VULKAN_Shutdown1(m_device);
        return false;
    }

    // Map our DLSSQualityMode enum to NGX PerfQuality enum
    // Ours: Off=0, MaxPerf=1, Balanced=2, MaxQuality=3, UltraQuality=4
    // NGX:  MaxPerf=0, Balanced=1, MaxQuality=2, UltraPerf=3, UltraQuality=4, DLAA=5
    NVSDK_NGX_PerfQuality_Value ngxQuality;
    switch (m_qualityMode) {
        case DLSSQualityMode::UltraPerformance: ngxQuality = NVSDK_NGX_PerfQuality_Value_UltraPerformance; break;
        case DLSSQualityMode::MaxPerformance:   ngxQuality = NVSDK_NGX_PerfQuality_Value_MaxPerf; break;
        case DLSSQualityMode::Balanced:         ngxQuality = NVSDK_NGX_PerfQuality_Value_Balanced; break;
        case DLSSQualityMode::MaxQuality:       ngxQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality; break;
        case DLSSQualityMode::UltraQuality:     ngxQuality = NVSDK_NGX_PerfQuality_Value_UltraQuality; break;
        case DLSSQualityMode::DLAA:             ngxQuality = NVSDK_NGX_PerfQuality_Value_DLAA; break;
        default:                                ngxQuality = NVSDK_NGX_PerfQuality_Value_Balanced; break;
    }

    NVSDK_NGX_DLSS_Create_Params dlssCreateParams = {};
    dlssCreateParams.Feature.InWidth = m_renderWidth;
    dlssCreateParams.Feature.InHeight = m_renderHeight;
    dlssCreateParams.Feature.InTargetWidth = m_displayWidth;
    dlssCreateParams.Feature.InTargetHeight = m_displayHeight;
    dlssCreateParams.Feature.InPerfQualityValue = ngxQuality;
    // MVLowRes: motion vectors are at render resolution (not display resolution)
    // AutoExposure: let DLSS handle exposure internally (no exposure texture provided)
    // No IsHDR: input is LDR tonemapped RGBA8
    dlssCreateParams.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes
                                          | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure
                                          | NVSDK_NGX_DLSS_Feature_Flags_IsHDR;

    ngxResult = NGX_VULKAN_CREATE_DLSS_EXT1(
        m_device,    // VkDevice for proper device association
        tempCmdBuf,  // Temporary command buffer
        1,  // CreationNodeMask
        1,  // VisibilityNodeMask
        (NVSDK_NGX_Handle**)&m_ngxFeature,
        (NVSDK_NGX_Parameter*)m_ngxParameters,
        &dlssCreateParams
    );

    if ((ngxResult != NVSDK_NGX_Result_Success)) {
        Log(L"[DLSS-NGX] ERROR: Failed to create DLSS feature (error %d)\n", ngxResult);
        vkEndCommandBuffer(tempCmdBuf);
        vkFreeCommandBuffers(m_device, commandPool, 1, &tempCmdBuf);
        NVSDK_NGX_VULKAN_DestroyParameters((NVSDK_NGX_Parameter*)m_ngxParameters);
        NVSDK_NGX_VULKAN_Shutdown1(m_device);
        return false;
    }

    // End and submit command buffer
    vkResult = vkEndCommandBuffer(tempCmdBuf);
    if (vkResult != VK_SUCCESS) {
        Log(L"[DLSS-NGX] ERROR: Failed to end command buffer (result=%d)\n", vkResult);
        vkFreeCommandBuffers(m_device, commandPool, 1, &tempCmdBuf);
        NVSDK_NGX_VULKAN_DestroyParameters((NVSDK_NGX_Parameter*)m_ngxParameters);
        NVSDK_NGX_VULKAN_Shutdown1(m_device);
        return false;
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &tempCmdBuf;

    vkResult = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (vkResult != VK_SUCCESS) {
        Log(L"[DLSS-NGX] ERROR: Failed to submit command buffer (result=%d)\n", vkResult);
        vkFreeCommandBuffers(m_device, commandPool, 1, &tempCmdBuf);
        NVSDK_NGX_VULKAN_DestroyParameters((NVSDK_NGX_Parameter*)m_ngxParameters);
        NVSDK_NGX_VULKAN_Shutdown1(m_device);
        return false;
    }

    // Wait for completion
    vkQueueWaitIdle(queue);

    // Free temporary command buffer
    vkFreeCommandBuffers(m_device, commandPool, 1, &tempCmdBuf);
    Log(L"[DLSS-NGX] DLSS feature created successfully\n");

    Log(L"[DLSS-NGX] DLSS SR initialized OK\n");

    m_initialized = true;
    m_dlssSupported = true;
    m_activeMode = DLSSMode::SR;

    return true;
#endif
}

void DLSS_NGX::Shutdown() {
    if (!m_initialized) {
        return;
    }

    Log(L"[DLSS-NGX] Shutting down...\n");

#ifdef ACPT_HAVE_NGX
    if (m_ngxFeatureRR) {
        NVSDK_NGX_VULKAN_ReleaseFeature((NVSDK_NGX_Handle*)m_ngxFeatureRR);
        m_ngxFeatureRR = nullptr;
    }

    if (m_ngxFeature) {
        NVSDK_NGX_VULKAN_ReleaseFeature((NVSDK_NGX_Handle*)m_ngxFeature);
        m_ngxFeature = nullptr;
    }

    if (m_ngxParameters) {
        NVSDK_NGX_VULKAN_DestroyParameters((NVSDK_NGX_Parameter*)m_ngxParameters);
        m_ngxParameters = nullptr;
    }

    if (m_device != VK_NULL_HANDLE) {
        NVSDK_NGX_VULKAN_Shutdown1(m_device);
    }

    Log(L"[DLSS-NGX] Shutdown complete\n");
#endif

    m_initialized = false;
    m_dlssSupported = false;
    m_rrSupported = false;
    m_activeMode = DLSSMode::Off;
}

void DLSS_NGX::Evaluate(
    VkCommandBuffer cmdBuf,
    VkImage colorInputImage,
    VkImageView colorInputView,
    VkImage depthInputImage,
    VkImageView depthInputView,
    VkImage motionVectorsImage,
    VkImageView motionVectorsView,
    VkImage outputImage,
    VkImageView outputView,
    VkFormat colorFormat,
    VkFormat depthFormat,
    VkFormat motionFormat,
    float jitterX,
    float jitterY,
    float deltaTime,
    float sharpness,
    bool reset,
    VkImage reactiveMaskImage,
    VkImageView reactiveMaskView
) {
#ifdef ACPT_HAVE_NGX
    if (!m_initialized || !m_dlssSupported) {
        return;
    }

    m_frameIndex++;

    // Create NVSDK_NGX_Resource_VK for each input/output using helpers
    VkImageSubresourceRange fullRange = {};
    fullRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    fullRange.baseMipLevel = 0;
    fullRange.levelCount = 1;
    fullRange.baseArrayLayer = 0;
    fullRange.layerCount = 1;

    NVSDK_NGX_Resource_VK colorResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        colorInputView, colorInputImage, fullRange, colorFormat,
        m_renderWidth, m_renderHeight, false  // read-only
    );

    // Depth resource (optional - can be null)
    NVSDK_NGX_Resource_VK depthResource = {};
    if (depthInputView != VK_NULL_HANDLE && depthInputImage != VK_NULL_HANDLE) {
        VkImageSubresourceRange depthRange = fullRange;
        depthRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // AC-PT uses R32_SFLOAT (Color) for depth
        depthResource = NVSDK_NGX_Create_ImageView_Resource_VK(
            depthInputView, depthInputImage, depthRange, depthFormat,
            m_renderWidth, m_renderHeight, false  // read-only
        );
    }

    // Motion vectors resource (optional - can be null)
    NVSDK_NGX_Resource_VK motionResource = {};
    if (motionVectorsView != VK_NULL_HANDLE && motionVectorsImage != VK_NULL_HANDLE) {
        motionResource = NVSDK_NGX_Create_ImageView_Resource_VK(
            motionVectorsView, motionVectorsImage, fullRange, motionFormat,
            m_renderWidth, m_renderHeight, false  // read-only
        );
    }

    NVSDK_NGX_Resource_VK outputResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        outputView, outputImage, fullRange, colorFormat,
        m_displayWidth, m_displayHeight, true  // read-write
    );

    // Prepare DLSS evaluation parameters
    NVSDK_NGX_VK_DLSS_Eval_Params evalParams = {};
    evalParams.Feature.pInColor = &colorResource;
    evalParams.Feature.pInOutput = &outputResource;
    evalParams.Feature.InSharpness = sharpness;
    evalParams.pInDepth = (depthInputView != VK_NULL_HANDLE) ? &depthResource : nullptr;
    evalParams.pInMotionVectors = (motionVectorsView != VK_NULL_HANDLE) ? &motionResource : nullptr;
    // DLSS jitter: pixel-space offset [-0.5, 0.5].
    evalParams.InJitterOffsetX = jitterX;
    evalParams.InJitterOffsetY = jitterY;
    evalParams.InRenderSubrectDimensions.Width = m_renderWidth;
    evalParams.InRenderSubrectDimensions.Height = m_renderHeight;
    evalParams.InReset = (reset || m_frameIndex <= 1) ? 1 : 0;  // Reset on first frame for clean history

    // Motion vectors are in UV space [0,1] (backward: prevUV - currUV), Y-down.
    // Our shader already Y-flips when computing prevUV (0.5 - prevNDC.y * 0.5),
    // so MVs are in screen-space UV matching pixel-space Y direction. Positive scale.
    evalParams.InMVScaleX = (float)m_renderWidth;
    evalParams.InMVScaleY = (float)m_renderHeight;
    // Frame time delta helps DLSS with temporal accumulation quality
    evalParams.InFrameTimeDeltaInMsec = deltaTime * 1000.0f;

    // Reactive mask disabled — pInBiasCurrentColorMask causes artifacts with current format
    // TODO: investigate correct usage for DLSS SR reactive mask


    // Execute DLSS
    NVSDK_NGX_Result ngxResult = NGX_VULKAN_EVALUATE_DLSS_EXT(
        cmdBuf,
        (NVSDK_NGX_Handle*)m_ngxFeature,
        (NVSDK_NGX_Parameter*)m_ngxParameters,
        &evalParams
    );

    if (ngxResult != NVSDK_NGX_Result_Success) {
        Log(L"[DLSS-NGX] ERROR: DLSS evaluation failed (error %d)\n", ngxResult);
    }
#endif
}

bool DLSS_NGX::InitializeRR() {
#ifndef ACPT_HAVE_NGX
    return false;
#else
    if (!m_initialized || !m_dlssSupported) {
        Log(L"[DLSS-RR] Cannot init RR: DLSS not initialized\n");
        return false;
    }

    Log(L"[DLSS-RR] Initializing Ray Reconstruction (%ux%u -> %ux%u)...\n",
        m_renderWidth, m_renderHeight, m_displayWidth, m_displayHeight);

    // Allocate fresh parameters for RR (don't reuse SR params — causes UnsupportedParameter)
    NVSDK_NGX_Parameter* rrParams = nullptr;
    NVSDK_NGX_Result allocResult = NVSDK_NGX_VULKAN_AllocateParameters(&rrParams);
    if (allocResult != NVSDK_NGX_Result_Success || !rrParams) {
        Log(L"[DLSS-RR] ERROR: Failed to allocate RR parameters (error %d)\n", allocResult);
        m_rrSupported = false;
        return false;
    }

    // Map quality mode
    NVSDK_NGX_PerfQuality_Value ngxQuality;
    switch (m_qualityMode) {
        case DLSSQualityMode::UltraPerformance: ngxQuality = NVSDK_NGX_PerfQuality_Value_UltraPerformance; break;
        case DLSSQualityMode::MaxPerformance:   ngxQuality = NVSDK_NGX_PerfQuality_Value_MaxPerf; break;
        case DLSSQualityMode::Balanced:         ngxQuality = NVSDK_NGX_PerfQuality_Value_Balanced; break;
        case DLSSQualityMode::MaxQuality:       ngxQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality; break;
        case DLSSQualityMode::UltraQuality:     ngxQuality = NVSDK_NGX_PerfQuality_Value_UltraQuality; break;
        case DLSSQualityMode::DLAA:             ngxQuality = NVSDK_NGX_PerfQuality_Value_DLAA; break;
        default:                                ngxQuality = NVSDK_NGX_PerfQuality_Value_Balanced; break;
    }

    // Set RR creation parameters on fresh parameter block
    // RR uses the same parameter names as DLSS SR for creation
    unsigned int flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes
                       | NVSDK_NGX_DLSS_Feature_Flags_IsHDR
                       | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;

    rrParams->Set(NVSDK_NGX_Parameter_Width, (unsigned int)m_renderWidth);
    rrParams->Set(NVSDK_NGX_Parameter_Height, (unsigned int)m_renderHeight);
    rrParams->Set(NVSDK_NGX_Parameter_OutWidth, (unsigned int)m_displayWidth);
    rrParams->Set(NVSDK_NGX_Parameter_OutHeight, (unsigned int)m_displayHeight);
    rrParams->Set(NVSDK_NGX_Parameter_PerfQualityValue, (unsigned int)ngxQuality);
    rrParams->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);
    // Required for RR: tell NGX to use the DL unified denoiser
    rrParams->Set(NVSDK_NGX_Parameter_DLSS_Denoise_Mode, (int)NVSDK_NGX_DLSS_Denoise_Mode_DLUnified);
    // Roughness is packed in normalRoughness.a
    rrParams->Set(NVSDK_NGX_Parameter_DLSS_Roughness_Mode, (unsigned int)NVSDK_NGX_DLSS_Roughness_Mode_Packed);
    rrParams->Set(NVSDK_NGX_Parameter_Use_HW_Depth, (unsigned int)1);  // NDC depth [0,1]
    // RR model preset: E = latest transformer model (D and E are the only valid RR presets)
    rrParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA, (unsigned int)NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E);
    rrParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality, (unsigned int)NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E);
    rrParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced, (unsigned int)NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E);
    rrParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance, (unsigned int)NVSDK_NGX_RayReconstruction_Hint_Render_Preset_D);
    rrParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance, (unsigned int)NVSDK_NGX_RayReconstruction_Hint_Render_Preset_D);
    rrParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality, (unsigned int)NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E);
    // Output subrects (required by DLSSD)
    rrParams->Set(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
    // CreationNodeMask / VisibilityNodeMask
    rrParams->Set(NVSDK_NGX_Parameter_CreationNodeMask, (unsigned int)1);
    rrParams->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, (unsigned int)1);

    // Create temporary command buffer for feature creation
    VkCommandBufferAllocateInfo cmdBufAllocInfo{};
    cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocInfo.commandPool = m_commandPool;
    cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocInfo.commandBufferCount = 1;

    VkCommandBuffer tempCmdBuf = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &cmdBufAllocInfo, &tempCmdBuf) != VK_SUCCESS) {
        Log(L"[DLSS-RR] ERROR: Failed to allocate command buffer\n");
        NVSDK_NGX_VULKAN_DestroyParameters(rrParams);
        m_rrSupported = false;
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(tempCmdBuf, &beginInfo);

    // Create Ray Reconstruction feature using generic API
    NVSDK_NGX_Result ngxResult = NVSDK_NGX_VULKAN_CreateFeature1(
        m_device,
        tempCmdBuf,
        NVSDK_NGX_Feature_RayReconstruction,
        rrParams,
        (NVSDK_NGX_Handle**)&m_ngxFeatureRR
    );

    vkEndCommandBuffer(tempCmdBuf);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &tempCmdBuf;
    vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_queue);
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &tempCmdBuf);
    NVSDK_NGX_VULKAN_DestroyParameters(rrParams);

    if (ngxResult != NVSDK_NGX_Result_Success) {
        const wchar_t* errName = L"Unknown";
        uint32_t code = (uint32_t)ngxResult & 0xFF;
        switch (code) {
            case 1:  errName = L"FeatureNotSupported"; break;
            case 2:  errName = L"PlatformError"; break;
            case 3:  errName = L"FeatureAlreadyExists"; break;
            case 4:  errName = L"FeatureNotFound"; break;
            case 5:  errName = L"InvalidParameter"; break;
            case 6:  errName = L"ScratchBufferTooSmall"; break;
            case 7:  errName = L"NotInitialized"; break;
            case 8:  errName = L"UnsupportedInputFormat"; break;
            case 9:  errName = L"RWFlagMissing"; break;
            case 10: errName = L"MissingInput"; break;
            case 11: errName = L"UnableToInitializeFeature"; break;
            case 12: errName = L"OutOfDate"; break;
            case 13: errName = L"OutOfGPUMemory"; break;
            case 14: errName = L"UnsupportedFormat"; break;
            case 16: errName = L"UnsupportedParameter"; break;
        }
        Log(L"[DLSS-RR] Ray Reconstruction creation FAILED: %ls (0x%08X) — falling back to NRD + SR\n",
            errName, (uint32_t)ngxResult);
        m_ngxFeatureRR = nullptr;
        m_rrSupported = false;
        return false;
    }

    m_rrSupported = true;
    m_activeMode = DLSSMode::RayReconstruction;
    Log(L"[DLSS-RR] Ray Reconstruction initialized successfully\n");
    return true;
#endif
}

void DLSS_NGX::EvaluateRR(
    VkCommandBuffer cmdBuf,
    VkImage colorInputImage,
    VkImageView colorInputView,
    VkImage outputImage,
    VkImageView outputView,
    VkImage depthImage,
    VkImageView depthView,
    VkImage motionVectorsImage,
    VkImageView motionVectorsView,
    VkImage normalsImage,
    VkImageView normalsView,
    VkImage albedoImage,
    VkImageView albedoView,
    float jitterX,
    float jitterY,
    float deltaTime,
    const float* viewMatrix,
    const float* projMatrix,
    VkImage specularAlbedoImage,
    VkImageView specularAlbedoView,
    VkImage specularMVImage,
    VkImageView specularMVView,
    VkImage diffuseHitDistImage,
    VkImageView diffuseHitDistView,
    VkImage specularHitDistImage,
    VkImageView specularHitDistView,
    bool reset,
    VkImage reactiveMaskImage,
    VkImageView reactiveMaskView
) {
#ifdef ACPT_HAVE_NGX
    if (!m_initialized || !m_rrSupported || !m_ngxFeatureRR) {
        return;
    }

    m_frameIndex++;

    VkImageSubresourceRange fullRange{};
    fullRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    fullRange.baseMipLevel = 0;
    fullRange.levelCount = 1;
    fullRange.baseArrayLayer = 0;
    fullRange.layerCount = 1;

    // Create NGX resources for each input/output
    NVSDK_NGX_Resource_VK colorResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        colorInputView, colorInputImage, fullRange, VK_FORMAT_R16G16B16A16_SFLOAT,
        m_renderWidth, m_renderHeight, false);

    NVSDK_NGX_Resource_VK outputResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        outputView, outputImage, fullRange, VK_FORMAT_R16G16B16A16_SFLOAT,
        m_displayWidth, m_displayHeight, true);

    NVSDK_NGX_Resource_VK depthResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        depthView, depthImage, fullRange, VK_FORMAT_R32_SFLOAT,
        m_renderWidth, m_renderHeight, false);

    NVSDK_NGX_Resource_VK mvResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        motionVectorsView, motionVectorsImage, fullRange, VK_FORMAT_R16G16B16A16_SFLOAT,
        m_renderWidth, m_renderHeight, false);

    NVSDK_NGX_Resource_VK normalsResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        normalsView, normalsImage, fullRange, VK_FORMAT_R16G16B16A16_SFLOAT,
        m_renderWidth, m_renderHeight, false);

    NVSDK_NGX_Resource_VK albedoResource = NVSDK_NGX_Create_ImageView_Resource_VK(
        albedoView, albedoImage, fullRange, VK_FORMAT_R16G16B16A16_SFLOAT,
        m_renderWidth, m_renderHeight, false);

    // Use the official DLSSD eval params struct
    NVSDK_NGX_VK_DLSSD_Eval_Params evalParams = {};
    evalParams.pInColor = &colorResource;
    evalParams.pInOutput = &outputResource;
    evalParams.pInDepth = &depthResource;
    evalParams.pInMotionVectors = &mvResource;
    evalParams.pInNormals = &normalsResource;
    evalParams.pInDiffuseAlbedo = &albedoResource;
    // Specular albedo: use EnvBRDFApprox if available, fallback to diffuse albedo
    NVSDK_NGX_Resource_VK specAlbedoResource = {};
    if (specularAlbedoView != VK_NULL_HANDLE) {
        specAlbedoResource = NVSDK_NGX_Create_ImageView_Resource_VK(
            specularAlbedoView, specularAlbedoImage, fullRange, VK_FORMAT_R16G16B16A16_SFLOAT,
            m_renderWidth, m_renderHeight, false);
        evalParams.pInSpecularAlbedo = &specAlbedoResource;
    } else {
        evalParams.pInSpecularAlbedo = &albedoResource;
    }
    evalParams.pInRoughness = &normalsResource;  // roughness packed in normalRoughness.a
    // GBufferSurface attribs (also required by the helper — sets GBuffer_Normals, _Roughness, _Albedo params)
    evalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_ALBEDO] = &albedoResource;
    evalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_NORMALS] = &normalsResource;
    evalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_ROUGHNESS] = &normalsResource;
    evalParams.InJitterOffsetX = jitterX;
    evalParams.InJitterOffsetY = jitterY;
    // MVs are in UV space (prevUV - currUV), scaled to pixel space
    evalParams.InMVScaleX = (float)m_renderWidth;
    evalParams.InMVScaleY = (float)m_renderHeight;
    evalParams.InReset = (reset || m_frameIndex <= 1) ? 1 : 0;
    evalParams.InFrameTimeDeltaInMsec = deltaTime * 1000.0f;
    evalParams.InRenderSubrectDimensions.Width = m_renderWidth;
    evalParams.InRenderSubrectDimensions.Height = m_renderHeight;
    evalParams.InPreExposure = 1.0f;
    evalParams.InExposureScale = 1.0f;
    evalParams.InToneMapperType = NVSDK_NGX_TONEMAPPER_ACES;

    // Camera matrices for temporal reprojection
    if (viewMatrix) evalParams.pInWorldToViewMatrix = const_cast<float*>(viewMatrix);
    if (projMatrix) evalParams.pInViewToClipMatrix = const_cast<float*>(projMatrix);

    // Hit distance for edge-stopping
    NVSDK_NGX_Resource_VK diffHitDistResource = {};
    NVSDK_NGX_Resource_VK specHitDistResource = {};
    if (diffuseHitDistView != VK_NULL_HANDLE) {
        diffHitDistResource = NVSDK_NGX_Create_ImageView_Resource_VK(
            diffuseHitDistView, diffuseHitDistImage, fullRange, VK_FORMAT_R16G16B16A16_SFLOAT,
            m_renderWidth, m_renderHeight, false);
        evalParams.pInDiffuseHitDistance = &diffHitDistResource;
    }
    if (specularHitDistView != VK_NULL_HANDLE) {
        specHitDistResource = NVSDK_NGX_Create_ImageView_Resource_VK(
            specularHitDistView, specularHitDistImage, fullRange, VK_FORMAT_R16G16B16A16_SFLOAT,
            m_renderWidth, m_renderHeight, false);
        evalParams.pInSpecularHitDistance = &specHitDistResource;
    }
    // Specular MVs: using Path B (hit distance + matrices) instead of direct MVs

    // Reactive mask: disabled for now — pInBiasCurrentColorMask causes artifacts.
    // TODO: investigate correct field and format for RR reactive mask.
    // (void)reactiveMaskImage;
    // (void)reactiveMaskView;

    NVSDK_NGX_Result ngxResult = NGX_VULKAN_EVALUATE_DLSSD_EXT(
        cmdBuf,
        (NVSDK_NGX_Handle*)m_ngxFeatureRR,
        (NVSDK_NGX_Parameter*)m_ngxParameters,
        &evalParams
    );

    if (ngxResult != NVSDK_NGX_Result_Success && m_frameIndex <= 3) {
        Log(L"[DLSS-RR] ERROR: Evaluation failed (error %d)\n", ngxResult);
    }
#endif
}
