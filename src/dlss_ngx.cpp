#include "dlss_ngx.h"
#include "ignis_log.h"
#include <string>

#ifdef ACPT_HAVE_NGX
#include "ngx/nvsdk_ngx.h"
#include "ngx/nvsdk_ngx_vk.h"
#include "ngx/nvsdk_ngx_defs.h"
#include "ngx/nvsdk_ngx_helpers_vk.h"
#endif

using namespace acpt;

DLSS_NGX::DLSS_NGX()
    : m_initialized(false)
    , m_dlssSupported(false)
    , m_qualityMode(DLSSQualityMode::Balanced)
    , m_displayWidth(0)
    , m_displayHeight(0)
    , m_renderWidth(0)
    , m_renderHeight(0)
    , m_frameIndex(0)
    , m_instance(VK_NULL_HANDLE)
    , m_device(VK_NULL_HANDLE)
    , m_physicalDevice(VK_NULL_HANDLE)
    , m_ngxParameters(nullptr)
    , m_ngxFeature(nullptr)
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
    // DLSS quality mode scaling ratios (approximate)
    float scalingRatio = 1.0f;
    switch (qualityMode) {
        case DLSSQualityMode::UltraQuality:  scalingRatio = 1.0f; break;  // Native / DLAA
        case DLSSQualityMode::MaxQuality:    scalingRatio = 1.5f; break;  // Quality
        case DLSSQualityMode::Balanced:      scalingRatio = 1.7f; break;  // Balanced
        case DLSSQualityMode::MaxPerformance: scalingRatio = 3.0f; break; // Ultra Performance
        default:                             scalingRatio = 1.7f; break;  // Default to Balanced
    }

    *outRenderWidth = static_cast<uint32_t>(displayWidth / scalingRatio);
    *outRenderHeight = static_cast<uint32_t>(displayHeight / scalingRatio);

    // Align to 2 pixels (DLSS requirement), but never exceed display resolution
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
    m_displayWidth = displayWidth;
    m_displayHeight = displayHeight;
    m_qualityMode = qualityMode;

    // Use proper render resolution for upscaling
    GetRenderResolution(displayWidth, displayHeight, qualityMode, &m_renderWidth, &m_renderHeight);

    Log(L"[DLSS-NGX] Initializing NVIDIA DLSS via NGX SDK:\n");
    Log(L"[DLSS-NGX]   Display: %ux%u\n", m_displayWidth, m_displayHeight);
    Log(L"[DLSS-NGX]   Render:  %ux%u\n", m_renderWidth, m_renderHeight);
    Log(L"[DLSS-NGX]   Quality: %d (1=UltraPerf, 2=Balanced, 3=Quality, 4=UltraQuality)\n",
        static_cast<int>(m_qualityMode));

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
    Log(L"[DLSS-NGX] Initializing NGX SDK...\n");

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
    Log(L"[DLSS-NGX] NGX SDK initialized successfully\n");

    // Allocate NGX parameters
    ngxResult = NVSDK_NGX_VULKAN_AllocateParameters((NVSDK_NGX_Parameter**)&m_ngxParameters);
    if ((ngxResult != NVSDK_NGX_Result_Success)) {
        Log(L"[DLSS-NGX] ERROR: Failed to allocate NGX parameters (error %d)\n", ngxResult);
        NVSDK_NGX_VULKAN_Shutdown1(m_device);
        return false;
    }

    // Check DLSS support
    Log(L"[DLSS-NGX] Checking DLSS support...\n");
    NVSDK_NGX_Parameter* capabilityParams = nullptr;
    ngxResult = NVSDK_NGX_VULKAN_GetCapabilityParameters(&capabilityParams);
    if ((ngxResult == NVSDK_NGX_Result_Success) && capabilityParams) {
        int dlssAvailable = 0;
        NVSDK_NGX_Result getResult = capabilityParams->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvailable);
        Log(L"[DLSS-NGX] SuperSampling_Available: result=%d, available=%d\n", getResult, dlssAvailable);

        // Also check for detailed error info
        int needsUpdatedDriver = 0;
        capabilityParams->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver);
        unsigned int minDriverMajor = 0, minDriverMinor = 0;
        capabilityParams->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &minDriverMajor);
        capabilityParams->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minDriverMinor);
        int featureInitResult = 0;
        capabilityParams->Get(NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &featureInitResult);
        Log(L"[DLSS-NGX] NeedsUpdatedDriver=%d, MinDriver=%u.%u, FeatureInitResult=%d\n",
            needsUpdatedDriver, minDriverMajor, minDriverMinor, featureInitResult);

        if ((getResult == NVSDK_NGX_Result_Success) && dlssAvailable) {
            Log(L"[DLSS-NGX] DLSS is supported on this GPU!\n");
            m_dlssSupported = true;
        } else {
            Log(L"[DLSS-NGX] WARNING: DLSS not supported (result=%d, available=%d)\n", getResult, dlssAvailable);
            Log(L"[DLSS-NGX]   Possible reasons:\n");
            Log(L"[DLSS-NGX]     - nvngx_dlss.dll not found in app directory\n");
            Log(L"[DLSS-NGX]     - GPU doesn't support DLSS (need RTX GPU)\n");
            Log(L"[DLSS-NGX]     - NVIDIA drivers out of date (need %u.%u+)\n", minDriverMajor, minDriverMinor);
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
    Log(L"[DLSS-NGX] Creating DLSS feature...\n");

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
        case DLSSQualityMode::MaxPerformance: ngxQuality = NVSDK_NGX_PerfQuality_Value_MaxPerf; break;
        case DLSSQualityMode::Balanced:       ngxQuality = NVSDK_NGX_PerfQuality_Value_Balanced; break;
        case DLSSQualityMode::MaxQuality:     ngxQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality; break;
        case DLSSQualityMode::UltraQuality:   ngxQuality = NVSDK_NGX_PerfQuality_Value_UltraQuality; break;
        default:                              ngxQuality = NVSDK_NGX_PerfQuality_Value_Balanced; break;
    }
    Log(L"[DLSS-NGX] Quality mapping: ours=%d -> NGX=%d\n", (int)m_qualityMode, (int)ngxQuality);

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

    Log(L"[DLSS-NGX] ========== INITIALIZATION COMPLETE ==========\n");
    Log(L"[DLSS-NGX]   Display Resolution: %ux%u\n", m_displayWidth, m_displayHeight);
    Log(L"[DLSS-NGX]   Render Resolution:  %ux%u (%.1f%% pixels)\n",
        m_renderWidth, m_renderHeight,
        (float)(m_renderWidth * m_renderHeight) / (m_displayWidth * m_displayHeight) * 100.0f);
    Log(L"[DLSS-NGX]   Quality Mode: %d\n", static_cast<int>(m_qualityMode));

    m_initialized = true;
    m_dlssSupported = true;

    return true;
#endif
}

void DLSS_NGX::Shutdown() {
    if (!m_initialized) {
        return;
    }

    Log(L"[DLSS-NGX] Shutting down...\n");

#ifdef ACPT_HAVE_NGX
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

    // Reactive mask (pInBiasCurrentColorMask) — hints DLSS about dynamic regions
    NVSDK_NGX_Resource_VK reactiveMaskResource = {};
    if (reactiveMaskView != VK_NULL_HANDLE && reactiveMaskImage != VK_NULL_HANDLE) {
        reactiveMaskResource = NVSDK_NGX_Create_ImageView_Resource_VK(
            reactiveMaskView, reactiveMaskImage, fullRange, VK_FORMAT_R8_UNORM,
            m_renderWidth, m_renderHeight, false  // read-only
        );
        evalParams.pInBiasCurrentColorMask = &reactiveMaskResource;
    }

    // DIAGNOSTIC: Log all DLSS parameters on first few frames
    if (m_frameIndex <= 3) {
        Log(L"[DLSS-Diag] Frame %u: jitter=(%.4f, %.4f) reset=%d deltaMs=%.1f\n",
            m_frameIndex, jitterX, jitterY, evalParams.InReset, evalParams.InFrameTimeDeltaInMsec);
        Log(L"[DLSS-Diag]   MVScale=(%.1f, %.1f) SubRect=%ux%u\n",
            evalParams.InMVScaleX, evalParams.InMVScaleY,
            evalParams.InRenderSubrectDimensions.Width, evalParams.InRenderSubrectDimensions.Height);
        Log(L"[DLSS-Diag]   Color=%p Depth=%p MV=%p Output=%p\n",
            evalParams.Feature.pInColor, evalParams.pInDepth,
            evalParams.pInMotionVectors, evalParams.Feature.pInOutput);
    }

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
