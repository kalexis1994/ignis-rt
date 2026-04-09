// Streamline Frame Generation + Reflex integration
// Compiled when IGNIS_USE_STREAMLINE=ON (ACPT_HAVE_STREAMLINE defined)

#ifdef ACPT_HAVE_STREAMLINE

#include "sl_frame_gen.h"
#include "../include/ignis_log.h"

// Streamline SDK headers
#include <sl.h>
#include <sl_consts.h>
#include <sl_core_api.h>
#include <sl_core_types.h>
#include <sl_dlss_g.h>
#include <sl_reflex.h>
#include <sl_pcl.h>

#include <windows.h>
#include <cstring>

namespace acpt {

// ============================================================================
// Helpers
// ============================================================================

/// Detect NVIDIA GPU architecture from VkPhysicalDeviceProperties
/// Uses PCI Device ID ranges to identify Ada Lovelace (RTX 40) vs Blackwell (RTX 50)
static FrameGenGPUCap DetectGPUCapFromDeviceID(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);

    // Must be NVIDIA (vendor 0x10DE)
    if (props.vendorID != 0x10DE) {
        Log(L"[SL FrameGen] Non-NVIDIA GPU (vendor 0x%04X) — FG unsupported\n", props.vendorID);
        return FrameGenGPUCap::Unsupported;
    }

    Log(L"[SL FrameGen] NVIDIA GPU: %S (deviceID=0x%04X)\n", props.deviceName, props.deviceID);

    // Preliminary check — Streamline's slIsFeatureSupported is authoritative.
    // PCI device ID ranges (approximate):
    //   RTX 20xx (Turing):  0x1E00-0x1FFF
    //   RTX 30xx (Ampere):  0x2200-0x25FF
    //   RTX 40xx (Ada):     0x2600-0x28FF
    //   RTX 50xx (Blackwell): 0x2900+ (tentative)

    uint32_t id = props.deviceID;

    if (id >= 0x2900) {
        Log(L"[SL FrameGen] Detected Blackwell (RTX 50xx) — Multi Frame Generation\n");
        return FrameGenGPUCap::MultiFrame;
    } else if (id >= 0x2600) {
        Log(L"[SL FrameGen] Detected Ada Lovelace (RTX 40xx) — Single Frame Generation\n");
        return FrameGenGPUCap::SingleFrame;
    } else {
        Log(L"[SL FrameGen] Pre-Ada GPU (deviceID 0x%04X) — FG unsupported\n", id);
        return FrameGenGPUCap::Unsupported;
    }
}

// ============================================================================
// SLFrameGen
// ============================================================================

SLFrameGen::SLFrameGen()
    : available_(false), active_(false), reflexReady_(false),
      gpuCap_(FrameGenGPUCap::Unsupported), mode_(FrameGenMode::Off),
      framesToGenerate_(0), maxFramesToGenerate_(0),
      displayWidth_(0), displayHeight_(0), viewportId_(0),
      instance_(VK_NULL_HANDLE), physicalDevice_(VK_NULL_HANDLE),
      device_(VK_NULL_HANDLE), slInterposerLib_(nullptr) {}

SLFrameGen::~SLFrameGen() {
    Shutdown();
}

bool SLFrameGen::Initialize(
    VkInstance instance,
    VkPhysicalDevice physicalDevice,
    VkDevice device,
    VkQueue graphicsQueue,
    uint32_t graphicsQueueFamily,
    uint32_t displayWidth,
    uint32_t displayHeight)
{
    if (available_) return true;

    instance_ = instance;
    physicalDevice_ = physicalDevice;
    device_ = device;
    displayWidth_ = displayWidth;
    displayHeight_ = displayHeight;
    viewportId_ = 0;

    // ---- Step 1: Preliminary GPU check (fast, before loading DLLs) ----
    gpuCap_ = DetectGPUCapFromDeviceID(physicalDevice);
    if (gpuCap_ == FrameGenGPUCap::Unsupported) {
        Log(L"[SL FrameGen] GPU does not support Frame Generation\n");
        return false;
    }

    // ---- Step 2: Load Streamline interposer (manual hooking) ----
    HMODULE slLib = LoadLibraryW(L"sl.interposer.dll");
    if (!slLib) {
        Log(L"[SL FrameGen] sl.interposer.dll not found — Streamline SDK not available\n");
        return false;
    }
    slInterposerLib_ = slLib;

    // ---- Step 3: Initialize Streamline ----
    sl::Preferences prefs{};
    prefs.showConsole = false;
    prefs.logLevel = sl::LogLevel::eOff;
    prefs.pathsToPlugins = nullptr;
    prefs.numPathsToPlugins = 0;
    prefs.applicationId = 0x1337BEEF;  // Same app ID as NGX
    prefs.engineType = sl::EngineType::eCustom;

    // Request DLSS-G, Reflex, and PCL features
    sl::Feature features[] = { sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL };
    prefs.featuresToLoad = features;
    prefs.numFeaturesToLoad = 3;

    // Vulkan + manual hooking (only intercepts 8 swapchain/present calls)
    prefs.renderAPI = sl::RenderAPI::eVulkan;
    prefs.flags = sl::PreferenceFlags::eUseManualHooking;

    sl::Result r = slInit(prefs);
    if (r != sl::Result::eOk) {
        Log(L"[SL FrameGen] slInit failed: %d\n", (int)r);
        FreeLibrary(slLib);
        slInterposerLib_ = nullptr;
        return false;
    }

    Log(L"[SL FrameGen] Streamline initialized\n");

    // ---- Step 4: Check actual feature support via Streamline ----
    sl::FeatureRequirements fgReqs{};
    r = slGetFeatureRequirements(sl::kFeatureDLSS_G, fgReqs);
    if (r != sl::Result::eOk ||
        !(fgReqs.flags & sl::FeatureRequirementFlags::eVulkanSupported)) {
        Log(L"[SL FrameGen] DLSS-G not supported on this system\n");
        slShutdown();
        FreeLibrary(slLib);
        slInterposerLib_ = nullptr;
        gpuCap_ = FrameGenGPUCap::Unsupported;
        return false;
    }

    // ---- Step 5: Initialize Reflex (mandatory for Frame Generation) ----
    sl::ReflexOptions reflexOpts{};
    reflexOpts.mode = sl::ReflexMode::eLowLatency;
    reflexOpts.frameLimitUs = 0;
    r = slReflexSetOptions(reflexOpts);
    if (r != sl::Result::eOk) {
        Log(L"[SL FrameGen] Reflex init failed: %d — continuing without Reflex\n", (int)r);
        reflexReady_ = false;
    } else {
        reflexReady_ = true;
        Log(L"[SL FrameGen] Reflex initialized (Low Latency mode)\n");
    }

    // ---- Step 6: Query Frame Generation state for max frames ----
    sl::ViewportHandle viewport(viewportId_);
    sl::DLSSGState fgState{};
    r = slDLSSGGetState(viewport, fgState, nullptr);
    if (r == sl::Result::eOk && fgState.numFramesToGenerateMax > 0) {
        maxFramesToGenerate_ = fgState.numFramesToGenerateMax;
        if (maxFramesToGenerate_ >= 3) {
            gpuCap_ = FrameGenGPUCap::MultiFrame;
        } else {
            gpuCap_ = FrameGenGPUCap::SingleFrame;
        }
        Log(L"[SL FrameGen] Max frames to generate: %u\n", maxFramesToGenerate_);
    } else {
        // Fallback from device ID heuristic
        maxFramesToGenerate_ = (gpuCap_ == FrameGenGPUCap::MultiFrame) ? 3 : 1;
        Log(L"[SL FrameGen] State query unavailable, using heuristic: max=%u\n", maxFramesToGenerate_);
    }

    available_ = true;
    Log(L"[SL FrameGen] Ready — GPU cap: %s, max frames: %u\n",
        gpuCap_ == FrameGenGPUCap::MultiFrame ? L"MultiFrame" : L"SingleFrame",
        maxFramesToGenerate_);

    return true;
}

void SLFrameGen::Shutdown() {
    if (!available_) return;

    // Disable FG before shutdown
    if (active_) {
        sl::DLSSGOptions opts{};
        opts.mode = sl::DLSSGMode::eOff;
        sl::ViewportHandle viewport(viewportId_);
        slDLSSGSetOptions(viewport, opts);
        active_ = false;
    }

    slShutdown();

    if (slInterposerLib_) {
        FreeLibrary((HMODULE)slInterposerLib_);
        slInterposerLib_ = nullptr;
    }

    available_ = false;
    reflexReady_ = false;
    gpuCap_ = FrameGenGPUCap::Unsupported;
    maxFramesToGenerate_ = 0;
    framesToGenerate_ = 0;
    mode_ = FrameGenMode::Off;

    Log(L"[SL FrameGen] Shutdown\n");
}

bool SLFrameGen::SetOptions(FrameGenMode mode, uint32_t framesToGenerate) {
    if (!available_) return false;

    if (mode != FrameGenMode::Off) {
        if (framesToGenerate < 1 || framesToGenerate > maxFramesToGenerate_) {
            Log(L"[SL FrameGen] Invalid framesToGenerate=%u (max=%u)\n",
                framesToGenerate, maxFramesToGenerate_);
            return false;
        }
    }

    sl::DLSSGOptions opts{};
    switch (mode) {
        case FrameGenMode::Off:
            opts.mode = sl::DLSSGMode::eOff;
            break;
        case FrameGenMode::On:
            opts.mode = sl::DLSSGMode::eOn;
            opts.numFramesToGenerate = framesToGenerate;
            break;
        case FrameGenMode::Auto:
            opts.mode = sl::DLSSGMode::eAuto;
            opts.numFramesToGenerate = framesToGenerate;
            break;
    }

    sl::ViewportHandle viewport(viewportId_);
    sl::Result r = slDLSSGSetOptions(viewport, opts);
    if (r != sl::Result::eOk) {
        Log(L"[SL FrameGen] SetOptions failed: %d\n", (int)r);
        return false;
    }

    mode_ = mode;
    framesToGenerate_ = (mode == FrameGenMode::Off) ? 0 : framesToGenerate;
    active_ = (mode != FrameGenMode::Off);

    Log(L"[SL FrameGen] Mode=%d, frames=%u\n", (int)mode, framesToGenerate_);
    return true;
}

void SLFrameGen::TagResources(
    VkCommandBuffer cmd,
    uint32_t frameIndex,
    VkImage hudlessColor,    VkImageView hudlessColorView,
    VkImage depth,           VkImageView depthView,
    VkImage motionVectors,   VkImageView motionVectorsView,
    VkImage uiColor,         VkImageView uiColorView)
{
    if (!available_ || !active_) return;

    // Get frame token for this frame
    sl::FrameToken* frameToken = nullptr;
    slGetNewFrameToken(frameToken, &frameIndex);
    if (!frameToken) return;

    sl::ViewportHandle viewport(viewportId_);

    // Set common constants for this frame
    sl::Constants consts{};
    consts.reset = sl::Boolean::eFalse;
    consts.mvecScale = { 1.0f, 1.0f };
    consts.jitterOffset = { 0.0f, 0.0f };
    consts.depthInverted = sl::Boolean::eFalse;
    consts.cameraMotionIncluded = sl::Boolean::eTrue;
    slSetConstants(consts, *frameToken, viewport);

    // Tag HUD-less color (mandatory)
    sl::Resource hudlessRes(sl::ResourceType::eTex2d, hudlessColor,
                            (uint32_t)VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    sl::ResourceTag hudlessTag(&hudlessRes, sl::kBufferTypeHUDLessColor,
                               sl::ResourceLifecycle::eValidUntilPresent);

    // Tag depth
    sl::Resource depthRes(sl::ResourceType::eTex2d, depth,
                          (uint32_t)VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    sl::ResourceTag depthTag(&depthRes, sl::kBufferTypeDepth,
                             sl::ResourceLifecycle::eValidUntilPresent);

    // Tag motion vectors
    sl::Resource mvRes(sl::ResourceType::eTex2d, motionVectors,
                       (uint32_t)VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    sl::ResourceTag mvTag(&mvRes, sl::kBufferTypeMotionVectors,
                          sl::ResourceLifecycle::eValidUntilPresent);

    // Build tag array
    sl::ResourceTag tags[4];
    uint32_t tagCount = 0;
    tags[tagCount++] = hudlessTag;
    tags[tagCount++] = depthTag;
    tags[tagCount++] = mvTag;

    // Optional: UI overlay
    if (uiColor != VK_NULL_HANDLE) {
        sl::Resource uiRes(sl::ResourceType::eTex2d, uiColor,
                           (uint32_t)VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        sl::ResourceTag uiTag(&uiRes, sl::kBufferTypeUIColorAndAlpha,
                              sl::ResourceLifecycle::eValidUntilPresent);
        tags[tagCount++] = uiTag;
    }

    slSetTagForFrame(*frameToken, viewport, tags, tagCount, (sl::CommandBuffer*)cmd);
}

void SLFrameGen::SetReflexMarker(ReflexMarker marker, uint64_t frameId) {
    if (!reflexReady_) return;

    sl::PCLMarker pclMarker;
    switch (marker) {
        case ReflexMarker::SimulationStart: pclMarker = sl::PCLMarker::eSimulationStart;    break;
        case ReflexMarker::SimulationEnd:   pclMarker = sl::PCLMarker::eSimulationEnd;      break;
        case ReflexMarker::RenderStart:     pclMarker = sl::PCLMarker::eRenderSubmitStart;  break;
        case ReflexMarker::RenderEnd:       pclMarker = sl::PCLMarker::eRenderSubmitEnd;    break;
        case ReflexMarker::PresentStart:    pclMarker = sl::PCLMarker::ePresentStart;       break;
        case ReflexMarker::PresentEnd:      pclMarker = sl::PCLMarker::ePresentEnd;         break;
        default: return;
    }

    uint32_t idx = (uint32_t)frameId;
    sl::FrameToken* frameToken = nullptr;
    slGetNewFrameToken(frameToken, &idx);
    if (frameToken) {
        slPCLSetMarker(pclMarker, *frameToken);
    }
}

void SLFrameGen::ReflexSleep() {
    if (!reflexReady_) return;

    // slReflexSleep requires a valid FrameToken
    uint32_t idx = 0;
    sl::FrameToken* frameToken = nullptr;
    slGetNewFrameToken(frameToken, &idx);
    if (frameToken) {
        slReflexSleep(*frameToken);
    }
}

} // namespace acpt

#endif // ACPT_HAVE_STREAMLINE
