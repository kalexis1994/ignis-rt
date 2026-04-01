// nrc_integration.cpp — NVIDIA Neural Radiance Cache integration

#ifdef IGNIS_HAVE_NRC

#include "nrc_integration.h"
#include "vk_context.h"
#include "../../include/ignis_log.h"

#include <NrcVk.h>
#include <NrcCommon.h>
#include <algorithm>

namespace acpt {
namespace vk {

// NRC log callback
static void NrcLogCallback(const char* message, nrc::LogLevel level) {
    const wchar_t* prefix = L"[NRC]";
    switch (level) {
        case nrc::LogLevel::Error:   prefix = L"[NRC ERROR]"; break;
        case nrc::LogLevel::Warning: prefix = L"[NRC WARN]";  break;
        case nrc::LogLevel::Info:    prefix = L"[NRC]";        break;
        case nrc::LogLevel::Debug:   prefix = L"[NRC DBG]";    break;
    }
    // Convert char* to wchar_t* for our logger
    wchar_t wmsg[512];
    mbstowcs(wmsg, message, 511);
    wmsg[511] = 0;
    Log(L"%s %s\n", prefix, wmsg);
}

bool NrcIntegration::Initialize(Context* ctx, uint32_t renderWidth, uint32_t renderHeight,
                                 uint32_t spp, uint32_t maxBounces,
                                 const float sceneMin[3], const float sceneMax[3]) {
    context_ = ctx;

    // 1. Initialize NRC library (once globally)
    nrc::GlobalSettings globalSettings = {};
    globalSettings.majorVersion = NRC_VERSION_MAJOR;
    globalSettings.minorVersion = NRC_VERSION_MINOR;
    globalSettings.loggerFn = NrcLogCallback;
    globalSettings.enableGPUMemoryAllocation = true;
    globalSettings.maxNumFramesInFlight = 2;

    // Tell NRC where CUDA runtime DLLs are (same dir as ignis_rt.dll)
    static wchar_t nrcDepsPath[512] = {};
    {
        HMODULE hm = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR)&NrcLogCallback, &hm);
        if (hm) {
            GetModuleFileNameW(hm, nrcDepsPath, 511);
            wchar_t* lastSlash = wcsrchr(nrcDepsPath, L'\\');
            if (lastSlash) *(lastSlash + 1) = 0;
            globalSettings.depsDirectoryPath = nrcDepsPath;
            Log(L"[NRC] Deps path: %s\n", nrcDepsPath);
        }
    }

    nrc::Status status = nrc::vulkan::Initialize(globalSettings);
    if (status != nrc::Status::OK) {
        Log(L"[NRC] Initialize failed: %d\n", (int)status);
        if (status == nrc::Status::UnsupportedHardware) {
            Log(L"[NRC] GPU does not support NRC (needs Tensor Cores)\n");
        }
        return false;
    }
    Log(L"[NRC] Library initialized (v%d.%d)\n", NRC_VERSION_MAJOR, NRC_VERSION_MINOR);

    // 2. Create NRC context
    status = nrc::vulkan::Context::Create(
        ctx->GetDevice(), ctx->GetPhysicalDevice(), ctx->GetInstance(), nrcContext_);
    if (status != nrc::Status::OK) {
        Log(L"[NRC] Context creation failed: %d\n", (int)status);
        nrc::vulkan::Shutdown();
        return false;
    }
    Log(L"[NRC] Context created\n");

    // 3. Configure
    contextSettings_ = {};
    contextSettings_.learnIrradiance = true;   // demodulate albedo → better detail retention
    contextSettings_.includeDirectLighting = false;  // cache only indirect → direct comes from NEE
    contextSettings_.frameDimensions = { renderWidth, renderHeight };
    // Training resolution: keep small to minimize update pass cost
    contextSettings_.trainingDimensions = nrc::ComputeIdealTrainingDimensions(
        { renderWidth, renderHeight }, 2);  // 2 training iterations — cheap
    contextSettings_.samplesPerPixel = spp;
    contextSettings_.maxPathVertices = std::max(maxBounces, 4u);  // 4 bounces for training
    contextSettings_.sceneBoundsMin = { sceneMin[0], sceneMin[1], sceneMin[2] };
    contextSettings_.sceneBoundsMax = { sceneMax[0], sceneMax[1], sceneMax[2] };
    contextSettings_.smallestResolvableFeatureSize = 0.05f;

    status = nrcContext_->Configure(contextSettings_);
    if (status != nrc::Status::OK) {
        Log(L"[NRC] Configure failed: %d\n", (int)status);
        nrc::vulkan::Context::Destroy(*nrcContext_);
        nrcContext_ = nullptr;
        nrc::vulkan::Shutdown();
        return false;
    }

    // 4. Frame settings — tuned for interactive viewport
    frameSettings_ = {};
    frameSettings_.maxExpectedAverageRadianceValue = 1.0f;
    frameSettings_.terminationHeuristicThreshold = 1000.0f;  // start disabled, ramp down after warmup
    frameSettings_.trainingTerminationHeuristicThreshold = 0.1f;
    frameSettings_.trainTheCache = true;
    frameSettings_.learningRate = 1e-2f;
    frameSettings_.numTrainingIterations = 2;  // keep training cheap
    frameSettings_.skipDeltaVertices = true;   // don't terminate on mirrors/glass

    nrcReady_ = true;
    Log(L"[NRC] Ready: render=%ux%u training=%ux%u spp=%u maxPathVerts=%u (input bounces=%u)\n",
        renderWidth, renderHeight,
        contextSettings_.trainingDimensions.x, contextSettings_.trainingDimensions.y,
        spp, contextSettings_.maxPathVertices, maxBounces);
    return true;
}

void NrcIntegration::Shutdown() {
    if (nrcContext_) {
        nrc::vulkan::Context::Destroy(*nrcContext_);
        nrcContext_ = nullptr;
    }
    nrc::vulkan::Shutdown();
    nrcReady_ = false;
    Log(L"[NRC] Shutdown\n");
}

void NrcIntegration::BeginFrame(VkCommandBuffer cmd) {
    if (!nrcReady_) return;

    // Warmup: let the network train for ~90 frames (~3s at 30fps) before
    // enabling early path termination. Ramp threshold from 1000 → 0.15.
    warmupFrames_++;
    const uint32_t WARMUP_FRAMES = 90;
    const float TARGET_THRESHOLD = 0.5f;  // very conservative — only terminate extremely spread paths
    if (warmupFrames_ < WARMUP_FRAMES) {
        frameSettings_.terminationHeuristicThreshold = 1000.0f;  // no termination
    } else if (warmupFrames_ < WARMUP_FRAMES + 30) {
        // Ramp over 30 frames (1s)
        float t = float(warmupFrames_ - WARMUP_FRAMES) / 30.0f;
        frameSettings_.terminationHeuristicThreshold = 1000.0f * (1.0f - t) + TARGET_THRESHOLD * t;
    } else {
        frameSettings_.terminationHeuristicThreshold = TARGET_THRESHOLD;
    }

    nrc::Status s = nrcContext_->BeginFrame(cmd, frameSettings_);
    if (s != nrc::Status::OK) {
        Log(L"[NRC] BeginFrame failed: %d\n", (int)s);
    }
}

void NrcIntegration::QueryAndTrain(VkCommandBuffer cmd) {
    if (!nrcReady_) return;
    nrc::Status s = nrcContext_->QueryAndTrain(cmd, nullptr);
    if (s != nrc::Status::OK) {
        Log(L"[NRC] QueryAndTrain failed: %d\n", (int)s);
    }
}

void NrcIntegration::Resolve(VkCommandBuffer cmd, VkImageView outputImage) {
    if (!nrcReady_) return;
    nrc::Status s = nrcContext_->Resolve(cmd, outputImage);
    if (s != nrc::Status::OK) {
        Log(L"[NRC] Resolve failed: %d\n", (int)s);
    }
}

void NrcIntegration::EndFrame(VkQueue queue) {
    if (!nrcReady_) return;
    nrc::Status s = nrcContext_->EndFrame(queue);
    if (s != nrc::Status::OK) {
        Log(L"[NRC] EndFrame failed: %d\n", (int)s);
    }
}

void NrcIntegration::Reconfigure(uint32_t renderWidth, uint32_t renderHeight,
                                  const float sceneMin[3], const float sceneMax[3]) {
    if (!nrcReady_) return;
    contextSettings_.frameDimensions = { renderWidth, renderHeight };
    contextSettings_.trainingDimensions = nrc::ComputeIdealTrainingDimensions(
        { renderWidth, renderHeight }, 0);
    contextSettings_.sceneBoundsMin = { sceneMin[0], sceneMin[1], sceneMin[2] };
    contextSettings_.sceneBoundsMax = { sceneMax[0], sceneMax[1], sceneMax[2] };
    contextSettings_.requestReset = true;

    nrc::Status s = nrcContext_->Configure(contextSettings_);
    if (s != nrc::Status::OK) {
        Log(L"[NRC] Reconfigure failed: %d\n", (int)s);
    }
    contextSettings_.requestReset = false;
    Log(L"[NRC] Reconfigured: %ux%u\n", renderWidth, renderHeight);
}

void NrcIntegration::SyncSettings(int maxBounces, int spp, const float sceneMin[3], const float sceneMax[3]) {
    if (!nrcReady_) return;

    uint32_t newMaxVerts = std::max((uint32_t)maxBounces, 2u);
    uint32_t newSpp = std::max((uint32_t)spp, 1u);
    nrc_float3 newMin = { sceneMin[0], sceneMin[1], sceneMin[2] };
    nrc_float3 newMax = { sceneMax[0], sceneMax[1], sceneMax[2] };

    // Only reconfigure if something actually changed
    if (newMaxVerts == contextSettings_.maxPathVertices &&
        newSpp == contextSettings_.samplesPerPixel &&
        newMin.x == contextSettings_.sceneBoundsMin.x &&
        newMin.y == contextSettings_.sceneBoundsMin.y &&
        newMin.z == contextSettings_.sceneBoundsMin.z &&
        newMax.x == contextSettings_.sceneBoundsMax.x &&
        newMax.y == contextSettings_.sceneBoundsMax.y &&
        newMax.z == contextSettings_.sceneBoundsMax.z) {
        return;  // no change
    }

    contextSettings_.maxPathVertices = newMaxVerts;
    contextSettings_.samplesPerPixel = newSpp;
    contextSettings_.sceneBoundsMin = newMin;
    contextSettings_.sceneBoundsMax = newMax;
    contextSettings_.requestReset = true;

    nrc::Status s = nrcContext_->Configure(contextSettings_);
    if (s != nrc::Status::OK) {
        Log(L"[NRC] SyncSettings reconfigure failed: %d\n", (int)s);
    } else {
        Log(L"[NRC] Reconfigured: bounces=%u spp=%u\n", newMaxVerts, newSpp);
        warmupFrames_ = 0;  // reset warmup — network needs to relearn
    }
    contextSettings_.requestReset = false;
}

bool NrcIntegration::PopulateShaderConstants(NrcConstants& outConstants) const {
    if (!nrcReady_) return false;
    nrc::Status s = nrcContext_->PopulateShaderConstants(outConstants);
    return s == nrc::Status::OK;
}

const nrc::vulkan::Buffers* NrcIntegration::GetBuffers() const {
    if (!nrcReady_) return nullptr;
    return nrcContext_->GetBuffers();
}

} // namespace vk
} // namespace acpt

#endif // IGNIS_HAVE_NRC
