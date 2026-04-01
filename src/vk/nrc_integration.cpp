// nrc_integration.cpp — NVIDIA Neural Radiance Cache integration

#ifdef IGNIS_HAVE_NRC

#include "nrc_integration.h"
#include "vk_context.h"
#include "../../include/ignis_log.h"

#include <NrcVk.h>
#include <NrcCommon.h>

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
    contextSettings_.learnIrradiance = false;
    contextSettings_.includeDirectLighting = false;
    contextSettings_.frameDimensions = { renderWidth, renderHeight };
    contextSettings_.trainingDimensions = nrc::ComputeIdealTrainingDimensions(
        { renderWidth, renderHeight }, 0);
    contextSettings_.samplesPerPixel = spp;
    contextSettings_.maxPathVertices = maxBounces;
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

    // 4. Default frame settings
    frameSettings_ = {};
    frameSettings_.maxExpectedAverageRadianceValue = 1.0f;
    frameSettings_.terminationHeuristicThreshold = 0.1f;
    frameSettings_.trainingTerminationHeuristicThreshold = 0.1f;
    frameSettings_.trainTheCache = true;
    frameSettings_.learningRate = 1e-2f;

    nrcReady_ = true;
    Log(L"[NRC] Ready: render=%ux%u training=%ux%u spp=%u bounces=%u\n",
        renderWidth, renderHeight,
        contextSettings_.trainingDimensions.x, contextSettings_.trainingDimensions.y,
        spp, maxBounces);
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
