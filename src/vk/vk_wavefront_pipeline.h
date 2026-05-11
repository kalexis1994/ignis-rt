#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

namespace acpt {
namespace vk {

class Context;
class RTResources;

/// Wavefront path tracing pipeline — replaces monolithic raygen with
/// multiple compute kernels for better GPU occupancy.
///
/// Kernels: K0(camera rays) → K1(intersect) → K2(shade+NEE) →
///          K3(shadow intersect) → K4(accumulate) → K5(output)
/// Bounced K1-K4 loop with path compaction between iterations.
class WavefrontPipeline {
public:
    bool Initialize(Context* context, RTResources* rtPipeline,
                    uint32_t width, uint32_t height, uint32_t maxBounces);
    void Shutdown();

    /// Record the bounce loop + ReSTIR PT, stopping BEFORE K5 (output).
    /// Caller must follow with any extra pixelRadiance contributors
    /// (e.g. the DI port) and then RecordOutput to consume the buffer.
    /// Uses the same descriptor set 0 as RTResources for scene data.
    void RecordDispatch(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                        VkDescriptorSet sceneDescSet, uint32_t maxBounces,
                        uint32_t spp = 1);

    /// Dispatch K5 (wf_output): consume pixelRadiance + write the final
    /// interop image. Must be called once after RecordDispatch and any
    /// DI-port additive passes. Owns the surface-history ping-pong
    /// flip + frameIndex_ increment at the tail.
    void RecordOutput(VkCommandBuffer cmd, VkDescriptorSet sceneDescSet);

    /// Phase 4b: dispatch wf_prepare_lights over the polymorphic light
    /// buffer. lightCount = number of valid entries; the rest of the
    /// capacity stays untouched. No-op when lightCount == 0.
    void RecordPrepareLights(VkCommandBuffer cmd, VkDescriptorSet diDescSet,
                             uint32_t lightCount);

    /// Phase 4c: dispatch wf_di_initial_samples over primary pixels.
    /// Generates the per-pixel DI reservoir at diPolyInitial (set 2 b=1).
    /// Sky pixels and zero-light frames produce empty reservoirs.
    /// Must be called AFTER RecordDispatch so primaryGBuf is populated.
    /// No-op when lightCount == 0.
    void RecordDIInitialSamples(VkCommandBuffer cmd,
                                 VkDescriptorSet sceneDescSet,
                                 VkDescriptorSet diDescSet,
                                 uint32_t width, uint32_t height,
                                 uint32_t frameIndex,
                                 uint32_t lightCount);

    /// Phase 4d: dispatch wf_di_temporal over primary pixels.
    /// Reads diPolyInitial + diPolyFinalPrev + diGBufPrev (via set 2),
    /// writes diPolyScratch + diGBufCurr. Must be called AFTER
    /// RecordDIInitialSamples (depends on initial reservoir).
    /// No-op when lightCount == 0.
    void RecordDITemporal(VkCommandBuffer cmd,
                           VkDescriptorSet sceneDescSet,
                           VkDescriptorSet diDescSet,
                           uint32_t width, uint32_t height,
                           uint32_t frameIndex,
                           uint32_t lightCount);

    /// Phase 4e: dispatch wf_di_spatial over primary pixels.
    /// Reads diPolyScratch (the temporal output) and writes
    /// diPolyFinalCurr (the closes-the-loop final reservoir for
    /// next-frame temporal reuse). Must be called AFTER
    /// RecordDITemporal. No-op when lightCount == 0.
    void RecordDISpatial(VkCommandBuffer cmd,
                          VkDescriptorSet sceneDescSet,
                          VkDescriptorSet diDescSet,
                          uint32_t width, uint32_t height,
                          uint32_t frameIndex,
                          uint32_t lightCount);

    /// Phase 4f: dispatch wf_di_shade over primary pixels.
    /// Reads diPolyFinalCurr, atomic-adds BSDF × Li × geometry × W
    /// into pixelRadiance's diffuse slots. Currently runs additively
    /// on top of the inline NEE in wf_shade — directly-lit primary
    /// surfaces will look ~2× as bright until the inline path is
    /// gated off at bounce 0 in a follow-up.
    /// Must be called AFTER RecordDISpatial. No-op when
    /// lightCount == 0.
    void RecordDIShade(VkCommandBuffer cmd,
                        VkDescriptorSet sceneDescSet,
                        VkDescriptorSet diDescSet,
                        uint32_t width, uint32_t height,
                        uint32_t frameIndex,
                        uint32_t lightCount);

    bool IsReady() const { return ready_; }

private:
    bool CreateBuffers(uint32_t pixelCount);
    bool CreateDescriptorSet();
    bool CreatePipelines();
    bool LoadComputeShader(const char* path, VkShaderModule* outModule);

    Context* context_ = nullptr;
    RTResources* rtResources_ = nullptr;
    bool ready_ = false;
    uint32_t maxPixels_ = 0;
    uint32_t frameIndex_ = 0;

    // Wavefront SSBO buffers — PathState SoA (double-buffered per field)
    // originDir: origin.xyz + direction.xyz = 24 bytes/path
    // pixelRng:  pixelIndex + rngState = 8 bytes/path
    // throughput: throughput.xyz = 12 bytes/path
    // flags:     flags = 4 bytes/path
    VkBuffer originDirBuffer_[2] = {};
    VkDeviceMemory originDirMemory_[2] = {};
    VkBuffer pixelRngBuffer_[2] = {};
    VkDeviceMemory pixelRngMemory_[2] = {};
    VkBuffer throughputBuffer_[2] = {};
    VkDeviceMemory throughputMemory_[2] = {};
    VkBuffer flagsBuffer_[2] = {};

    VkDeviceMemory flagsMemory_[2] = {};
    // Per-path firefly_k tracker (RTXPT). 1 float/path, ping-pong R/W bindings 22/23.
    VkBuffer fireflyKBuffer_[2] = {};
    VkDeviceMemory fireflyKMemory_[2] = {};
    uint32_t pathStateCurrent_ = 0;                    // index of current read buffer
    VkBuffer hitResultBuffer_ = VK_NULL_HANDLE;        // HitResult[]
    VkDeviceMemory hitResultMemory_ = VK_NULL_HANDLE;
    VkBuffer shadowRayBuffer_ = VK_NULL_HANDLE;        // ShadowRay[]
    VkDeviceMemory shadowRayMemory_ = VK_NULL_HANDLE;
    VkBuffer pixelRadianceBuffer_ = VK_NULL_HANDLE;    // PixelRadiance[]
    VkDeviceMemory pixelRadianceMemory_ = VK_NULL_HANDLE;
    VkBuffer primaryGBufBuffer_ = VK_NULL_HANDLE;      // PrimaryGBuffer[]
    VkDeviceMemory primaryGBufMemory_ = VK_NULL_HANDLE;
    VkBuffer countersBuffer_ = VK_NULL_HANDLE;          // WavefrontCounters
    VkDeviceMemory countersMemory_ = VK_NULL_HANDLE;
    VkBuffer indirectDispatchBuffer_ = VK_NULL_HANDLE;  // VkDispatchIndirectCommand[3]
    VkDeviceMemory indirectDispatchMemory_ = VK_NULL_HANDLE;
    VkBuffer sharcStateBuffer_ = VK_NULL_HANDLE;       // SharcState[] per-pixel (persists across bounces)
    VkDeviceMemory sharcStateMemory_ = VK_NULL_HANDLE;

    // ReSTIR PT buffers
    VkBuffer ptReservoirBuffer_[2] = {};               // ping-pong reservoirs (128 bytes/pixel)
    VkDeviceMemory ptReservoirMemory_[2] = {};
    VkBuffer ptPathRecordBuffer_ = VK_NULL_HANDLE;     // path records (96 bytes/pixel)
    VkDeviceMemory ptPathRecordMemory_ = VK_NULL_HANDLE;
    uint32_t ptReservoirCurrent_ = 0;                  // ping-pong index

    // Stable Planes buffers
    VkBuffer spHeaderBuffer_ = VK_NULL_HANDLE;         // uvec4 per pixel (branchIDs + dominant)
    VkDeviceMemory spHeaderMemory_ = VK_NULL_HANDLE;
    VkBuffer spDataBuffer_ = VK_NULL_HANDLE;           // 24 vec4s per pixel (3 planes × 8 vec4s)
    VkDeviceMemory spDataMemory_ = VK_NULL_HANDLE;

    // Surface History (ping-pong, RTXPT-style temporal-reuse veto).
    // 5 uints/pixel: validFlags, instanceId, materialId, primitiveId, customIndex.
    // Written by wf_output (curr), read by wf_pt_temporal/spatial (prev) to
    // veto reservoir reuse when the reprojected surface is no longer the same
    // instance/primitive — kills the wake/smear of objects across reveal edges.
    VkBuffer surfaceHistoryBuffer_[2] = {};
    VkDeviceMemory surfaceHistoryMemory_[2] = {};
    uint32_t surfaceHistoryCurrent_ = 0;               // ping-pong index for "curr"

    // Descriptor sets for ping-pong (2 sets, no host updates during recording)
    // SoA bindings: 0=originDir(R), 7=originDir(W), 9=pixelRng(R), 10=pixelRng(W),
    //               11=throughput(R), 12=throughput(W), 13=flags(R), 14=flags(W)
    // Plus: 1=hitResult, 2=shadowRay, 3=pixelRadiance, 4=primaryGBuf, 5=counters, 6=indirect, 8=sharc
    VkDescriptorSetLayout wfDescSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool wfDescPool_ = VK_NULL_HANDLE;
    VkDescriptorSet wfDescSet_[2] = {};  // [0] = A, [1] = B

    // Compute pipelines (K0-K5 + compact)
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;  // shared layout for all kernels
    VkPipeline pipelineK0_ = VK_NULL_HANDLE;  // camera rays
    VkPipeline pipelineK1_ = VK_NULL_HANDLE;  // intersect
    VkPipeline pipelineK2_ = VK_NULL_HANDLE;  // shade + NEE
    VkPipeline pipelineK3_ = VK_NULL_HANDLE;  // shadow intersect
    VkPipeline pipelineK4_ = VK_NULL_HANDLE;  // accumulate
    VkPipeline pipelineK5_ = VK_NULL_HANDLE;  // output
    VkPipeline pipelineCompact_ = VK_NULL_HANDLE;  // prepare indirect + compact

    // ReSTIR PT compute pipelines
    VkPipeline pipelinePTTemporal_ = VK_NULL_HANDLE;
    VkPipeline pipelinePTSpatial_ = VK_NULL_HANDLE;
    VkPipeline pipelinePTFinal_ = VK_NULL_HANDLE;

    // Stable Planes compute pipeline
    VkPipeline pipelineStablePlanes_ = VK_NULL_HANDLE;

    // Phase 4b — RTXDI DI port: wf_prepare_lights compute pass.
    // Standalone layout (descriptor set 2 only + push_constant uint).
    VkPipelineLayout pipelineLayoutDIPrepare_ = VK_NULL_HANDLE;
    VkPipeline       pipelinePrepareLights_   = VK_NULL_HANDLE;

    // Phase 4c/4d — DI port compute pipelines. All share the same
    // 3-set + 4×u32-push layout. Initial / temporal differ only in
    // which set 2 bindings they touch.
    VkPipelineLayout pipelineLayoutDI_     = VK_NULL_HANDLE;
    VkPipeline       pipelineDIInitial_    = VK_NULL_HANDLE;
    VkPipeline       pipelineDITemporal_   = VK_NULL_HANDLE;
    VkPipeline       pipelineDISpatial_    = VK_NULL_HANDLE;
    VkPipeline       pipelineDIShade_      = VK_NULL_HANDLE;

    // RT pipeline for K2 (shade) — enables hardware SER via reorderThreadNV
    VkPipeline pipelineK2RT_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayoutRT_ = VK_NULL_HANDLE;
    VkBuffer sbtK2Buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory sbtK2Memory_ = VK_NULL_HANDLE;
    VkStridedDeviceAddressRegionKHR sbtK2RaygenRegion_{};
    VkStridedDeviceAddressRegionKHR sbtK2MissRegion_{};
    VkStridedDeviceAddressRegionKHR sbtK2HitRegion_{};
    VkStridedDeviceAddressRegionKHR sbtK2CallableRegion_{};
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR_ = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR_ = nullptr;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR_ = nullptr;
    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR_ = nullptr;
    bool serAvailable_ = false;
    bool CreateK2RTResources();

    // Cached push-constant + workgroup-count snapshot from the most
    // recent RecordDispatch — consumed by RecordOutput when K5 fires
    // after the DI port. We snapshot rather than recompute so K5 sees
    // the exact same frameIndex / spp / sampleIdx that the bounce loop
    // and PT passes used in this frame.
    struct OutputPush {
        uint32_t width, height, frameIndex, maxBounces, currentBounce, spp, sampleIdx;
    };
    OutputPush outputPush_   = {};
    uint32_t   outputGroupsX_ = 0;
    uint32_t   outputGroupsY_ = 0;
    uint32_t   outputShSetIdx_ = 0;
    bool       outputPending_  = false;

    static constexpr uint32_t WORKGROUP_SIZE = 256;
    static constexpr uint32_t MAX_SHADOW_RAYS_PER_PATH = 10; // sun + 8 lights + emissive
};

} // namespace vk
} // namespace acpt
