// nirc_integration.h — Custom NIRC (Neural Incident Radiance Cache)
// Our own implementation: hash grid + MLP + training, no external SDK.

#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

namespace acpt {
namespace vk {

class Context;

class NircIntegration {
public:
    bool Initialize(Context* ctx, uint32_t renderWidth, uint32_t renderHeight,
                    const float sceneMin[3], const float sceneMax[3]);
    void Shutdown();

    // Per-frame: train network from samples written by path tracer
    void Train(VkCommandBuffer cmd, uint32_t sampleCount, uint32_t frameIndex);

    // Per-frame: run inference for query points
    void Infer(VkCommandBuffer cmd, uint32_t queryCount);

    // Reconfigure on scene/resolution change
    void Reconfigure(const float sceneMin[3], const float sceneMax[3]);

    bool IsReady() const { return ready_; }

    // Buffer accessors for raygen shader binding
    VkBuffer GetTrainingSampleBuffer() const { return trainingSampleBuf_; }
    VkBuffer GetHashFeatureBuffer() const { return hashFeatureBuf_; }
    VkBuffer GetWeightBuffer() const { return weightBuf_; }
    VkBuffer GetQueryInputBuffer() const { return queryInputBuf_; }
    VkBuffer GetQueryOutputBuffer() const { return queryOutputBuf_; }

    VkDeviceSize GetTrainingSampleBufferSize() const { return trainingSampleBufSize_; }
    VkDeviceSize GetHashFeatureBufferSize() const { return hashFeatureBufSize_; }
    VkDeviceSize GetWeightBufferSize() const { return weightBufSize_; }
    VkDeviceSize GetQueryInputBufferSize() const { return queryInputBufSize_; }
    VkDeviceSize GetQueryOutputBufferSize() const { return queryOutputBufSize_; }

    // Scene bounds (for shader push constants)
    float scenePosScale[3] = {};
    float scenePosOffset[3] = {};

private:
    Context* ctx_ = nullptr;
    bool ready_ = false;

    // Hash grid feature table: 12 levels × 512K × 2 floats
    static constexpr uint32_t HASH_LEVELS = 12;
    static constexpr uint32_t HASH_TABLE_SIZE = 524288;  // 2^19
    static constexpr uint32_t FEATURES_PER_ENTRY = 2;
    VkBuffer hashFeatureBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory hashFeatureMem_ = VK_NULL_HANDLE;
    VkDeviceSize hashFeatureBufSize_ = 0;

    // MLP weights + biases: 579 floats (2 layers × 16 hidden)
    static constexpr uint32_t TOTAL_WEIGHTS = 579;
    VkBuffer weightBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory weightMem_ = VK_NULL_HANDLE;
    VkDeviceSize weightBufSize_ = 0;

    // Adam optimizer state (momentum + variance)
    VkBuffer adamMBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory adamMMem_ = VK_NULL_HANDLE;
    VkBuffer adamVBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory adamVMem_ = VK_NULL_HANDLE;

    // Training samples (written by path tracer, read by training shader)
    static constexpr uint32_t MAX_TRAINING_SAMPLES = 65536;
    VkBuffer trainingSampleBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory trainingSampleMem_ = VK_NULL_HANDLE;
    VkDeviceSize trainingSampleBufSize_ = 0;

    // Query input/output (written by path tracer, processed by inference)
    static constexpr uint32_t MAX_QUERIES = 2 * 1024 * 1024;  // 2M queries max
    VkBuffer queryInputBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory queryInputMem_ = VK_NULL_HANDLE;
    VkDeviceSize queryInputBufSize_ = 0;
    VkBuffer queryOutputBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory queryOutputMem_ = VK_NULL_HANDLE;
    VkDeviceSize queryOutputBufSize_ = 0;

    // Training compute pipeline
    VkPipeline trainPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout trainPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout trainDescSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool trainDescPool_ = VK_NULL_HANDLE;
    VkDescriptorSet trainDescSet_ = VK_NULL_HANDLE;

    // Inference compute pipeline
    VkPipeline inferPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout inferPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout inferDescSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool inferDescPool_ = VK_NULL_HANDLE;
    VkDescriptorSet inferDescSet_ = VK_NULL_HANDLE;

    bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags memProps,
                      VkBuffer& outBuf, VkDeviceMemory& outMem);
    void DestroyBuffer(VkBuffer& buf, VkDeviceMemory& mem);
    bool CreateTrainPipeline();
    bool CreateInferPipeline();
    void InitializeWeights();
    void ComputeSceneBounds(const float sceneMin[3], const float sceneMax[3]);
};

} // namespace vk
} // namespace acpt
