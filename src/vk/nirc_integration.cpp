// nirc_integration.cpp — Custom NIRC implementation

#include "nirc_integration.h"
#include "vk_context.h"
#include "../../include/ignis_log.h"
#include <cstring>
#include <cmath>
#include <random>
#include <fstream>
#include <vector>

namespace acpt {
namespace vk {

bool NircIntegration::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                    VkMemoryPropertyFlags memProps,
                                    VkBuffer& outBuf, VkDeviceMemory& outMem) {
    VkDevice device = ctx_->GetDevice();

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufInfo, nullptr, &outBuf) != VK_SUCCESS) return false;

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, outBuf, &reqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = reqs.size;
    allocInfo.memoryTypeIndex = ctx_->FindMemoryType(reqs.memoryTypeBits, memProps);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &outMem) != VK_SUCCESS) {
        vkDestroyBuffer(device, outBuf, nullptr);
        outBuf = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(device, outBuf, outMem, 0);
    return true;
}

void NircIntegration::DestroyBuffer(VkBuffer& buf, VkDeviceMemory& mem) {
    VkDevice device = ctx_->GetDevice();
    if (buf) { vkDestroyBuffer(device, buf, nullptr); buf = VK_NULL_HANDLE; }
    if (mem) { vkFreeMemory(device, mem, nullptr); mem = VK_NULL_HANDLE; }
}

void NircIntegration::ComputeSceneBounds(const float sceneMin[3], const float sceneMax[3]) {
    // Map world positions to [0,1]: pos_norm = (pos - min) / (max - min)
    // scenePosScale = 1 / (max - min), scenePosOffset = -min / (max - min)
    for (int i = 0; i < 3; i++) {
        float range = sceneMax[i] - sceneMin[i];
        if (range < 0.01f) range = 100.0f;  // fallback for degenerate AABB
        scenePosScale[i] = 1.0f / range;
        scenePosOffset[i] = -sceneMin[i] / range;
    }
}

void NircIntegration::InitializeWeights() {
    // Xavier initialization for MLP weights
    VkDevice device = ctx_->GetDevice();
    std::vector<float> weights(TOTAL_WEIGHTS, 0.0f);
    std::mt19937 rng(42);

    // Layer 1: 32 → 64
    float scale1 = sqrtf(2.0f / 32.0f);
    std::normal_distribution<float> dist1(0.0f, scale1);
    for (uint32_t i = 0; i < 32 * 64; i++) weights[i] = dist1(rng);

    // Layer 2: 64 → 64
    float scale2 = sqrtf(2.0f / 64.0f);
    std::normal_distribution<float> dist2(0.0f, scale2);
    for (uint32_t i = 2048 + 64; i < 2048 + 64 + 64 * 64; i++) weights[i] = dist2(rng);

    // Layer 3: 64 → 64
    for (uint32_t i = 6208 + 64; i < 6208 + 64 + 64 * 64; i++) weights[i] = dist2(rng);

    // Layer 4: 64 → 3
    float scale4 = sqrtf(2.0f / 64.0f);
    std::normal_distribution<float> dist4(0.0f, scale4);
    for (uint32_t i = 10368 + 64; i < 10368 + 64 + 64 * 3; i++) weights[i] = dist4(rng);

    // Upload to GPU
    void* mapped;
    vkMapMemory(device, weightMem_, 0, weightBufSize_, 0, &mapped);
    memcpy(mapped, weights.data(), weightBufSize_);
    vkUnmapMemory(device, weightMem_);

    // Zero-initialize Adam state
    void* mMapped;
    vkMapMemory(device, adamMMem_, 0, weightBufSize_, 0, &mMapped);
    memset(mMapped, 0, weightBufSize_);
    vkUnmapMemory(device, adamMMem_);

    void* vMapped;
    vkMapMemory(device, adamVMem_, 0, weightBufSize_, 0, &vMapped);
    memset(vMapped, 0, weightBufSize_);
    vkUnmapMemory(device, adamVMem_);

    // Zero-initialize hash features (small random init)
    {
        uint32_t hashEntries = HASH_LEVELS * HASH_TABLE_SIZE * FEATURES_PER_ENTRY;
        std::vector<float> hashInit(hashEntries, 0.0f);
        std::uniform_real_distribution<float> hashDist(-0.0001f, 0.0001f);
        for (uint32_t i = 0; i < hashEntries; i++) hashInit[i] = hashDist(rng);

        void* hMapped;
        vkMapMemory(device, hashFeatureMem_, 0, hashFeatureBufSize_, 0, &hMapped);
        memcpy(hMapped, hashInit.data(), hashFeatureBufSize_);
        vkUnmapMemory(device, hashFeatureMem_);
    }

    Log(L"[NIRC] Weights initialized (Xavier), hash grid randomized\n");
}

bool NircIntegration::Initialize(Context* ctx, uint32_t renderWidth, uint32_t renderHeight,
                                  const float sceneMin[3], const float sceneMax[3]) {
    ctx_ = ctx;
    ComputeSceneBounds(sceneMin, sceneMax);

    VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags gpuMem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    // Hash grid features: 12 levels × 512K × 2 floats × 4 bytes = ~48 MB
    hashFeatureBufSize_ = (VkDeviceSize)HASH_LEVELS * HASH_TABLE_SIZE * FEATURES_PER_ENTRY * sizeof(float);
    if (!CreateBuffer(hashFeatureBufSize_, storageUsage, hostVisible, hashFeatureBuf_, hashFeatureMem_)) {
        Log(L"[NIRC] Failed to create hash feature buffer (%llu bytes)\n", hashFeatureBufSize_);
        return false;
    }

    // MLP weights
    weightBufSize_ = TOTAL_WEIGHTS * sizeof(float);
    if (!CreateBuffer(weightBufSize_, storageUsage, hostVisible, weightBuf_, weightMem_)) {
        Log(L"[NIRC] Failed to create weight buffer\n");
        return false;
    }

    // Adam state
    if (!CreateBuffer(weightBufSize_, storageUsage, hostVisible, adamMBuf_, adamMMem_) ||
        !CreateBuffer(weightBufSize_, storageUsage, hostVisible, adamVBuf_, adamVMem_)) {
        Log(L"[NIRC] Failed to create Adam buffers\n");
        return false;
    }

    // Training samples: 16 floats per sample × MAX_TRAINING_SAMPLES
    trainingSampleBufSize_ = MAX_TRAINING_SAMPLES * 16 * sizeof(float);
    if (!CreateBuffer(trainingSampleBufSize_, storageUsage, gpuMem, trainingSampleBuf_, trainingSampleMem_)) {
        Log(L"[NIRC] Failed to create training sample buffer\n");
        return false;
    }

    // Query input: position(3) + direction(3) + roughness(1) + pad(1) = 8 floats per query
    queryInputBufSize_ = (VkDeviceSize)renderWidth * renderHeight * 8 * sizeof(float);
    if (!CreateBuffer(queryInputBufSize_, storageUsage, gpuMem, queryInputBuf_, queryInputMem_)) {
        Log(L"[NIRC] Failed to create query input buffer\n");
        return false;
    }

    // Query output: RGB per pixel = 4 floats (padded)
    queryOutputBufSize_ = (VkDeviceSize)renderWidth * renderHeight * 4 * sizeof(float);
    if (!CreateBuffer(queryOutputBufSize_, storageUsage, gpuMem, queryOutputBuf_, queryOutputMem_)) {
        Log(L"[NIRC] Failed to create query output buffer\n");
        return false;
    }

    InitializeWeights();

    if (!CreateTrainPipeline()) {
        Log(L"[NIRC] Failed to create training pipeline\n");
        return false;
    }

    if (!CreateInferPipeline()) {
        Log(L"[NIRC] Failed to create inference pipeline\n");
        return false;
    }

    ready_ = true;
    Log(L"[NIRC] Ready: hashGrid=%.1fMB weights=%uB queries=%ux%u\n",
        hashFeatureBufSize_ / 1048576.0f, TOTAL_WEIGHTS * 4, renderWidth, renderHeight);
    return true;
}

bool NircIntegration::CreateTrainPipeline() {
    VkDevice device = ctx_->GetDevice();

    // Descriptor set layout: 5 SSBOs (training samples, hash features, weights, adam_m, adam_v)
    VkDescriptorSetLayoutBinding bindings[5] = {};
    for (int i = 0; i < 5; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = 5;
    layoutCI.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &trainDescSetLayout_) != VK_SUCCESS)
        return false;

    // Push constants: sampleCount(4) + frameIndex(4) + learningRate(4) + scenePosScale(12) + scenePosOffset(12) = 36 bytes
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 48;  // padded to 48 for alignment

    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts = &trainDescSetLayout_;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &plCI, nullptr, &trainPipelineLayout_) != VK_SUCCESS)
        return false;

    // Load shader
    std::string shaderPath = IgnisResolvePath("shaders/nirc_train.comp.spv");
    std::ifstream file(shaderPath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[NIRC] Cannot open %hs\n", shaderPath.c_str());
        return false;
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> codeBytes(fileSize);
    file.seekg(0);
    file.read(codeBytes.data(), fileSize);
    file.close();
    VkShaderModule shaderModule = VK_NULL_HANDLE;

    VkShaderModuleCreateInfo smCI{};
    smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCI.codeSize = fileSize;
    smCI.pCode = reinterpret_cast<const uint32_t*>(codeBytes.data());
    if (vkCreateShaderModule(device, &smCI, nullptr, &shaderModule) != VK_SUCCESS)
        return false;

    VkComputePipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCI.stage.module = shaderModule;
    pipelineCI.stage.pName = "main";
    pipelineCI.layout = trainPipelineLayout_;

    VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &trainPipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    if (r != VK_SUCCESS) return false;

    // Descriptor pool + set
    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 };
    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(device, &poolCI, nullptr, &trainDescPool_) != VK_SUCCESS)
        return false;

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = trainDescPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &trainDescSetLayout_;
    if (vkAllocateDescriptorSets(device, &allocInfo, &trainDescSet_) != VK_SUCCESS)
        return false;

    // Update descriptors
    VkBuffer bufs[5] = { trainingSampleBuf_, hashFeatureBuf_, weightBuf_, adamMBuf_, adamVBuf_ };
    VkDeviceSize sizes[5] = { trainingSampleBufSize_, hashFeatureBufSize_, weightBufSize_, weightBufSize_, weightBufSize_ };

    VkDescriptorBufferInfo bufInfos[5] = {};
    VkWriteDescriptorSet writes[5] = {};
    for (int i = 0; i < 5; i++) {
        bufInfos[i].buffer = bufs[i];
        bufInfos[i].offset = 0;
        bufInfos[i].range = sizes[i];
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = trainDescSet_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);

    Log(L"[NIRC] Training pipeline created\n");
    return true;
}

bool NircIntegration::CreateInferPipeline() {
    VkDevice device = ctx_->GetDevice();

    // Descriptor set layout: 4 SSBOs (query input, hash features, weights, query output)
    VkDescriptorSetLayoutBinding bindings[4] = {};
    for (int i = 0; i < 4; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = 4;
    layoutCI.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &inferDescSetLayout_) != VK_SUCCESS)
        return false;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 48;

    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts = &inferDescSetLayout_;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &plCI, nullptr, &inferPipelineLayout_) != VK_SUCCESS)
        return false;

    // Load shader
    std::string shaderPath = IgnisResolvePath("shaders/nirc_inference.comp.spv");
    std::ifstream file(shaderPath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[NIRC] Cannot open inference shader %hs\n", shaderPath.c_str());
        return false;
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> codeBytes(fileSize);
    file.seekg(0);
    file.read(codeBytes.data(), fileSize);
    file.close();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo smCI{};
    smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCI.codeSize = fileSize;
    smCI.pCode = reinterpret_cast<const uint32_t*>(codeBytes.data());
    if (vkCreateShaderModule(device, &smCI, nullptr, &shaderModule) != VK_SUCCESS)
        return false;

    VkComputePipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCI.stage.module = shaderModule;
    pipelineCI.stage.pName = "main";
    pipelineCI.layout = inferPipelineLayout_;

    VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &inferPipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    if (r != VK_SUCCESS) return false;

    // Descriptor pool + set
    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(device, &poolCI, nullptr, &inferDescPool_) != VK_SUCCESS)
        return false;

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = inferDescPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &inferDescSetLayout_;
    if (vkAllocateDescriptorSets(device, &allocInfo, &inferDescSet_) != VK_SUCCESS)
        return false;

    // Update descriptors: 0=queryInput, 1=hashFeatures, 2=weights, 3=queryOutput
    VkBuffer bufs[4] = { queryInputBuf_, hashFeatureBuf_, weightBuf_, queryOutputBuf_ };
    VkDeviceSize sizes[4] = { queryInputBufSize_, hashFeatureBufSize_, weightBufSize_, queryOutputBufSize_ };

    VkDescriptorBufferInfo bufInfos[4] = {};
    VkWriteDescriptorSet writes[4] = {};
    for (int i = 0; i < 4; i++) {
        bufInfos[i] = { bufs[i], 0, sizes[i] };
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = inferDescSet_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);

    Log(L"[NIRC] Inference pipeline created\n");
    return true;
}

void NircIntegration::Train(VkCommandBuffer cmd, uint32_t sampleCount, uint32_t frameIndex) {
    if (!ready_ || sampleCount == 0) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trainPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        trainPipelineLayout_, 0, 1, &trainDescSet_, 0, nullptr);

    struct {
        uint32_t sampleCount;
        uint32_t frameIndex;
        float learningRate;
        float pad0;
        float scenePosScale[3];
        float pad1;
        float scenePosOffset[3];
        float pad2;
    } pc;
    pc.sampleCount = sampleCount;
    pc.frameIndex = frameIndex;
    pc.learningRate = 0.01f;
    pc.pad0 = 0;
    memcpy(pc.scenePosScale, scenePosScale, 12);
    pc.pad1 = 0;
    memcpy(pc.scenePosOffset, scenePosOffset, 12);
    pc.pad2 = 0;

    vkCmdPushConstants(cmd, trainPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (sampleCount + 255) / 256, 1, 1);
}

void NircIntegration::Infer(VkCommandBuffer cmd, uint32_t queryCount) {
    if (!ready_ || queryCount == 0 || !inferPipeline_) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, inferPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        inferPipelineLayout_, 0, 1, &inferDescSet_, 0, nullptr);

    struct {
        uint32_t queryCount;
        float pad0;
        float scenePosScale[3];
        float pad1;
        float scenePosOffset[3];
        float pad2;
    } pc;
    pc.queryCount = queryCount;
    pc.pad0 = 0;
    memcpy(pc.scenePosScale, scenePosScale, 12);
    pc.pad1 = 0;
    memcpy(pc.scenePosOffset, scenePosOffset, 12);
    pc.pad2 = 0;

    vkCmdPushConstants(cmd, inferPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (queryCount + 255) / 256, 1, 1);
}

void NircIntegration::Reconfigure(const float sceneMin[3], const float sceneMax[3]) {
    ComputeSceneBounds(sceneMin, sceneMax);
    Log(L"[NIRC] Reconfigured scene bounds\n");
}

void NircIntegration::Shutdown() {
    if (!ctx_) return;
    VkDevice device = ctx_->GetDevice();
    vkDeviceWaitIdle(device);

    if (trainPipeline_) vkDestroyPipeline(device, trainPipeline_, nullptr);
    if (trainPipelineLayout_) vkDestroyPipelineLayout(device, trainPipelineLayout_, nullptr);
    if (trainDescPool_) vkDestroyDescriptorPool(device, trainDescPool_, nullptr);
    if (trainDescSetLayout_) vkDestroyDescriptorSetLayout(device, trainDescSetLayout_, nullptr);

    if (inferPipeline_) vkDestroyPipeline(device, inferPipeline_, nullptr);
    if (inferPipelineLayout_) vkDestroyPipelineLayout(device, inferPipelineLayout_, nullptr);
    if (inferDescPool_) vkDestroyDescriptorPool(device, inferDescPool_, nullptr);
    if (inferDescSetLayout_) vkDestroyDescriptorSetLayout(device, inferDescSetLayout_, nullptr);

    DestroyBuffer(hashFeatureBuf_, hashFeatureMem_);
    DestroyBuffer(weightBuf_, weightMem_);
    DestroyBuffer(adamMBuf_, adamMMem_);
    DestroyBuffer(adamVBuf_, adamVMem_);
    DestroyBuffer(trainingSampleBuf_, trainingSampleMem_);
    DestroyBuffer(queryInputBuf_, queryInputMem_);
    DestroyBuffer(queryOutputBuf_, queryOutputMem_);

    ready_ = false;
    Log(L"[NIRC] Shutdown\n");
}

} // namespace vk
} // namespace acpt
