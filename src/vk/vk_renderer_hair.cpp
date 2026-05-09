// vk_renderer_hair.cpp — GPU hair generation (compute) + hair contour pass.
// Extracted from vk_renderer.cpp.

#include "vk_renderer.h"
#include "vk_context.h"
#include "vk_accel_structure.h"
#include "vk_rt_resources.h"
#include "ignis_log.h"
#include <vector>
#include <fstream>
#include <cstring>

namespace acpt {
namespace vk {

// ── GPU Hair Generation ──

bool Renderer::CreateHairComputePipeline() {
    if (hairComputeReady_) return true;
    VkDevice device = context_->GetDevice();

    // Load compute shader
    std::string shaderPath = IgnisResolvePath("shaders/hair_generate.comp.spv");
    std::ifstream file(shaderPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        Log(L"[Hair] Cannot open hair_generate.comp.spv\n");
        return false;
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> code(fileSize);
    file.seekg(0);
    file.read(code.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = code.size();
    moduleInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        Log(L"[Hair] Failed to create shader module\n");
        return false;
    }

    // 9 storage buffers: parents, emitterV, emitterT, CDF, pos(out), nrm(out), idx(out), uv(out), frandTable
    VkDescriptorSetLayoutBinding bindings[9] = {};
    for (int i = 0; i < 9; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 9;
    layoutInfo.pBindings = bindings;
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &hairComputeDescSetLayout_);

    // Push constants (22 floats/uints = 88 bytes)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 100; // 25 fields × 4 bytes

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &hairComputeDescSetLayout_;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &hairComputePipelineLayout_);

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage = stageInfo;
    pipeInfo.layout = hairComputePipelineLayout_;
    auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &hairComputePipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        Log(L"[Hair] Failed to create compute pipeline\n");
        return false;
    }

    // Descriptor pool
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &hairComputeDescPool_);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = hairComputeDescPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &hairComputeDescSetLayout_;
    vkAllocateDescriptorSets(device, &allocInfo, &hairComputeDescSet_);

    hairComputeReady_ = true;
    Log(L"[Hair] Compute pipeline created\n");
    return true;
}

void Renderer::DestroyHairComputePipeline() {
    VkDevice device = context_->GetDevice();
    if (hairComputePipeline_) vkDestroyPipeline(device, hairComputePipeline_, nullptr);
    if (hairComputePipelineLayout_) vkDestroyPipelineLayout(device, hairComputePipelineLayout_, nullptr);
    if (hairComputeDescPool_) vkDestroyDescriptorPool(device, hairComputeDescPool_, nullptr);
    if (hairComputeDescSetLayout_) vkDestroyDescriptorSetLayout(device, hairComputeDescSetLayout_, nullptr);
    hairComputePipeline_ = VK_NULL_HANDLE;
    hairComputeReady_ = false;
}

int Renderer::GenerateHairGPU(const float* parentKeys, uint32_t nParents,
                               uint32_t keysPerStrand, uint32_t childrenPerParent,
                               const float* emitterVerts, uint32_t nEmitterVerts,
                               const uint32_t* emitterTris, uint32_t nEmitterTris,
                               const float* emitterCDF,
                               float rootRadius, float tipFactor,
                               float clumpNoiseSize, float childRoundness,
                               float childLength, float avgSpacing,
                               float kinkAmplitude, float kinkFrequency,
                               float clumpFactor, float clumpShape,
                               float rough1, float rough1Size,
                               float rough2, float roughEnd,
                               uint32_t childMode,
                               float kinkShape, float kinkFlat, float kinkAmpRandom,
                               bool opaqueHair,
                               float childSizeRandom, bool useParentParticles,
                               bool precomputedStrands,
                               uint32_t blenderSeed,
                               const float* frandTable, uint32_t frandCount) {
    if (!accelBuilder_) return -1;
    if (!CreateHairComputePipeline()) return -1;

    VkDevice device = context_->GetDevice();
    uint32_t totalChildren = nParents * childrenPerParent;
    if (useParentParticles) totalChildren += nParents; // parents rendered as first nParents strands
    // Catmull-Rom subdivision: 8 sub-segments per key segment (must match shader SUBDIV)
    const uint32_t SUBDIV = 16;
    const uint32_t TIP_EXTRA = 4;  // extra tip subdivisions for curved closure
    uint32_t subdivPoints = precomputedStrands ? keysPerStrand :
        ((keysPerStrand - 1) * SUBDIV + 1 + TIP_EXTRA);
    // DOTS: ribbon A + ribbon B (both continuous strips, 4 verts per cross-section)
    uint32_t vertsPerChild = subdivPoints * 4;
    uint32_t trisPerChild = (subdivPoints - 1) * 4;
    uint32_t totalVerts = totalChildren * vertsPerChild;
    uint32_t totalIndices = totalChildren * trisPerChild * 3;

    VkDeviceSize parentSize = nParents * keysPerStrand * 4 * sizeof(float);  // vec4 per key
    VkDeviceSize posSize = totalVerts * 3 * sizeof(float);
    VkDeviceSize nrmSize = totalVerts * 3 * sizeof(float);
    VkDeviceSize uvSize  = totalVerts * 2 * sizeof(float);
    VkDeviceSize idxSize = totalIndices * sizeof(uint32_t);

    Log(L"[Hair] Generating %u children from %u parents (%u keys each)\n",
        totalChildren, nParents, keysPerStrand);
    Log(L"[Hair] Output: %u verts, %u indices (%.1f MB)\n",
        totalVerts, totalIndices, (float)(posSize + nrmSize + idxSize) / (1024*1024));

    // Create GPU buffers
    auto parentBuf = accelBuilder_->CreateAccelBuffer(parentSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto posBuf = accelBuilder_->CreateAccelBuffer(posSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto nrmBuf = accelBuilder_->CreateAccelBuffer(nrmSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto uvBuf = accelBuilder_->CreateAccelBuffer(uvSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto idxBuf = accelBuilder_->CreateAccelBuffer(idxSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Emitter mesh buffers
    VkDeviceSize emVSize = std::max((VkDeviceSize)(nEmitterVerts * 4 * sizeof(float)), (VkDeviceSize)16);
    VkDeviceSize emTSize = std::max((VkDeviceSize)(nEmitterTris * 4 * sizeof(uint32_t)), (VkDeviceSize)16);
    VkDeviceSize emCSize = std::max((VkDeviceSize)(nEmitterTris * sizeof(float)), (VkDeviceSize)16);
    auto emVertBuf = accelBuilder_->CreateAccelBuffer(emVSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto emTriBuf = accelBuilder_->CreateAccelBuffer(emTSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto emCDFBuf = accelBuilder_->CreateAccelBuffer(emCSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!parentBuf.buffer || !posBuf.buffer || !nrmBuf.buffer || !uvBuf.buffer || !idxBuf.buffer) {
        Log(L"[Hair] Failed to create GPU buffers\n");
        accelBuilder_->DestroyAccelBuffer(parentBuf);
        accelBuilder_->DestroyAccelBuffer(posBuf);
        accelBuilder_->DestroyAccelBuffer(nrmBuf);
        accelBuilder_->DestroyAccelBuffer(uvBuf);
        accelBuilder_->DestroyAccelBuffer(idxBuf);
        return -1;
    }

    // Upload parent keys to GPU (convert float3 → vec4 for alignment)
    auto stagingBuf = accelBuilder_->CreateAccelBuffer(parentSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped;
    vkMapMemory(device, stagingBuf.memory, 0, parentSize, 0, &mapped);
    float* dst = reinterpret_cast<float*>(mapped);
    for (uint32_t i = 0; i < nParents * keysPerStrand; i++) {
        dst[i * 4 + 0] = parentKeys[i * 3 + 0];
        dst[i * 4 + 1] = parentKeys[i * 3 + 1];
        dst[i * 4 + 2] = parentKeys[i * 3 + 2];
        dst[i * 4 + 3] = 0.0f;
    }
    vkUnmapMemory(device, stagingBuf.memory);

    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    VkBufferCopy copy{};
    copy.size = parentSize;
    vkCmdCopyBuffer(cmd, stagingBuf.buffer, parentBuf.buffer, 1, &copy);

    // Upload emitter mesh data
    // Upload emitter data — staging buffers must survive until EndSingleTimeCommands
    vk::AccelBuffer emVStaging{}, emTStaging{}, emCStaging{};
    if (nEmitterVerts > 0 && emitterVerts) {
        emVStaging = accelBuilder_->CreateAccelBuffer(emVSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* emVM;
        vkMapMemory(device, emVStaging.memory, 0, emVSize, 0, &emVM);
        float* evDst = reinterpret_cast<float*>(emVM);
        for (uint32_t i = 0; i < nEmitterVerts; i++) {
            evDst[i*4+0] = emitterVerts[i*3+0];
            evDst[i*4+1] = emitterVerts[i*3+1];
            evDst[i*4+2] = emitterVerts[i*3+2];
            evDst[i*4+3] = 0.0f;
        }
        vkUnmapMemory(device, emVStaging.memory);
        copy.size = emVSize;
        vkCmdCopyBuffer(cmd, emVStaging.buffer, emVertBuf.buffer, 1, &copy);
    }
    if (nEmitterTris > 0 && emitterTris) {
        emTStaging = accelBuilder_->CreateAccelBuffer(emTSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* emTM;
        vkMapMemory(device, emTStaging.memory, 0, emTSize, 0, &emTM);
        uint32_t* etDst = reinterpret_cast<uint32_t*>(emTM);
        for (uint32_t i = 0; i < nEmitterTris; i++) {
            etDst[i*4+0] = emitterTris[i*3+0];
            etDst[i*4+1] = emitterTris[i*3+1];
            etDst[i*4+2] = emitterTris[i*3+2];
            etDst[i*4+3] = 0;
        }
        vkUnmapMemory(device, emTStaging.memory);
        copy.size = emTSize;
        vkCmdCopyBuffer(cmd, emTStaging.buffer, emTriBuf.buffer, 1, &copy);
    }
    if (nEmitterTris > 0 && emitterCDF) {
        emCStaging = accelBuilder_->CreateAccelBuffer(emCSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* emCM;
        vkMapMemory(device, emCStaging.memory, 0, emCSize, 0, &emCM);
        memcpy(emCM, emitterCDF, nEmitterTris * sizeof(float));
        vkUnmapMemory(device, emCStaging.memory);
        copy.size = emCSize;
        vkCmdCopyBuffer(cmd, emCStaging.buffer, emCDFBuf.buffer, 1, &copy);
    }

    // Upload Blender frand table (1024 floats = 4KB)
    VkDeviceSize frandSize = std::max((VkDeviceSize)(frandCount * sizeof(float)), (VkDeviceSize)16);
    auto frandBuf = accelBuilder_->CreateAccelBuffer(frandSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk::AccelBuffer frandStaging{};
    if (frandTable && frandCount > 0) {
        frandStaging = accelBuilder_->CreateAccelBuffer(frandSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* frandM;
        vkMapMemory(device, frandStaging.memory, 0, frandSize, 0, &frandM);
        memcpy(frandM, frandTable, frandCount * sizeof(float));
        vkUnmapMemory(device, frandStaging.memory);
        copy.size = frandSize;
        vkCmdCopyBuffer(cmd, frandStaging.buffer, frandBuf.buffer, 1, &copy);
    }

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    // Update descriptor set (9 bindings matching shader layout)
    VkDescriptorBufferInfo bufInfos[9] = {
        {parentBuf.buffer, 0, parentSize},      // binding 0: parent keys
        {emVertBuf.buffer, 0, emVSize},          // binding 1: emitter verts
        {emTriBuf.buffer, 0, emTSize},           // binding 2: emitter tris
        {emCDFBuf.buffer, 0, emCSize},           // binding 3: emitter CDF
        {posBuf.buffer, 0, posSize},             // binding 4: output positions
        {nrmBuf.buffer, 0, nrmSize},             // binding 5: output normals
        {idxBuf.buffer, 0, idxSize},             // binding 6: output indices
        {uvBuf.buffer, 0, uvSize},               // binding 7: output UVs
        {frandBuf.buffer, 0, frandSize},         // binding 8: Blender frand table
    };
    VkWriteDescriptorSet writes[9] = {};
    for (int i = 0; i < 9; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = hairComputeDescSet_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(device, 9, writes, 0, nullptr);

    // Dispatch compute shader
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hairComputePipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        hairComputePipelineLayout_, 0, 1, &hairComputeDescSet_, 0, nullptr);

    struct {
        uint32_t nParents;
        uint32_t keysPerStrand;
        uint32_t totalChildren;
        uint32_t nEmitterTris;
        float rootRadius;
        float tipFactor;
        float clumpNoiseSize, childRoundness, childLength;
        float avgSpacing;
        float kinkAmplitude;
        float kinkFrequency;
        float clumpFactor;
        float clumpShape;
        float rough1;
        float rough1Size;
        float rough2;
        float roughEnd;
        uint32_t childMode;
        float kinkShape;
        float kinkFlat;
        float kinkAmpRandom;
        float childSizeRandom;
        uint32_t useParentParticles;
        uint32_t precomputedStrands;
        uint32_t blenderSeed;
    } pc = {nParents, keysPerStrand, totalChildren, nEmitterTris,
            rootRadius, tipFactor, clumpNoiseSize, childRoundness, childLength, avgSpacing,
            kinkAmplitude, kinkFrequency,
            clumpFactor, clumpShape, rough1, rough1Size, rough2, roughEnd,
            childMode, kinkShape, kinkFlat, kinkAmpRandom,
            childSizeRandom, useParentParticles ? 1u : 0u,
            precomputedStrands ? 1u : 0u, blenderSeed};
    vkCmdPushConstants(cmd, hairComputePipelineLayout_,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t groups = (totalChildren + 63) / 64;
    vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier: compute → BLAS build
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    // Record BLAS build into the same command buffer to avoid a second
    // submit + queue idle per hair system during staged load.
    int blasIdx = accelBuilder_->BuildBLASFromGPUBuffers(
        posBuf.deviceAddress, totalVerts,
        idxBuf.deviceAddress, totalIndices,
        -1, cmd, opaqueHair);

    context_->EndSingleTimeCommands(cmd);
    // Safe to destroy staging now (GPU finished)
    accelBuilder_->DestroyAccelBuffer(stagingBuf);
    if (emVStaging.buffer) accelBuilder_->DestroyAccelBuffer(emVStaging);
    if (emTStaging.buffer) accelBuilder_->DestroyAccelBuffer(emTStaging);
    if (emCStaging.buffer) accelBuilder_->DestroyAccelBuffer(emCStaging);
    if (frandStaging.buffer) accelBuilder_->DestroyAccelBuffer(frandStaging);

    if (blasIdx >= 0) {
        // Store normal buffer address in BLAS entry for shader access
        auto& blas = const_cast<vk::BLAS&>(accelBuilder_->GetBLASList()[blasIdx]);
        blas.normalBuf = nrmBuf;
        blas.uvBuf = uvBuf;
        blas.vertexBuf = posBuf;
        blas.indexBuf = idxBuf;
        blas.vertexCount = totalVerts;
        blas.indexCount = totalIndices;
        blas.isHair = true;
        Log(L"[Hair] BLAS %d built: %u verts, %u tris\n",
            blasIdx, totalVerts, totalIndices / 3);
    } else {
        Log(L"[Hair] BLAS build failed\n");
        accelBuilder_->DestroyAccelBuffer(posBuf);
        accelBuilder_->DestroyAccelBuffer(nrmBuf);
        accelBuilder_->DestroyAccelBuffer(uvBuf);
        accelBuilder_->DestroyAccelBuffer(idxBuf);
    }

    accelBuilder_->DestroyAccelBuffer(parentBuf);
    accelBuilder_->DestroyAccelBuffer(emVertBuf);
    accelBuilder_->DestroyAccelBuffer(emTriBuf);
    accelBuilder_->DestroyAccelBuffer(emCDFBuf);
    return blasIdx;
}

bool Renderer::CreateHairContourPipeline() {
    VkDevice device = context_->GetDevice();
    if (!rtResources_) return false;

    // Reuse the RT pipeline's descriptor set layout (already has binding 34 for hairV)
    VkDescriptorSetLayout rtDescSetLayout = rtResources_->GetDescriptorSetLayout();

    // Pipeline layout with push constants (width, height)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(uint32_t) * 2;  // width, height

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &rtDescSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &hairContourPipelineLayout_) != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create hair contour pipeline layout\n");
        return false;
    }

    // Load compute shader
    std::ifstream file(IgnisResolvePath("shaders/hair_contour.comp.spv"), std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[VK Renderer] WARNING: hair_contour.comp.spv not found\n");
        return false;
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> code(fileSize);
    file.seekg(0);
    file.read(code.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = code.size();
    moduleInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create hair contour shader module\n");
        return false;
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = hairContourPipelineLayout_;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &hairContourPipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create hair contour pipeline\n");
        return false;
    }

    hairContourReady_ = true;
    Log(L"[VK Renderer] Hair contour pipeline created\n");
    return true;
}

void Renderer::ShutdownHairContour() {
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) return;

    if (hairContourPipeline_) { vkDestroyPipeline(device, hairContourPipeline_, nullptr); hairContourPipeline_ = VK_NULL_HANDLE; }
    if (hairContourPipelineLayout_) { vkDestroyPipelineLayout(device, hairContourPipelineLayout_, nullptr); hairContourPipelineLayout_ = VK_NULL_HANDLE; }
    hairContourReady_ = false;
}

} // namespace vk
} // namespace acpt
