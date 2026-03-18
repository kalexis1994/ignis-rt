#include "vk_rasterizer.h"
#include "vk_context.h"
#include "vk_pipeline.h"
#include "vk_geometry.h"
#include "../../include/ignis_log.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace acpt {
namespace vk {

// ============================================================
// Helper: build a lookAt view matrix (column-major)
// ============================================================
static void BuildLookAt(float* out, float eyeX, float eyeY, float eyeZ,
                        float targetX, float targetY, float targetZ) {
    float zAxis[3] = {eyeX - targetX, eyeY - targetY, eyeZ - targetZ};
    float len = sqrtf(zAxis[0]*zAxis[0] + zAxis[1]*zAxis[1] + zAxis[2]*zAxis[2]);
    if (len < 1e-6f) len = 1e-6f;
    zAxis[0] /= len; zAxis[1] /= len; zAxis[2] /= len;

    float worldUp[3] = {0, 1, 0};
    // If looking straight up/down, use a different up vector
    if (fabsf(zAxis[1]) > 0.99f) { worldUp[0] = 0; worldUp[1] = 0; worldUp[2] = 1; }

    float xAxis[3] = {
        worldUp[1]*zAxis[2] - worldUp[2]*zAxis[1],
        worldUp[2]*zAxis[0] - worldUp[0]*zAxis[2],
        worldUp[0]*zAxis[1] - worldUp[1]*zAxis[0]
    };
    len = sqrtf(xAxis[0]*xAxis[0] + xAxis[1]*xAxis[1] + xAxis[2]*xAxis[2]);
    if (len < 1e-6f) len = 1e-6f;
    xAxis[0] /= len; xAxis[1] /= len; xAxis[2] /= len;

    float yAxis[3] = {
        zAxis[1]*xAxis[2] - zAxis[2]*xAxis[1],
        zAxis[2]*xAxis[0] - zAxis[0]*xAxis[2],
        zAxis[0]*xAxis[1] - zAxis[1]*xAxis[0]
    };

    memset(out, 0, sizeof(float)*16);
    out[0] = xAxis[0]; out[1] = yAxis[0]; out[2] = zAxis[0];
    out[4] = xAxis[1]; out[5] = yAxis[1]; out[6] = zAxis[1];
    out[8] = xAxis[2]; out[9] = yAxis[2]; out[10] = zAxis[2];
    out[12] = -(xAxis[0]*eyeX + xAxis[1]*eyeY + xAxis[2]*eyeZ);
    out[13] = -(yAxis[0]*eyeX + yAxis[1]*eyeY + yAxis[2]*eyeZ);
    out[14] = -(zAxis[0]*eyeX + zAxis[1]*eyeY + zAxis[2]*eyeZ);
    out[15] = 1.0f;
}

// ============================================================
// Helper: multiply two 4x4 matrices (column-major), result = A * B
// ============================================================
static void Mat4Multiply(float* result, const float* A, const float* B) {
    float tmp[16];
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            tmp[c*4+r] = A[0*4+r]*B[c*4+0] + A[1*4+r]*B[c*4+1] +
                         A[2*4+r]*B[c*4+2] + A[3*4+r]*B[c*4+3];
        }
    }
    memcpy(result, tmp, sizeof(float)*16);
}

// ============================================================
// Initialization
// ============================================================
bool Rasterizer::Initialize(Context* context, Pipeline* pipeline, Geometry* geometry) {
    context_ = context;
    pipeline_ = pipeline;
    geometry_ = geometry;

    if (!CreateDepthResources()) return false;
    if (!CreateFramebuffers()) return false;
    if (!CreateShadowResources()) return false;
    if (!CreateDummyAlbedoTexture()) return false;
    if (!CreateDummyNormalTexture()) return false;
    if (!CreateDummySpecTexture()) return false;
    if (!CreateDummyBarkAlbedoTexture()) return false;
    if (!CreateDummyBarkNormalTexture()) return false;
    if (!CreateDummyBarkSpecTexture()) return false;
    if (!CreateUniformBuffers()) return false;
    if (!CreateDescriptorPool()) return false;
    if (!CreateDescriptorSets()) return false;

    Log(L"[VK Rasterizer] Initialized with shadow mapping\n");
    return true;
}

void Rasterizer::Shutdown() {
    if (context_->GetDevice() != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(context_->GetDevice());

        // Billboard offscreen resources
        DestroyBillboardResources();

        // Shadow resources
        if (shadowPipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(context_->GetDevice(), shadowPipeline_, nullptr);
        if (shadowPipelineLayout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(context_->GetDevice(), shadowPipelineLayout_, nullptr);
        if (shadowDescSetLayout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(context_->GetDevice(), shadowDescSetLayout_, nullptr);
        if (shadowDescPool_ != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(context_->GetDevice(), shadowDescPool_, nullptr);
        if (shadowFramebuffer_ != VK_NULL_HANDLE)
            vkDestroyFramebuffer(context_->GetDevice(), shadowFramebuffer_, nullptr);
        if (shadowRenderPass_ != VK_NULL_HANDLE)
            vkDestroyRenderPass(context_->GetDevice(), shadowRenderPass_, nullptr);
        if (shadowMapSampler_ != VK_NULL_HANDLE)
            vkDestroySampler(context_->GetDevice(), shadowMapSampler_, nullptr);
        if (shadowMapView_ != VK_NULL_HANDLE)
            vkDestroyImageView(context_->GetDevice(), shadowMapView_, nullptr);
        if (shadowMapImage_ != VK_NULL_HANDLE)
            vkDestroyImage(context_->GetDevice(), shadowMapImage_, nullptr);
        if (shadowMapMemory_ != VK_NULL_HANDLE)
            vkFreeMemory(context_->GetDevice(), shadowMapMemory_, nullptr);

        for (size_t i = 0; i < shadowUniformBuffers_.size(); i++) {
            vkDestroyBuffer(context_->GetDevice(), shadowUniformBuffers_[i], nullptr);
            vkFreeMemory(context_->GetDevice(), shadowUniformBuffersMemory_[i], nullptr);
        }

        // Uploaded textures
        DestroyUploadedAlbedo();
        DestroyUploadedNormal();
        DestroyUploadedSpec();
        DestroyUploadedBarkAlbedo();
        DestroyUploadedBarkNormal();
        DestroyUploadedBarkSpec();

        // Dummy albedo texture
        if (dummyAlbedoSampler_ != VK_NULL_HANDLE)
            vkDestroySampler(context_->GetDevice(), dummyAlbedoSampler_, nullptr);
        if (dummyAlbedoView_ != VK_NULL_HANDLE)
            vkDestroyImageView(context_->GetDevice(), dummyAlbedoView_, nullptr);
        if (dummyAlbedoImage_ != VK_NULL_HANDLE)
            vkDestroyImage(context_->GetDevice(), dummyAlbedoImage_, nullptr);
        if (dummyAlbedoMemory_ != VK_NULL_HANDLE)
            vkFreeMemory(context_->GetDevice(), dummyAlbedoMemory_, nullptr);

        // Dummy normal texture
        if (dummyNormalSampler_ != VK_NULL_HANDLE)
            vkDestroySampler(context_->GetDevice(), dummyNormalSampler_, nullptr);
        if (dummyNormalView_ != VK_NULL_HANDLE)
            vkDestroyImageView(context_->GetDevice(), dummyNormalView_, nullptr);
        if (dummyNormalImage_ != VK_NULL_HANDLE)
            vkDestroyImage(context_->GetDevice(), dummyNormalImage_, nullptr);
        if (dummyNormalMemory_ != VK_NULL_HANDLE)
            vkFreeMemory(context_->GetDevice(), dummyNormalMemory_, nullptr);

        // Dummy specular texture
        if (dummySpecSampler_ != VK_NULL_HANDLE)
            vkDestroySampler(context_->GetDevice(), dummySpecSampler_, nullptr);
        if (dummySpecView_ != VK_NULL_HANDLE)
            vkDestroyImageView(context_->GetDevice(), dummySpecView_, nullptr);
        if (dummySpecImage_ != VK_NULL_HANDLE)
            vkDestroyImage(context_->GetDevice(), dummySpecImage_, nullptr);
        if (dummySpecMemory_ != VK_NULL_HANDLE)
            vkFreeMemory(context_->GetDevice(), dummySpecMemory_, nullptr);

        // Dummy bark albedo texture
        if (dummyBarkAlbedoSampler_ != VK_NULL_HANDLE)
            vkDestroySampler(context_->GetDevice(), dummyBarkAlbedoSampler_, nullptr);
        if (dummyBarkAlbedoView_ != VK_NULL_HANDLE)
            vkDestroyImageView(context_->GetDevice(), dummyBarkAlbedoView_, nullptr);
        if (dummyBarkAlbedoImage_ != VK_NULL_HANDLE)
            vkDestroyImage(context_->GetDevice(), dummyBarkAlbedoImage_, nullptr);
        if (dummyBarkAlbedoMemory_ != VK_NULL_HANDLE)
            vkFreeMemory(context_->GetDevice(), dummyBarkAlbedoMemory_, nullptr);

        // Dummy bark normal texture
        if (dummyBarkNormalSampler_ != VK_NULL_HANDLE)
            vkDestroySampler(context_->GetDevice(), dummyBarkNormalSampler_, nullptr);
        if (dummyBarkNormalView_ != VK_NULL_HANDLE)
            vkDestroyImageView(context_->GetDevice(), dummyBarkNormalView_, nullptr);
        if (dummyBarkNormalImage_ != VK_NULL_HANDLE)
            vkDestroyImage(context_->GetDevice(), dummyBarkNormalImage_, nullptr);
        if (dummyBarkNormalMemory_ != VK_NULL_HANDLE)
            vkFreeMemory(context_->GetDevice(), dummyBarkNormalMemory_, nullptr);

        // Dummy bark specular texture
        if (dummyBarkSpecSampler_ != VK_NULL_HANDLE)
            vkDestroySampler(context_->GetDevice(), dummyBarkSpecSampler_, nullptr);
        if (dummyBarkSpecView_ != VK_NULL_HANDLE)
            vkDestroyImageView(context_->GetDevice(), dummyBarkSpecView_, nullptr);
        if (dummyBarkSpecImage_ != VK_NULL_HANDLE)
            vkDestroyImage(context_->GetDevice(), dummyBarkSpecImage_, nullptr);
        if (dummyBarkSpecMemory_ != VK_NULL_HANDLE)
            vkFreeMemory(context_->GetDevice(), dummyBarkSpecMemory_, nullptr);

        // Main resources
        for (size_t i = 0; i < uniformBuffers_.size(); i++) {
            vkDestroyBuffer(context_->GetDevice(), uniformBuffers_[i], nullptr);
            vkFreeMemory(context_->GetDevice(), uniformBuffersMemory_[i], nullptr);
        }

        if (descriptorPool_ != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(context_->GetDevice(), descriptorPool_, nullptr);

        for (auto framebuffer : framebuffers_)
            vkDestroyFramebuffer(context_->GetDevice(), framebuffer, nullptr);

        if (depthImageView_ != VK_NULL_HANDLE)
            vkDestroyImageView(context_->GetDevice(), depthImageView_, nullptr);
        if (depthImage_ != VK_NULL_HANDLE)
            vkDestroyImage(context_->GetDevice(), depthImage_, nullptr);
        if (depthImageMemory_ != VK_NULL_HANDLE)
            vkFreeMemory(context_->GetDevice(), depthImageMemory_, nullptr);
    }

    Log(L"[VK Rasterizer] Shutdown\n");
}

void Rasterizer::SetScene(Mesh* sphere, Mesh* plane) {
    sphereMesh_ = sphere;
    planeMesh_ = plane;
}

void Rasterizer::SetDrawCalls(const std::vector<RasterDrawCall>& draws) {
    drawCalls_ = draws;
}

// ============================================================
// Shadow mapping - compute light matrices
// ============================================================
void Rasterizer::ComputeLightMatrices(float* outLightView, float* outLightProj, float* outLightViewProj) {
    // Light direction (same as UBO: towards surface)
    float ld[3] = {0.4f, -0.7f, 0.3f};
    float len = sqrtf(ld[0]*ld[0] + ld[1]*ld[1] + ld[2]*ld[2]);
    ld[0] /= len; ld[1] /= len; ld[2] /= len;

    // Scene center (tree center)
    float centerX = 0.0f, centerY = cameraTargetY_, centerZ = 0.0f;

    // Light position: far along negative light direction from scene center
    float lightDist = 25.0f;
    float lightPosX = centerX - ld[0] * lightDist;
    float lightPosY = centerY - ld[1] * lightDist;
    float lightPosZ = centerZ - ld[2] * lightDist;

    // Light view matrix
    BuildLookAt(outLightView, lightPosX, lightPosY, lightPosZ, centerX, centerY, centerZ);

    // Orthographic projection (Vulkan: Y-flipped, Z in [0,1])
    float halfSize = 12.0f;
    float nearP = 0.1f, farP = 50.0f;

    memset(outLightProj, 0, sizeof(float)*16);
    outLightProj[0]  = 1.0f / halfSize;
    outLightProj[5]  = -1.0f / halfSize;          // Y flip for Vulkan
    outLightProj[10] = 1.0f / (nearP - farP);     // maps [near,far] to [0,1]
    outLightProj[14] = nearP / (nearP - farP);
    outLightProj[15] = 1.0f;

    // lightViewProj = lightProj * lightView
    Mat4Multiply(outLightViewProj, outLightProj, outLightView);
}

// ============================================================
// Shadow pass rendering
// ============================================================
void Rasterizer::RenderShadowPass(VkCommandBuffer commandBuffer) {
    // Update shadow UBO with light matrices
    float lightView[16], lightProj[16], lightViewProj[16];
    ComputeLightMatrices(lightView, lightProj, lightViewProj);

    // Write light view/proj to shadow UBO (same struct layout, view/proj slots)
    // We reuse all swapchain images' shadow UBOs (index 0 is fine since we only need one)
    for (size_t i = 0; i < shadowUniformBuffers_.size(); i++) {
        UniformBufferObject shadowUbo{};
        memcpy(shadowUbo.view, lightView, sizeof(lightView));
        memcpy(shadowUbo.projection, lightProj, sizeof(lightProj));
        memcpy(shadowUniformBuffersMapped_[i], &shadowUbo, sizeof(shadowUbo));
    }

    // Begin shadow render pass
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = shadowRenderPass_;
    renderPassInfo.framebuffer = shadowFramebuffer_;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};

    VkClearValue clearValue;
    clearValue.depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearValue;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_);

    // Set viewport and scissor for shadow map
    VkViewport viewport{};
    viewport.width = (float)SHADOW_MAP_SIZE;
    viewport.height = (float)SHADOW_MAP_SIZE;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // Bind shadow descriptor set (uses first available, all have same data)
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelineLayout_,
                           0, 1, &shadowDescSets_[0], 0, nullptr);

    // Draw all geometry into shadow map
    if (!drawCalls_.empty()) {
        for (const auto& draw : drawCalls_) {
            if (!draw.mesh || draw.mesh->vertexBuffer == VK_NULL_HANDLE) continue;

            RasterPushConstants push{};
            memcpy(push.model, draw.modelMatrix, sizeof(push.model));
            memcpy(push.color, draw.color, sizeof(push.color));

            vkCmdPushConstants(commandBuffer, shadowPipelineLayout_,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(RasterPushConstants), &push);

            VkBuffer vertexBuffers[] = {draw.mesh->vertexBuffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, draw.mesh->indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(draw.mesh->indices.size()), 1, 0, 0, 0);
        }
    }

    vkCmdEndRenderPass(commandBuffer);

    // Barrier: shadow map depth writes -> fragment shader reads
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = shadowMapImage_;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// ============================================================
// Main render
// ============================================================
void Rasterizer::Render(uint32_t imageIndex, VkCommandBuffer commandBuffer) {
    // 1) Shadow pass
    RenderShadowPass(commandBuffer);

    // 2) Main pass
    UpdateUniformBuffer(imageIndex);

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = pipeline_->GetRenderPass();
    renderPassInfo.framebuffer = framebuffers_[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = context_->GetSwapchainExtent();

    VkClearValue clearValues[2];
    clearValues[0].color = {{0.45f, 0.60f, 0.82f, 1.0f}};  // Light sky blue
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->GetPipeline());

    // Set dynamic viewport/scissor
    VkViewport vp{};
    VkRect2D sc{};
    if (viewportOverride_) {
        vp.x = viewportX_;
        vp.y = viewportY_;
        vp.width = viewportW_;
        vp.height = viewportH_;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        sc.offset = {(int32_t)viewportX_, (int32_t)viewportY_};
        sc.extent = {(uint32_t)viewportW_, (uint32_t)viewportH_};
    } else {
        vp.x = 0.0f;
        vp.y = 0.0f;
        vp.width = (float)context_->GetSwapchainExtent().width;
        vp.height = (float)context_->GetSwapchainExtent().height;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        sc.offset = {0, 0};
        sc.extent = context_->GetSwapchainExtent();
    }
    vkCmdSetViewport(commandBuffer, 0, 1, &vp);
    vkCmdSetScissor(commandBuffer, 0, 1, &sc);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->GetPipelineLayout(),
                           0, 1, &descriptorSets_[imageIndex], 0, nullptr);

    // If draw calls list is populated, use it; otherwise fall back to legacy scene
    if (!drawCalls_.empty()) {
        for (const auto& draw : drawCalls_) {
            if (!draw.mesh || draw.mesh->vertexBuffer == VK_NULL_HANDLE) continue;

            RasterPushConstants push{};
            memcpy(push.model, draw.modelMatrix, sizeof(push.model));
            memcpy(push.color, draw.color, sizeof(push.color));

            vkCmdPushConstants(commandBuffer, pipeline_->GetPipelineLayout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(RasterPushConstants), &push);

            VkBuffer vertexBuffers[] = {draw.mesh->vertexBuffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, draw.mesh->indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(draw.mesh->indices.size()), 1, 0, 0, 0);
        }
    } else {
        // Legacy path: identity model + white color for sphere/plane
        RasterPushConstants push{};
        memset(push.model, 0, sizeof(push.model));
        push.model[0] = push.model[5] = push.model[10] = push.model[15] = 1.0f;
        push.color[0] = 0.7f; push.color[1] = 0.7f; push.color[2] = 0.8f; push.color[3] = 1.0f;

        vkCmdPushConstants(commandBuffer, pipeline_->GetPipelineLayout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(RasterPushConstants), &push);

        if (sphereMesh_ && sphereMesh_->vertexBuffer != VK_NULL_HANDLE) {
            VkBuffer vertexBuffers[] = {sphereMesh_->vertexBuffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, sphereMesh_->indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(sphereMesh_->indices.size()), 1, 0, 0, 0);
        }

        push.color[0] = 0.9f; push.color[1] = 0.9f; push.color[2] = 0.95f;
        vkCmdPushConstants(commandBuffer, pipeline_->GetPipelineLayout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(RasterPushConstants), &push);

        if (planeMesh_ && planeMesh_->vertexBuffer != VK_NULL_HANDLE) {
            VkBuffer vertexBuffers[] = {planeMesh_->vertexBuffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, planeMesh_->indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(planeMesh_->indices.size()), 1, 0, 0, 0);
        }
    }

    vkCmdEndRenderPass(commandBuffer);
}

void Rasterizer::UpdateCamera(float distance, float angle, float targetY, float pitch) {
    cameraDistance_ = distance;
    cameraAngle_ = angle;
    cameraTargetY_ = targetY;
    cameraPitch_ = pitch;
}

void Rasterizer::SetViewportOverride(float x, float y, float w, float h) {
    viewportX_ = x;
    viewportY_ = y;
    viewportW_ = w;
    viewportH_ = h;
    viewportOverride_ = true;
}

void Rasterizer::ClearViewportOverride() {
    viewportOverride_ = false;
}

// ============================================================
// Camera depth resources (existing)
// ============================================================
bool Rasterizer::CreateDepthResources() {
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = context_->GetSwapchainExtent().width;
    imageInfo.extent.height = context_->GetSwapchainExtent().height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = depthFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(context_->GetDevice(), &imageInfo, nullptr, &depthImage_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create depth image\n");
        return false;
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(context_->GetDevice(), depthImage_, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(context_->GetDevice(), &allocInfo, nullptr, &depthImageMemory_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to allocate depth image memory\n");
        return false;
    }

    vkBindImageMemory(context_->GetDevice(), depthImage_, depthImageMemory_, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = depthImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = depthFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(context_->GetDevice(), &viewInfo, nullptr, &depthImageView_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create depth image view\n");
        return false;
    }

    Log(L"[VK Rasterizer] Depth resources created\n");
    return true;
}

bool Rasterizer::CreateFramebuffers() {
    const auto& swapchainImageViews = context_->GetSwapchainImageViews();
    framebuffers_.resize(swapchainImageViews.size());

    for (size_t i = 0; i < swapchainImageViews.size(); i++) {
        VkImageView attachments[] = {swapchainImageViews[i], depthImageView_};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = pipeline_->GetRenderPass();
        framebufferInfo.attachmentCount = 2;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = context_->GetSwapchainExtent().width;
        framebufferInfo.height = context_->GetSwapchainExtent().height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(context_->GetDevice(), &framebufferInfo, nullptr, &framebuffers_[i]) != VK_SUCCESS) {
            Log(L"[VK Rasterizer] ERROR: Failed to create framebuffer %zu\n", i);
            return false;
        }
    }

    Log(L"[VK Rasterizer] Framebuffers created\n");
    return true;
}

// ============================================================
// Shadow map resources
// ============================================================
// ============================================================
// Dummy 1x1 white albedo texture (default when no texture bound)
// ============================================================
bool Rasterizer::CreateDummyAlbedoTexture() {
    VkDevice device = context_->GetDevice();

    // Create 1x1 RGBA8 image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {1, 1, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &dummyAlbedoImage_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create dummy albedo image\n");
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, dummyAlbedoImage_, &memReq);

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize = memReq.size;
    memAlloc.memoryTypeIndex = context_->FindMemoryType(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &memAlloc, nullptr, &dummyAlbedoMemory_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to allocate dummy albedo memory\n");
        return false;
    }
    vkBindImageMemory(device, dummyAlbedoImage_, dummyAlbedoMemory_, 0);

    // Write white pixel
    void* mapped;
    vkMapMemory(device, dummyAlbedoMemory_, 0, memReq.size, 0, &mapped);
    uint32_t white = 0xFFFFFFFF;
    memcpy(mapped, &white, 4);
    vkUnmapMemory(device, dummyAlbedoMemory_);

    // Transition to SHADER_READ_ONLY_OPTIMAL
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = context_->GetCommandPool();
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = dummyAlbedoImage_;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(context_->GetGraphicsQueue());
    vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

    // Image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = dummyAlbedoImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &dummyAlbedoView_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create dummy albedo image view\n");
        return false;
    }

    // Sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &dummyAlbedoSampler_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create dummy albedo sampler\n");
        return false;
    }

    // Start with dummy as active
    activeAlbedoView_ = VK_NULL_HANDLE;
    activeAlbedoSampler_ = VK_NULL_HANDLE;

    Log(L"[VK Rasterizer] Dummy albedo texture created (1x1 white)\n");
    return true;
}

void Rasterizer::SetAlbedoTexture(VkImageView imageView, VkSampler sampler) {
    activeAlbedoView_ = imageView;
    activeAlbedoSampler_ = sampler;

    // Update binding 2 in all descriptor sets
    VkDevice device = context_->GetDevice();
    for (size_t i = 0; i < descriptorSets_.size(); i++) {
        VkDescriptorImageInfo albedoInfo{};
        albedoInfo.sampler = sampler;
        albedoInfo.imageView = imageView;
        albedoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSets_[i];
        write.dstBinding = 2;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &albedoInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

void Rasterizer::ClearAlbedoTexture() {
    activeAlbedoView_ = VK_NULL_HANDLE;
    activeAlbedoSampler_ = VK_NULL_HANDLE;

    // Revert to dummy
    VkDevice device = context_->GetDevice();
    for (size_t i = 0; i < descriptorSets_.size(); i++) {
        VkDescriptorImageInfo albedoInfo{};
        albedoInfo.sampler = dummyAlbedoSampler_;
        albedoInfo.imageView = dummyAlbedoView_;
        albedoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSets_[i];
        write.dstBinding = 2;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &albedoInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

void Rasterizer::DestroyUploadedAlbedo() {
    VkDevice device = context_->GetDevice();
    if (uploadedAlbedoSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device, uploadedAlbedoSampler_, nullptr);
        uploadedAlbedoSampler_ = VK_NULL_HANDLE;
    }
    if (uploadedAlbedoView_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device, uploadedAlbedoView_, nullptr);
        uploadedAlbedoView_ = VK_NULL_HANDLE;
    }
    if (uploadedAlbedoImage_ != VK_NULL_HANDLE) {
        vkDestroyImage(device, uploadedAlbedoImage_, nullptr);
        uploadedAlbedoImage_ = VK_NULL_HANDLE;
    }
    if (uploadedAlbedoMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, uploadedAlbedoMemory_, nullptr);
        uploadedAlbedoMemory_ = VK_NULL_HANDLE;
    }
}

bool Rasterizer::UploadAlbedoFromPixels(const uint8_t* rgba, int width, int height) {
    if (!context_ || !rgba || width <= 0 || height <= 0) return false;

    // Destroy previous uploaded texture
    DestroyUploadedAlbedo();

    VkDevice device = context_->GetDevice();
    VkDeviceSize imageSize = (VkDeviceSize)width * height * 4;

    // Create staging buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = imageSize;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufInfo, nullptr, &stagingBuffer) != VK_SUCCESS)
        return false;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &memReq);

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize = memReq.size;
    memAlloc.memoryTypeIndex = context_->FindMemoryType(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &memAlloc, nullptr, &stagingMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        return false;
    }
    vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

    // Copy pixels to staging
    void* mapped;
    vkMapMemory(device, stagingMemory, 0, imageSize, 0, &mapped);
    memcpy(mapped, rgba, imageSize);
    vkUnmapMemory(device, stagingMemory);

    // Create VkImage (OPTIMAL tiling for GPU sampling)
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {(uint32_t)width, (uint32_t)height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &uploadedAlbedoImage_) != VK_SUCCESS) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
        return false;
    }

    VkMemoryRequirements imgMemReq;
    vkGetImageMemoryRequirements(device, uploadedAlbedoImage_, &imgMemReq);

    VkMemoryAllocateInfo imgMemAlloc{};
    imgMemAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imgMemAlloc.allocationSize = imgMemReq.size;
    imgMemAlloc.memoryTypeIndex = context_->FindMemoryType(imgMemReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &imgMemAlloc, nullptr, &uploadedAlbedoMemory_) != VK_SUCCESS) {
        vkDestroyImage(device, uploadedAlbedoImage_, nullptr);
        uploadedAlbedoImage_ = VK_NULL_HANDLE;
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
        return false;
    }
    vkBindImageMemory(device, uploadedAlbedoImage_, uploadedAlbedoMemory_, 0);

    // Transition UNDEFINED -> TRANSFER_DST, copy, then TRANSFER_DST -> SHADER_READ
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = context_->GetCommandPool();
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Barrier: UNDEFINED -> TRANSFER_DST
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = uploadedAlbedoImage_;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Copy buffer to image
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {(uint32_t)width, (uint32_t)height, 1};
    vkCmdCopyBufferToImage(cmd, stagingBuffer, uploadedAlbedoImage_,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Barrier: TRANSFER_DST -> SHADER_READ_ONLY
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(context_->GetGraphicsQueue());
    vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

    // Destroy staging
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = uploadedAlbedoImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &uploadedAlbedoView_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create uploaded albedo image view\n");
        return false;
    }

    // Create sampler (clamp to edge for leaf textures)
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &uploadedAlbedoSampler_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create uploaded albedo sampler\n");
        return false;
    }

    // Bind as active albedo texture
    SetAlbedoTexture(uploadedAlbedoView_, uploadedAlbedoSampler_);

    Log(L"[VK Rasterizer] Uploaded leaf albedo texture %dx%d\n", width, height);
    return true;
}

// ============================================================
// Generic texture upload helper (shared by albedo/normal/specular)
// ============================================================
bool Rasterizer::UploadTextureFromPixels(const uint8_t* rgba, int width, int height,
    VkImage& outImage, VkDeviceMemory& outMemory,
    VkImageView& outView, VkSampler& outSampler,
    VkSamplerAddressMode addressMode) {
    if (!context_ || !rgba || width <= 0 || height <= 0) return false;

    VkDevice device = context_->GetDevice();
    VkDeviceSize imageSize = (VkDeviceSize)width * height * 4;

    // Create staging buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = imageSize;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufInfo, nullptr, &stagingBuffer) != VK_SUCCESS)
        return false;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &memReq);

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize = memReq.size;
    memAlloc.memoryTypeIndex = context_->FindMemoryType(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &memAlloc, nullptr, &stagingMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        return false;
    }
    vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

    void* mapped;
    vkMapMemory(device, stagingMemory, 0, imageSize, 0, &mapped);
    memcpy(mapped, rgba, imageSize);
    vkUnmapMemory(device, stagingMemory);

    // Create VkImage
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {(uint32_t)width, (uint32_t)height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &outImage) != VK_SUCCESS) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
        return false;
    }

    VkMemoryRequirements imgMemReq;
    vkGetImageMemoryRequirements(device, outImage, &imgMemReq);

    VkMemoryAllocateInfo imgMemAlloc{};
    imgMemAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imgMemAlloc.allocationSize = imgMemReq.size;
    imgMemAlloc.memoryTypeIndex = context_->FindMemoryType(imgMemReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &imgMemAlloc, nullptr, &outMemory) != VK_SUCCESS) {
        vkDestroyImage(device, outImage, nullptr);
        outImage = VK_NULL_HANDLE;
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
        return false;
    }
    vkBindImageMemory(device, outImage, outMemory, 0);

    // Transfer
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = context_->GetCommandPool();
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = outImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {(uint32_t)width, (uint32_t)height, 1};
    vkCmdCopyBufferToImage(cmd, stagingBuffer, outImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(context_->GetGraphicsQueue());
    vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);

    // Image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = outImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &outView) != VK_SUCCESS)
        return false;

    // Sampler (addressMode parameterized: CLAMP for leaves, REPEAT for bark)
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = addressMode;
    samplerInfo.addressModeV = addressMode;
    samplerInfo.addressModeW = addressMode;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &outSampler) != VK_SUCCESS)
        return false;

    return true;
}

// ============================================================
// Dummy normal map (flat: 128,128,255,255 = tangent-space up)
// ============================================================
bool Rasterizer::CreateDummyNormalTexture() {
    VkDevice device = context_->GetDevice();

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {1, 1, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &dummyNormalImage_) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, dummyNormalImage_, &memReq);

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize = memReq.size;
    memAlloc.memoryTypeIndex = context_->FindMemoryType(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &memAlloc, nullptr, &dummyNormalMemory_) != VK_SUCCESS) return false;
    vkBindImageMemory(device, dummyNormalImage_, dummyNormalMemory_, 0);

    // Flat normal: (0.5, 0.5, 1.0, 1.0) = (128, 128, 255, 255)
    void* mapped;
    vkMapMemory(device, dummyNormalMemory_, 0, memReq.size, 0, &mapped);
    uint32_t flatNormal = 0xFFFF8080; // ABGR: A=FF, B=FF, G=80, R=80
    memcpy(mapped, &flatNormal, 4);
    vkUnmapMemory(device, dummyNormalMemory_);

    // Transition
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = context_->GetCommandPool();
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = dummyNormalImage_;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(context_->GetGraphicsQueue());
    vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

    // Image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = dummyNormalImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(device, &viewInfo, nullptr, &dummyNormalView_) != VK_SUCCESS) return false;

    // Sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &dummyNormalSampler_) != VK_SUCCESS) return false;

    activeNormalView_ = VK_NULL_HANDLE;
    activeNormalSampler_ = VK_NULL_HANDLE;

    Log(L"[VK Rasterizer] Dummy normal texture created (1x1 flat)\n");
    return true;
}

// ============================================================
// Dummy specular map (1x1 white = full specular)
// ============================================================
bool Rasterizer::CreateDummySpecTexture() {
    VkDevice device = context_->GetDevice();

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {1, 1, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &dummySpecImage_) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, dummySpecImage_, &memReq);

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize = memReq.size;
    memAlloc.memoryTypeIndex = context_->FindMemoryType(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &memAlloc, nullptr, &dummySpecMemory_) != VK_SUCCESS) return false;
    vkBindImageMemory(device, dummySpecImage_, dummySpecMemory_, 0);

    void* mapped;
    vkMapMemory(device, dummySpecMemory_, 0, memReq.size, 0, &mapped);
    uint32_t white = 0xFFFFFFFF;
    memcpy(mapped, &white, 4);
    vkUnmapMemory(device, dummySpecMemory_);

    // Transition
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = context_->GetCommandPool();
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = dummySpecImage_;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(context_->GetGraphicsQueue());
    vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = dummySpecImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(device, &viewInfo, nullptr, &dummySpecView_) != VK_SUCCESS) return false;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &dummySpecSampler_) != VK_SUCCESS) return false;

    activeSpecView_ = VK_NULL_HANDLE;
    activeSpecSampler_ = VK_NULL_HANDLE;

    Log(L"[VK Rasterizer] Dummy specular texture created (1x1 white)\n");
    return true;
}

// ============================================================
// Normal map set/destroy/upload
// ============================================================
void Rasterizer::SetNormalTexture(VkImageView imageView, VkSampler sampler) {
    activeNormalView_ = imageView;
    activeNormalSampler_ = sampler;

    VkDevice device = context_->GetDevice();
    for (size_t i = 0; i < descriptorSets_.size(); i++) {
        VkDescriptorImageInfo normalInfo{};
        normalInfo.sampler = sampler;
        normalInfo.imageView = imageView;
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSets_[i];
        write.dstBinding = 3;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &normalInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

void Rasterizer::DestroyUploadedNormal() {
    VkDevice device = context_->GetDevice();
    if (uploadedNormalSampler_ != VK_NULL_HANDLE) { vkDestroySampler(device, uploadedNormalSampler_, nullptr); uploadedNormalSampler_ = VK_NULL_HANDLE; }
    if (uploadedNormalView_ != VK_NULL_HANDLE) { vkDestroyImageView(device, uploadedNormalView_, nullptr); uploadedNormalView_ = VK_NULL_HANDLE; }
    if (uploadedNormalImage_ != VK_NULL_HANDLE) { vkDestroyImage(device, uploadedNormalImage_, nullptr); uploadedNormalImage_ = VK_NULL_HANDLE; }
    if (uploadedNormalMemory_ != VK_NULL_HANDLE) { vkFreeMemory(device, uploadedNormalMemory_, nullptr); uploadedNormalMemory_ = VK_NULL_HANDLE; }
}

bool Rasterizer::UploadNormalFromPixels(const uint8_t* rgba, int width, int height) {
    DestroyUploadedNormal();
    if (!UploadTextureFromPixels(rgba, width, height,
        uploadedNormalImage_, uploadedNormalMemory_, uploadedNormalView_, uploadedNormalSampler_))
        return false;
    SetNormalTexture(uploadedNormalView_, uploadedNormalSampler_);
    Log(L"[VK Rasterizer] Uploaded normal map %dx%d\n", width, height);
    return true;
}

// ============================================================
// Specular map set/destroy/upload
// ============================================================
void Rasterizer::SetSpecTexture(VkImageView imageView, VkSampler sampler) {
    activeSpecView_ = imageView;
    activeSpecSampler_ = sampler;

    VkDevice device = context_->GetDevice();
    for (size_t i = 0; i < descriptorSets_.size(); i++) {
        VkDescriptorImageInfo specInfo{};
        specInfo.sampler = sampler;
        specInfo.imageView = imageView;
        specInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSets_[i];
        write.dstBinding = 4;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &specInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

void Rasterizer::DestroyUploadedSpec() {
    VkDevice device = context_->GetDevice();
    if (uploadedSpecSampler_ != VK_NULL_HANDLE) { vkDestroySampler(device, uploadedSpecSampler_, nullptr); uploadedSpecSampler_ = VK_NULL_HANDLE; }
    if (uploadedSpecView_ != VK_NULL_HANDLE) { vkDestroyImageView(device, uploadedSpecView_, nullptr); uploadedSpecView_ = VK_NULL_HANDLE; }
    if (uploadedSpecImage_ != VK_NULL_HANDLE) { vkDestroyImage(device, uploadedSpecImage_, nullptr); uploadedSpecImage_ = VK_NULL_HANDLE; }
    if (uploadedSpecMemory_ != VK_NULL_HANDLE) { vkFreeMemory(device, uploadedSpecMemory_, nullptr); uploadedSpecMemory_ = VK_NULL_HANDLE; }
}

bool Rasterizer::UploadSpecularFromPixels(const uint8_t* rgba, int width, int height) {
    DestroyUploadedSpec();
    if (!UploadTextureFromPixels(rgba, width, height,
        uploadedSpecImage_, uploadedSpecMemory_, uploadedSpecView_, uploadedSpecSampler_))
        return false;
    SetSpecTexture(uploadedSpecView_, uploadedSpecSampler_);
    Log(L"[VK Rasterizer] Uploaded specular map %dx%d\n", width, height);
    return true;
}

// ============================================================
// Bark albedo dummy (1x1 mid-gray)
// ============================================================
bool Rasterizer::CreateDummyBarkAlbedoTexture() {
    VkDevice device = context_->GetDevice();

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {1, 1, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &dummyBarkAlbedoImage_) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, dummyBarkAlbedoImage_, &memReq);

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize = memReq.size;
    memAlloc.memoryTypeIndex = context_->FindMemoryType(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &memAlloc, nullptr, &dummyBarkAlbedoMemory_) != VK_SUCCESS) return false;
    vkBindImageMemory(device, dummyBarkAlbedoImage_, dummyBarkAlbedoMemory_, 0);

    void* mapped;
    vkMapMemory(device, dummyBarkAlbedoMemory_, 0, memReq.size, 0, &mapped);
    uint32_t gray = 0xFF808080; // mid-gray
    memcpy(mapped, &gray, 4);
    vkUnmapMemory(device, dummyBarkAlbedoMemory_);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = context_->GetCommandPool();
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = dummyBarkAlbedoImage_;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(context_->GetGraphicsQueue());
    vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = dummyBarkAlbedoImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(device, &viewInfo, nullptr, &dummyBarkAlbedoView_) != VK_SUCCESS) return false;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &dummyBarkAlbedoSampler_) != VK_SUCCESS) return false;

    activeBarkAlbedoView_ = VK_NULL_HANDLE;
    activeBarkAlbedoSampler_ = VK_NULL_HANDLE;
    return true;
}

bool Rasterizer::CreateDummyBarkNormalTexture() {
    VkDevice device = context_->GetDevice();

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {1, 1, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &dummyBarkNormalImage_) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, dummyBarkNormalImage_, &memReq);

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize = memReq.size;
    memAlloc.memoryTypeIndex = context_->FindMemoryType(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &memAlloc, nullptr, &dummyBarkNormalMemory_) != VK_SUCCESS) return false;
    vkBindImageMemory(device, dummyBarkNormalImage_, dummyBarkNormalMemory_, 0);

    void* mapped;
    vkMapMemory(device, dummyBarkNormalMemory_, 0, memReq.size, 0, &mapped);
    uint32_t flatNormal = 0xFFFF8080; // (128,128,255,255) = flat normal
    memcpy(mapped, &flatNormal, 4);
    vkUnmapMemory(device, dummyBarkNormalMemory_);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = context_->GetCommandPool();
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = dummyBarkNormalImage_;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(context_->GetGraphicsQueue());
    vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = dummyBarkNormalImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(device, &viewInfo, nullptr, &dummyBarkNormalView_) != VK_SUCCESS) return false;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &dummyBarkNormalSampler_) != VK_SUCCESS) return false;

    activeBarkNormalView_ = VK_NULL_HANDLE;
    activeBarkNormalSampler_ = VK_NULL_HANDLE;
    return true;
}

bool Rasterizer::CreateDummyBarkSpecTexture() {
    VkDevice device = context_->GetDevice();

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {1, 1, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &dummyBarkSpecImage_) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, dummyBarkSpecImage_, &memReq);

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize = memReq.size;
    memAlloc.memoryTypeIndex = context_->FindMemoryType(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &memAlloc, nullptr, &dummyBarkSpecMemory_) != VK_SUCCESS) return false;
    vkBindImageMemory(device, dummyBarkSpecImage_, dummyBarkSpecMemory_, 0);

    void* mapped;
    vkMapMemory(device, dummyBarkSpecMemory_, 0, memReq.size, 0, &mapped);
    uint32_t lowSpec = 0xFF262626; // low specular (bark is rough)
    memcpy(mapped, &lowSpec, 4);
    vkUnmapMemory(device, dummyBarkSpecMemory_);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = context_->GetCommandPool();
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = dummyBarkSpecImage_;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(context_->GetGraphicsQueue());
    vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = dummyBarkSpecImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(device, &viewInfo, nullptr, &dummyBarkSpecView_) != VK_SUCCESS) return false;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &dummyBarkSpecSampler_) != VK_SUCCESS) return false;

    activeBarkSpecView_ = VK_NULL_HANDLE;
    activeBarkSpecSampler_ = VK_NULL_HANDLE;
    return true;
}

// ============================================================
// Bark texture set/destroy/upload
// ============================================================
void Rasterizer::SetBarkAlbedoTexture(VkImageView imageView, VkSampler sampler) {
    activeBarkAlbedoView_ = imageView;
    activeBarkAlbedoSampler_ = sampler;
    VkDevice device = context_->GetDevice();
    for (size_t i = 0; i < descriptorSets_.size(); i++) {
        VkDescriptorImageInfo info{};
        info.sampler = sampler;
        info.imageView = imageView;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSets_[i];
        write.dstBinding = 5;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &info;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

void Rasterizer::SetBarkNormalTexture(VkImageView imageView, VkSampler sampler) {
    activeBarkNormalView_ = imageView;
    activeBarkNormalSampler_ = sampler;
    VkDevice device = context_->GetDevice();
    for (size_t i = 0; i < descriptorSets_.size(); i++) {
        VkDescriptorImageInfo info{};
        info.sampler = sampler;
        info.imageView = imageView;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSets_[i];
        write.dstBinding = 6;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &info;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

void Rasterizer::SetBarkSpecTexture(VkImageView imageView, VkSampler sampler) {
    activeBarkSpecView_ = imageView;
    activeBarkSpecSampler_ = sampler;
    VkDevice device = context_->GetDevice();
    for (size_t i = 0; i < descriptorSets_.size(); i++) {
        VkDescriptorImageInfo info{};
        info.sampler = sampler;
        info.imageView = imageView;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSets_[i];
        write.dstBinding = 7;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &info;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

void Rasterizer::DestroyUploadedBarkAlbedo() {
    VkDevice device = context_->GetDevice();
    if (uploadedBarkAlbedoSampler_ != VK_NULL_HANDLE) { vkDestroySampler(device, uploadedBarkAlbedoSampler_, nullptr); uploadedBarkAlbedoSampler_ = VK_NULL_HANDLE; }
    if (uploadedBarkAlbedoView_ != VK_NULL_HANDLE) { vkDestroyImageView(device, uploadedBarkAlbedoView_, nullptr); uploadedBarkAlbedoView_ = VK_NULL_HANDLE; }
    if (uploadedBarkAlbedoImage_ != VK_NULL_HANDLE) { vkDestroyImage(device, uploadedBarkAlbedoImage_, nullptr); uploadedBarkAlbedoImage_ = VK_NULL_HANDLE; }
    if (uploadedBarkAlbedoMemory_ != VK_NULL_HANDLE) { vkFreeMemory(device, uploadedBarkAlbedoMemory_, nullptr); uploadedBarkAlbedoMemory_ = VK_NULL_HANDLE; }
}

void Rasterizer::DestroyUploadedBarkNormal() {
    VkDevice device = context_->GetDevice();
    if (uploadedBarkNormalSampler_ != VK_NULL_HANDLE) { vkDestroySampler(device, uploadedBarkNormalSampler_, nullptr); uploadedBarkNormalSampler_ = VK_NULL_HANDLE; }
    if (uploadedBarkNormalView_ != VK_NULL_HANDLE) { vkDestroyImageView(device, uploadedBarkNormalView_, nullptr); uploadedBarkNormalView_ = VK_NULL_HANDLE; }
    if (uploadedBarkNormalImage_ != VK_NULL_HANDLE) { vkDestroyImage(device, uploadedBarkNormalImage_, nullptr); uploadedBarkNormalImage_ = VK_NULL_HANDLE; }
    if (uploadedBarkNormalMemory_ != VK_NULL_HANDLE) { vkFreeMemory(device, uploadedBarkNormalMemory_, nullptr); uploadedBarkNormalMemory_ = VK_NULL_HANDLE; }
}

void Rasterizer::DestroyUploadedBarkSpec() {
    VkDevice device = context_->GetDevice();
    if (uploadedBarkSpecSampler_ != VK_NULL_HANDLE) { vkDestroySampler(device, uploadedBarkSpecSampler_, nullptr); uploadedBarkSpecSampler_ = VK_NULL_HANDLE; }
    if (uploadedBarkSpecView_ != VK_NULL_HANDLE) { vkDestroyImageView(device, uploadedBarkSpecView_, nullptr); uploadedBarkSpecView_ = VK_NULL_HANDLE; }
    if (uploadedBarkSpecImage_ != VK_NULL_HANDLE) { vkDestroyImage(device, uploadedBarkSpecImage_, nullptr); uploadedBarkSpecImage_ = VK_NULL_HANDLE; }
    if (uploadedBarkSpecMemory_ != VK_NULL_HANDLE) { vkFreeMemory(device, uploadedBarkSpecMemory_, nullptr); uploadedBarkSpecMemory_ = VK_NULL_HANDLE; }
}

bool Rasterizer::UploadBarkAlbedoFromPixels(const uint8_t* rgba, int width, int height) {
    DestroyUploadedBarkAlbedo();
    if (!UploadTextureFromPixels(rgba, width, height,
        uploadedBarkAlbedoImage_, uploadedBarkAlbedoMemory_, uploadedBarkAlbedoView_, uploadedBarkAlbedoSampler_,
        VK_SAMPLER_ADDRESS_MODE_REPEAT))
        return false;
    SetBarkAlbedoTexture(uploadedBarkAlbedoView_, uploadedBarkAlbedoSampler_);
    Log(L"[VK Rasterizer] Uploaded bark albedo %dx%d\n", width, height);
    return true;
}

bool Rasterizer::UploadBarkNormalFromPixels(const uint8_t* rgba, int width, int height) {
    DestroyUploadedBarkNormal();
    if (!UploadTextureFromPixels(rgba, width, height,
        uploadedBarkNormalImage_, uploadedBarkNormalMemory_, uploadedBarkNormalView_, uploadedBarkNormalSampler_,
        VK_SAMPLER_ADDRESS_MODE_REPEAT))
        return false;
    SetBarkNormalTexture(uploadedBarkNormalView_, uploadedBarkNormalSampler_);
    Log(L"[VK Rasterizer] Uploaded bark normal %dx%d\n", width, height);
    return true;
}

bool Rasterizer::UploadBarkSpecularFromPixels(const uint8_t* rgba, int width, int height) {
    DestroyUploadedBarkSpec();
    if (!UploadTextureFromPixels(rgba, width, height,
        uploadedBarkSpecImage_, uploadedBarkSpecMemory_, uploadedBarkSpecView_, uploadedBarkSpecSampler_,
        VK_SAMPLER_ADDRESS_MODE_REPEAT))
        return false;
    SetBarkSpecTexture(uploadedBarkSpecView_, uploadedBarkSpecSampler_);
    Log(L"[VK Rasterizer] Uploaded bark specular %dx%d\n", width, height);
    return true;
}

bool Rasterizer::CreateShadowResources() {
    VkDevice device = context_->GetDevice();

    // --- Shadow map depth image ---
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_D32_SFLOAT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &shadowMapImage_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create shadow map image\n");
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, shadowMapImage_, &memReq);

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize = memReq.size;
    memAlloc.memoryTypeIndex = context_->FindMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &memAlloc, nullptr, &shadowMapMemory_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to allocate shadow map memory\n");
        return false;
    }
    vkBindImageMemory(device, shadowMapImage_, shadowMapMemory_, 0);

    // --- Shadow map image view ---
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = shadowMapImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_D32_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &shadowMapView_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create shadow map image view\n");
        return false;
    }

    // --- Shadow map sampler ---
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; // 1.0 = no shadow outside map
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &shadowMapSampler_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create shadow map sampler\n");
        return false;
    }

    // --- Shadow render pass (depth-only) ---
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 0;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 0;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dependencies[2]{};
    // Dependency 0: external -> shadow pass
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Dependency 1: shadow pass -> main pass fragment shader reads
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &depthAttachment;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies = dependencies;

    if (vkCreateRenderPass(device, &rpInfo, nullptr, &shadowRenderPass_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create shadow render pass\n");
        return false;
    }

    // --- Shadow framebuffer ---
    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = shadowRenderPass_;
    fbInfo.attachmentCount = 1;
    fbInfo.pAttachments = &shadowMapView_;
    fbInfo.width = SHADOW_MAP_SIZE;
    fbInfo.height = SHADOW_MAP_SIZE;
    fbInfo.layers = 1;

    if (vkCreateFramebuffer(device, &fbInfo, nullptr, &shadowFramebuffer_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create shadow framebuffer\n");
        return false;
    }

    // --- Shadow descriptor set layout (binding 0 = UBO only) ---
    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding = 0;
    uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslInfo{};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = 1;
    dslInfo.pBindings = &uboBinding;

    if (vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &shadowDescSetLayout_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create shadow descriptor set layout\n");
        return false;
    }

    // --- Shadow pipeline layout ---
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(RasterPushConstants);

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &shadowDescSetLayout_;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(device, &plInfo, nullptr, &shadowPipelineLayout_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create shadow pipeline layout\n");
        return false;
    }

    // --- Shadow pipeline (depth-only, vertex-only) ---
    std::vector<char> vertShaderCode;
    if (!pipeline_->LoadShader("shaders/basic.vert.spv", vertShaderCode)) {
        Log(L"[VK Rasterizer] ERROR: Failed to load vertex shader for shadow pipeline\n");
        return false;
    }
    VkShaderModule vertModule = pipeline_->CreateShaderModule(vertShaderCode);
    if (vertModule == VK_NULL_HANDLE) return false;

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";

    auto bindingDesc = Vertex::GetBindingDescription();
    auto attrDescs = Vertex::GetAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInput.pVertexAttributeDescriptions = attrDescs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Dynamic viewport/scissor
    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 4.0f;
    rasterizer.depthBiasSlopeFactor = 3.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 0; // depth-only, no color

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 1; // vertex only
    pipelineInfo.pStages = &vertStage;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = shadowPipelineLayout_;
    pipelineInfo.renderPass = shadowRenderPass_;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowPipeline_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create shadow pipeline\n");
        vkDestroyShaderModule(device, vertModule, nullptr);
        return false;
    }
    vkDestroyShaderModule(device, vertModule, nullptr);

    // --- Shadow UBOs ---
    size_t imageCount = context_->GetSwapchainImageViews().size();
    shadowUniformBuffers_.resize(imageCount);
    shadowUniformBuffersMemory_.resize(imageCount);
    shadowUniformBuffersMapped_.resize(imageCount);

    for (size_t i = 0; i < imageCount; i++) {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = sizeof(UniformBufferObject);
        bufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufInfo, nullptr, &shadowUniformBuffers_[i]) != VK_SUCCESS) return false;

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(device, shadowUniformBuffers_[i], &mr);

        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = context_->FindMemoryType(mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(device, &ai, nullptr, &shadowUniformBuffersMemory_[i]) != VK_SUCCESS) return false;
        vkBindBufferMemory(device, shadowUniformBuffers_[i], shadowUniformBuffersMemory_[i], 0);
        vkMapMemory(device, shadowUniformBuffersMemory_[i], 0, sizeof(UniformBufferObject), 0, &shadowUniformBuffersMapped_[i]);
    }

    // --- Shadow descriptor pool + sets ---
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(imageCount);

    VkDescriptorPoolCreateInfo dpInfo{};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;
    dpInfo.maxSets = static_cast<uint32_t>(imageCount);

    if (vkCreateDescriptorPool(device, &dpInfo, nullptr, &shadowDescPool_) != VK_SUCCESS) return false;

    std::vector<VkDescriptorSetLayout> layouts(imageCount, shadowDescSetLayout_);
    VkDescriptorSetAllocateInfo dsAlloc{};
    dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAlloc.descriptorPool = shadowDescPool_;
    dsAlloc.descriptorSetCount = static_cast<uint32_t>(imageCount);
    dsAlloc.pSetLayouts = layouts.data();

    shadowDescSets_.resize(imageCount);
    if (vkAllocateDescriptorSets(device, &dsAlloc, shadowDescSets_.data()) != VK_SUCCESS) return false;

    for (size_t i = 0; i < imageCount; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = shadowUniformBuffers_[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = shadowDescSets_[i];
        write.dstBinding = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    Log(L"[VK Rasterizer] Shadow mapping resources created (%dx%d)\n", SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    return true;
}

// ============================================================
// Uniform buffers (camera)
// ============================================================
bool Rasterizer::CreateUniformBuffers() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    size_t imageCount = context_->GetSwapchainImageViews().size();
    uniformBuffers_.resize(imageCount);
    uniformBuffersMemory_.resize(imageCount);
    uniformBuffersMapped_.resize(imageCount);

    for (size_t i = 0; i < imageCount; i++) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(context_->GetDevice(), &bufferInfo, nullptr, &uniformBuffers_[i]) != VK_SUCCESS) {
            Log(L"[VK Rasterizer] ERROR: Failed to create uniform buffer\n");
            return false;
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(context_->GetDevice(), uniformBuffers_[i], &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = context_->FindMemoryType(memRequirements.memoryTypeBits,
                                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(context_->GetDevice(), &allocInfo, nullptr, &uniformBuffersMemory_[i]) != VK_SUCCESS) {
            Log(L"[VK Rasterizer] ERROR: Failed to allocate uniform buffer memory\n");
            return false;
        }

        vkBindBufferMemory(context_->GetDevice(), uniformBuffers_[i], uniformBuffersMemory_[i], 0);
        vkMapMemory(context_->GetDevice(), uniformBuffersMemory_[i], 0, bufferSize, 0, &uniformBuffersMapped_[i]);
    }

    Log(L"[VK Rasterizer] Uniform buffers created\n");
    return true;
}

// ============================================================
// Descriptor pool (main - UBO + shadow sampler)
// ============================================================
bool Rasterizer::CreateDescriptorPool() {
    size_t imageCount = context_->GetSwapchainImageViews().size();

    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(imageCount);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(imageCount * 7); // shadow + leaf[albedo/normal/spec] + bark[albedo/normal/spec]

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT; // allow updates
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = static_cast<uint32_t>(imageCount);

    if (vkCreateDescriptorPool(context_->GetDevice(), &poolInfo, nullptr, &descriptorPool_) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to create descriptor pool\n");
        return false;
    }

    Log(L"[VK Rasterizer] Descriptor pool created\n");
    return true;
}

// ============================================================
// Descriptor sets (main - UBO + shadow map sampler)
// ============================================================
bool Rasterizer::CreateDescriptorSets() {
    size_t imageCount = context_->GetSwapchainImageViews().size();
    std::vector<VkDescriptorSetLayout> layouts(imageCount, pipeline_->GetDescriptorSetLayout());

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets_.resize(imageCount);
    if (vkAllocateDescriptorSets(context_->GetDevice(), &allocInfo, descriptorSets_.data()) != VK_SUCCESS) {
        Log(L"[VK Rasterizer] ERROR: Failed to allocate descriptor sets\n");
        return false;
    }

    for (size_t i = 0; i < imageCount; i++) {
        // Binding 0: Camera UBO
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers_[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        // Binding 1: Shadow map sampler
        VkDescriptorImageInfo shadowInfo{};
        shadowInfo.sampler = shadowMapSampler_;
        shadowInfo.imageView = shadowMapView_;
        shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        // Binding 2: Albedo texture (dummy white by default)
        VkDescriptorImageInfo albedoInfo{};
        albedoInfo.sampler = activeAlbedoSampler_ ? activeAlbedoSampler_ : dummyAlbedoSampler_;
        albedoInfo.imageView = activeAlbedoView_ ? activeAlbedoView_ : dummyAlbedoView_;
        albedoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Binding 3: Normal map (dummy flat normal by default)
        VkDescriptorImageInfo normalInfo{};
        normalInfo.sampler = activeNormalSampler_ ? activeNormalSampler_ : dummyNormalSampler_;
        normalInfo.imageView = activeNormalView_ ? activeNormalView_ : dummyNormalView_;
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Binding 4: Specular map (dummy white by default)
        VkDescriptorImageInfo specInfo{};
        specInfo.sampler = activeSpecSampler_ ? activeSpecSampler_ : dummySpecSampler_;
        specInfo.imageView = activeSpecView_ ? activeSpecView_ : dummySpecView_;
        specInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Binding 5: Bark albedo (dummy by default)
        VkDescriptorImageInfo barkAlbedoInfo{};
        barkAlbedoInfo.sampler = activeBarkAlbedoSampler_ ? activeBarkAlbedoSampler_ : dummyBarkAlbedoSampler_;
        barkAlbedoInfo.imageView = activeBarkAlbedoView_ ? activeBarkAlbedoView_ : dummyBarkAlbedoView_;
        barkAlbedoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Binding 6: Bark normal (dummy by default)
        VkDescriptorImageInfo barkNormalInfo{};
        barkNormalInfo.sampler = activeBarkNormalSampler_ ? activeBarkNormalSampler_ : dummyBarkNormalSampler_;
        barkNormalInfo.imageView = activeBarkNormalView_ ? activeBarkNormalView_ : dummyBarkNormalView_;
        barkNormalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Binding 7: Bark specular (dummy by default)
        VkDescriptorImageInfo barkSpecInfo{};
        barkSpecInfo.sampler = activeBarkSpecSampler_ ? activeBarkSpecSampler_ : dummyBarkSpecSampler_;
        barkSpecInfo.imageView = activeBarkSpecView_ ? activeBarkSpecView_ : dummyBarkSpecView_;
        barkSpecInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet writes[8]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSets_[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &bufferInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptorSets_[i];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &shadowInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = descriptorSets_[i];
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &albedoInfo;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = descriptorSets_[i];
        writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].descriptorCount = 1;
        writes[3].pImageInfo = &normalInfo;

        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = descriptorSets_[i];
        writes[4].dstBinding = 4;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[4].descriptorCount = 1;
        writes[4].pImageInfo = &specInfo;

        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet = descriptorSets_[i];
        writes[5].dstBinding = 5;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[5].descriptorCount = 1;
        writes[5].pImageInfo = &barkAlbedoInfo;

        writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[6].dstSet = descriptorSets_[i];
        writes[6].dstBinding = 6;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[6].descriptorCount = 1;
        writes[6].pImageInfo = &barkNormalInfo;

        writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[7].dstSet = descriptorSets_[i];
        writes[7].dstBinding = 7;
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[7].descriptorCount = 1;
        writes[7].pImageInfo = &barkSpecInfo;

        vkUpdateDescriptorSets(context_->GetDevice(), 8, writes, 0, nullptr);
    }

    Log(L"[VK Rasterizer] Descriptor sets created (UBO + shadow + leaf[albedo/normal/spec] + bark[albedo/normal/spec])\n");
    return true;
}

// ============================================================
// Update camera UBO (includes lightViewProj for shadow sampling)
// ============================================================
void Rasterizer::UpdateUniformBuffer(uint32_t currentImage) {
    UniformBufferObject ubo{};

    // View matrix (orbital camera looking at target)
    float horizDist = cameraDistance_ * cosf(cameraPitch_);
    float camX = horizDist * sinf(cameraAngle_);
    float camY = cameraTargetY_ + cameraDistance_ * sinf(cameraPitch_);
    float camZ = horizDist * cosf(cameraAngle_);

    BuildLookAt(ubo.view, camX, camY, camZ, 0.0f, cameraTargetY_, 0.0f);

    // Projection matrix (perspective)
    float fov = 45.0f * 3.14159f / 180.0f;
    float aspect = viewportOverride_
        ? (viewportW_ / viewportH_)
        : (float)context_->GetSwapchainExtent().width / (float)context_->GetSwapchainExtent().height;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;

    float tanHalfFov = tanf(fov / 2.0f);
    memset(ubo.projection, 0, sizeof(ubo.projection));
    ubo.projection[0] = 1.0f / (aspect * tanHalfFov);
    ubo.projection[5] = -1.0f / tanHalfFov;  // Flip Y for Vulkan
    ubo.projection[10] = farPlane / (nearPlane - farPlane);
    ubo.projection[11] = -1.0f;
    ubo.projection[14] = -(farPlane * nearPlane) / (farPlane - nearPlane);

    // Light direction (sun from upper-right)
    float ld[3] = {0.4f, -0.7f, 0.3f};
    float len = sqrtf(ld[0]*ld[0] + ld[1]*ld[1] + ld[2]*ld[2]);
    ubo.lightDir[0] = ld[0]/len;
    ubo.lightDir[1] = ld[1]/len;
    ubo.lightDir[2] = ld[2]/len;
    ubo.lightDir[3] = 0.0f;

    // Camera position (for view-dependent effects)
    ubo.cameraPos[0] = camX;
    ubo.cameraPos[1] = camY;
    ubo.cameraPos[2] = camZ;
    ubo.cameraPos[3] = 0.0f;

    // Light view-projection matrix (for shadow mapping in fragment shader)
    float lightView[16], lightProj[16];
    ComputeLightMatrices(lightView, lightProj, ubo.lightViewProj);

    memcpy(uniformBuffersMapped_[currentImage], &ubo, sizeof(ubo));
}

// ============================================================
// Billboard offscreen capture resources
// ============================================================
void Rasterizer::DestroyBillboardResources() {
    VkDevice device = context_->GetDevice();
    if (billboardReadbackBuffer_ != VK_NULL_HANDLE) { vkDestroyBuffer(device, billboardReadbackBuffer_, nullptr); billboardReadbackBuffer_ = VK_NULL_HANDLE; }
    if (billboardReadbackMemory_ != VK_NULL_HANDLE) { vkFreeMemory(device, billboardReadbackMemory_, nullptr); billboardReadbackMemory_ = VK_NULL_HANDLE; }
    if (billboardFramebuffer_ != VK_NULL_HANDLE) { vkDestroyFramebuffer(device, billboardFramebuffer_, nullptr); billboardFramebuffer_ = VK_NULL_HANDLE; }
    if (billboardRenderPass_ != VK_NULL_HANDLE) { vkDestroyRenderPass(device, billboardRenderPass_, nullptr); billboardRenderPass_ = VK_NULL_HANDLE; }
    if (billboardDepthView_ != VK_NULL_HANDLE) { vkDestroyImageView(device, billboardDepthView_, nullptr); billboardDepthView_ = VK_NULL_HANDLE; }
    if (billboardDepthImage_ != VK_NULL_HANDLE) { vkDestroyImage(device, billboardDepthImage_, nullptr); billboardDepthImage_ = VK_NULL_HANDLE; }
    if (billboardDepthMemory_ != VK_NULL_HANDLE) { vkFreeMemory(device, billboardDepthMemory_, nullptr); billboardDepthMemory_ = VK_NULL_HANDLE; }
    if (billboardColorView_ != VK_NULL_HANDLE) { vkDestroyImageView(device, billboardColorView_, nullptr); billboardColorView_ = VK_NULL_HANDLE; }
    if (billboardColorImage_ != VK_NULL_HANDLE) { vkDestroyImage(device, billboardColorImage_, nullptr); billboardColorImage_ = VK_NULL_HANDLE; }
    if (billboardColorMemory_ != VK_NULL_HANDLE) { vkFreeMemory(device, billboardColorMemory_, nullptr); billboardColorMemory_ = VK_NULL_HANDLE; }
    billboardSize_ = 0;
}

bool Rasterizer::CreateBillboardResources(uint32_t size) {
    if (billboardSize_ == size && billboardColorImage_ != VK_NULL_HANDLE) return true;
    DestroyBillboardResources();

    VkDevice device = context_->GetDevice();
    VkFormat colorFormat = context_->GetSwapchainFormat(); // match swapchain for pipeline compatibility
    billboardSize_ = size;

    // --- Color image (swapchain format, color attachment + transfer src for readback) ---
    {
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.extent = {size, size, 1};
        ci.mipLevels = 1; ci.arrayLayers = 1;
        ci.format = colorFormat;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(device, &ci, nullptr, &billboardColorImage_) != VK_SUCCESS) return false;

        VkMemoryRequirements mr; vkGetImageMemoryRequirements(device, billboardColorImage_, &mr);
        VkMemoryAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = context_->FindMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &ai, nullptr, &billboardColorMemory_) != VK_SUCCESS) return false;
        vkBindImageMemory(device, billboardColorImage_, billboardColorMemory_, 0);

        VkImageViewCreateInfo vi{}; vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = billboardColorImage_; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = colorFormat;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device, &vi, nullptr, &billboardColorView_) != VK_SUCCESS) return false;
    }

    // --- Depth image ---
    {
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.extent = {size, size, 1};
        ci.mipLevels = 1; ci.arrayLayers = 1;
        ci.format = VK_FORMAT_D32_SFLOAT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(device, &ci, nullptr, &billboardDepthImage_) != VK_SUCCESS) return false;

        VkMemoryRequirements mr; vkGetImageMemoryRequirements(device, billboardDepthImage_, &mr);
        VkMemoryAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = context_->FindMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &ai, nullptr, &billboardDepthMemory_) != VK_SUCCESS) return false;
        vkBindImageMemory(device, billboardDepthImage_, billboardDepthMemory_, 0);

        VkImageViewCreateInfo vi{}; vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = billboardDepthImage_; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_D32_SFLOAT;
        vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device, &vi, nullptr, &billboardDepthView_) != VK_SUCCESS) return false;
    }

    // --- Render pass (same format as main pipeline for compatibility) ---
    {
        VkAttachmentDescription attachments[2]{};
        // Color — same format as swapchain so pipeline is compatible
        attachments[0].format = colorFormat;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Depth
        attachments[1].format = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 2;
        rpInfo.pAttachments = attachments;
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        if (vkCreateRenderPass(device, &rpInfo, nullptr, &billboardRenderPass_) != VK_SUCCESS) return false;
    }

    // --- Framebuffer ---
    {
        VkImageView fbAttachments[2] = {billboardColorView_, billboardDepthView_};
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = billboardRenderPass_;
        fbInfo.attachmentCount = 2;
        fbInfo.pAttachments = fbAttachments;
        fbInfo.width = size; fbInfo.height = size; fbInfo.layers = 1;
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &billboardFramebuffer_) != VK_SUCCESS) return false;
    }

    // --- Readback staging buffer ---
    {
        VkDeviceSize bufSize = (VkDeviceSize)size * size * 4;
        VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = bufSize;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &bi, nullptr, &billboardReadbackBuffer_) != VK_SUCCESS) return false;

        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device, billboardReadbackBuffer_, &mr);
        VkMemoryAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = context_->FindMemoryType(mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(device, &ai, nullptr, &billboardReadbackMemory_) != VK_SUCCESS) return false;
        vkBindBufferMemory(device, billboardReadbackBuffer_, billboardReadbackMemory_, 0);
    }

    Log(L"[VK Rasterizer] Billboard capture resources created (%dx%d, format=%d)\n", size, size, (int)colorFormat);
    return true;
}

// ============================================================
// Billboard capture: render current draw calls with ortho camera
// ============================================================
bool Rasterizer::CaptureBillboardTexture(uint32_t size, float treeWidth, float treeHeight,
                                          float treeBaseY, std::vector<uint8_t>& outPixels) {
    Log(L"[VK Billboard] CaptureBillboardTexture: size=%u, treeW=%.2f, treeH=%.2f, baseY=%.2f, drawCalls=%zu\n",
        size, treeWidth, treeHeight, treeBaseY, drawCalls_.size());

    if (!context_) { Log(L"[VK Billboard] ERROR: no context\n"); return false; }
    if (!pipeline_) { Log(L"[VK Billboard] ERROR: no pipeline\n"); return false; }
    if (drawCalls_.empty()) { Log(L"[VK Billboard] ERROR: no draw calls\n"); return false; }

    // If caller bounds are invalid, compute directly from draw call vertices
    if (treeWidth < 0.1f || treeHeight < 0.1f) {
        float minX = 1e9f, maxX = -1e9f;
        float minY = 1e9f, maxY = -1e9f;
        float minZ = 1e9f, maxZ = -1e9f;
        for (const auto& dc : drawCalls_) {
            if (!dc.mesh) continue;
            for (const auto& v : dc.mesh->vertices) {
                float x = v.position[0], y = v.position[1], z = v.position[2];
                if (x < minX) minX = x; if (x > maxX) maxX = x;
                if (y < minY) minY = y; if (y > maxY) maxY = y;
                if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
            }
        }
        treeWidth  = (std::max)(0.5f, (std::max)(maxX - minX, maxZ - minZ));
        treeHeight = (std::max)(0.5f, maxY - minY);
        treeBaseY  = minY;
        Log(L"[VK Billboard] Recomputed bounds from draw calls: W=%.2f H=%.2f baseY=%.2f\n",
            treeWidth, treeHeight, treeBaseY);
    }

    // Create/resize offscreen resources
    if (!CreateBillboardResources(size)) {
        Log(L"[VK Billboard] ERROR: Failed to create billboard resources\n");
        return false;
    }

    VkDevice device = context_->GetDevice();

    // Allocate one-shot command buffer
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandPool = context_->GetCommandPool();
    cmdAlloc.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VkResult allocRes = vkAllocateCommandBuffers(device, &cmdAlloc, &cmd);
    if (allocRes != VK_SUCCESS) {
        Log(L"[VK Billboard] ERROR: vkAllocateCommandBuffers failed: %d\n", (int)allocRes);
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Set up orthographic UBO (using frame 0's uniform buffer)
    {
        float halfW = treeWidth * 0.5f * 1.1f;  // 10% padding
        float halfH = treeHeight * 0.5f * 1.1f;
        float centerY = treeBaseY + treeHeight * 0.5f;

        UniformBufferObject ubo{};

        // View: look from front (Z+) toward origin
        BuildLookAt(ubo.view, 0.0f, centerY, 10.0f, 0.0f, centerY, 0.0f);

        // Orthographic projection (Vulkan clip: Y flipped)
        memset(ubo.projection, 0, sizeof(ubo.projection));
        ubo.projection[0]  =  1.0f / halfW;           // 2/(r-l) with r=-l=halfW -> 1/halfW
        ubo.projection[5]  = -1.0f / halfH;           // Y flipped for Vulkan
        ubo.projection[10] = -1.0f / 20.0f;           // -1/(far-near), near=0.1, far~20
        ubo.projection[14] = -0.1f / 20.0f;           // -near/(far-near)
        ubo.projection[15] =  1.0f;                   // orthographic w=1

        // Light from above-front for good billboard shading
        ubo.lightDir[0] = 0.3f; ubo.lightDir[1] = -0.8f; ubo.lightDir[2] = 0.5f;
        float len = sqrtf(ubo.lightDir[0]*ubo.lightDir[0] + ubo.lightDir[1]*ubo.lightDir[1] + ubo.lightDir[2]*ubo.lightDir[2]);
        ubo.lightDir[0] /= len; ubo.lightDir[1] /= len; ubo.lightDir[2] /= len;

        ubo.cameraPos[0] = 0; ubo.cameraPos[1] = centerY; ubo.cameraPos[2] = 10.0f;

        // Zero out lightViewProj (no shadows needed for billboard capture)
        memset(ubo.lightViewProj, 0, sizeof(ubo.lightViewProj));

        memcpy(uniformBuffersMapped_[0], &ubo, sizeof(ubo));
    }

    // Begin render pass (clear to transparent black)
    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = billboardRenderPass_;
    rpBegin.framebuffer = billboardFramebuffer_;
    rpBegin.renderArea.offset = {0, 0};
    rpBegin.renderArea.extent = {size, size};

    VkClearValue clearValues[2];
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};  // transparent black
    clearValues[1].depthStencil = {1.0f, 0};
    rpBegin.clearValueCount = 2;
    rpBegin.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->GetPipeline());

    VkViewport vp{0, 0, (float)size, (float)size, 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {size, size}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->GetPipelineLayout(),
                           0, 1, &descriptorSets_[0], 0, nullptr);

    // Draw all current draw calls
    // Draw all current draw calls with raw albedo output (no lighting baked in)
    // so the path tracer's SHADER_TREE can apply lighting consistently with 3D trees.
    // Alpha overrides: leaf (1.25) → 1.65 (raw albedoMap), bark (2.25) → 0.65 (raw barkAlbedoMap)
    for (const auto& draw : drawCalls_) {
        if (!draw.mesh || draw.mesh->vertexBuffer == VK_NULL_HANDLE) continue;

        float origAlpha = draw.color[3];

        // Skip ground plane (alpha ~0) — we only want the tree silhouette
        if (origAlpha < 0.1f) continue;

        RasterPushConstants push{};
        memcpy(push.model, draw.modelMatrix, sizeof(push.model));
        memcpy(push.color, draw.color, sizeof(push.color));

        // Override alpha for raw output modes
        if (origAlpha > 1.1f && origAlpha < 1.4f) {
            push.color[3] = 1.65f;  // procedural leaf → raw albedoMap
        } else if (origAlpha > 2.1f && origAlpha < 2.4f) {
            push.color[3] = 0.65f;  // bark → raw barkAlbedoMap
        }

        vkCmdPushConstants(cmd, pipeline_->GetPipelineLayout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(RasterPushConstants), &push);

        VkBuffer vertexBuffers[] = {draw.mesh->vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(cmd, draw.mesh->indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(draw.mesh->indices.size()), 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);

    // Barrier: transition color image from COLOR_ATTACHMENT_OPTIMAL to TRANSFER_SRC_OPTIMAL
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = billboardColorImage_;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    // Copy color image to staging buffer
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {size, size, 1};

    vkCmdCopyImageToBuffer(cmd, billboardColorImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           billboardReadbackBuffer_, 1, &region);

    vkEndCommandBuffer(cmd);

    // Submit and wait
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(context_->GetGraphicsQueue());

    vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

    // Read back pixels
    outPixels.resize((size_t)size * size * 4);
    void* mapped = nullptr;
    vkMapMemory(device, billboardReadbackMemory_, 0, (VkDeviceSize)size * size * 4, 0, &mapped);
    memcpy(outPixels.data(), mapped, outPixels.size());
    vkUnmapMemory(device, billboardReadbackMemory_);

    // If swapchain format is BGRA, swizzle to RGBA for DDS storage
    VkFormat fmt = context_->GetSwapchainFormat();
    if (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_SRGB) {
        for (size_t i = 0; i < outPixels.size(); i += 4) {
            std::swap(outPixels[i], outPixels[i + 2]); // B <-> R
        }
    }

    // Diagnostic: count non-zero and non-transparent pixels
    uint32_t nonZeroCount = 0;
    uint32_t opaqueCount = 0;
    int firstNonZeroX = -1, firstNonZeroY = -1;
    uint8_t firstR = 0, firstG = 0, firstB = 0, firstA = 0;
    for (uint32_t py = 0; py < size; py++) {
        for (uint32_t px = 0; px < size; px++) {
            size_t idx = ((size_t)py * size + px) * 4;
            uint8_t r = outPixels[idx], g = outPixels[idx+1], b = outPixels[idx+2], a = outPixels[idx+3];
            if (r | g | b | a) {
                nonZeroCount++;
                if (a > 0) opaqueCount++;
                if (firstNonZeroX < 0) {
                    firstNonZeroX = (int)px; firstNonZeroY = (int)py;
                    firstR = r; firstG = g; firstB = b; firstA = a;
                }
            }
        }
    }
    Log(L"[VK Billboard] Readback: %u non-zero pixels, %u with alpha>0, first=(%d,%d) rgba=(%d,%d,%d,%d), format=%d\n",
        nonZeroCount, opaqueCount, firstNonZeroX, firstNonZeroY, firstR, firstG, firstB, firstA, (int)fmt);
    return true;
}

} // namespace vk
} // namespace acpt
