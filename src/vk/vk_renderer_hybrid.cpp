// vk_renderer_hybrid.cpp — hybrid G-buffer rasterization pre-pass.
// Rasterizes primary visibility (primID, instanceInfo, linearDepth) so the
// path tracer can skip the first ray. Extracted from vk_renderer.cpp.

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

bool Renderer::CreateHybridGBufferPipeline() {
    if (hybridGBufferReady_) return true;
    VkDevice device = context_->GetDevice();
    if (!device || renderWidth_ == 0 || renderHeight_ == 0) return false;

    // Helper: create image + memory + view
    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& img, VkDeviceMemory& mem, VkImageView& view) -> bool {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = { renderWidth_, renderHeight_, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &ici, nullptr, &img) != VK_SUCCESS) return false;

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device, img, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = context_->FindMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
        vkBindImageMemory(device, img, mem, 0);

        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange = { aspect, 0, 1, 0, 1 };
        return vkCreateImageView(device, &vci, nullptr, &view) == VK_SUCCESS;
    };

    // Create G-buffer images
    if (!createImage(VK_FORMAT_R32_UINT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            hybridPrimIdImage_, hybridPrimIdMemory_, hybridPrimIdView_))
        return false;

    if (!createImage(VK_FORMAT_R32G32_UINT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            hybridInstanceInfoImage_, hybridInstanceInfoMemory_, hybridInstanceInfoView_))
        return false;

    // Depth output as R32_SFLOAT color attachment (universally compatible with storage images)
    if (!createImage(VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            hybridDepthImage_, hybridDepthMemory_, hybridDepthView_))
        return false;

    // Z-buffer for depth testing (not read by shaders — just for rasterizer z-test)
    if (!createImage(VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            hybridZBuffer_, hybridZBufferMemory_, hybridZBufferView_))
        return false;

    // Create render pass: 3 color (primID + instanceInfo + linearDepth) + 1 depth (z-test)
    VkAttachmentDescription attachments[4] = {};
    // Attachment 0: PrimID (R32_UINT)
    attachments[0].format = VK_FORMAT_R32_UINT;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    // Attachment 1: InstanceInfo (RG32_UINT)
    attachments[1] = attachments[0];
    attachments[1].format = VK_FORMAT_R32G32_UINT;
    // Attachment 2: LinearDepth (R32_SFLOAT) — written by fragment shader, read by raygen
    attachments[2] = attachments[0];
    attachments[2].format = VK_FORMAT_R32_SFLOAT;
    // Attachment 3: Z-buffer (D32_SFLOAT) — for rasterizer depth test only
    attachments[3].format = VK_FORMAT_D32_SFLOAT;
    attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // not read after
    attachments[3].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[3].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRefs[3] = {
        { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
        { 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
        { 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL }
    };
    VkAttachmentReference depthRef = { 3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 3;
    subpass.pColorAttachments = colorRefs;
    subpass.pDepthStencilAttachment = &depthRef;

    VkRenderPassCreateInfo rpci{};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 4;
    rpci.pAttachments = attachments;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    if (vkCreateRenderPass(device, &rpci, nullptr, &hybridRenderPass_) != VK_SUCCESS) return false;

    // Framebuffer
    VkImageView fbViews[4] = { hybridPrimIdView_, hybridInstanceInfoView_, hybridDepthView_, hybridZBufferView_ };
    VkFramebufferCreateInfo fbci{};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = hybridRenderPass_;
    fbci.attachmentCount = 4;
    fbci.pAttachments = fbViews;
    fbci.width = renderWidth_;
    fbci.height = renderHeight_;
    fbci.layers = 1;
    if (vkCreateFramebuffer(device, &fbci, nullptr, &hybridFramebuffer_) != VK_SUCCESS) return false;

    // Push constant layout
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(HybridPushConstants);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &hybridPipelineLayout_) != VK_SUCCESS) return false;

    // Load shader modules
    auto loadSPV = [&](const char* filename) -> VkShaderModule {
        std::string path = IgnisResolvePath(filename);
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open()) { Log(L"[Hybrid] Failed to open %hs\n", filename); return VK_NULL_HANDLE; }
        size_t sz = (size_t)file.tellg();
        std::vector<char> code(sz);
        file.seekg(0);
        file.read(code.data(), sz);
        VkShaderModuleCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sci.codeSize = sz;
        sci.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule mod;
        return (vkCreateShaderModule(device, &sci, nullptr, &mod) == VK_SUCCESS) ? mod : VK_NULL_HANDLE;
    };

    VkShaderModule vertMod = loadSPV("shaders/gbuffer_hybrid.vert.spv");
    VkShaderModule fragMod = loadSPV("shaders/gbuffer_hybrid.frag.spv");
    if (!vertMod || !fragMod) {
        if (vertMod) vkDestroyShaderModule(device, vertMod, nullptr);
        if (fragMod) vkDestroyShaderModule(device, fragMod, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    // Vertex input: position only (float3, 12 bytes stride)
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = 12; // 3 * sizeof(float)
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attr{};
    attr.location = 0;
    attr.binding = 0;
    attr.format = VK_FORMAT_R32G32B32_SFLOAT;
    attr.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 1;
    vertexInput.pVertexAttributeDescriptions = &attr;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = { 0, 0, (float)renderWidth_, (float)renderHeight_, 0.0f, 1.0f };
    VkRect2D scissor = { {0, 0}, {renderWidth_, renderHeight_} };
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterState{};
    rasterState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterState.cullMode = VK_CULL_MODE_NONE; // double-sided like Cycles
    rasterState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterState.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo msaa{};
    msaa.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    // No blending for uint/float attachments
    VkPipelineColorBlendAttachmentState blendAttach[3] = {};
    blendAttach[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    blendAttach[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
    blendAttach[2].colorWriteMask = VK_COLOR_COMPONENT_R_BIT; // linear depth

    VkPipelineColorBlendStateCreateInfo blendState{};
    blendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendState.attachmentCount = 3;
    blendState.pAttachments = blendAttach;

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &vertexInput;
    gpci.pInputAssemblyState = &inputAssembly;
    gpci.pViewportState = &viewportState;
    gpci.pRasterizationState = &rasterState;
    gpci.pMultisampleState = &msaa;
    gpci.pDepthStencilState = &depthStencil;
    gpci.pColorBlendState = &blendState;
    gpci.layout = hybridPipelineLayout_;
    gpci.renderPass = hybridRenderPass_;
    gpci.subpass = 0;

    VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr, &hybridPipeline_);
    vkDestroyShaderModule(device, vertMod, nullptr);
    vkDestroyShaderModule(device, fragMod, nullptr);

    if (result != VK_SUCCESS) return false;

    // Update RT descriptor set with hybrid G-buffer bindings
    if (rtResources_) {
        rtResources_->UpdateHybridGBufferDescriptors(
            hybridPrimIdView_, hybridInstanceInfoView_, hybridDepthView_);
    }

    hybridGBufferReady_ = true;
    Log(L"[VK Renderer] Hybrid G-buffer pipeline created (%ux%u)\n", renderWidth_, renderHeight_);
    return true;
}

void Renderer::RecordHybridGBufferPass(VkCommandBuffer cmd) {
    if (!hybridGBufferReady_ || !accelBuilder_ || cachedTLASInstances_.empty()) return;

    const auto& blasList = accelBuilder_->GetBLASList();

    VkClearValue clearValues[4] = {};
    clearValues[0].color.uint32[0] = 0xFFFFFFFFu; // primID = invalid
    clearValues[1].color.uint32[0] = 0xFFFFFFFFu; // instanceInfo = invalid
    clearValues[1].color.uint32[1] = 0xFFFFFFFFu;
    clearValues[2].color.float32[0] = 99999.0f;    // linearDepth = far (no raster hit)
    clearValues[3].depthStencil = { 1.0f, 0 };    // z-buffer clear

    VkRenderPassBeginInfo rpbi{};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = hybridRenderPass_;
    rpbi.framebuffer = hybridFramebuffer_;
    rpbi.renderArea = { {0, 0}, {renderWidth_, renderHeight_} };
    rpbi.clearValueCount = 4;
    rpbi.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, hybridPipeline_);

    // Build view*proj matrix for Vulkan rasterization.
    // Apply the SAME sub-pixel jitter as the ray tracer so depths match exactly.
    // Without this, the rasterizer's depth disagrees with the RT ray direction,
    // causing black glitches at grazing angles.
    float viewProj[16];
    {
        float rasterProj[16];
        memcpy(rasterProj, lastProj_, sizeof(float) * 16);
        rasterProj[5] = -rasterProj[5]; // Flip Y for Vulkan clip space

        // Jitter: shift projection by sub-pixel offset (Halton sequence from DLSS)
        // proj[8] and proj[9] are the NDC-space X/Y offsets in column-major layout
        // jitterX_/jitterY_ are in pixel space; convert to NDC: 2*jitter/resolution
        if (renderWidth_ > 0 && renderHeight_ > 0) {
            float jitterNDC_X =  2.0f * jitterX_ / (float)renderWidth_;
            float jitterNDC_Y =  2.0f * jitterY_ / (float)renderHeight_;
            rasterProj[8]  += jitterNDC_X;  // proj[2][0] in column-major
            rasterProj[9]  += jitterNDC_Y;  // proj[2][1] in column-major
        }

        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                float sum = 0;
                for (int k = 0; k < 4; k++)
                    sum += rasterProj[r + k * 4] * lastView_[k + c * 4];
                viewProj[r + c * 4] = sum;
            }
        }
    }

    // Draw each TLAS instance (skip hair)
    for (size_t i = 0; i < cachedTLASInstances_.size(); i++) {
        const auto& inst = cachedTLASInstances_[i];
        if (inst.blasIndex < 0 || inst.blasIndex >= (int)blasList.size()) continue;
        const auto& blas = blasList[inst.blasIndex];
        if (blas.isHair || !blas.built || !blas.vertexBuf.buffer || !blas.indexBuf.buffer) continue;

        float model[16] = {
            inst.transform[0], inst.transform[4], inst.transform[8],  0,
            inst.transform[1], inst.transform[5], inst.transform[9],  0,
            inst.transform[2], inst.transform[6], inst.transform[10], 0,
            inst.transform[3], inst.transform[7], inst.transform[11], 1
        };
        HybridPushConstants pc;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                float sum = 0;
                for (int k = 0; k < 4; k++)
                    sum += viewProj[r + k * 4] * model[k + c * 4];
                pc.mvp[r + c * 4] = sum;
            }
        }
        pc.camPos[0] = camWorldPos_[0];
        pc.camPos[1] = camWorldPos_[1];
        pc.camPos[2] = camWorldPos_[2];
        pc.camPos[3] = 0;
        pc.instanceIndex = (uint32_t)i;
        pc.blasIndex = inst.customIndex;

        vkCmdPushConstants(cmd, hybridPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(HybridPushConstants), &pc);

        VkBuffer vb = blas.vertexBuf.buffer;
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &offset);
        vkCmdBindIndexBuffer(cmd, blas.indexBuf.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, blas.indexCount, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
}

void Renderer::ShutdownHybridGBuffer() {
    if (!hybridGBufferReady_ && !hybridRenderPass_) return;
    VkDevice device = context_->GetDevice();

    if (hybridPipeline_) { vkDestroyPipeline(device, hybridPipeline_, nullptr); hybridPipeline_ = VK_NULL_HANDLE; }
    if (hybridPipelineLayout_) { vkDestroyPipelineLayout(device, hybridPipelineLayout_, nullptr); hybridPipelineLayout_ = VK_NULL_HANDLE; }
    if (hybridFramebuffer_) { vkDestroyFramebuffer(device, hybridFramebuffer_, nullptr); hybridFramebuffer_ = VK_NULL_HANDLE; }
    if (hybridRenderPass_) { vkDestroyRenderPass(device, hybridRenderPass_, nullptr); hybridRenderPass_ = VK_NULL_HANDLE; }

    auto destroyImg = [&](VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        if (view) { vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
        if (img) { vkDestroyImage(device, img, nullptr); img = VK_NULL_HANDLE; }
        if (mem) { vkFreeMemory(device, mem, nullptr); mem = VK_NULL_HANDLE; }
    };
    destroyImg(hybridPrimIdImage_, hybridPrimIdMemory_, hybridPrimIdView_);
    destroyImg(hybridInstanceInfoImage_, hybridInstanceInfoMemory_, hybridInstanceInfoView_);
    destroyImg(hybridDepthImage_, hybridDepthMemory_, hybridDepthView_);
    destroyImg(hybridZBuffer_, hybridZBufferMemory_, hybridZBufferView_);

    hybridGBufferReady_ = false;
    hybridGBufferRendered_ = false;
    Log(L"[VK Renderer] Hybrid G-buffer shutdown\n");
}

} // namespace vk
} // namespace acpt
