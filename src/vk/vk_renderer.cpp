#include "vk_renderer.h"
#include "vk_context.h"
#include "vk_pipeline.h"
#include "vk_geometry.h"
#include "vk_rasterizer.h"
#include "vk_accel_structure.h"
#include "vk_rt_pipeline.h"
#include "vk_wavefront_pipeline.h"
#include "vk_interop.h"
#include "vk_texture_manager.h"
#include "ignis_log.h"
#include "ignis_config.h"
#include "vk_check.h"
#include "nrd_vulkan_integration.h"
#include <vector>
#include <fstream>
#include <cstring>
#include <chrono>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Forward declaration — defined in ignis_api.cpp
namespace acpt {
    PathTracerConfig* VK_GetConfig();
}

// ImGui stub declarations (implemented in imgui_stub.cpp)
bool ImGui_Init(HWND hwnd, VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                VkQueue queue, uint32_t queueFamily, VkRenderPass renderPass,
                VkDescriptorPool descriptorPool);
void ImGui_NewFrame();
void ImGui_Render(VkCommandBuffer cmd);
void ImGui_Shutdown();
bool ImGui_WantCaptureMouse();

namespace acpt {
namespace vk {

bool Renderer::Initialize(HWND hwnd, uint32_t width, uint32_t height) {
    width_ = width;
    height_ = height;

    // Create modules
    context_ = new Context();
    pipeline_ = new Pipeline();
    geometry_ = new Geometry();
    rasterizer_ = new Rasterizer();

    // Initialize context (instance, device, swapchain)
    if (!context_->Initialize(hwnd, width, height)) {
        Log(L"[VK Renderer] ERROR: Failed to initialize context\n");
        return false;
    }

    // Initialize pipeline (shaders, render pass)
    if (!pipeline_->Initialize(context_)) {
        Log(L"[VK Renderer] ERROR: Failed to initialize pipeline\n");
        return false;
    }

    if (!pipeline_->CreateGraphicsPipeline("shaders/basic.vert.spv", "shaders/basic.frag.spv")) {
        Log(L"[VK Renderer] ERROR: Failed to create graphics pipeline\n");
        return false;
    }

    // Initialize geometry
    if (!geometry_->Initialize(context_)) {
        Log(L"[VK Renderer] ERROR: Failed to initialize geometry\n");
        return false;
    }

    // Create scene meshes
    sphereMesh_ = new Mesh();
    *sphereMesh_ = Geometry::CreateSphere(1.0f, 32, 32);
    if (!geometry_->UploadMesh(*sphereMesh_)) {
        Log(L"[VK Renderer] ERROR: Failed to upload sphere mesh\n");
        return false;
    }
    Log(L"[VK Renderer] Sphere mesh created (%zu vertices, %zu indices)\n",
        sphereMesh_->vertices.size(), sphereMesh_->indices.size());

    planeMesh_ = new Mesh();
    *planeMesh_ = Geometry::CreatePlane(10.0f);
    if (!geometry_->UploadMesh(*planeMesh_)) {
        Log(L"[VK Renderer] ERROR: Failed to upload plane mesh\n");
        return false;
    }
    Log(L"[VK Renderer] Plane mesh created (%zu vertices, %zu indices)\n",
        planeMesh_->vertices.size(), planeMesh_->indices.size());

    // Initialize rasterizer (framebuffers, uniforms, descriptors)
    if (!rasterizer_->Initialize(context_, pipeline_, geometry_)) {
        Log(L"[VK Renderer] ERROR: Failed to initialize rasterizer\n");
        return false;
    }

    rasterizer_->SetScene(sphereMesh_, planeMesh_);

    // Create command buffers and sync objects
    if (!CreateCommandBuffers()) return false;
    if (!CreateSyncObjects()) return false;

    // Try to initialize RT modules
    if (context_->IsRayQuerySupported()) {
        InitRT();
    } else {
        Log(L"[VK Renderer] RT not supported, using rasterization fallback\n");
    }

    Log(L"[VK Renderer] ========== INITIALIZATION COMPLETE ==========\n");
    return true;
}

bool Renderer::InitRT() {
    // Create interop (shared image for D3D11) — always at display resolution
    interop_ = new Interop();
    if (!interop_->Initialize(context_, width_, height_)) {
        Log(L"[VK Renderer] WARNING: Interop initialization failed\n");
        delete interop_;
        interop_ = nullptr;
        return false;
    }

    // Initialize DLSS (before NRD so we know render resolution)
    renderWidth_ = width_;
    renderHeight_ = height_;

    PathTracerConfig* cfg = VK_GetConfig();
    if (cfg && cfg->dlssEnabled) {
        dlss_ = new DLSS_NGX();
        DLSSQualityMode mode = static_cast<DLSSQualityMode>(cfg->dlssQualityMode);
        if (dlss_->Initialize(context_->GetInstance(), context_->GetDevice(),
                              context_->GetPhysicalDevice(), context_->GetCommandPool(),
                              context_->GetGraphicsQueue(), width_, height_, mode)) {
            if (dlss_->IsSupported()) {
                dlss_->GetCurrentRenderResolution(&renderWidth_, &renderHeight_);
                dlssActive_ = true;
                Log(L"[VK Renderer] DLSS active: render %ux%u -> display %ux%u\n",
                    renderWidth_, renderHeight_, width_, height_);
            } else {
                Log(L"[VK Renderer] DLSS not supported on this GPU, using native resolution\n");
            }
        } else {
            Log(L"[VK Renderer] DLSS initialization failed, using native resolution\n");
            delete dlss_;
            dlss_ = nullptr;
        }
    }

    // Create intermediate DLSS color input image (render resolution, RGBA16F for HDR)
    if (dlssActive_) {
        VkDevice device = context_->GetDevice();

        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        imgInfo.extent = { renderWidth_, renderHeight_, 1 };
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(device, &imgInfo, nullptr, &dlssColorInput_) != VK_SUCCESS) {
            Log(L"[VK Renderer] ERROR: Failed to create DLSS color input image\n");
            dlssActive_ = false;
            renderWidth_ = width_;
            renderHeight_ = height_;
        } else {
            VkMemoryRequirements memReqs;
            vkGetImageMemoryRequirements(device, dlssColorInput_, &memReqs);

            VkPhysicalDeviceMemoryProperties memProps;
            vkGetPhysicalDeviceMemoryProperties(context_->GetPhysicalDevice(), &memProps);
            uint32_t memTypeIdx = UINT32_MAX;
            for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
                if ((memReqs.memoryTypeBits & (1 << i)) &&
                    (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                    memTypeIdx = i;
                    break;
                }
            }

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memReqs.size;
            allocInfo.memoryTypeIndex = memTypeIdx;

            if (vkAllocateMemory(device, &allocInfo, nullptr, &dlssColorInputMemory_) != VK_SUCCESS ||
                vkBindImageMemory(device, dlssColorInput_, dlssColorInputMemory_, 0) != VK_SUCCESS) {
                Log(L"[VK Renderer] ERROR: Failed to allocate DLSS color input memory\n");
                vkDestroyImage(device, dlssColorInput_, nullptr);
                dlssColorInput_ = VK_NULL_HANDLE;
                dlssActive_ = false;
                renderWidth_ = width_;
                renderHeight_ = height_;
            } else {
                VkImageViewCreateInfo viewInfo{};
                viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                viewInfo.image = dlssColorInput_;
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
                viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

                if (vkCreateImageView(device, &viewInfo, nullptr, &dlssColorInputView_) != VK_SUCCESS) {
                    Log(L"[VK Renderer] ERROR: Failed to create DLSS color input view\n");
                    vkFreeMemory(device, dlssColorInputMemory_, nullptr);
                    vkDestroyImage(device, dlssColorInput_, nullptr);
                    dlssColorInput_ = VK_NULL_HANDLE;
                    dlssColorInputMemory_ = VK_NULL_HANDLE;
                    dlssActive_ = false;
                    renderWidth_ = width_;
                    renderHeight_ = height_;
                }
            }
        }

        // Create DLSS HDR output image (display resolution, RGBA16F)
        if (dlssActive_) {
            VkImageCreateInfo hdrOutInfo{};
            hdrOutInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            hdrOutInfo.imageType = VK_IMAGE_TYPE_2D;
            hdrOutInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            hdrOutInfo.extent = { width_, height_, 1 };
            hdrOutInfo.mipLevels = 1;
            hdrOutInfo.arrayLayers = 1;
            hdrOutInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            hdrOutInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            hdrOutInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            hdrOutInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            hdrOutInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            if (vkCreateImage(device, &hdrOutInfo, nullptr, &dlssHdrOutput_) != VK_SUCCESS) {
                Log(L"[VK Renderer] ERROR: Failed to create DLSS HDR output image\n");
            } else {
                VkMemoryRequirements memReqs;
                vkGetImageMemoryRequirements(device, dlssHdrOutput_, &memReqs);

                VkPhysicalDeviceMemoryProperties memProps;
                vkGetPhysicalDeviceMemoryProperties(context_->GetPhysicalDevice(), &memProps);
                uint32_t memTypeIdx = UINT32_MAX;
                for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
                    if ((memReqs.memoryTypeBits & (1 << i)) &&
                        (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                        memTypeIdx = i;
                        break;
                    }
                }

                VkMemoryAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                allocInfo.allocationSize = memReqs.size;
                allocInfo.memoryTypeIndex = memTypeIdx;

                if (vkAllocateMemory(device, &allocInfo, nullptr, &dlssHdrOutputMemory_) != VK_SUCCESS ||
                    vkBindImageMemory(device, dlssHdrOutput_, dlssHdrOutputMemory_, 0) != VK_SUCCESS) {
                    Log(L"[VK Renderer] ERROR: Failed to allocate DLSS HDR output memory\n");
                    vkDestroyImage(device, dlssHdrOutput_, nullptr);
                    dlssHdrOutput_ = VK_NULL_HANDLE;
                } else {
                    VkImageViewCreateInfo viewInfo{};
                    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                    viewInfo.image = dlssHdrOutput_;
                    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
                    viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

                    if (vkCreateImageView(device, &viewInfo, nullptr, &dlssHdrOutputView_) != VK_SUCCESS) {
                        Log(L"[VK Renderer] ERROR: Failed to create DLSS HDR output view\n");
                        vkFreeMemory(device, dlssHdrOutputMemory_, nullptr);
                        vkDestroyImage(device, dlssHdrOutput_, nullptr);
                        dlssHdrOutput_ = VK_NULL_HANDLE;
                        dlssHdrOutputMemory_ = VK_NULL_HANDLE;
                    } else {
                        Log(L"[VK Renderer] DLSS HDR output image created (%ux%u RGBA16F)\n", width_, height_);
                    }
                }
            }
        }

        // Transition dlssColorInput_ and dlssHdrOutput_ to GENERAL layout
        if (dlssActive_) {
            VkCommandBuffer cmd;
            VkCommandBufferAllocateInfo cmdInfo{};
            cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmdInfo.commandPool = context_->GetCommandPool();
            cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdInfo.commandBufferCount = 1;
            vkAllocateCommandBuffers(device, &cmdInfo, &cmd);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            VkImageMemoryBarrier barriers[2] = {};
            uint32_t barrierCount = 1;

            barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].image = dlssColorInput_;
            barriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            barriers[0].srcAccessMask = 0;
            barriers[0].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

            if (dlssHdrOutput_) {
                barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[1].image = dlssHdrOutput_;
                barriers[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                barriers[1].srcAccessMask = 0;
                barriers[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrierCount = 2;
            }

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, barrierCount, barriers);

            vkEndCommandBuffer(cmd);
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(context_->GetGraphicsQueue());
            vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);
        }
    }

    // Try Ray Reconstruction if requested (after DLSS SR init)
    if (cfg && cfg->dlssRREnabled && dlssActive_ && dlss_) {
        if (dlss_->InitializeRR()) {
            dlssRRActive_ = true;
            Log(L"[VK Renderer] DLSS Ray Reconstruction active — will skip NRD\n");
        } else {
            Log(L"[VK Renderer] RR unavailable, falling back to NRD + SR\n");
        }
    }

    // Create acceleration structure builder
    accelBuilder_ = new AccelStructureBuilder();
    if (!accelBuilder_->Initialize(context_)) {
        Log(L"[VK Renderer] WARNING: AccelStructure initialization failed\n");
        delete accelBuilder_;
        accelBuilder_ = nullptr;
        return false;
    }

    // Create RT pipeline
    rtPipeline_ = new RTPipeline();
    if (!rtPipeline_->Initialize(context_, accelBuilder_, interop_)) {
        Log(L"[VK Renderer] WARNING: RT pipeline initialization failed\n");
        delete rtPipeline_;
        rtPipeline_ = nullptr;
        return false;
    }

    // When DLSS active, point RT storage image to intermediate buffer
    if (dlssActive_) {
        rtPipeline_->UpdateStorageImage(dlssColorInputView_);
    }

    // Create G-buffers and initialize NRD denoiser (at render resolution)
    if (!InitNRD()) {
        // NRD failed — create G-buffers anyway so shader bindings are valid
        rtPipeline_->CreateGBuffers(renderWidth_, renderHeight_);
        Log(L"[VK Renderer] RT modules initialized (no denoiser)\n");
    } else {
        Log(L"[VK Renderer] RT modules initialized with NRD denoiser\n");
    }

    // Initialize wavefront pipeline (experimental)
    if (cfg && cfg->useWavefront) {
        wavefrontPipeline_ = new WavefrontPipeline();
        if (!wavefrontPipeline_->Initialize(context_, rtPipeline_, renderWidth_, renderHeight_, cfg->maxBounces)) {
            Log(L"[VK Renderer] WARNING: Wavefront init failed, using monolithic raygen\n");
            delete wavefrontPipeline_;
            wavefrontPipeline_ = nullptr;
        }
    }

    return true;
}

bool Renderer::BuildAccelStructure(const float* vertices, uint32_t vertexCount,
                                    const uint32_t* indices, uint32_t indexCount) {
    if (!accelBuilder_) return false;
    int blasIdx = accelBuilder_->BuildBLAS(vertices, vertexCount, indices, indexCount, false);
    return blasIdx >= 0;
}

bool Renderer::BuildTLAS() {
    if (!accelBuilder_) return false;
    if (!accelBuilder_->BuildTLAS()) return false;
    if (rtPipeline_) {
        rtPipeline_->UpdateTLASDescriptor();
    }
    rtReady_ = true;
    Log(L"[VK Renderer] RT ready for dispatch\n");
    return true;
}

int Renderer::BuildBLAS(const float* vertices, uint32_t vertexCount,
                         const uint32_t* indices, uint32_t indexCount) {
    if (!accelBuilder_) return -1;
    return accelBuilder_->BuildBLAS(vertices, vertexCount, indices, indexCount, false);
}

bool Renderer::RefitBLAS(int blasIndex, const float* vertices, uint32_t vertexCount,
                          const uint32_t* indices, uint32_t indexCount) {
    if (!accelBuilder_) return false;
    return accelBuilder_->RefitBLAS(blasIndex, vertices, vertexCount, indices, indexCount);
}

bool Renderer::UploadBLASAttributes(int blasIndex, const float* normals, const float* uvs, uint32_t vertexCount) {
    if (!accelBuilder_) return false;
    return accelBuilder_->UploadBLASAttributes(blasIndex, normals, uvs, vertexCount);
}

bool Renderer::UploadBLASPrimitiveMaterials(int blasIndex, const uint32_t* materialIds, uint32_t primitiveCount) {
    if (!accelBuilder_) return false;
    return accelBuilder_->UploadBLASPrimitiveMaterials(blasIndex, materialIds, primitiveCount);
}

bool Renderer::UploadBLASPrimitiveYBounds(int blasIndex, const float* yBounds, uint32_t primitiveCount) {
    if (!accelBuilder_) return false;
    return accelBuilder_->UploadBLASPrimitiveYBounds(blasIndex, yBounds, primitiveCount);
}

void Renderer::UploadMaterialBuffer(const void* materials, uint32_t count) {
    if (rtPipeline_) {
        rtPipeline_->UpdateMaterialBuffer(static_cast<const vk::GPUMaterial*>(materials), count);
    }
}

void Renderer::UploadEmissiveTriangles(const float* data, uint32_t triangleCount) {
    if (rtPipeline_) {
        rtPipeline_->UpdateEmissiveTriangleBuffer(data, triangleCount);
    }
}

void Renderer::UploadLightTree(const void* nodes, uint32_t nodeCount,
                                const void* emitters, uint32_t emitterCount) {
    if (rtPipeline_) {
        rtPipeline_->UpdateLightTreeBuffer(nodes, nodeCount);
    }
}

void Renderer::UpdateTextureDescriptors(void* texManager) {
    if (rtPipeline_) {
        rtPipeline_->UpdateTextureDescriptors(static_cast<vk::TextureManager*>(texManager));
    }
}

bool Renderer::BuildTLASInstanced(const std::vector<vk::TLASInstance>& instances) {
    if (!accelBuilder_) return false;
    if (!accelBuilder_->BuildTLAS(instances)) return false;
    if (rtPipeline_) {
        rtPipeline_->UpdateTLASDescriptor();
    }

    // Capture per-instance transforms for motion vectors.
    // prev is managed by RenderFrameRT (swapped after each render), so we
    // only update curr here.  On first build or instance-count change we
    // initialise prev = curr so the shader sees zero motion.
    currInstanceTransforms_.resize(instances.size() * 12);
    for (size_t i = 0; i < instances.size(); i++) {
        memcpy(&currInstanceTransforms_[i * 12], instances[i].transform, 12 * sizeof(float));
    }
    if (instances.size() != instanceTransformCount_ || prevInstanceTransforms_.empty()) {
        // No valid history — initialise prev = curr (zero motion vectors)
        prevInstanceTransforms_ = currInstanceTransforms_;
    }
    instanceTransformCount_ = (uint32_t)instances.size();

    rtReady_ = true;
    return true;
}

void Renderer::UpdateCamera(const CameraUBO& camera) {
    if (rtPipeline_) {
        rtPipeline_->UpdateCamera(camera);
    }
    // Track the Python-side frame index for shader use.
    // Do NOT reset renderer frameIndex_ here — NRD's anti-lag handles
    // shadow changes naturally without destructive CLEAR_AND_RESTART.

    // Store matrices for NRD
    memcpy(lastView_, camera.view, 64);
    memcpy(lastProj_, camera.proj, 64);
    memcpy(lastViewPrev_, camera.viewPrev, 64);
    memcpy(lastProjPrev_, camera.projPrev, 64);

    // Store jitter for NRD and DLSS
    prevJitterX_ = jitterX_;
    prevJitterY_ = jitterY_;
    jitterX_ = camera.jitterData[0];
    jitterY_ = camera.jitterData[1];

}

bool Renderer::ReadPickResult(uint32_t& outCustomIndex, uint32_t& outPrimitiveId, uint32_t& outMaterialId) {
    if (!rtPipeline_) return false;
    auto result = rtPipeline_->ReadPickResult();
    if (!result.valid) return false;
    outCustomIndex = result.customIndex;
    outPrimitiveId = result.primitiveId;
    outMaterialId = result.materialId;
    rtPipeline_->ResetPickBuffer();
    return true;
}

void Renderer::RenderFrameRT() {
    if (!rtReady_ || !rtPipeline_ || !interop_) {
        Log(L"[VK Renderer] RenderFrameRT SKIPPED: rtReady=%d rtPipeline=%p interop=%p\n",
            (int)rtReady_, (void*)rtPipeline_, (void*)interop_);
        return;
    }


    VkDevice device = context_->GetDevice();
    VkCommandBuffer cmd = commandBuffers_[currentFrame_];

    // Wait for previous frame
    vkWaitForFences(device, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &inFlightFences_[currentFrame_]);

    // Single command buffer: RT → NRD → Composite → ImGui → Readback
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (!VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo))) return;

    // Check debug view mode early (needed for DLSS bypass)
    PathTracerConfig* rtCfgEarly = VK_GetConfig();
    bool debugViewActive = rtCfgEarly && rtCfgEarly->debugView > 0;

    // DLSS debug bypass: when debug views are active, route RT output directly
    // to the interop image instead of through DLSS (which would show stale frames)
    if (dlssActive_ && rtPipeline_) {
        bool wantDebugBypass = debugViewActive;
        if (wantDebugBypass != dlssDebugBypass_) {
            VkImageView targetView = wantDebugBypass
                ? interop_->GetSharedImageView()
                : dlssColorInputView_;
            if (targetView != VK_NULL_HANDLE) {
                rtPipeline_->UpdateStorageImage(targetView);
                dlssDebugBypass_ = wantDebugBypass;
            }
        }
    }

    // Upload previous-frame instance transforms for per-object motion vectors.
    // This must happen every frame so that once an object stops moving,
    // prev == curr and the shader outputs zero motion vectors (no ghosting).
    if (rtPipeline_ && instanceTransformCount_ > 0) {
        rtPipeline_->UpdatePrevTransforms(prevInstanceTransforms_.data(), instanceTransformCount_);
    }

    bool diagFlush = false;  // Set true to flush GPU between stages for crash isolation

    // 1. Path tracing dispatch (wavefront or monolithic)
    interop_->TransitionForRTWrite(cmd);
    if (wavefrontPipeline_ && wavefrontPipeline_->IsReady()) {
        PathTracerConfig* wfCfg = VK_GetConfig();
        if (frameIndex_ < 3) {
            Log(L"[WF] frame %u: dispatch %ux%u dlssActive=%d dlssRR=%d tonemapReady=%d\n",
                frameIndex_, renderWidth_, renderHeight_, (int)dlssActive_, (int)dlssRRActive_, (int)tonemapReady_);
        }
        wavefrontPipeline_->RecordDispatch(cmd, renderWidth_, renderHeight_,
            rtPipeline_->GetDescriptorSet(), wfCfg ? wfCfg->maxBounces : 2);
    } else {
        rtPipeline_->RecordDispatch(cmd, renderWidth_, renderHeight_);
    }

    if (diagFlush) {
        if (!VK_CHECK(vkEndCommandBuffer(cmd))) return;
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        Log(L"[DIAG] frame %u: submitting RT dispatch (%ux%u, %u instances)...\n",
            frameIndex_, renderWidth_, renderHeight_, instanceTransformCount_);
        VkResult r = vkQueueSubmit(context_->GetGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
        if (r != VK_SUCCESS) { Log(L"[DIAG] RT submit FAILED: %d\n", r); return; }
        r = vkQueueWaitIdle(context_->GetGraphicsQueue());
        if (r != VK_SUCCESS) { Log(L"[DIAG] RT waitIdle FAILED (DEVICE_LOST): %d\n", r); return; }
        Log(L"[DIAG] frame %u: RT dispatch OK\n", frameIndex_);
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi2{}; bi2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &bi2);
    }

    // SHARC resolve: merge write buffer → read buffer (EMA + aging)
    // TODO: SHARC causes VK_ERROR_DEVICE_LOST on complex scenes — disabled until investigated
    if (false && sharcResolveReady_ && rtPipeline_->HasSHARCBuffers()) {
        VkMemoryBarrier rtToSharc{};
        rtToSharc.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        rtToSharc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        rtToSharc.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &rtToSharc, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sharcResolvePipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            sharcResolvePipelineLayout_, 0, 1, &sharcResolveDescriptorSet_, 0, nullptr);
        vkCmdPushConstants(cmd, sharcResolvePipelineLayout_,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &frameIndex_);

        // 2^22 entries / 256 threads per group = 16384 groups
        vkCmdDispatch(cmd, 16384, 1, 1);
    }

    // (SHARC diag flush removed — SHARC is disabled)

    // Wavefront path: skip NRD denoise + composite — K5 wrote G-buffers + output
    // But DLSS SR + tonemap still need to run if DLSS is active
    bool wavefrontActive = wavefrontPipeline_ && wavefrontPipeline_->IsReady();

    if (wavefrontActive && dlssActive_ && dlss_ && dlss_->IsInitialized() && dlss_->IsSupported() && !dlssRRActive_) {
        if (frameIndex_ < 3) Log(L"[WF] frame %u: running DLSS SR + tonemap path\n", frameIndex_);
        // Barrier: K5 compute writes → DLSS reads
        VkMemoryBarrier wfBarrier{};
        wfBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        wfBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        wfBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &wfBarrier, 0, nullptr, 0, nullptr);

        // DLSS SR upscaling (K5 wrote to dlssColorInput in HDR)
        static auto s_lastWfDlssTime = std::chrono::high_resolution_clock::now();
        auto wfDlssNow = std::chrono::high_resolution_clock::now();
        float wfDlssDelta = std::chrono::duration<float, std::milli>(wfDlssNow - s_lastWfDlssTime).count();
        s_lastWfDlssTime = wfDlssNow;
        if (wfDlssDelta < 1.0f) wfDlssDelta = 1.0f;
        if (wfDlssDelta > 100.0f) wfDlssDelta = 100.0f;

        dlss_->Evaluate(cmd,
            dlssColorInput_, dlssColorInputView_,
            rtPipeline_->GetDlssDepthImage(), rtPipeline_->GetDlssDepthView(),
            rtPipeline_->GetMotionVectorsImage(), rtPipeline_->GetMotionVectorsView(),
            dlssHdrOutput_, dlssHdrOutputView_,
            VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
            jitterX_, jitterY_, wfDlssDelta / 1000.0f, 0.0f, false,
            rtPipeline_->GetReactiveMaskImage(), rtPipeline_->GetReactiveMaskView());

        // Barrier: DLSS → tonemap
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &wfBarrier, 0, nullptr, 0, nullptr);

        // Tonemap: DLSS HDR → LDR interop
        PathTracerConfig* wfCfg2 = VK_GetConfig();
        if (tonemapReady_ && wfCfg2) {
            UpdateTonemapDescriptors();
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tonemapPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                tonemapPipelineLayout_, 0, 1, &tonemapDescSet_, 0, nullptr);
            struct { uint32_t mode; float exposure, saturation, contrast; } tp;
            tp.mode = static_cast<uint32_t>(wfCfg2->ptTonemapMode);
            tp.exposure = wfCfg2->ptAutoExposure ? computedExposure_ : wfCfg2->ptExposure;
            tp.saturation = wfCfg2->ptSaturation;
            tp.contrast = wfCfg2->ptContrast;
            vkCmdPushConstants(cmd, tonemapPipelineLayout_,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tp), &tp);
            vkCmdDispatch(cmd, (width_ + 7) / 8, (height_ + 7) / 8, 1);

            VkMemoryBarrier tmBarrier{};
            tmBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            tmBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            tmBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &tmBarrier, 0, nullptr, 0, nullptr);
        }
    }

    // 2. Ray Reconstruction path (replaces NRD + composite + DLSS SR)
    // RR still runs with wavefront — it denoises + upscales K5's noisy output
    if (dlssRRActive_ && dlss_ && dlss_->IsRRActive()) {
        // Barrier: RT/compute writes → RR reads (G-buffers and noisy color)
        VkMemoryBarrier rtToRR{};
        rtToRR.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        rtToRR.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        rtToRR.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        VkPipelineStageFlags srcStage = wavefrontActive ?
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT :
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
        vkCmdPipelineBarrier(cmd, srcStage,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &rtToRR, 0, nullptr, 0, nullptr);

        // Measure frame delta
        static auto s_lastRRTime = std::chrono::high_resolution_clock::now();
        auto rrNow = std::chrono::high_resolution_clock::now();
        float rrDeltaMs = std::chrono::duration<float, std::milli>(rrNow - s_lastRRTime).count();
        s_lastRRTime = rrNow;
        if (rrDeltaMs < 1.0f) rrDeltaMs = 1.0f;
        if (rrDeltaMs > 100.0f) rrDeltaMs = 100.0f;

        bool rrReset = (frameIndex_ == 0);

        dlss_->EvaluateRR(cmd,
            dlssColorInput_, dlssColorInputView_,           // noisy color (render res)
            dlssHdrOutput_, dlssHdrOutputView_,              // output (display res)
            rtPipeline_->GetDlssDepthImage(),                // NDC depth
            rtPipeline_->GetDlssDepthView(),
            rtPipeline_->GetMotionVectorsImage(),            // MVs
            rtPipeline_->GetMotionVectorsView(),
            rtPipeline_->GetNormalRoughnessImage(),          // normals + roughness
            rtPipeline_->GetNormalRoughnessView(),
            rtPipeline_->GetAlbedoBufferImage(),             // albedo
            rtPipeline_->GetAlbedoBufferView(),
            jitterX_, jitterY_,
            rrDeltaMs / 1000.0f,
            lastView_, lastProj_,
            rtPipeline_->GetSpecularAlbedoImage(),           // EnvBRDFApprox specular albedo
            rtPipeline_->GetSpecularAlbedoView(),
            rtPipeline_->GetSpecularMVImage(),               // specular motion vectors
            rtPipeline_->GetSpecularMVView(),
            rtPipeline_->GetDiffuseRadianceImage(),          // diffuse hit distance (.a)
            rtPipeline_->GetDiffuseRadianceView(),
            rtPipeline_->GetSpecularRadianceImage(),         // specular hit distance (.a)
            rtPipeline_->GetSpecularRadianceView(),
            rrReset);

        // Barrier: RR writes → tonemap reads
        VkMemoryBarrier rrBarrier{};
        rrBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        rrBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        rrBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &rrBarrier, 0, nullptr, 0, nullptr);

        // Tonemap: RR HDR output → LDR interop
        PathTracerConfig* cfg = VK_GetConfig();
        if (tonemapReady_ && cfg) {
            UpdateTonemapDescriptors();
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tonemapPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                tonemapPipelineLayout_, 0, 1, &tonemapDescSet_, 0, nullptr);

            struct { uint32_t mode; float exposure, saturation, contrast; } tonemapPush;
            tonemapPush.mode = static_cast<uint32_t>(cfg->ptTonemapMode);
            tonemapPush.exposure = cfg->ptAutoExposure ? computedExposure_ : cfg->ptExposure;
            tonemapPush.saturation = cfg->ptSaturation;
            tonemapPush.contrast = cfg->ptContrast;
            vkCmdPushConstants(cmd, tonemapPipelineLayout_,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tonemapPush), &tonemapPush);

            vkCmdDispatch(cmd, (width_ + 7) / 8, (height_ + 7) / 8, 1);

            // Barrier: tonemap writes → ImGui/readback reads
            VkMemoryBarrier tonemapBarrier{};
            tonemapBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            tonemapBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            tonemapBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 1, &tonemapBarrier, 0, nullptr, 0, nullptr);
        }
    }
    // 2b. NRD denoise path (traditional pipeline)
    else if (!wavefrontActive && nrdInitialized_) {
        // Barrier: RT writes → NRD reads (G-buffers and output image)
        VkMemoryBarrier rtToNrd{};
        rtToNrd.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        rtToNrd.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        rtToNrd.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &rtToNrd, 0, nullptr, 0, nullptr);

        // Measure real frame delta for NRD temporal accumulation
        static auto s_lastFrameTime = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        float frameDeltaMs = std::chrono::duration<float, std::milli>(now - s_lastFrameTime).count();
        s_lastFrameTime = now;
        // Clamp to reasonable range (1ms = 1000fps, 100ms = 10fps)
        if (frameDeltaMs < 1.0f) frameDeltaMs = 1.0f;
        if (frameDeltaMs > 100.0f) frameDeltaMs = 100.0f;

        // Set sun direction for SIGMA shadow denoiser
        {
            PathTracerConfig* nrdCfg = VK_GetConfig();
            float nrdSunDir[3] = {0.4f, 0.85f, 0.35f};
            if (nrdCfg) {
                float az = nrdCfg->sunAzimuth * 3.14159265f / 180.0f;
                float el = nrdCfg->sunElevation * 3.14159265f / 180.0f;
                nrdSunDir[0] = sinf(az) * cosf(el);
                nrdSunDir[1] = sinf(el);
                nrdSunDir[2] = cosf(az) * cosf(el);
            }
            acpt::NRD_Vulkan_SetSunDirection(nrdSunDir[0], nrdSunDir[1], nrdSunDir[2]);
        }

        // Pass actual sub-pixel jitter to NRD for correct temporal reprojection.
        // NRD expects pixel-space jitter matching what was applied in the shader.
        acpt::NRD_Vulkan_Denoise(cmd, frameIndex_,
            lastView_, lastProj_, lastViewPrev_, lastProjPrev_,
            jitterX_, jitterY_, prevJitterX_, prevJitterY_, frameDeltaMs,
            dlssActive_);

        if (diagFlush) {
            if (!VK_CHECK(vkEndCommandBuffer(cmd))) return;
            VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
            Log(L"[DIAG] frame %u: submitting NRD denoise...\n", frameIndex_);
            VkResult r = vkQueueSubmit(context_->GetGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
            if (r != VK_SUCCESS) { Log(L"[DIAG] NRD submit FAILED: %d\n", r); return; }
            vkQueueWaitIdle(context_->GetGraphicsQueue());
            Log(L"[DIAG] frame %u: NRD denoise OK\n", frameIndex_);
            vkResetCommandBuffer(cmd, 0);
            VkCommandBufferBeginInfo bi2{}; bi2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            vkBeginCommandBuffer(cmd, &bi2);
        }

        // 3. Composite dispatch (reads NRD outputs, writes to interop image)
        // Skip when debug view is active — raygen already wrote final output directly
        if (compositeReady_ && !debugViewActive) {
            // Barrier: NRD writes → Composite reads
            VkMemoryBarrier nrdToComposite{};
            nrdToComposite.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            nrdToComposite.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            nrdToComposite.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &nrdToComposite, 0, nullptr, 0, nullptr);

            // Update composite descriptors (binds NRD outputs + interop image)
            UpdateCompositeDescriptors();

            // Bind composite pipeline and dispatch
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compositePipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                compositePipelineLayout_, 0, 1, &compositeDescriptorSet_, 0, nullptr);

            PathTracerConfig* cfg = VK_GetConfig();

            // CPU readback of auto-exposure from previous frame (1-frame delay, standard practice)
            if (cfg->ptAutoExposure && exposureResolveReady_ && exposureStagingSSBO_.memory != VK_NULL_HANDLE) {
                void* mapped;
                if (vkMapMemory(device, exposureStagingSSBO_.memory, 0, 12, 0, &mapped) == VK_SUCCESS) {
                    float* data = reinterpret_cast<float*>(static_cast<uint8_t*>(mapped) + 8);  // offset to currentExposure
                    computedExposure_ = *data;
                    vkUnmapMemory(device, exposureStagingSSBO_.memory);
                }
            }

            struct { uint32_t mode; uint32_t tonemapMode; float exposure; float saturation; float contrast; uint32_t hdrOutput; } compositeParams;
            compositeParams.mode = 1;  // Normal NRD composite
            compositeParams.tonemapMode = static_cast<uint32_t>(cfg->ptTonemapMode);
            compositeParams.exposure = cfg->ptAutoExposure ? computedExposure_ : cfg->ptExposure;
            compositeParams.saturation = cfg->ptSaturation;
            compositeParams.contrast = cfg->ptContrast;
            compositeParams.hdrOutput = dlssActive_ ? 1 : 0;
            vkCmdPushConstants(cmd, compositePipelineLayout_,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compositeParams), &compositeParams);

            vkCmdDispatch(cmd, (renderWidth_ + 7) / 8, (renderHeight_ + 7) / 8, 1);

            // Auto-exposure resolve: read accumulated luminance, compute EMA exposure, reset accumulators
            if (exposureResolveReady_ && cfg->ptAutoExposure) {
                // Barrier: composite SSBO writes → resolve reads/writes
                VkMemoryBarrier exposureBarrier{};
                exposureBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                exposureBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                exposureBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &exposureBarrier, 0, nullptr, 0, nullptr);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposureResolvePipeline_);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    exposureResolvePipelineLayout_, 0, 1, &exposureResolveDescSet_, 0, nullptr);

                float dt = 1.0f / 60.0f;  // approximate; good enough for EMA
                struct { float key, speed, minE, maxE; } resolvePush;
                resolvePush.key = cfg->ptAutoExposureKey;
                resolvePush.speed = 1.0f - expf(-cfg->ptAutoExposureSpeed * dt);  // frame-rate independent EMA
                resolvePush.minE = cfg->ptAutoExposureMin;
                resolvePush.maxE = cfg->ptAutoExposureMax;
                vkCmdPushConstants(cmd, exposureResolvePipelineLayout_,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(resolvePush), &resolvePush);

                vkCmdDispatch(cmd, 1, 1, 1);

                // Barrier: resolve writes → staging copy
                VkMemoryBarrier resolveCopyBarrier{};
                resolveCopyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                resolveCopyBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                resolveCopyBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &resolveCopyBarrier, 0, nullptr, 0, nullptr);

                // Copy SSBO to staging for CPU readback next frame
                VkBufferCopy copyRegion{};
                copyRegion.size = 12;
                vkCmdCopyBuffer(cmd, exposureSSBO_.buffer, exposureStagingSSBO_.buffer, 1, &copyRegion);
            }

            // Barrier: Composite writes → Rain/DLSS/ImGui/readback reads
            VkMemoryBarrier compositeBarrier{};
            compositeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            compositeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            compositeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 1, &compositeBarrier, 0, nullptr, 0, nullptr);

            if (diagFlush) {
                if (!VK_CHECK(vkEndCommandBuffer(cmd))) return;
                VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
                Log(L"[DIAG] frame %u: submitting composite...\n", frameIndex_);
                VkResult r = vkQueueSubmit(context_->GetGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
                if (r != VK_SUCCESS) { Log(L"[DIAG] composite submit FAILED: %d\n", r); return; }
                vkQueueWaitIdle(context_->GetGraphicsQueue());
                Log(L"[DIAG] frame %u: composite OK\n", frameIndex_);
                vkResetCommandBuffer(cmd, 0);
                VkCommandBufferBeginInfo bi2{}; bi2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                vkBeginCommandBuffer(cmd, &bi2);
            }

            // 3b. DLSS upscaling (render res → display res)
            if (dlssActive_ && dlss_ && dlss_->IsInitialized() && dlss_->IsSupported()) {
                // Measure real frame delta for DLSS
                static auto s_lastDlssTime = std::chrono::high_resolution_clock::now();
                auto dlssNow = std::chrono::high_resolution_clock::now();
                float dlssDeltaMs = std::chrono::duration<float, std::milli>(dlssNow - s_lastDlssTime).count();
                s_lastDlssTime = dlssNow;
                if (dlssDeltaMs < 1.0f) dlssDeltaMs = 1.0f;
                if (dlssDeltaMs > 100.0f) dlssDeltaMs = 100.0f;

                dlss_->Evaluate(cmd,
                    dlssColorInput_, dlssColorInputView_,          // HDR color (RGBA16F, render res)
                    rtPipeline_->GetDlssDepthImage(),              // NDC depth [0,1] (binding 22)
                    rtPipeline_->GetDlssDepthView(),
                    rtPipeline_->GetMotionVectorsImage(),          // MV
                    rtPipeline_->GetMotionVectorsView(),
                    dlssHdrOutput_, dlssHdrOutputView_,            // HDR output (RGBA16F, display res)
                    VK_FORMAT_R16G16B16A16_SFLOAT,                 // color format (HDR)
                    VK_FORMAT_R32_SFLOAT,                          // depth format
                    VK_FORMAT_R16G16B16A16_SFLOAT,                 // MV format
                    jitterX_, jitterY_,                            // pixel-space jitter
                    dlssDeltaMs / 1000.0f,                         // deltaTime (seconds)
                    0.0f,                                          // sharpness
                    false,                                         // reset
                    rtPipeline_->GetReactiveMaskImage(),            // reactive mask
                    rtPipeline_->GetReactiveMaskView());            // reactive mask view

                // Barrier: DLSS writes → tonemap reads
                VkMemoryBarrier dlssBarrier{};
                dlssBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                dlssBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                dlssBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 1, &dlssBarrier, 0, nullptr, 0, nullptr);

                // Tonemap: DLSS HDR output → LDR interop
                if (tonemapReady_) {
                    UpdateTonemapDescriptors();
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tonemapPipeline_);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                        tonemapPipelineLayout_, 0, 1, &tonemapDescSet_, 0, nullptr);

                    struct { uint32_t mode; float exposure, saturation, contrast; } tonemapPush;
                    tonemapPush.mode = static_cast<uint32_t>(cfg->ptTonemapMode);
                    tonemapPush.exposure = cfg->ptAutoExposure ? computedExposure_ : cfg->ptExposure;
                    tonemapPush.saturation = cfg->ptSaturation;
                    tonemapPush.contrast = cfg->ptContrast;
                    vkCmdPushConstants(cmd, tonemapPipelineLayout_,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tonemapPush), &tonemapPush);

                    vkCmdDispatch(cmd, (width_ + 7) / 8, (height_ + 7) / 8, 1);

                    // Barrier: tonemap writes → ImGui/readback reads
                    VkMemoryBarrier tonemapBarrier{};
                    tonemapBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                    tonemapBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    tonemapBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
                    vkCmdPipelineBarrier(cmd,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        0, 1, &tonemapBarrier, 0, nullptr, 0, nullptr);
                }
            }
        }
    }

    // 4. ImGui overlay (renders on top of final output)
    if (imguiReady_) {
        RenderImGuiOverlay(cmd);
    }

    // 5. Readback
    if (!useDirectInterop_) {
        interop_->RecordReadbackCopy(cmd);
    } else {
        interop_->TransitionForExternalRead(cmd);
    }

    if (!VK_CHECK(vkEndCommandBuffer(cmd))) return;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkResult submitResult = vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, inFlightFences_[currentFrame_]);
    if (submitResult != VK_SUCCESS) {
        if (nrdInitialized_) {
            Log(L"[VK Renderer] Submit failed (%d) — disabling NRD\n", (int)submitResult);
            nrdInitialized_ = false;
        }
        return;
    }

    // Wait for completion (needed for readback)
    vkQueueWaitIdle(context_->GetGraphicsQueue());

    // After rendering, current transforms become previous for the next frame.
    // Without this, prevTransforms would stay stale after an object stops
    // moving, causing perpetual non-zero motion vectors and ghosting.
    if (instanceTransformCount_ > 0) {
        prevInstanceTransforms_ = currInstanceTransforms_;
    }

    // Swap GI reservoir ping-pong buffers (current → previous for next frame)
    if (rtPipeline_ && rtPipeline_->HasGIReservoirBuffers()) {
        rtPipeline_->SwapGIReservoirBuffers();
    }

    frameIndex_++;
    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

HANDLE Renderer::GetInteropNTHandle() const {
    return interop_ ? interop_->GetNTHandle() : nullptr;
}

bool Renderer::ReadbackPixels(void* outData, uint32_t bufferSize) {
    // Fast path: readback was already recorded in RenderFrameRT command buffer
    // After vkQueueWaitIdle, just copy from persistent mapped memory
    return interop_ ? interop_->CopyReadbackResult(outData, bufferSize) : false;
}

bool Renderer::ImportD3D11Texture(HANDLE ntHandle, uint32_t width, uint32_t height) {
    if (!interop_) return false;
    bool result = interop_->ImportD3D11Texture(ntHandle, width, height);
    if (result && rtPipeline_) {
        // When DLSS active, RT writes to intermediate image (not interop)
        if (dlssActive_ && dlssColorInputView_) {
            rtPipeline_->UpdateStorageImage(dlssColorInputView_);
        } else {
            rtPipeline_->UpdateStorageImage(interop_->GetSharedImageView());
        }
    }
    return result;
}

bool Renderer::IsRTSupported() const {
    return context_ && context_->IsRayQuerySupported();
}

bool Renderer::InitGLInterop() {
    if (useDirectInterop_) return true;
    if (glInteropFailed_) return false;
    if (!interop_) { glInteropFailed_ = true; return false; }

    if (interop_->InitGLInterop()) {
        useDirectInterop_ = true;
        return true;
    }

    glInteropFailed_ = true;
    return false;
}

void Renderer::DrawGL(uint32_t w, uint32_t h) {
    if (interop_) interop_->DrawGL(w, h);
}

bool Renderer::InitImGui(HWND hwnd, bool forceRasterPath) {
    VkDevice device = context_->GetDevice();

    // Create descriptor pool for ImGui (256 sets for texture thumbnails in material inspector)
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256 },
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 256;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &imguiDescriptorPool_) != VK_SUCCESS) {
        Log(L"[VK Renderer] Failed to create ImGui descriptor pool\n");
        return false;
    }

    // Determine format and layouts based on whether we have RT interop or rasterizer
    bool useRTPath = (interop_ != nullptr) && !forceRasterPath;

    VkAttachmentDescription colorAttachment{};
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;      // Preserve previous output
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

    if (useRTPath) {
        colorAttachment.format = VK_FORMAT_R8G8B8A8_UNORM;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        dep.srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    } else {
        colorAttachment.format = context_->GetSwapchainFormat();
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    }

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &colorAttachment;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dep;

    if (vkCreateRenderPass(device, &rpInfo, nullptr, &imguiRenderPass_) != VK_SUCCESS) {
        Log(L"[VK Renderer] Failed to create ImGui render pass\n");
        return false;
    }

    if (useRTPath) {
        // RT path: single framebuffer using shared image view
        VkImageView imageView = interop_->GetSharedImageView();
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = imguiRenderPass_;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &imageView;
        fbInfo.width = width_;
        fbInfo.height = height_;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &imguiFramebuffer_) != VK_SUCCESS) {
            Log(L"[VK Renderer] Failed to create ImGui framebuffer\n");
            return false;
        }
    } else {
        // Rasterizer path: per-swapchain-image framebuffers
        const auto& swapViews = context_->GetSwapchainImageViews();
        imguiSwapchainFramebuffers_.resize(swapViews.size());
        for (size_t i = 0; i < swapViews.size(); i++) {
            VkImageView iv = swapViews[i];
            VkFramebufferCreateInfo fbInfo{};
            fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fbInfo.renderPass = imguiRenderPass_;
            fbInfo.attachmentCount = 1;
            fbInfo.pAttachments = &iv;
            fbInfo.width = context_->GetSwapchainExtent().width;
            fbInfo.height = context_->GetSwapchainExtent().height;
            fbInfo.layers = 1;

            if (vkCreateFramebuffer(device, &fbInfo, nullptr, &imguiSwapchainFramebuffers_[i]) != VK_SUCCESS) {
                Log(L"[VK Renderer] Failed to create ImGui swapchain framebuffer %zu\n", i);
                return false;
            }
        }
    }

    // Initialize ImGui with Vulkan backend
    if (!ImGui_Init(hwnd,
                    context_->GetInstance(),
                    context_->GetPhysicalDevice(),
                    device,
                    context_->GetGraphicsQueue(),
                    context_->GetGraphicsQueueFamily(),
                    imguiRenderPass_,
                    imguiDescriptorPool_)) {
        Log(L"[VK Renderer] Failed to initialize ImGui\n");
        return false;
    }

    imguiReady_ = true;
    externalCameraControl_ = true; // Tree editor controls the camera
    Log(L"[VK Renderer] ImGui overlay initialized (%ux%u, %s path)\n",
        width_, height_, useRTPath ? L"RT" : L"raster");
    return true;
}

void Renderer::RenderImGuiOverlay(VkCommandBuffer cmd) {
    if (!imguiReady_) return;

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = imguiRenderPass_;

    if (imguiFramebuffer_ != VK_NULL_HANDLE) {
        // RT path
        rpBegin.framebuffer = imguiFramebuffer_;
        rpBegin.renderArea.extent = {width_, height_};
    } else if (!imguiSwapchainFramebuffers_.empty()) {
        // Raster path: use current frame's swapchain framebuffer
        rpBegin.framebuffer = imguiSwapchainFramebuffers_[imguiCurrentImageIndex_];
        rpBegin.renderArea.extent = context_->GetSwapchainExtent();
    } else {
        return;
    }

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_Render(cmd);
    vkCmdEndRenderPass(cmd);
}

bool Renderer::InitNRD() {
    if (nrdInitialized_) return true;
    if (!rtPipeline_ || !context_) return false;

    // Create G-buffers first (at render resolution — may be < display when DLSS active)
    // RR also needs G-buffers (normals, albedo, depth, MVs)
    if (!rtPipeline_->CreateGBuffers(renderWidth_, renderHeight_)) {
        Log(L"[VK Renderer] WARNING: Failed to create G-buffers\n");
        return false;
    }

    // When Ray Reconstruction is active, skip NRD and composite — RR replaces them.
    // Still create G-buffers (done above) and tonemap pipeline (RR outputs HDR).
    if (dlssRRActive_) {
        Log(L"[VK Renderer] RR active — skipping NRD init, creating tonemap only\n");

        // Create tonemap pipeline (RR outputs HDR to dlssHdrOutput_, needs tonemap to LDR)
        if (dlssActive_ && dlssHdrOutput_) {
            if (!CreateTonemapPipeline()) {
                Log(L"[VK Renderer] WARNING: Tonemap pipeline creation failed\n");
            }
        }

        // Mark as initialized so RenderFrameRT proceeds
        nrdInitialized_ = false;  // NRD itself is NOT initialized
        return true;
    }

    // Initialize NRD (at render resolution)
    if (!acpt::NRD_Vulkan_Init(context_->GetPhysicalDevice(), context_->GetDevice(),
                                context_->GetGraphicsQueue(), context_->GetCommandPool(),
                                renderWidth_, renderHeight_)) {
        Log(L"[VK Renderer] WARNING: NRD init failed, running without denoiser\n");
        return false;
    }

    // Register G-buffer images
    acpt::NRD_GBufferImages gbuffers;
    gbuffers.normalRoughness = rtPipeline_->GetNormalRoughnessImage();
    gbuffers.viewDepth = rtPipeline_->GetViewDepthImage();
    gbuffers.motionVectors = rtPipeline_->GetMotionVectorsImage();
    gbuffers.diffuseRadiance = rtPipeline_->GetDiffuseRadianceImage();
    gbuffers.specularRadiance = rtPipeline_->GetSpecularRadianceImage();
    gbuffers.albedoBuffer = rtPipeline_->GetAlbedoBufferImage();
    gbuffers.penumbraBuffer = rtPipeline_->GetPenumbraImage();
    gbuffers.diffuseConfidence = rtPipeline_->GetDiffConfidenceImage();
    gbuffers.specularConfidence = rtPipeline_->GetSpecConfidenceImage();

    if (!acpt::NRD_Vulkan_SetGBuffers(gbuffers)) {
        Log(L"[VK Renderer] WARNING: NRD SetGBuffers failed\n");
        acpt::NRD_Vulkan_Shutdown();
        return false;
    }

    nrdInitialized_ = true;
    Log(L"[VK Renderer] NRD initialized successfully\n");

    // Create composite pipeline
    if (!CreateCompositePipeline()) {
        Log(L"[VK Renderer] WARNING: Composite pipeline creation failed\n");
    }

    // Create auto-exposure resolve pipeline (after composite, uses same SSBO)
    if (!CreateExposureResolvePipeline()) {
        Log(L"[VK Renderer] WARNING: Auto-exposure resolve pipeline creation failed (non-fatal)\n");
    }

    // Create tonemap pipeline (post-DLSS HDR → LDR, only when DLSS active)
    if (dlssActive_ && dlssHdrOutput_) {
        if (!CreateTonemapPipeline()) {
            Log(L"[VK Renderer] WARNING: Tonemap pipeline creation failed\n");
        }
    }

    // Create SHARC resolve pipeline
    if (!CreateSHARCResolvePipeline()) {
        Log(L"[VK Renderer] WARNING: SHARC resolve pipeline creation failed (non-fatal)\n");
    }

    return true;
}

bool Renderer::CreateCompositePipeline() {
    VkDevice device = context_->GetDevice();

    // Sampler for NRD denoised textures
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &compositeSampler_) != VK_SUCCESS) {
        return false;
    }

    // Descriptor set layout: 10 bindings matching nrd_composite.comp
    VkDescriptorSetLayoutBinding bindings[10] = {};
    // binding 0: denoised diffuse (sampler2D)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 1: denoised specular (sampler2D)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 2: final output (storage image, read-write for cloud blending)
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 3: raw PT output (sampler2D)
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 4: albedo buffer (sampler2D)
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 5: denoised shadow from SIGMA (sampler2D)
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 6: exposure SSBO (auto-exposure luminance accumulation)
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 7: cloud buffer (volumetric clouds, full-res RGBA16F)
    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 8: cloud depth (first-hit distance, full-res R32F)
    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 9: scene depth (linear view-space Z, R32F)
    bindings[9].binding = 9;
    bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[9].descriptorCount = 1;
    bindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 10;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &compositeDescriptorSetLayout_) != VK_SUCCESS) {
        return false;
    }

    // Push constant: useNRD (uint32_t)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 6 * sizeof(uint32_t);  // mode + tonemapMode + exposure + saturation + contrast + hdrOutput

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &compositeDescriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &compositePipelineLayout_) != VK_SUCCESS) {
        return false;
    }

    // Load compute shader
    std::ifstream file(IgnisResolvePath("shaders/nrd_composite.spv"), std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[VK Renderer] WARNING: nrd_composite.spv not found\n");
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
    pipelineInfo.layout = compositePipelineLayout_;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &compositePipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create composite pipeline\n");
        return false;
    }

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},  // 8 samplers (diffuse, specular, rawPT, albedo, shadow, clouds, cloudDepth, sceneDepth)
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},          // exposure SSBO
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &compositeDescriptorPool_) != VK_SUCCESS) {
        return false;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = compositeDescriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &compositeDescriptorSetLayout_;
    if (vkAllocateDescriptorSets(device, &allocInfo, &compositeDescriptorSet_) != VK_SUCCESS) {
        return false;
    }

    compositeReady_ = true;
    Log(L"[VK Renderer] Composite pipeline created\n");
    return true;
}

void Renderer::UpdateCompositeDescriptors() {
    if (!compositeReady_ || !nrdInitialized_) return;

    VkDevice device = context_->GetDevice();

    VkImageView denoisedDiffuseView, denoisedSpecularView;
    acpt::NRD_Vulkan_GetDenoisedOutputs(denoisedDiffuseView, denoisedSpecularView);
    VkImageView albedoView = acpt::NRD_Vulkan_GetAlbedoBufferView();
    VkImageView shadowView;
    acpt::NRD_Vulkan_GetDenoisedShadow(shadowView);

    if (!denoisedDiffuseView || !denoisedSpecularView) return;

    // binding 0: denoised diffuse (NRD output textures stay in GENERAL)
    VkDescriptorImageInfo diffuseInfo{};
    diffuseInfo.imageView = denoisedDiffuseView;
    diffuseInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    diffuseInfo.sampler = compositeSampler_;

    // binding 1: denoised specular
    VkDescriptorImageInfo specularInfo{};
    specularInfo.imageView = denoisedSpecularView;
    specularInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    specularInfo.sampler = compositeSampler_;

    // binding 2: final output — when DLSS active, write to intermediate image (render res)
    //            otherwise write directly to interop image (display res)
    VkImageView compositeOutputView = dlssActive_ ? dlssColorInputView_ : interop_->GetSharedImageView();
    VkDescriptorImageInfo outputInfo{};
    outputInfo.imageView = compositeOutputView;
    outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // binding 3: raw PT output — same destination as composite output
    VkDescriptorImageInfo rawPTInfo{};
    rawPTInfo.imageView = compositeOutputView;
    rawPTInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    rawPTInfo.sampler = compositeSampler_;

    // binding 4: albedo buffer
    VkDescriptorImageInfo albedoInfo{};
    albedoInfo.imageView = albedoView;
    albedoInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    albedoInfo.sampler = compositeSampler_;

    // binding 5: denoised shadow (SIGMA output)
    VkDescriptorImageInfo shadowInfo{};
    shadowInfo.imageView = shadowView ? shadowView : denoisedDiffuseView;  // fallback if no shadow
    shadowInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    shadowInfo.sampler = compositeSampler_;

    // binding 6: exposure SSBO
    VkDescriptorBufferInfo exposureBufferInfo{};
    exposureBufferInfo.buffer = exposureSSBO_.buffer;
    exposureBufferInfo.offset = 0;
    exposureBufferInfo.range = VK_WHOLE_SIZE;

    // binding 7: cloud buffer (volumetric clouds — stubbed out, always fallback)
    VkDescriptorImageInfo cloudInfo{};
    cloudInfo.imageView = denoisedDiffuseView;
    cloudInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    cloudInfo.sampler = compositeSampler_;

    // binding 8: cloud depth (stubbed out, always fallback)
    VkDescriptorImageInfo cloudDepthInfo{};
    cloudDepthInfo.imageView = denoisedDiffuseView;
    cloudDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    cloudDepthInfo.sampler = compositeSampler_;

    // binding 9: scene depth (linear view-space Z)
    VkDescriptorImageInfo sceneDepthInfo{};
    sceneDepthInfo.imageView = rtPipeline_ ? rtPipeline_->GetViewDepthView() : denoisedDiffuseView;
    sceneDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    sceneDepthInfo.sampler = compositeSampler_;

    VkWriteDescriptorSet writes[10] = {};
    for (int i = 0; i < 10; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = compositeDescriptorSet_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &diffuseInfo;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &specularInfo;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[2].pImageInfo = &outputInfo;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &rawPTInfo;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].pImageInfo = &albedoInfo;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[5].pImageInfo = &shadowInfo;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[6].pBufferInfo = &exposureBufferInfo;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[7].pImageInfo = &cloudInfo;
    writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[8].pImageInfo = &cloudDepthInfo;
    writes[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[9].pImageInfo = &sceneDepthInfo;

    uint32_t writeCount = exposureSSBO_.buffer ? 10u : 6u;
    vkUpdateDescriptorSets(device, writeCount, writes, 0, nullptr);
}


bool Renderer::CreateExposureResolvePipeline() {
    VkDevice device = context_->GetDevice();

    // Create 12-byte device-local SSBO: { uint luminanceSum, uint pixelCount, float currentExposure }
    exposureSSBO_ = accelBuilder_->CreateAccelBuffer(
        12,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!exposureSSBO_.buffer) {
        Log(L"[VK Renderer] WARNING: Failed to create exposure SSBO\n");
        return false;
    }

    // Create 12-byte host-visible staging buffer for CPU readback
    exposureStagingSSBO_ = accelBuilder_->CreateAccelBuffer(
        12,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!exposureStagingSSBO_.buffer) {
        Log(L"[VK Renderer] WARNING: Failed to create exposure staging buffer\n");
        return false;
    }

    // Initialize SSBO: luminanceSum=0, pixelCount=0, currentExposure=0.55
    {
        struct { uint32_t lumSum; uint32_t pixCount; float exposure; } initData = { 0, 0, 0.55f };
        void* mapped;
        vkMapMemory(device, exposureStagingSSBO_.memory, 0, 12, 0, &mapped);
        memcpy(mapped, &initData, 12);
        vkUnmapMemory(device, exposureStagingSSBO_.memory);

        VkCommandBuffer initCmd = context_->BeginSingleTimeCommands();
        VkBufferCopy copyRegion{};
        copyRegion.size = 12;
        vkCmdCopyBuffer(initCmd, exposureStagingSSBO_.buffer, exposureSSBO_.buffer, 1, &copyRegion);
        context_->EndSingleTimeCommands(initCmd);
    }

    // Descriptor set layout: 1 SSBO binding
    VkDescriptorSetLayoutBinding ssboBinding{};
    ssboBinding.binding = 0;
    ssboBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ssboBinding.descriptorCount = 1;
    ssboBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descLayoutInfo{};
    descLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descLayoutInfo.bindingCount = 1;
    descLayoutInfo.pBindings = &ssboBinding;
    if (vkCreateDescriptorSetLayout(device, &descLayoutInfo, nullptr, &exposureResolveDescSetLayout_) != VK_SUCCESS) {
        return false;
    }

    // Push constant: 4 floats (keyValue, adaptSpeed, minExposure, maxExposure)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 4 * sizeof(float);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &exposureResolveDescSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &exposureResolvePipelineLayout_) != VK_SUCCESS) {
        return false;
    }

    // Load compute shader
    std::ifstream file(IgnisResolvePath("shaders/exposure_resolve.comp.spv"), std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[VK Renderer] WARNING: exposure_resolve.comp.spv not found\n");
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
    pipelineInfo.layout = exposureResolvePipelineLayout_;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &exposureResolvePipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create exposure resolve pipeline\n");
        return false;
    }

    // Descriptor pool + set
    VkDescriptorPoolSize resolvePoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo resolvePoolInfo{};
    resolvePoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    resolvePoolInfo.maxSets = 1;
    resolvePoolInfo.poolSizeCount = 1;
    resolvePoolInfo.pPoolSizes = &resolvePoolSize;
    if (vkCreateDescriptorPool(device, &resolvePoolInfo, nullptr, &exposureResolveDescPool_) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorSetAllocateInfo resolveAllocInfo{};
    resolveAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    resolveAllocInfo.descriptorPool = exposureResolveDescPool_;
    resolveAllocInfo.descriptorSetCount = 1;
    resolveAllocInfo.pSetLayouts = &exposureResolveDescSetLayout_;
    if (vkAllocateDescriptorSets(device, &resolveAllocInfo, &exposureResolveDescSet_) != VK_SUCCESS) {
        return false;
    }

    // Write SSBO descriptor
    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = exposureSSBO_.buffer;
    bufInfo.offset = 0;
    bufInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet ssboWrite{};
    ssboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ssboWrite.dstSet = exposureResolveDescSet_;
    ssboWrite.dstBinding = 0;
    ssboWrite.descriptorCount = 1;
    ssboWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ssboWrite.pBufferInfo = &bufInfo;
    vkUpdateDescriptorSets(device, 1, &ssboWrite, 0, nullptr);

    exposureResolveReady_ = true;
    Log(L"[VK Renderer] Auto-exposure resolve pipeline created\n");
    return true;
}

void Renderer::ShutdownExposureResolve() {
    if (!exposureResolveReady_) return;
    VkDevice device = context_->GetDevice();

    if (exposureResolvePipeline_) { vkDestroyPipeline(device, exposureResolvePipeline_, nullptr); exposureResolvePipeline_ = VK_NULL_HANDLE; }
    if (exposureResolvePipelineLayout_) { vkDestroyPipelineLayout(device, exposureResolvePipelineLayout_, nullptr); exposureResolvePipelineLayout_ = VK_NULL_HANDLE; }
    if (exposureResolveDescPool_) { vkDestroyDescriptorPool(device, exposureResolveDescPool_, nullptr); exposureResolveDescPool_ = VK_NULL_HANDLE; }
    if (exposureResolveDescSetLayout_) { vkDestroyDescriptorSetLayout(device, exposureResolveDescSetLayout_, nullptr); exposureResolveDescSetLayout_ = VK_NULL_HANDLE; }

    if (exposureSSBO_.buffer) accelBuilder_->DestroyAccelBuffer(exposureSSBO_);
    if (exposureStagingSSBO_.buffer) accelBuilder_->DestroyAccelBuffer(exposureStagingSSBO_);

    exposureResolveReady_ = false;
    Log(L"[VK Renderer] Auto-exposure resolve shutdown\n");
}

bool Renderer::CreateTonemapPipeline() {
    VkDevice device = context_->GetDevice();

    // Sampler for HDR input
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &tonemapSampler_) != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create tonemap sampler\n");
        return false;
    }

    // Descriptor set layout: binding 0 = sampler (HDR input), binding 1 = storage image (LDR output)
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &tonemapDescSetLayout_) != VK_SUCCESS) {
        return false;
    }

    // Push constant: tonemapMode (uint) + exposure (float) + saturation (float) + contrast (float) = 16 bytes
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 4 * sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &tonemapDescSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &tonemapPipelineLayout_) != VK_SUCCESS) {
        return false;
    }

    // Load compute shader
    std::ifstream file(IgnisResolvePath("shaders/tonemap.comp.spv"), std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[VK Renderer] WARNING: tonemap.comp.spv not found\n");
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
    pipelineInfo.layout = tonemapPipelineLayout_;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &tonemapPipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create tonemap pipeline\n");
        return false;
    }

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &tonemapDescPool_) != VK_SUCCESS) {
        return false;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = tonemapDescPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &tonemapDescSetLayout_;
    if (vkAllocateDescriptorSets(device, &allocInfo, &tonemapDescSet_) != VK_SUCCESS) {
        return false;
    }

    tonemapReady_ = true;
    Log(L"[VK Renderer] Tonemap pipeline created (post-DLSS HDR->LDR)\n");
    return true;
}

void Renderer::UpdateTonemapDescriptors() {
    if (!tonemapReady_ || !dlssHdrOutputView_ || !interop_) return;

    VkDevice device = context_->GetDevice();

    // binding 0: DLSS HDR output (sampler)
    VkDescriptorImageInfo hdrInfo{};
    hdrInfo.imageView = dlssHdrOutputView_;
    hdrInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    hdrInfo.sampler = tonemapSampler_;

    // binding 1: interop (storage image, LDR output)
    VkDescriptorImageInfo ldrInfo{};
    ldrInfo.imageView = interop_->GetSharedImageView();
    ldrInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = tonemapDescSet_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &hdrInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = tonemapDescSet_;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &ldrInfo;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

void Renderer::ShutdownTonemap() {
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) return;

    if (tonemapPipeline_) { vkDestroyPipeline(device, tonemapPipeline_, nullptr); tonemapPipeline_ = VK_NULL_HANDLE; }
    if (tonemapPipelineLayout_) { vkDestroyPipelineLayout(device, tonemapPipelineLayout_, nullptr); tonemapPipelineLayout_ = VK_NULL_HANDLE; }
    if (tonemapDescPool_) { vkDestroyDescriptorPool(device, tonemapDescPool_, nullptr); tonemapDescPool_ = VK_NULL_HANDLE; }
    if (tonemapDescSetLayout_) { vkDestroyDescriptorSetLayout(device, tonemapDescSetLayout_, nullptr); tonemapDescSetLayout_ = VK_NULL_HANDLE; }
    if (tonemapSampler_) { vkDestroySampler(device, tonemapSampler_, nullptr); tonemapSampler_ = VK_NULL_HANDLE; }
    tonemapReady_ = false;
}

bool Renderer::CreateSHARCResolvePipeline() {
    if (!rtPipeline_ || !rtPipeline_->HasSHARCBuffers()) return false;

    VkDevice device = context_->GetDevice();

    // Descriptor set layout: 2 SSBOs (write buffer, read buffer)
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &sharcResolveDescriptorSetLayout_) != VK_SUCCESS) {
        return false;
    }

    // Push constant: frameIndex (uint32_t)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &sharcResolveDescriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &sharcResolvePipelineLayout_) != VK_SUCCESS) {
        return false;
    }

    // Load compute shader
    std::ifstream file(IgnisResolvePath("shaders/sharc_resolve.comp.spv"), std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        Log(L"[VK Renderer] WARNING: sharc_resolve.comp.spv not found\n");
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
    pipelineInfo.layout = sharcResolvePipelineLayout_;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &sharcResolvePipeline_);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        Log(L"[VK Renderer] WARNING: Failed to create SHARC resolve pipeline\n");
        return false;
    }

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &sharcResolveDescriptorPool_) != VK_SUCCESS) {
        return false;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = sharcResolveDescriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &sharcResolveDescriptorSetLayout_;
    if (vkAllocateDescriptorSets(device, &allocInfo, &sharcResolveDescriptorSet_) != VK_SUCCESS) {
        return false;
    }

    UpdateSHARCResolveDescriptors();

    sharcResolveReady_ = true;
    Log(L"[VK Renderer] SHARC resolve pipeline created\n");
    return true;
}

void Renderer::UpdateSHARCResolveDescriptors() {
    if (!rtPipeline_ || !rtPipeline_->HasSHARCBuffers()) return;

    VkDevice device = context_->GetDevice();

    VkDescriptorBufferInfo writeInfo{};
    writeInfo.buffer = rtPipeline_->GetSHARCWriteBuffer();
    writeInfo.offset = 0;
    writeInfo.range = RTPipeline::GetSHARCBufferSize();

    VkDescriptorBufferInfo readInfo{};
    readInfo.buffer = rtPipeline_->GetSHARCReadBuffer();
    readInfo.offset = 0;
    readInfo.range = RTPipeline::GetSHARCBufferSize();

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = sharcResolveDescriptorSet_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &writeInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = sharcResolveDescriptorSet_;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &readInfo;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

void Renderer::ShutdownSHARCResolve() {
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) return;

    if (sharcResolvePipeline_) { vkDestroyPipeline(device, sharcResolvePipeline_, nullptr); sharcResolvePipeline_ = VK_NULL_HANDLE; }
    if (sharcResolvePipelineLayout_) { vkDestroyPipelineLayout(device, sharcResolvePipelineLayout_, nullptr); sharcResolvePipelineLayout_ = VK_NULL_HANDLE; }
    if (sharcResolveDescriptorPool_) { vkDestroyDescriptorPool(device, sharcResolveDescriptorPool_, nullptr); sharcResolveDescriptorPool_ = VK_NULL_HANDLE; }
    if (sharcResolveDescriptorSetLayout_) { vkDestroyDescriptorSetLayout(device, sharcResolveDescriptorSetLayout_, nullptr); sharcResolveDescriptorSetLayout_ = VK_NULL_HANDLE; }
    sharcResolveReady_ = false;
}

void Renderer::ShutdownDLSS() {
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;

    // Shutdown tonemap pipeline (depends on DLSS HDR output)
    ShutdownTonemap();

    if (dlss_) {
        dlss_->Shutdown();
        delete dlss_;
        dlss_ = nullptr;
    }

    if (device != VK_NULL_HANDLE) {
        if (dlssColorInputView_) { vkDestroyImageView(device, dlssColorInputView_, nullptr); dlssColorInputView_ = VK_NULL_HANDLE; }
        if (dlssColorInput_) { vkDestroyImage(device, dlssColorInput_, nullptr); dlssColorInput_ = VK_NULL_HANDLE; }
        if (dlssColorInputMemory_) { vkFreeMemory(device, dlssColorInputMemory_, nullptr); dlssColorInputMemory_ = VK_NULL_HANDLE; }

        if (dlssHdrOutputView_) { vkDestroyImageView(device, dlssHdrOutputView_, nullptr); dlssHdrOutputView_ = VK_NULL_HANDLE; }
        if (dlssHdrOutput_) { vkDestroyImage(device, dlssHdrOutput_, nullptr); dlssHdrOutput_ = VK_NULL_HANDLE; }
        if (dlssHdrOutputMemory_) { vkFreeMemory(device, dlssHdrOutputMemory_, nullptr); dlssHdrOutputMemory_ = VK_NULL_HANDLE; }
    }

    dlssActive_ = false;
    dlssRRActive_ = false;
    Log(L"[VK Renderer] DLSS shutdown\n");
}

void Renderer::ShutdownNRD() {
    if (!nrdInitialized_) return;
    VkDevice device = context_->GetDevice();

    acpt::NRD_Vulkan_Shutdown();
    nrdInitialized_ = false;

    // Destroy composite pipeline
    if (compositePipeline_) { vkDestroyPipeline(device, compositePipeline_, nullptr); compositePipeline_ = VK_NULL_HANDLE; }
    if (compositePipelineLayout_) { vkDestroyPipelineLayout(device, compositePipelineLayout_, nullptr); compositePipelineLayout_ = VK_NULL_HANDLE; }
    if (compositeDescriptorPool_) { vkDestroyDescriptorPool(device, compositeDescriptorPool_, nullptr); compositeDescriptorPool_ = VK_NULL_HANDLE; }
    if (compositeDescriptorSetLayout_) { vkDestroyDescriptorSetLayout(device, compositeDescriptorSetLayout_, nullptr); compositeDescriptorSetLayout_ = VK_NULL_HANDLE; }
    if (compositeSampler_) { vkDestroySampler(device, compositeSampler_, nullptr); compositeSampler_ = VK_NULL_HANDLE; }
    compositeReady_ = false;

    Log(L"[VK Renderer] NRD shutdown\n");
}

void Renderer::ShutdownImGui() {
    if (!imguiReady_) return;
    VkDevice device = context_->GetDevice();

    ImGui_Shutdown();

    if (imguiFramebuffer_) vkDestroyFramebuffer(device, imguiFramebuffer_, nullptr);
    for (auto fb : imguiSwapchainFramebuffers_) {
        if (fb) vkDestroyFramebuffer(device, fb, nullptr);
    }
    imguiSwapchainFramebuffers_.clear();
    if (imguiRenderPass_) vkDestroyRenderPass(device, imguiRenderPass_, nullptr);
    if (imguiDescriptorPool_) vkDestroyDescriptorPool(device, imguiDescriptorPool_, nullptr);
    imguiReady_ = false;
    Log(L"[VK Renderer] ImGui overlay shutdown\n");
}

void Renderer::Shutdown() {
    ShutdownImGui();
    if (context_ && context_->GetDevice() != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(context_->GetDevice());

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (renderFinishedSemaphores_[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(context_->GetDevice(), renderFinishedSemaphores_[i], nullptr);
            }
            if (imageAvailableSemaphores_[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(context_->GetDevice(), imageAvailableSemaphores_[i], nullptr);
            }
            if (inFlightFences_[i] != VK_NULL_HANDLE) {
                vkDestroyFence(context_->GetDevice(), inFlightFences_[i], nullptr);
            }
        }
    }

    // Shutdown SHARC resolve + auto-exposure + NRD + composite + DLSS
    ShutdownSHARCResolve();
    ShutdownExposureResolve();
    ShutdownNRD();
    ShutdownDLSS();

    // Shutdown wavefront + RT modules
    if (wavefrontPipeline_) { wavefrontPipeline_->Shutdown(); delete wavefrontPipeline_; wavefrontPipeline_ = nullptr; }
    if (rtPipeline_) { rtPipeline_->Shutdown(); delete rtPipeline_; rtPipeline_ = nullptr; }
    if (accelBuilder_) { accelBuilder_->Shutdown(); delete accelBuilder_; accelBuilder_ = nullptr; }
    if (interop_) { interop_->Shutdown(); delete interop_; interop_ = nullptr; }

    if (sphereMesh_) {
        if (geometry_) geometry_->DestroyMesh(*sphereMesh_);
        delete sphereMesh_;
    }
    if (planeMesh_) {
        if (geometry_) geometry_->DestroyMesh(*planeMesh_);
        delete planeMesh_;
    }

    if (rasterizer_) {
        rasterizer_->Shutdown();
        delete rasterizer_;
    }
    if (geometry_) {
        geometry_->Shutdown();
        delete geometry_;
    }
    if (pipeline_) {
        pipeline_->Shutdown();
        delete pipeline_;
    }
    if (context_) {
        context_->Shutdown();
        delete context_;
    }

    Log(L"[VK Renderer] Shutdown complete\n");
}

void Renderer::RenderFrame() {
    // Wait for previous frame
    vkWaitForFences(context_->GetDevice(), 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);

    // Acquire next image
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(context_->GetDevice(), context_->GetSwapchain(), UINT64_MAX,
                                           imageAvailableSemaphores_[currentFrame_], VK_NULL_HANDLE, &imageIndex);

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        Log(L"[VK Renderer] ERROR: Failed to acquire swapchain image (VkResult=%d)\n", (int)result);
        return;
    }

    vkResetFences(context_->GetDevice(), 1, &inFlightFences_[currentFrame_]);

    // Update camera (auto-rotate only if tree editor isn't controlling it)
    if (!externalCameraControl_) {
        cameraAngle_ += 0.01f;
        rasterizer_->UpdateCamera(cameraDistance_, cameraAngle_);
    }

    // Record command buffer
    vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffers_[currentFrame_], &beginInfo) != VK_SUCCESS) {
        Log(L"[VK Renderer] ERROR: Failed to begin recording command buffer\n");
        return;
    }

    rasterizer_->Render(imageIndex, commandBuffers_[currentFrame_]);

    // ImGui overlay (if initialized)
    if (imguiReady_) {
        imguiCurrentImageIndex_ = imageIndex;
        RenderImGuiOverlay(commandBuffers_[currentFrame_]);
    } else {
        // No ImGui: transition from COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC_KHR
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = context_->GetSwapchainImages()[imageIndex];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(commandBuffers_[currentFrame_],
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    if (vkEndCommandBuffer(commandBuffers_[currentFrame_]) != VK_SUCCESS) {
        Log(L"[VK Renderer] ERROR: Failed to record command buffer\n");
        return;
    }

    // Submit
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores_[currentFrame_]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers_[currentFrame_];

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores_[currentFrame_]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(context_->GetGraphicsQueue(), 1, &submitInfo, inFlightFences_[currentFrame_]) != VK_SUCCESS) {
        Log(L"[VK Renderer] ERROR: Failed to submit draw command buffer\n");
        return;
    }

    // Present
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {context_->GetSwapchain()};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    vkQueuePresentKHR(context_->GetPresentQueue(), &presentInfo);

    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

bool Renderer::CreateCommandBuffers() {
    commandBuffers_.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = context_->GetCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers_.size();

    if (vkAllocateCommandBuffers(context_->GetDevice(), &allocInfo, commandBuffers_.data()) != VK_SUCCESS) {
        Log(L"[VK Renderer] ERROR: Failed to allocate command buffers\n");
        return false;
    }

    Log(L"[VK Renderer] Command buffers created\n");
    return true;
}

bool Renderer::CreateSyncObjects() {
    imageAvailableSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences_.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(context_->GetDevice(), &semaphoreInfo, nullptr, &imageAvailableSemaphores_[i]) != VK_SUCCESS ||
            vkCreateSemaphore(context_->GetDevice(), &semaphoreInfo, nullptr, &renderFinishedSemaphores_[i]) != VK_SUCCESS ||
            vkCreateFence(context_->GetDevice(), &fenceInfo, nullptr, &inFlightFences_[i]) != VK_SUCCESS) {
            Log(L"[VK Renderer] ERROR: Failed to create sync objects\n");
            return false;
        }
    }

    Log(L"[VK Renderer] Sync objects created\n");
    return true;
}

} // namespace vk
} // namespace acpt

