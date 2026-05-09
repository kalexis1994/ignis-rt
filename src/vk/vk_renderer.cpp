#include "vk_renderer.h"
#include "vk_context.h"
#include "vk_pipeline.h"
#include "vk_geometry.h"
#include "vk_rasterizer.h"
#include "vk_accel_structure.h"
#include "vk_rt_resources.h"
#include "vk_wavefront_pipeline.h"
#include "vk_interop.h"
#include "vk_texture_manager.h"
#include "ignis_log.h"
#include "ignis_config.h"
#include "vk_check.h"
#include "nirc_integration.h"
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <chrono>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Forward declaration — defined in ignis_api.cpp
namespace acpt {
    PathTracerConfig* VK_GetConfig();
}

namespace acpt { extern PathTracerConfig g_config; }

namespace acpt {
namespace vk {

namespace {
float HalfBitsToFloat(uint16_t h)
{
    const uint32_t sign = (uint32_t(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    }
    else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    }
    else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}
}  // namespace

const char* Renderer::InitializeStep(HWND hwnd, uint32_t width, uint32_t height) {
    // Phased initialization — one step per call for smooth loading screen.
    // Returns step name, or nullptr when complete.

    if (initStep_ == 0) {
        // Step 0: Vulkan context
        width_ = width; height_ = height;
        initHwnd_ = hwnd;
        context_ = new Context();
        pipeline_ = new Pipeline();
        geometry_ = new Geometry();
        rasterizer_ = new Rasterizer();
        if (!context_->Initialize(hwnd, width, height)) {
            Log(L"[VK Renderer] ERROR: Failed to initialize context\n");
            return nullptr;
        }
        initStep_ = 1;
        return "Vulkan context ready";
    }

    if (initStep_ == 1) {
        // Step 1: Basic pipelines + geometry
        if (!pipeline_->Initialize(context_) ||
            !pipeline_->CreateGraphicsPipeline("shaders/basic.vert.spv", "shaders/basic.frag.spv") ||
            !geometry_->Initialize(context_)) {
            Log(L"[VK Renderer] ERROR: Failed basic pipeline/geometry init\n");
            return nullptr;
        }
        sphereMesh_ = new Mesh(); *sphereMesh_ = Geometry::CreateSphere(1.0f, 32, 32);
        geometry_->UploadMesh(*sphereMesh_);
        planeMesh_ = new Mesh(); *planeMesh_ = Geometry::CreatePlane(10.0f);
        geometry_->UploadMesh(*planeMesh_);
        rasterizer_->Initialize(context_, pipeline_, geometry_);
        rasterizer_->SetScene(sphereMesh_, planeMesh_);
        CreateCommandBuffers();
        CreateSyncObjects();
        initStep_ = 2;
        return "Pipelines ready";
    }

    if (initStep_ == 2) {
        // Step 2: RT pipeline + interop
        if (!context_->IsRayQuerySupported()) {
            Log(L"[VK Renderer] RT not supported\n");
            initStep_ = 5; // done
            return nullptr;
        }
        interop_ = new Interop();
        if (!interop_->Initialize(context_, width_, height_)) {
            delete interop_; interop_ = nullptr;
            initStep_ = 5;
            return nullptr;
        }
        renderWidth_ = width_; renderHeight_ = height_;
        initStep_ = 3;
        return "RT interop ready";
    }

    if (initStep_ == 3) {
        // Step 3: DLSS initialization
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
                }
            }
            if (!dlssActive_ && dlss_) { delete dlss_; dlss_ = nullptr; }
        }
        // Create DLSS intermediate images if active
        if (dlssActive_) {
            // (DLSS color/HDR images are created in InitRT — we call it from step 4)
        }
        initStep_ = 4;
        return "DLSS ready";
    }

    if (initStep_ == 4) {
        // Step 4: G-buffers + RT resources + wavefront pipeline
        // (interop + DLSS already done above)
        InitRT_Remaining();
        Log(L"[VK Renderer] ========== INITIALIZATION COMPLETE ==========\n");
        initStep_ = 5;
        return nullptr; // done
    }

    return nullptr; // already complete
}

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

    // Initialize DLSS so we know render resolution before allocating G-buffers
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
            Log(L"[VK Renderer] DLSS Ray Reconstruction active\n");
        } else {
            Log(L"[VK Renderer] RR unavailable, using DLSS SR only\n");
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
    rtResources_ = new RTResources();
    if (!rtResources_->Initialize(context_, accelBuilder_, interop_)) {
        Log(L"[VK Renderer] WARNING: RT pipeline initialization failed\n");
        delete rtResources_;
        rtResources_ = nullptr;
        return false;
    }

    // When DLSS active, point RT storage image to intermediate buffer
    if (dlssActive_) {
        rtResources_->UpdateStorageImage(dlssColorInputView_);
    }

    // Create render-resolution G-buffers (normals, depth, MVs, albedo, ...)
    // consumed by DLSS Ray Reconstruction and the wavefront kernels.
    rtResources_->CreateGBuffers(renderWidth_, renderHeight_);

    // Post-PT compute pipelines. Each is independent — failures are non-fatal.
    if (dlssActive_ && dlssHdrOutput_ && !CreateTonemapPipeline())
        Log(L"[VK Renderer] WARNING: Tonemap pipeline creation failed\n");
    if (!CreateExposureResolvePipeline())
        Log(L"[VK Renderer] WARNING: Auto-exposure resolve pipeline creation failed\n");
    if (!CreateSHARCResolvePipeline())
        Log(L"[VK Renderer] WARNING: SHARC resolve pipeline creation failed\n");
    if (!CreateSurfelResolvePipeline())
        Log(L"[VK Renderer] WARNING: Surfel resolve pipeline creation failed\n");
    if (!CreateHairContourPipeline())
        Log(L"[VK Renderer] WARNING: Hair contour pipeline creation failed\n");

    Log(L"[VK Renderer] RT modules initialized\n");

    // Initialize NRC (Neural Radiance Cache)

    // Initialize wavefront pipeline (the only path tracer we have)
    wavefrontPipeline_ = new WavefrontPipeline();
    if (!wavefrontPipeline_->Initialize(context_, rtResources_, renderWidth_, renderHeight_, cfg ? cfg->maxBounces : 2)) {
        Log(L"[VK Renderer] ERROR: Wavefront init failed\n");
        delete wavefrontPipeline_;
        wavefrontPipeline_ = nullptr;
        return false;
    }

    return true;
}

void Renderer::InitRT_Remaining() {
    // Called from phased init step 4 — interop + DLSS already initialized.
    // Handles: DLSS images, RR, AccelStruct, RT resources, G-buffers, Wavefront.
    PathTracerConfig* cfg = VK_GetConfig();
    VkDevice device = context_->GetDevice();

    // Create DLSS intermediate images if active (same code as InitRT)
    if (dlssActive_) {
        // Color input image (render resolution)
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        imgInfo.extent = { renderWidth_, renderHeight_, 1 };
        imgInfo.mipLevels = 1; imgInfo.arrayLayers = 1;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(device, &imgInfo, nullptr, &dlssColorInput_) == VK_SUCCESS) {
            VkMemoryRequirements memReqs;
            vkGetImageMemoryRequirements(device, dlssColorInput_, &memReqs);
            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memReqs.size;
            allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (vkAllocateMemory(device, &allocInfo, nullptr, &dlssColorInputMemory_) == VK_SUCCESS) {
                vkBindImageMemory(device, dlssColorInput_, dlssColorInputMemory_, 0);
                VkImageViewCreateInfo viewInfo{};
                viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                viewInfo.image = dlssColorInput_;
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
                viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                vkCreateImageView(device, &viewInfo, nullptr, &dlssColorInputView_);
            }
        }

        // HDR output image (display resolution)
        imgInfo.extent = { width_, height_, 1 };
        imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (vkCreateImage(device, &imgInfo, nullptr, &dlssHdrOutput_) == VK_SUCCESS) {
            VkMemoryRequirements memReqs;
            vkGetImageMemoryRequirements(device, dlssHdrOutput_, &memReqs);
            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memReqs.size;
            allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (vkAllocateMemory(device, &allocInfo, nullptr, &dlssHdrOutputMemory_) == VK_SUCCESS) {
                vkBindImageMemory(device, dlssHdrOutput_, dlssHdrOutputMemory_, 0);
                VkImageViewCreateInfo viewInfo{};
                viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                viewInfo.image = dlssHdrOutput_;
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
                viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                vkCreateImageView(device, &viewInfo, nullptr, &dlssHdrOutputView_);
            }
        }

        // Transition to GENERAL layout
        VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
        VkImageMemoryBarrier barriers[2] = {};
        uint32_t bc = 0;
        if (dlssColorInput_) {
            barriers[bc].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[bc].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[bc].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[bc].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[bc].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[bc].image = dlssColorInput_;
            barriers[bc].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            bc++;
        }
        if (dlssHdrOutput_) {
            barriers[bc].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[bc].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[bc].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[bc].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[bc].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[bc].image = dlssHdrOutput_;
            barriers[bc].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            bc++;
        }
        if (bc > 0) {
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, bc, barriers);
        }
        context_->EndSingleTimeCommands(cmd);
    }

    // Ray Reconstruction
    if (cfg && cfg->dlssRREnabled && dlssActive_ && dlss_) {
        if (dlss_->InitializeRR()) {
            dlssRRActive_ = true;
            Log(L"[VK Renderer] DLSS Ray Reconstruction active\n");
        }
    }

    // Acceleration structure builder
    accelBuilder_ = new AccelStructureBuilder();
    accelBuilder_->Initialize(context_);

    // RT pipeline
    rtResources_ = new RTResources();
    rtResources_->Initialize(context_, accelBuilder_, interop_);
    if (dlssActive_ && dlssColorInputView_) {
        rtResources_->UpdateStorageImage(dlssColorInputView_);
    }

    // G-buffers for DLSS RR / wavefront
    rtResources_->CreateGBuffers(renderWidth_, renderHeight_);

    // Post-PT compute pipelines. Each is independent — failures are non-fatal.
    if (dlssActive_ && dlssHdrOutput_ && !CreateTonemapPipeline())
        Log(L"[VK Renderer] WARNING: Tonemap pipeline creation failed\n");
    if (!CreateExposureResolvePipeline())
        Log(L"[VK Renderer] WARNING: Auto-exposure resolve pipeline creation failed\n");
    if (!CreateSHARCResolvePipeline())
        Log(L"[VK Renderer] WARNING: SHARC resolve pipeline creation failed\n");
    if (!CreateSurfelResolvePipeline())
        Log(L"[VK Renderer] WARNING: Surfel resolve pipeline creation failed\n");
    if (!CreateHairContourPipeline())
        Log(L"[VK Renderer] WARNING: Hair contour pipeline creation failed\n");

    // NRC (Neural Radiance Cache)

    // NIRC (custom Neural Incident Radiance Cache)
    if (cfg && !nirc_) {
        nirc_ = new NircIntegration();
        float sMin[3] = { cfg->sceneAABBMin[0], cfg->sceneAABBMin[1], cfg->sceneAABBMin[2] };
        float sMax[3] = { cfg->sceneAABBMax[0], cfg->sceneAABBMax[1], cfg->sceneAABBMax[2] };
        if (sMin[0] == sMax[0]) { sMin[0] = -50; sMax[0] = 50; sMin[1] = -50; sMax[1] = 50; sMin[2] = -50; sMax[2] = 50; }
        if (!nirc_->Initialize(context_, renderWidth_, renderHeight_, sMin, sMax)) {
            Log(L"[VK Renderer] NIRC init failed\n");
            delete nirc_;
            nirc_ = nullptr;
        } else if (rtResources_) {
            // Bind NIRC buffers to RT pipeline descriptors (44-48)
            VkBuffer bufs[5] = {
                nirc_->GetTrainingSampleBuffer(), nirc_->GetHashFeatureBuffer(),
                nirc_->GetWeightBuffer(), nirc_->GetQueryInputBuffer(), nirc_->GetQueryOutputBuffer()
            };
            VkDeviceSize sizes[5] = {
                nirc_->GetTrainingSampleBufferSize(), nirc_->GetHashFeatureBufferSize(),
                nirc_->GetWeightBufferSize(), nirc_->GetQueryInputBufferSize(), nirc_->GetQueryOutputBufferSize()
            };
            VkDescriptorBufferInfo bufInfos[5] = {};
            VkWriteDescriptorSet writes[5] = {};
            for (int i = 0; i < 5; i++) {
                bufInfos[i] = { bufs[i], 0, sizes[i] };
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = rtResources_->GetDescriptorSet();
                writes[i].dstBinding = 44 + i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &bufInfos[i];
            }
            vkUpdateDescriptorSets(context_->GetDevice(), 5, writes, 0, nullptr);
            Log(L"[VK Renderer] NIRC buffers bound to descriptors 44-48\n");
        }
    }

    // Wavefront (the only path tracer we have)
    wavefrontPipeline_ = new WavefrontPipeline();
    if (!wavefrontPipeline_->Initialize(context_, rtResources_, renderWidth_, renderHeight_, cfg ? cfg->maxBounces : 2)) {
        Log(L"[VK Renderer] ERROR: Wavefront init failed (phased)\n");
        delete wavefrontPipeline_;
        wavefrontPipeline_ = nullptr;
        return;
    }

    rtReady_ = true;
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
    if (rtResources_) {
        rtResources_->UpdateTLASDescriptor();
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

bool Renderer::UploadBLASAttributes(int blasIndex, const float* normals, const float* uvs, uint32_t vertexCount, const float* colors) {
    if (!accelBuilder_) return false;
    return accelBuilder_->UploadBLASAttributes(blasIndex, normals, uvs, vertexCount, colors);
}

int Renderer::QueueBLAS(const float* vertices, uint32_t vertexCount,
                         const uint32_t* indices, uint32_t indexCount,
                         const float* normals, const float* uvs, const float* colors) {
    if (!accelBuilder_) return -1;
    return accelBuilder_->QueueBLAS(vertices, vertexCount, indices, indexCount, normals, uvs, colors);
}

int Renderer::FlushBLASBatch() {
    if (!accelBuilder_) return 0;
    return accelBuilder_->FlushBLASBatch();
}

void Renderer::FreeBLAS(int blasIndex) {
    if (accelBuilder_) accelBuilder_->FreeBLAS(blasIndex);
}

bool Renderer::UploadBLASPrimitiveMaterials(int blasIndex, const uint32_t* materialIds, uint32_t primitiveCount) {
    if (!accelBuilder_) return false;
    return accelBuilder_->UploadBLASPrimitiveMaterials(blasIndex, materialIds, primitiveCount);
}

bool Renderer::UploadBLASPrimitiveYBounds(int blasIndex, const float* yBounds, uint32_t primitiveCount) {
    if (!accelBuilder_) return false;
    return accelBuilder_->UploadBLASPrimitiveYBounds(blasIndex, yBounds, primitiveCount);
}

void Renderer::ClearGeometry() {
    if (context_) vkDeviceWaitIdle(context_->GetDevice());
    if (accelBuilder_) accelBuilder_->ClearBLAS();
    rtReady_ = false;
    instanceTransformCount_ = 0;
    prevInstanceTransforms_.clear();
    currInstanceTransforms_.clear();
    Log(L"[VK Renderer] Geometry cleared\n");
}

void Renderer::UploadMaterialBuffer(const void* materials, uint32_t count) {
    if (rtResources_) {
        rtResources_->UpdateMaterialBuffer(static_cast<const vk::GPUMaterial*>(materials), count);
    }
}

void Renderer::UploadEmissiveTriangles(const float* data, uint32_t triangleCount) {
    if (rtResources_) {
        rtResources_->UpdateEmissiveTriangleBuffer(data, triangleCount);
    }
}

void Renderer::UploadLightTree(const void* nodes, uint32_t nodeCount,
                                const void* emitters, uint32_t emitterCount) {
    if (rtResources_) {
        rtResources_->UpdateLightTreeBuffer(nodes, nodeCount);
    }
}

void Renderer::UpdateTextureDescriptors(void* texManager) {
    if (rtResources_) {
        rtResources_->UpdateTextureDescriptors(static_cast<vk::TextureManager*>(texManager));
    }
}

bool Renderer::BuildTLASInstanced(const std::vector<vk::TLASInstance>& instances) {
    if (!accelBuilder_) return false;
    if (!accelBuilder_->BuildTLAS(instances)) return false;
    if (rtResources_) {
        rtResources_->UpdateTLASDescriptor();
    }

    // Cache for partial updates
    cachedTLASInstances_ = instances;

    // Capture per-instance transforms for motion vectors.
    currInstanceTransforms_.resize(instances.size() * 12);
    for (size_t i = 0; i < instances.size(); i++) {
        memcpy(&currInstanceTransforms_[i * 12], instances[i].transform, 12 * sizeof(float));
    }
    if (instances.size() != instanceTransformCount_ || prevInstanceTransforms_.empty()) {
        prevInstanceTransforms_ = currInstanceTransforms_;
    }
    instanceTransformCount_ = (uint32_t)instances.size();

    rtReady_ = true;
    return true;
}

bool Renderer::UpdateInstanceTransforms(const uint32_t* indices, const float* transforms, uint32_t count) {
    if (!accelBuilder_ || cachedTLASInstances_.empty()) return false;

    // Patch cached instances at specified indices
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = indices[i];
        if (idx >= cachedTLASInstances_.size()) continue;
        memcpy(cachedTLASInstances_[idx].transform, &transforms[i * 12], 12 * sizeof(float));
    }

    // TLAS refit (UPDATE mode — faster than full rebuild)
    if (!accelBuilder_->UpdateTLAS(cachedTLASInstances_)) return false;

    // Update motion vector transforms
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = indices[i];
        if (idx < instanceTransformCount_) {
            memcpy(&currInstanceTransforms_[idx * 12], &transforms[i * 12], 12 * sizeof(float));
        }
    }

    return true;
}


void Renderer::UpdateCamera(const CameraUBO& camera) {
    if (rtResources_) {
        rtResources_->UpdateCamera(camera);
    }
    // Track the Python-side frame index for shader use.

    // Cache view/proj for downstream passes (DLSS RR jitter, hybrid raster MVPs)
    memcpy(lastView_, camera.view, 64);
    memcpy(lastProj_, camera.proj, 64);
    // Extract camera world position from viewInverse column 3 (column-major)
    camWorldPos_[0] = camera.viewInverse[12];
    camWorldPos_[1] = camera.viewInverse[13];
    camWorldPos_[2] = camera.viewInverse[14];

    // Store jitter for DLSS
    prevJitterX_ = jitterX_;
    prevJitterY_ = jitterY_;
    jitterX_ = camera.jitterData[0];
    jitterY_ = camera.jitterData[1];

}

bool Renderer::ReadPickResult(uint32_t& outCustomIndex, uint32_t& outPrimitiveId, uint32_t& outMaterialId) {
    if (!rtResources_) return false;
    auto result = rtResources_->ReadPickResult();
    if (!result.valid) return false;
    outCustomIndex = result.customIndex;
    outPrimitiveId = result.primitiveId;
    outMaterialId = result.materialId;
    rtResources_->ResetPickBuffer();
    return true;
}

void Renderer::RenderFrameRT() {
    if (!rtReady_ || !rtResources_ || !interop_ || !wavefrontPipeline_ || !wavefrontPipeline_->IsReady()) {
        Log(L"[VK Renderer] RenderFrameRT SKIPPED: rtReady=%d rtPipeline=%p interop=%p wavefront=%p wfReady=%d\n",
            (int)rtReady_, (void*)rtResources_, (void*)interop_,
            (void*)wavefrontPipeline_,
            (int)(wavefrontPipeline_ ? wavefrontPipeline_->IsReady() : false));
        return;
    }


    VkDevice device = context_->GetDevice();
    VkCommandBuffer cmd = commandBuffers_[currentFrame_];

    // Wait for ALL in-flight frames to complete.  With double-buffered interop,
    // GL reads the buffer from the most recent submit (prevSlot), so we must
    // ensure it's done before draw_gl runs after this call returns.
    // Waiting for both fences here keeps draw_gl non-blocking (~0ms).
    auto fenceT0 = std::chrono::high_resolution_clock::now();
    vkWaitForFences(device, MAX_FRAMES_IN_FLIGHT, inFlightFences_.data(), VK_TRUE, UINT64_MAX);
    auto fenceT1 = std::chrono::high_resolution_clock::now();
    float fenceWaitMs = std::chrono::duration<float, std::milli>(fenceT1 - fenceT0).count();
    if (fenceWaitMs > 5.0f && (frameIndex_ % 30 == 0 || fenceWaitMs > 50.0f)) {
        Log(L"[PERF] frame %u: fence wait %.1f ms\n", frameIndex_, fenceWaitMs);
    }
    vkResetFences(device, 1, &inFlightFences_[currentFrame_]);

    // GPU profiling: readback AFTER fence wait (all GPU work done, no blocking)
    if (!timestampReady_) InitTimestampQueries();
    if (timestampReady_ && frameIndex_ > 0) ReadbackTimestamps();

    if (tonemapReady_ && (frameIndex_ % 300) == 0) {
        ReloadAgXLutIfChanged();  // filesystem stat every ~10s, not every 2s
    }

    // Single command buffer: RT → DLSS (RR or SR+tonemap) → ImGui → Readback
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (!VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo))) return;

    // Reset timestamp queries for this frame
    if (timestampReady_)
        vkCmdResetQueryPool(cmd, timestampQueryPool_, 0, TS_COUNT);
        tsWritten_ = 0;

    // Check debug view mode early (needed for DLSS bypass).
    // Debug view 8 (Motion Vectors) is rendered via wf_output at render res
    // and upscaled by DLSS like a normal frame, so it must NOT trigger the
    // display-res bypass — doing so would dispatch wavefront at display res
    // against buffers sized for render res and crash on out-of-bounds writes.
    PathTracerConfig* rtCfgEarly = VK_GetConfig();
    bool debugViewActive = rtCfgEarly && rtCfgEarly->debugView >= 2
                                      && rtCfgEarly->debugView != 8;

    // DLSS debug bypass: when debug views are active, route RT output directly
    // to the interop image instead of through DLSS (which would show stale frames)
    if (dlssActive_ && rtResources_) {
        bool wantDebugBypass = debugViewActive;
        if (wantDebugBypass != dlssDebugBypass_) {
            VkImageView targetView = wantDebugBypass
                ? interop_->GetSharedImageView()
                : dlssColorInputView_;
            if (targetView != VK_NULL_HANDLE) {
                rtResources_->UpdateStorageImage(targetView);
                dlssDebugBypass_ = wantDebugBypass;
            }
        }
    }

    // Upload previous-frame instance transforms for per-object motion vectors.
    // This must happen every frame so that once an object stops moving,
    // prev == curr and the shader outputs zero motion vectors (no ghosting).
    if (rtResources_ && instanceTransformCount_ > 0) {
        rtResources_->UpdatePrevTransforms(prevInstanceTransforms_.data(), instanceTransformCount_);
        rtResources_->UpdateCurrTransforms(currInstanceTransforms_.data(), instanceTransformCount_);
    }

    bool diagFlush = false;  // Set true to flush GPU between stages for crash isolation

    // NRC: populate constants + begin frame (only when user enabled)

    WriteTimestamp(cmd, TS_START);

    // 0. Hybrid G-buffer rasterization pass (before RT dispatch)
    PathTracerConfig* hybridCfg = VK_GetConfig();
    bool hybridEnabled = hybridCfg && hybridCfg->hybridRasterization;
    if (hybridEnabled && !hybridGBufferReady_) {
        CreateHybridGBufferPipeline();
    }
    if (hybridEnabled && hybridGBufferReady_) {
        RecordHybridGBufferPass(cmd);

        VkMemoryBarrier rasterToRT{};
        rasterToRT.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        rasterToRT.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        rasterToRT.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &rasterToRT, 0, nullptr, 0, nullptr);
        hybridGBufferRendered_ = true;
    }
    WriteTimestamp(cmd, TS_HYBRID);

    // 0.5 NRC update pass: trace at training resolution to generate training data

    // 1. Path tracing dispatch — wavefront compute kernels
    interop_->TransitionForRTWrite(cmd);
    {
        PathTracerConfig* wfCfg = VK_GetConfig();
        if (frameIndex_ < 3) {
            Log(L"[WF] frame %u: dispatch %ux%u dlssActive=%d dlssRR=%d tonemapReady=%d\n",
                frameIndex_, renderWidth_, renderHeight_, (int)dlssActive_, (int)dlssRRActive_, (int)tonemapReady_);
        }
        uint32_t dispW = (dlssDebugBypass_ && debugViewActive) ? width_ : renderWidth_;
        uint32_t dispH = (dlssDebugBypass_ && debugViewActive) ? height_ : renderHeight_;
        wavefrontPipeline_->RecordDispatch(cmd, dispW, dispH,
            rtResources_->GetDescriptorSet(), wfCfg ? wfCfg->maxBounces : 2,
            wfCfg ? static_cast<uint32_t>(wfCfg->samplesPerPixel) : 1);
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

    WriteTimestamp(cmd, TS_RT);

    // SHARC resolve: merge accumulation → resolved (EMA + aging + eviction)
    if (sharcResolveReady_ && rtResources_->HasSHARCBuffers()) {
        VkMemoryBarrier rtToSharc{};
        rtToSharc.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        rtToSharc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        rtToSharc.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        // Source stage depends on whether wavefront (compute) or monolithic (raygen) ran
        bool wavefrontActive = wavefrontPipeline_ && wavefrontPipeline_->IsReady();
        VkPipelineStageFlags sharcSrcStage = wavefrontActive ?
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT :
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
        vkCmdPipelineBarrier(cmd,
            sharcSrcStage,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &rtToSharc, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sharcResolvePipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            sharcResolvePipelineLayout_, 0, 1, &sharcResolveDescriptorSet_, 0, nullptr);

        // Push constants: capacity, frameIndex, accumulationFrameMax, staleFrameMax, radianceScale
        struct { uint32_t capacity; uint32_t frameIndex; uint32_t accFrameMax; uint32_t staleMax; float radScale; } sharcPC;
        sharcPC.capacity = RTResources::SHARC_CAPACITY;
        sharcPC.frameIndex = frameIndex_;
        sharcPC.accFrameMax = 256;  // stable temporal accumulation (warmup burst disabled)
        sharcPC.staleMax = 128;
        sharcPC.radScale = 1000.0f;
        vkCmdPushConstants(cmd, sharcResolvePipelineLayout_,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sharcPC), &sharcPC);

        // capacity / 256 threads per group
        uint32_t groups = (RTResources::SHARC_CAPACITY + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
    }

    // Surfel GI resolve: merge surfel accumulation → resolved
    if (surfelResolveReady_ && rtResources_->HasSurfelBuffers()) {
        VkMemoryBarrier rtToSurfel{};
        rtToSurfel.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        rtToSurfel.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        rtToSurfel.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // after SHARC resolve
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &rtToSurfel, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, surfelResolvePipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            surfelResolvePipelineLayout_, 0, 1, &surfelResolveDescSet_, 0, nullptr);

        struct { uint32_t capacity; uint32_t frameIndex; uint32_t accFrameMax; uint32_t staleMax; float radScale; } surfelPC;
        surfelPC.capacity = RTResources::SURFEL_CAPACITY;
        surfelPC.frameIndex = frameIndex_;
        surfelPC.accFrameMax = 64;   // faster convergence than SHARC
        surfelPC.staleMax = 128;
        surfelPC.radScale = 1000.0f;
        vkCmdPushConstants(cmd, surfelResolvePipelineLayout_,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(surfelPC), &surfelPC);

        uint32_t groups = (RTResources::SURFEL_CAPACITY + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
    }

    // Hair contour detection: screen-space edge detection on hairV buffer
    if (hairContourReady_ && rtResources_ && rtResources_->HasGBuffers()) {
        VkMemoryBarrier rtToHairContour{};
        rtToHairContour.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        rtToHairContour.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        rtToHairContour.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &rtToHairContour, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hairContourPipeline_);
        VkDescriptorSet rtDescSet = rtResources_->GetDescriptorSet();
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            hairContourPipelineLayout_, 0, 1, &rtDescSet, 0, nullptr);

        struct { uint32_t width; uint32_t height; } hairContourPC;
        hairContourPC.width = renderWidth_;
        hairContourPC.height = renderHeight_;
        vkCmdPushConstants(cmd, hairContourPipelineLayout_,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(hairContourPC), &hairContourPC);

        uint32_t gx = (renderWidth_ + 7) / 8;
        uint32_t gy = (renderHeight_ + 7) / 8;
        vkCmdDispatch(cmd, gx, gy, 1);
    }

    WriteTimestamp(cmd, TS_HAIR);

    // K5 (wf_output) wrote G-buffers + the final HDR image into dlssColorInput.
    // From here we either feed DLSS SR + tonemap, or DLSS Ray Reconstruction.
    bool wavefrontActive = wavefrontPipeline_ && wavefrontPipeline_->IsReady();

    if (wavefrontActive && dlssActive_ && dlss_ && dlss_->IsInitialized() && dlss_->IsSupported() && !dlssRRActive_ && !debugViewActive) {
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
            rtResources_->GetDlssDepthImage(), rtResources_->GetDlssDepthView(),
            rtResources_->GetMotionVectorsImage(), rtResources_->GetMotionVectorsView(),
            dlssHdrOutput_, dlssHdrOutputView_,
            VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
            jitterX_, jitterY_, wfDlssDelta / 1000.0f, 0.0f, false,
            rtResources_->GetReactiveMaskImage(), rtResources_->GetReactiveMaskView());

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

    // 2. Ray Reconstruction path (denoises + upscales K5's noisy output in one pass)
    // Skip when debug view active — raygen wrote final LDR directly to interop
    if (dlssRRActive_ && dlss_ && dlss_->IsRRActive() && !debugViewActive) {
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
            rtResources_->GetViewDepthImage(),                // linear view-space depth (better for RR)
            rtResources_->GetViewDepthView(),
            rtResources_->GetMotionVectorsImage(),            // MVs
            rtResources_->GetMotionVectorsView(),
            rtResources_->GetNormalRoughnessImage(),          // normals + roughness
            rtResources_->GetNormalRoughnessView(),
            rtResources_->GetAlbedoBufferImage(),             // albedo
            rtResources_->GetAlbedoBufferView(),
            jitterX_, jitterY_,
            rrDeltaMs / 1000.0f,
            lastView_, lastProj_,
            rtResources_->GetSpecularAlbedoImage(),           // EnvBRDFApprox specular albedo
            rtResources_->GetSpecularAlbedoView(),
            rtResources_->GetSpecularMVImage(),               // specular motion vectors
            rtResources_->GetSpecularMVView(),
            rtResources_->GetDiffuseRadianceImage(),          // diffuse hit distance (.a)
            rtResources_->GetDiffuseRadianceView(),
            rtResources_->GetSpecularRadianceImage(),         // specular hit distance (.a)
            rtResources_->GetSpecularRadianceView(),
            rrReset,
            rtResources_->GetReactiveMaskImage(),             // reactive mask (dynamic objects)
            rtResources_->GetReactiveMaskView());

        WriteTimestamp(cmd, TS_DENOISE);
        WriteTimestamp(cmd, TS_COMPOSITE);  // no separate composite, marker only

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

    // NRC: query neural cache + train network

    // NIRC: train neural incident radiance cache from path tracer samples
    if (nirc_ && nirc_->IsReady()) {
        VkMemoryBarrier rtToNirc{};
        rtToNirc.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        rtToNirc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        rtToNirc.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &rtToNirc, 0, nullptr, 0, nullptr);
        // Training: 1/16 of pixels, capped at 64K
        uint32_t trainSamples = std::min(renderWidth_ * renderHeight_ / 16, 65536u);
        nirc_->Train(cmd, trainSamples, frameIndex_);

        // Barrier: training writes → inference reads (hash grid + weights updated)
        VkMemoryBarrier trainToInfer{};
        trainToInfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        trainToInfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        trainToInfer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &trainToInfer, 0, nullptr, 0, nullptr);

        // Inference: predict radiance for all pixels (next frame reads results)
        uint32_t queryCount = renderWidth_ * renderHeight_;
        nirc_->Infer(cmd, queryCount);
    }

    // Fill any unwritten timestamp slots (skipped stages → current time → 0ms delta).
    FillMissingTimestamps(cmd);

    // 4. ImGui overlay (renders on top of final output)
    if (imguiReady_) {
        RenderImGuiOverlay(cmd);
    }

    // 5. Readback
    if (!useDirectInterop_) {
        interop_->RecordReadbackCopy(cmd, inFlightFences_[currentFrame_]);
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
        Log(L"[VK Renderer] vkQueueSubmit failed: %d\n", (int)submitResult);
        return;
    }

    // NRC: end frame (must be called after queue submit)

    // No WaitIdle — double-buffered readback with per-buffer fence sync

    // After rendering, current transforms become previous for the next frame.
    // Without this, prevTransforms would stay stale after an object stops
    // moving, causing perpetual non-zero motion vectors and ghosting.
    if (instanceTransformCount_ > 0) {
        prevInstanceTransforms_ = currInstanceTransforms_;
    }

    // Swap reservoir ping-pong buffers (current → previous for next frame)
    if (rtResources_ && rtResources_->HasGIReservoirBuffers()) {
        rtResources_->SwapGIReservoirBuffers();   // DI (24-25)
        rtResources_->SwapGIWfReservoirBuffers(); // GI (49-50)
    }

    // Double-buffer swap: flip write/read indices so GL displays the completed frame
    // while RT writes to the other buffer next frame.
    // Note: the fence wait at the START of the next RenderFrameRT() guarantees the
    // read buffer's GPU work is complete before DrawGL() reads it.
    if (interop_) {
        interop_->SwapBuffers();
        // Update all descriptors that reference the interop image to point to new write slot
        if (rtResources_ && (!dlssActive_ || dlssDebugBypass_)) {
            rtResources_->UpdateStorageImage(interop_->GetSharedImageView());
        }
        if (tonemapReady_) {
            UpdateTonemapDescriptors();
        }
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
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    return interop_ ? interop_->CopyReadbackResult(outData, bufferSize, device) : false;
}

bool Renderer::ReadbackHDRPixelsFloat(float* outData, uint32_t pixelCount) {
    if (!context_ || !outData || pixelCount == 0) return false;

    VkImage srcImage = VK_NULL_HANDLE;
    uint32_t srcWidth = 0;
    uint32_t srcHeight = 0;

    if (dlssHdrOutput_) {
        srcImage = dlssHdrOutput_;
        srcWidth = width_;
        srcHeight = height_;
    }
    else if (dlssColorInput_) {
        srcImage = dlssColorInput_;
        srcWidth = renderWidth_;
        srcHeight = renderHeight_;
    }
    else {
        return false;
    }

    const uint32_t expectedPixels = srcWidth * srcHeight;
    if (pixelCount < expectedPixels) {
        Log(L"[VK Renderer] HDR readback buffer too small (%u < %u)\n", pixelCount, expectedPixels);
        return false;
    }

    VkDevice device = context_->GetDevice();
    VkQueue queue = context_->GetGraphicsQueue();
    if (device == VK_NULL_HANDLE || queue == VK_NULL_HANDLE) return false;

    // Exact viewport color path: wait for the submitted frame, then copy the linear HDR image.
    vkQueueWaitIdle(queue);

    const VkDeviceSize stagingSize = VkDeviceSize(expectedPixels) * 4u * sizeof(uint16_t);
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = stagingSize;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements memReqs{};
    vkGetBufferMemoryRequirements(device, stagingBuffer, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(
        memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &stagingMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        return false;
    }
    vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = srcImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { srcWidth, srcHeight, 1 };

    vkCmdCopyImageToBuffer(
        cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    context_->EndSingleTimeCommands(cmd);

    void* mapped = nullptr;
    bool ok = false;
    if (vkMapMemory(device, stagingMemory, 0, stagingSize, 0, &mapped) == VK_SUCCESS && mapped) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(mapped);
        const uint32_t totalFloats = expectedPixels * 4u;
        for (uint32_t i = 0; i < totalFloats; ++i) {
            outData[i] = HalfBitsToFloat(src[i]);
        }
        vkUnmapMemory(device, stagingMemory);
        ok = true;
    }

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
    return ok;
}

bool Renderer::ImportD3D11Texture(HANDLE ntHandle, uint32_t width, uint32_t height) {
    if (!interop_) return false;
    bool result = interop_->ImportD3D11Texture(ntHandle, width, height);
    if (result && rtResources_) {
        // When DLSS active, RT writes to intermediate image (not interop)
        if (dlssActive_ && dlssColorInputView_) {
            rtResources_->UpdateStorageImage(dlssColorInputView_);
        } else {
            rtResources_->UpdateStorageImage(interop_->GetSharedImageView());
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


void Renderer::WaitForReadBuffer() {
    // GL reads the buffer from the PREVIOUS submit.  Wait on that fence
    // (the "other" slot) to guarantee the read buffer is complete.
    // Much cheaper than vkQueueWaitIdle: only waits for one specific frame.
    if (!context_) return;
    uint32_t prevSlot = (currentFrame_ + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT;
    if (inFlightFences_[prevSlot] != VK_NULL_HANDLE) {
        vkWaitForFences(context_->GetDevice(), 1, &inFlightFences_[prevSlot], VK_TRUE, UINT64_MAX);
    }
}


void Renderer::Shutdown() {
    ShutdownImGui();
    ShutdownTimestampQueries();
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

    // Shutdown sub-systems (hair contour, SHARC resolve, auto-exposure, hybrid, DLSS)
    ShutdownHairContour();
    ShutdownSHARCResolve();
    ShutdownExposureResolve();
    ShutdownHybridGBuffer();
    ShutdownDLSS();
    if (nirc_) { nirc_->Shutdown(); delete nirc_; nirc_ = nullptr; }

    // Shutdown wavefront + RT modules
    if (wavefrontPipeline_) { wavefrontPipeline_->Shutdown(); delete wavefrontPipeline_; wavefrontPipeline_ = nullptr; }
    if (rtResources_) { rtResources_->Shutdown(); delete rtResources_; rtResources_ = nullptr; }
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

