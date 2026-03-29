#include "vk_interop.h"
#include "vk_context.h"
#include "../../include/ignis_log.h"

namespace acpt {
namespace vk {

bool Interop::CreateSharedImageSlot(int idx) {
    VkDevice device = context_->GetDevice();

    // Create shared image with external memory
    VkExternalMemoryImageCreateInfo externalInfo{};
    externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &externalInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &imageInfo, nullptr, &sharedImage_[idx]) != VK_SUCCESS) {
        Log(L"[VK Interop] ERROR: Failed to create shared image [%d]\n", idx);
        return false;
    }

    // Allocate exportable memory
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, sharedImage_[idx], &memReqs);

    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &exportInfo;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &sharedMemory_[idx]) != VK_SUCCESS) {
        Log(L"[VK Interop] ERROR: Failed to allocate shared memory [%d]\n", idx);
        return false;
    }

    vkBindImageMemory(device, sharedImage_[idx], sharedMemory_[idx], 0);
    allocationSize_[idx] = memReqs.size;

    // Export NT handle
    VkMemoryGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.memory = sharedMemory_[idx];
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    if (vkGetMemoryWin32HandleKHR_(device, &handleInfo, &ntHandle_[idx]) != VK_SUCCESS) {
        Log(L"[VK Interop] ERROR: Failed to export NT handle [%d]\n", idx);
        return false;
    }

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = sharedImage_[idx];
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &sharedImageView_[idx]) != VK_SUCCESS) {
        Log(L"[VK Interop] ERROR: Failed to create shared image view [%d]\n", idx);
        return false;
    }

    return true;
}

bool Interop::Initialize(Context* context, uint32_t width, uint32_t height) {
    context_ = context;
    width_ = width;
    height_ = height;

    VkDevice device = context_->GetDevice();

    // Load function pointer
    vkGetMemoryWin32HandleKHR_ = (PFN_vkGetMemoryWin32HandleKHR)
        vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
    if (!vkGetMemoryWin32HandleKHR_) {
        Log(L"[VK Interop] ERROR: vkGetMemoryWin32HandleKHR not available\n");
        return false;
    }

    // Create double-buffered shared images
    for (int i = 0; i < 2; i++) {
        if (!CreateSharedImageSlot(i)) return false;
    }
    writeIdx_ = 0;

    // Initial layout transition for both images
    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    for (int i = 0; i < 2; i++) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = sharedImage_[i];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
    context_->EndSingleTimeCommands(cmd);

    Log(L"[VK Interop] Double-buffered shared images created (%ux%u), handles=%p/%p\n",
        width, height, ntHandle_[0], ntHandle_[1]);
    return true;
}

void Interop::SwapBuffers() {
    writeIdx_ = 1 - writeIdx_;
}

bool Interop::ImportD3D11Texture(HANDLE ntHandle, uint32_t width, uint32_t height) {
    if (!context_ || !ntHandle) return false;

    VkDevice device = context_->GetDevice();

    // Destroy all existing shared images (D3D11 import uses single-buffer on slot 0)
    for (int i = 0; i < 2; i++) {
        if (sharedImageView_[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(device, sharedImageView_[i], nullptr);
            sharedImageView_[i] = VK_NULL_HANDLE;
        }
        if (sharedImage_[i] != VK_NULL_HANDLE) {
            vkDestroyImage(device, sharedImage_[i], nullptr);
            sharedImage_[i] = VK_NULL_HANDLE;
        }
        if (sharedMemory_[i] != VK_NULL_HANDLE) {
            vkFreeMemory(device, sharedMemory_[i], nullptr);
            sharedMemory_[i] = VK_NULL_HANDLE;
        }
        if (ntHandle_[i]) {
            CloseHandle(ntHandle_[i]);
            ntHandle_[i] = nullptr;
        }
    }
    writeIdx_ = 0;

    // Create image backed by D3D11 external memory (slot 0 only)
    VkExternalMemoryImageCreateInfo externalInfo{};
    externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &externalInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &imageInfo, nullptr, &sharedImage_[0]) != VK_SUCCESS) {
        Log(L"[VK Interop] ERROR: Failed to create image for D3D11 import\n");
        return false;
    }

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, sharedImage_[0], &memReqs);

    VkImportMemoryWin32HandleInfoKHR importInfo{};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
    importInfo.handle = ntHandle;

    VkMemoryDedicatedAllocateInfo dedicatedInfo{};
    dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedInfo.pNext = &importInfo;
    dedicatedInfo.image = sharedImage_[0];

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &dedicatedInfo;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &sharedMemory_[0]) != VK_SUCCESS) {
        Log(L"[VK Interop] ERROR: Failed to import D3D11 memory (vkAllocateMemory)\n");
        vkDestroyImage(device, sharedImage_[0], nullptr);
        sharedImage_[0] = VK_NULL_HANDLE;
        return false;
    }

    vkBindImageMemory(device, sharedImage_[0], sharedMemory_[0], 0);
    allocationSize_[0] = memReqs.size;

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = sharedImage_[0];
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &sharedImageView_[0]) != VK_SUCCESS) {
        Log(L"[VK Interop] ERROR: Failed to create image view for imported texture\n");
        return false;
    }

    width_ = width;
    height_ = height;

    // Initial layout transition
    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    TransitionForRTWrite(cmd);
    context_->EndSingleTimeCommands(cmd);

    Log(L"[VK Interop] D3D11 texture imported successfully (%ux%u) — zero-copy interop active\n", width, height);
    return true;
}

void Interop::Shutdown() {
    // Clean up GL interop first (GL objects reference the VK memory)
    ShutdownGL();

    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) return;

    for (int i = 0; i < 2; i++) {
        if (readbackMapped_[i] && readbackMemory_[i] != VK_NULL_HANDLE) {
            vkUnmapMemory(device, readbackMemory_[i]);
            readbackMapped_[i] = nullptr;
        }
        if (readbackBuffer_[i] != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, readbackBuffer_[i], nullptr);
            readbackBuffer_[i] = VK_NULL_HANDLE;
        }
        if (readbackMemory_[i] != VK_NULL_HANDLE) {
            vkFreeMemory(device, readbackMemory_[i], nullptr);
            readbackMemory_[i] = VK_NULL_HANDLE;
        }
    }

    for (int i = 0; i < 2; i++) {
        if (sharedImageView_[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(device, sharedImageView_[i], nullptr);
            sharedImageView_[i] = VK_NULL_HANDLE;
        }
        if (sharedImage_[i] != VK_NULL_HANDLE) {
            vkDestroyImage(device, sharedImage_[i], nullptr);
            sharedImage_[i] = VK_NULL_HANDLE;
        }
        if (sharedMemory_[i] != VK_NULL_HANDLE) {
            vkFreeMemory(device, sharedMemory_[i], nullptr);
            sharedMemory_[i] = VK_NULL_HANDLE;
        }
        if (ntHandle_[i]) {
            CloseHandle(ntHandle_[i]);
            ntHandle_[i] = nullptr;
        }
        allocationSize_[i] = 0;
    }

    Log(L"[VK Interop] Shutdown\n");
}

bool Interop::EnsureReadbackBuffer() {
    if (readbackBuffer_[0] != VK_NULL_HANDLE) return true;

    VkDevice device = context_->GetDevice();
    uint32_t imageSize = width_ * height_ * 4;

    for (int i = 0; i < 2; i++) {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = imageSize;
        bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufInfo, nullptr, &readbackBuffer_[i]) != VK_SUCCESS)
            return false;

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, readbackBuffer_[i], &memReqs);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &readbackMemory_[i]) != VK_SUCCESS) {
            vkDestroyBuffer(device, readbackBuffer_[i], nullptr);
            readbackBuffer_[i] = VK_NULL_HANDLE;
            return false;
        }
        vkBindBufferMemory(device, readbackBuffer_[i], readbackMemory_[i], 0);

        if (vkMapMemory(device, readbackMemory_[i], 0, imageSize, 0, &readbackMapped_[i]) != VK_SUCCESS) {
            readbackMapped_[i] = nullptr;
            return false;
        }
        // Zero-fill so first-frame readback doesn't show garbage
        memset(readbackMapped_[i], 0, imageSize);
    }

    return true;
}

void Interop::RecordReadbackCopy(VkCommandBuffer cmd, VkFence submitFence) {
    if (!EnsureReadbackBuffer()) return;

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = sharedImage_[writeIdx_];
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {width_, height_, 1};

    vkCmdCopyImageToBuffer(cmd, sharedImage_[writeIdx_], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            readbackBuffer_[readbackWriteIdx_], 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Track which fence protects this buffer
    readbackFence_[readbackWriteIdx_] = submitFence;

    // Swap for next frame
    readbackWriteIdx_ = 1 - readbackWriteIdx_;
    readbackFirstFrame_ = false;
}

bool Interop::CopyReadbackResult(void* outData, uint32_t bufferSize, VkDevice device) {
    if (readbackFirstFrame_) {
        Log(L"[VK Interop] CopyReadback: firstFrame=true\n");
        return false;
    }
    if (!outData) return false;

    // Read from the buffer NOT being written (previous frame's data)
    uint32_t readIdx = readbackWriteIdx_;  // after swap, this is the next write target = previous read
    if (!readbackMapped_[readIdx]) {
        Log(L"[VK Interop] CopyReadback: mapped[%u]=null\n", readIdx);
        return false;
    }

    // Wait for the fence that protects this buffer's GPU copy
    if (device != VK_NULL_HANDLE && readbackFence_[readIdx] != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &readbackFence_[readIdx], VK_TRUE, UINT64_MAX);
    }

    uint32_t imageSize = width_ * height_ * 4;
    if (bufferSize < imageSize) {
        Log(L"[VK Interop] CopyReadback: bufSize=%u < imgSize=%u (%ux%u)\n",
            bufferSize, imageSize, width_, height_);
        return false;
    }
    memcpy(outData, readbackMapped_[readIdx], imageSize);
    return true;
}

bool Interop::ReadbackPixels(void* outData, uint32_t bufferSize) {
    if (!context_ || !sharedImage_[writeIdx_] || !outData) return false;
    uint32_t imageSize = width_ * height_ * 4;
    if (bufferSize < imageSize) return false;

    // Fallback: separate command buffer (legacy path)
    VkCommandBuffer cmd = context_->BeginSingleTimeCommands();
    RecordReadbackCopy(cmd);
    context_->EndSingleTimeCommands(cmd);

    return CopyReadbackResult(outData, bufferSize);
}

void Interop::TransitionForRTWrite(VkCommandBuffer cmd) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = sharedImage_[writeIdx_];
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void Interop::TransitionForExternalRead(VkCommandBuffer cmd) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = sharedImage_[writeIdx_];
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// ============================================================================
// OpenGL interop — import Vulkan shared memory as GL texture, draw fullscreen
// ============================================================================

// GL types (minimal definitions — avoid including GL headers which conflict with Vulkan)
typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLint;
typedef int GLsizei;
typedef char GLchar;
typedef unsigned char GLboolean;
typedef uint64_t GLuint64;

// GL constants
#define GL_TRUE                  1
#define GL_FALSE                 0
#define GL_TEXTURE_2D            0x0DE1
#define GL_RGBA8                 0x8058
#define GL_TEXTURE_MIN_FILTER    0x2801
#define GL_TEXTURE_MAG_FILTER    0x2800
#define GL_TEXTURE_WRAP_S        0x2802
#define GL_TEXTURE_WRAP_T        0x2803
#define GL_NEAREST               0x2600
#define GL_CLAMP_TO_EDGE         0x812F
#define GL_BLEND                 0x0BE2
#define GL_DEPTH_TEST            0x0B71
#define GL_SCISSOR_TEST          0x0C11
#define GL_FRAMEBUFFER_SRGB      0x8DB9
#define GL_TRIANGLES             0x0004
#define GL_FRAGMENT_SHADER       0x8B30
#define GL_VERTEX_SHADER         0x8B31
#define GL_COMPILE_STATUS        0x8B81
#define GL_LINK_STATUS           0x8B82
#define GL_CURRENT_PROGRAM       0x8B8D
#define GL_VERTEX_ARRAY_BINDING  0x85B5
#define GL_ACTIVE_TEXTURE_       0x84E0   // underscore to avoid macro clash
#define GL_TEXTURE_BINDING_2D    0x8069
#define GL_TEXTURE0              0x84C0
#define GL_HANDLE_TYPE_OPAQUE_WIN32_EXT 0x9587

// GL function pointer types
typedef void     (APIENTRY* PFN_glGetIntegerv)(GLenum, GLint*);
typedef void     (APIENTRY* PFN_glBindTexture)(GLenum, GLuint);
typedef void     (APIENTRY* PFN_glGenTextures)(GLsizei, GLuint*);
typedef void     (APIENTRY* PFN_glDeleteTextures)(GLsizei, const GLuint*);
typedef void     (APIENTRY* PFN_glTexParameteri)(GLenum, GLenum, GLint);
typedef void     (APIENTRY* PFN_glEnable)(GLenum);
typedef void     (APIENTRY* PFN_glDisable)(GLenum);
typedef GLboolean(APIENTRY* PFN_glIsEnabled)(GLenum);
typedef void     (APIENTRY* PFN_glDrawArrays)(GLenum, GLint, GLsizei);
typedef GLenum   (APIENTRY* PFN_glGetError)();

typedef GLuint   (APIENTRY* PFN_glCreateShader)(GLenum);
typedef void     (APIENTRY* PFN_glShaderSource)(GLuint, GLsizei, const GLchar**, const GLint*);
typedef void     (APIENTRY* PFN_glCompileShader)(GLuint);
typedef void     (APIENTRY* PFN_glGetShaderiv)(GLuint, GLenum, GLint*);
typedef void     (APIENTRY* PFN_glGetShaderInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*);
typedef GLuint   (APIENTRY* PFN_glCreateProgram)();
typedef void     (APIENTRY* PFN_glAttachShader)(GLuint, GLuint);
typedef void     (APIENTRY* PFN_glLinkProgram)(GLuint);
typedef void     (APIENTRY* PFN_glGetProgramiv)(GLuint, GLenum, GLint*);
typedef void     (APIENTRY* PFN_glGetProgramInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*);
typedef void     (APIENTRY* PFN_glUseProgram)(GLuint);
typedef void     (APIENTRY* PFN_glDeleteShader)(GLuint);
typedef void     (APIENTRY* PFN_glDeleteProgram)(GLuint);
typedef GLint    (APIENTRY* PFN_glGetUniformLocation)(GLuint, const GLchar*);
typedef void     (APIENTRY* PFN_glUniform1i)(GLint, GLint);
typedef void     (APIENTRY* PFN_glActiveTexture)(GLenum);
typedef void     (APIENTRY* PFN_glGenVertexArrays)(GLsizei, GLuint*);
typedef void     (APIENTRY* PFN_glBindVertexArray)(GLuint);
typedef void     (APIENTRY* PFN_glDeleteVertexArrays)(GLsizei, const GLuint*);

typedef void     (APIENTRY* PFN_glCreateMemoryObjectsEXT)(GLsizei, GLuint*);
typedef void     (APIENTRY* PFN_glDeleteMemoryObjectsEXT)(GLsizei, const GLuint*);
typedef void     (APIENTRY* PFN_glTexStorageMem2DEXT)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLuint, GLuint64);
typedef void     (APIENTRY* PFN_glImportMemoryWin32HandleEXT)(GLuint, GLuint64, GLenum, void*);

// Static GL function pointers
static struct {
    PFN_glGetIntegerv GetIntegerv;
    PFN_glBindTexture BindTexture;
    PFN_glGenTextures GenTextures;
    PFN_glDeleteTextures DeleteTextures;
    PFN_glTexParameteri TexParameteri;
    PFN_glEnable Enable;
    PFN_glDisable Disable;
    PFN_glIsEnabled IsEnabled;
    PFN_glDrawArrays DrawArrays;
    PFN_glGetError GetError;

    PFN_glCreateShader CreateShader;
    PFN_glShaderSource ShaderSource;
    PFN_glCompileShader CompileShader;
    PFN_glGetShaderiv GetShaderiv;
    PFN_glGetShaderInfoLog GetShaderInfoLog;
    PFN_glCreateProgram CreateProgram;
    PFN_glAttachShader AttachShader;
    PFN_glLinkProgram LinkProgram;
    PFN_glGetProgramiv GetProgramiv;
    PFN_glGetProgramInfoLog GetProgramInfoLog;
    PFN_glUseProgram UseProgram;
    PFN_glDeleteShader DeleteShader;
    PFN_glDeleteProgram DeleteProgram;
    PFN_glGetUniformLocation GetUniformLocation;
    PFN_glUniform1i Uniform1i;
    PFN_glActiveTexture ActiveTexture;
    PFN_glGenVertexArrays GenVertexArrays;
    PFN_glBindVertexArray BindVertexArray;
    PFN_glDeleteVertexArrays DeleteVertexArrays;

    PFN_glCreateMemoryObjectsEXT CreateMemoryObjectsEXT;
    PFN_glDeleteMemoryObjectsEXT DeleteMemoryObjectsEXT;
    PFN_glTexStorageMem2DEXT TexStorageMem2DEXT;
    PFN_glImportMemoryWin32HandleEXT ImportMemoryWin32HandleEXT;

    bool loaded;
} gl = {};

static bool LoadGLFunctions() {
    if (gl.loaded) return true;

    HMODULE hGL = GetModuleHandleA("opengl32.dll");
    if (!hGL) {
        Log(L"[GL Interop] opengl32.dll not loaded in process\n");
        return false;
    }

    typedef PROC (WINAPI* PFN_wglGetProcAddress)(LPCSTR);
    auto wglGetProc = (PFN_wglGetProcAddress)GetProcAddress(hGL, "wglGetProcAddress");
    if (!wglGetProc) {
        Log(L"[GL Interop] wglGetProcAddress not found\n");
        return false;
    }

    // Core GL 1.x (from opengl32.dll)
    #define LOAD_GL(name) gl.name = (decltype(gl.name))GetProcAddress(hGL, "gl" #name)
    LOAD_GL(GetIntegerv);
    LOAD_GL(BindTexture);
    LOAD_GL(GenTextures);
    LOAD_GL(DeleteTextures);
    LOAD_GL(TexParameteri);
    LOAD_GL(Enable);
    LOAD_GL(Disable);
    LOAD_GL(IsEnabled);
    LOAD_GL(DrawArrays);
    LOAD_GL(GetError);
    #undef LOAD_GL

    // GL 2.0+ and extensions (via wglGetProcAddress)
    #define LOAD_WGL(name) gl.name = (decltype(gl.name))wglGetProc("gl" #name)
    LOAD_WGL(CreateShader);
    LOAD_WGL(ShaderSource);
    LOAD_WGL(CompileShader);
    LOAD_WGL(GetShaderiv);
    LOAD_WGL(GetShaderInfoLog);
    LOAD_WGL(CreateProgram);
    LOAD_WGL(AttachShader);
    LOAD_WGL(LinkProgram);
    LOAD_WGL(GetProgramiv);
    LOAD_WGL(GetProgramInfoLog);
    LOAD_WGL(UseProgram);
    LOAD_WGL(DeleteShader);
    LOAD_WGL(DeleteProgram);
    LOAD_WGL(GetUniformLocation);
    LOAD_WGL(Uniform1i);
    LOAD_WGL(ActiveTexture);
    LOAD_WGL(GenVertexArrays);
    LOAD_WGL(BindVertexArray);
    LOAD_WGL(DeleteVertexArrays);
    LOAD_WGL(CreateMemoryObjectsEXT);
    LOAD_WGL(DeleteMemoryObjectsEXT);
    LOAD_WGL(TexStorageMem2DEXT);
    LOAD_WGL(ImportMemoryWin32HandleEXT);
    #undef LOAD_WGL

    // Verify critical functions
    if (!gl.GetIntegerv || !gl.BindTexture || !gl.GenTextures || !gl.DrawArrays ||
        !gl.CreateShader || !gl.UseProgram || !gl.GenVertexArrays || !gl.BindVertexArray ||
        !gl.CreateMemoryObjectsEXT || !gl.ImportMemoryWin32HandleEXT || !gl.TexStorageMem2DEXT) {
        Log(L"[GL Interop] Required GL functions not available (missing GL_EXT_memory_object?)\n");
        return false;
    }

    gl.loaded = true;
    return true;
}

// Fullscreen triangle vertex shader (Y-flip for Vulkan → OpenGL coordinate convention)
static const char* kGLVertexShaderSrc = R"glsl(
#version 330 core
out vec2 vUV;
void main() {
    float x = (gl_VertexID == 1) ? 3.0 : -1.0;
    float y = (gl_VertexID == 2) ? 3.0 : -1.0;
    vUV = vec2((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
)glsl";

static const char* kGLFragmentShaderSrc = R"glsl(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexture;
void main() {
    fragColor = texture(uTexture, vUV);
}
)glsl";

static GLuint CompileGLShader(GLenum type, const char* src) {
    GLuint shader = gl.CreateShader(type);
    gl.ShaderSource(shader, 1, &src, nullptr);
    gl.CompileShader(shader);

    GLint success;
    gl.GetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        gl.GetShaderInfoLog(shader, 512, nullptr, infoLog);
        Log(L"[GL Interop] Shader compile error: %S\n", infoLog);
        gl.DeleteShader(shader);
        return 0;
    }
    return shader;
}

bool Interop::InitGLInterop() {
    if (glInteropReady_) return true;
    if (!sharedImage_[0] || !ntHandle_[0] || allocationSize_[0] == 0) {
        Log(L"[GL Interop] No shared image/handle to import\n");
        return false;
    }

    if (!LoadGLFunctions()) return false;

    // Query GL context info for diagnostics
    GLint glMajor = 0, glMinor = 0;
    gl.GetIntegerv(0x821B, &glMajor);  // GL_MAJOR_VERSION
    gl.GetIntegerv(0x821C, &glMinor);  // GL_MINOR_VERSION
    Log(L"[GL Interop] GL context: %d.%d\n", glMajor, glMinor);

    // 1. Import both Vulkan shared images as GL memory objects + textures
    int numSlots = (sharedImage_[1] != VK_NULL_HANDLE) ? 2 : 1;
    for (int i = 0; i < numSlots; i++) {
        gl.CreateMemoryObjectsEXT(1, &glMemoryObject_[i]);
        if (!glMemoryObject_[i]) {
            Log(L"[GL Interop] Failed to create GL memory object [%d]\n", i);
            ShutdownGL();
            return false;
        }

        // Duplicate NT handle (GL spec allows driver to consume it)
        HANDLE dupHandle = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), ntHandle_[i], GetCurrentProcess(),
                             &dupHandle, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
            Log(L"[GL Interop] Failed to duplicate NT handle [%d]\n", i);
            ShutdownGL();
            return false;
        }

        while (gl.GetError()) {}

        gl.ImportMemoryWin32HandleEXT(glMemoryObject_[i], (GLuint64)allocationSize_[i],
                                      GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, dupHandle);

        GLenum err = gl.GetError();
        if (err) {
            Log(L"[GL Interop] glImportMemoryWin32HandleEXT failed [%d] (GL error 0x%X)\n", i, err);
            CloseHandle(dupHandle);
            ShutdownGL();
            return false;
        }

        // Create GL texture backed by imported memory
        gl.GenTextures(1, &glTexture_[i]);
        gl.BindTexture(GL_TEXTURE_2D, glTexture_[i]);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        gl.TexStorageMem2DEXT(GL_TEXTURE_2D, 1, GL_RGBA8, width_, height_, glMemoryObject_[i], 0);

        err = gl.GetError();
        if (err) {
            Log(L"[GL Interop] glTexStorageMem2DEXT failed [%d] (GL error 0x%X)\n", i, err);
            ShutdownGL();
            return false;
        }
        gl.BindTexture(GL_TEXTURE_2D, 0);

        Log(L"[GL Interop] Slot %d: GL tex=%u, mem=%u, allocSize=%llu\n",
            i, glTexture_[i], glMemoryObject_[i], (unsigned long long)allocationSize_[i]);
    }

    // 2. Create shader program (shared by both textures)
    GLuint vs = CompileGLShader(GL_VERTEX_SHADER, kGLVertexShaderSrc);
    GLuint fs = CompileGLShader(GL_FRAGMENT_SHADER, kGLFragmentShaderSrc);
    if (!vs || !fs) {
        if (vs) gl.DeleteShader(vs);
        if (fs) gl.DeleteShader(fs);
        ShutdownGL();
        return false;
    }

    glShaderProgram_ = gl.CreateProgram();
    gl.AttachShader(glShaderProgram_, vs);
    gl.AttachShader(glShaderProgram_, fs);
    gl.LinkProgram(glShaderProgram_);
    gl.DeleteShader(vs);
    gl.DeleteShader(fs);

    GLint linkStatus;
    gl.GetProgramiv(glShaderProgram_, GL_LINK_STATUS, &linkStatus);
    if (!linkStatus) {
        char infoLog[512];
        if (gl.GetProgramInfoLog)
            gl.GetProgramInfoLog(glShaderProgram_, 512, nullptr, infoLog);
        Log(L"[GL Interop] Shader program link failed: %S\n", infoLog);
        ShutdownGL();
        return false;
    }

    gl.UseProgram(glShaderProgram_);
    GLint texLoc = gl.GetUniformLocation(glShaderProgram_, "uTexture");
    gl.Uniform1i(texLoc, 0);
    gl.UseProgram(0);

    // 3. Create empty VAO (required by core profile for gl_VertexID draws)
    gl.GenVertexArrays(1, &glVAO_);

    glInteropReady_ = true;
    Log(L"[GL Interop] Double-buffered interop initialized: %ux%u (%d slots)\n",
        width_, height_, numSlots);
    return true;
}

void Interop::DrawGL(uint32_t viewportW, uint32_t viewportH) {
    if (!glInteropReady_) return;

    // Save minimal GL state
    GLint prevProgram, prevVAO, prevActiveTexture, prevTexture;
    gl.GetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
    gl.GetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
    gl.GetIntegerv(GL_ACTIVE_TEXTURE_, &prevActiveTexture);
    gl.ActiveTexture(GL_TEXTURE0);
    gl.GetIntegerv(GL_TEXTURE_BINDING_2D, &prevTexture);
    GLboolean prevBlend = gl.IsEnabled(GL_BLEND);
    GLboolean prevDepthTest = gl.IsEnabled(GL_DEPTH_TEST);
    GLboolean prevScissorTest = gl.IsEnabled(GL_SCISSOR_TEST);
    GLboolean prevFramebufferSRGB = gl.IsEnabled(GL_FRAMEBUFFER_SRGB);

    // Set state for fullscreen draw
    gl.Disable(GL_BLEND);
    gl.Disable(GL_DEPTH_TEST);
    gl.Disable(GL_SCISSOR_TEST);
    // The interop texture already contains display-ready SDR values. If Blender's
    // viewport context keeps GL_FRAMEBUFFER_SRGB enabled, OpenGL encodes again on
    // framebuffer write and the image looks washed out / overly white.
    gl.Disable(GL_FRAMEBUFFER_SRGB);

    // Draw fullscreen triangle — read from the COMPLETED frame (readIdx)
    uint32_t readIdx = 1 - writeIdx_;
    GLuint readTex = glTexture_[readIdx] ? glTexture_[readIdx] : glTexture_[0];
    gl.UseProgram(glShaderProgram_);
    gl.BindVertexArray(glVAO_);
    gl.ActiveTexture(GL_TEXTURE0);
    gl.BindTexture(GL_TEXTURE_2D, readTex);
    gl.DrawArrays(GL_TRIANGLES, 0, 3);

    // Restore GL state
    gl.UseProgram(prevProgram);
    gl.BindVertexArray(prevVAO);
    gl.ActiveTexture(GL_TEXTURE0);
    gl.BindTexture(GL_TEXTURE_2D, prevTexture);
    gl.ActiveTexture(prevActiveTexture);
    if (prevBlend) gl.Enable(GL_BLEND); else gl.Disable(GL_BLEND);
    if (prevDepthTest) gl.Enable(GL_DEPTH_TEST); else gl.Disable(GL_DEPTH_TEST);
    if (prevScissorTest) gl.Enable(GL_SCISSOR_TEST); else gl.Disable(GL_SCISSOR_TEST);
    if (prevFramebufferSRGB) gl.Enable(GL_FRAMEBUFFER_SRGB); else gl.Disable(GL_FRAMEBUFFER_SRGB);
}

void Interop::ShutdownGL() {
    if (!glInteropReady_ && !glTexture_[0] && !glShaderProgram_ && !glVAO_ && !glMemoryObject_[0])
        return;

    // Check if a GL context is current (safe to call GL functions)
    bool canCallGL = false;
    if (gl.loaded) {
        HMODULE hGL = GetModuleHandleA("opengl32.dll");
        if (hGL) {
            typedef HGLRC (WINAPI* PFN_wglGetCurrentContext)();
            auto wglGetCurrent = (PFN_wglGetCurrentContext)GetProcAddress(hGL, "wglGetCurrentContext");
            canCallGL = wglGetCurrent && wglGetCurrent() != nullptr;
        }
    }

    if (canCallGL) {
        if (glVAO_) gl.DeleteVertexArrays(1, &glVAO_);
        if (glShaderProgram_) gl.DeleteProgram(glShaderProgram_);
        for (int i = 0; i < 2; i++) {
            if (glTexture_[i]) gl.DeleteTextures(1, &glTexture_[i]);
            if (glMemoryObject_[i]) gl.DeleteMemoryObjectsEXT(1, &glMemoryObject_[i]);
        }
    }

    glVAO_ = 0;
    glShaderProgram_ = 0;
    glTexture_[0] = 0; glTexture_[1] = 0;
    glMemoryObject_[0] = 0; glMemoryObject_[1] = 0;
    glInteropReady_ = false;
}

} // namespace vk
} // namespace acpt
