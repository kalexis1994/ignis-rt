#include "vk_texture_manager.h"
#include "vk_context.h"
#include "../../include/ignis_log.h"
#include "../../include/ignis_texture.h"
#include <cstring>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#include "../../include/stb_image.h"

// DDS header parser (originally in kn5_loader.cpp)
static bool ParseDDSHeader(const std::vector<uint8_t>& data, int& width, int& height, int& mipCount, unsigned int& dxgiFormat)
{
    if (data.size() < 128) return false;
    if (data[0] != 'D' || data[1] != 'D' || data[2] != 'S' || data[3] != ' ') return false;

    const uint8_t* header = data.data() + 4;
    height = *(int*)(header + 8);
    width = *(int*)(header + 12);
    mipCount = *(int*)(header + 24);
    if (mipCount == 0) mipCount = 1;

    const uint8_t* pixelFormat = header + 72;
    uint32_t pfFlags = *(uint32_t*)(pixelFormat + 4);
    uint32_t fourCC = *(uint32_t*)(pixelFormat + 8);

    dxgiFormat = 0;
    if (pfFlags & 0x4) {  // DDPF_FOURCC
        if (fourCC == 0x31545844)       dxgiFormat = 71;  // DXT1 → BC1
        else if (fourCC == 0x33545844)  dxgiFormat = 74;  // DXT3 → BC2
        else if (fourCC == 0x35545844)  dxgiFormat = 77;  // DXT5 → BC3
        else if (fourCC == 0x31495441)  dxgiFormat = 80;  // ATI1 → BC4
        else if (fourCC == 0x32495441)  dxgiFormat = 83;  // ATI2 → BC5
        else if (fourCC == 0x30315844) {  // DX10
            if (data.size() >= 148)
                dxgiFormat = *(uint32_t*)(data.data() + 128);
            else
                dxgiFormat = 28;
        } else {
            dxgiFormat = 28;  // Fallback RGBA8
        }
    } else {
        uint32_t rgbBitCount = *(uint32_t*)(pixelFormat + 12);
        uint32_t rMask = *(uint32_t*)(pixelFormat + 16);
        uint32_t gMask = *(uint32_t*)(pixelFormat + 20);
        uint32_t bMask = *(uint32_t*)(pixelFormat + 24);

        if (pfFlags & 0x20000) {  // DDPF_LUMINANCE
            if (rgbBitCount == 16 && (pfFlags & 0x1))
                dxgiFormat = 0xFFFF1010;  // L8A8
            else if (rgbBitCount == 8)
                dxgiFormat = 0xFFFF0008;  // L8
            else
                dxgiFormat = 28;
        } else if (rgbBitCount == 32) {
            dxgiFormat = (rMask == 0x00FF0000) ? 87 : 28;  // BGRA8 vs RGBA8
        } else if (rgbBitCount == 24) {
            dxgiFormat = 0xFFFF0018;  // RGB24 → expand to RGBA8
        } else if (rgbBitCount == 16) {
            dxgiFormat = 0xFFFF0010;  // R5G6B5 etc → expand
        } else if (rgbBitCount == 8) {
            dxgiFormat = 0xFFFF0008;
        } else {
            dxgiFormat = 28;
        }
    }
    return true;
}

namespace acpt {
namespace vk {

VkFormat TextureManager::DXGIFormatToVulkan(unsigned int dxgiFormat) {
    switch (dxgiFormat) {
        case 71:  return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;   // D3D11 BC1 exposes 1-bit alpha
        case 72:  return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
        case 74:  return VK_FORMAT_BC2_UNORM_BLOCK;
        case 75:  return VK_FORMAT_BC2_SRGB_BLOCK;
        case 77:  return VK_FORMAT_BC3_UNORM_BLOCK;
        case 78:  return VK_FORMAT_BC3_SRGB_BLOCK;
        case 80:  return VK_FORMAT_BC4_UNORM_BLOCK;
        case 81:  return VK_FORMAT_BC4_SNORM_BLOCK;
        case 83:  return VK_FORMAT_BC5_UNORM_BLOCK;
        case 84:  return VK_FORMAT_BC5_SNORM_BLOCK;
        case 95:  return VK_FORMAT_BC6H_UFLOAT_BLOCK;
        case 96:  return VK_FORMAT_BC6H_SFLOAT_BLOCK;
        case 98:  return VK_FORMAT_BC7_UNORM_BLOCK;
        case 99:  return VK_FORMAT_BC7_SRGB_BLOCK;
        case 28:  return VK_FORMAT_R8G8B8A8_UNORM;
        case 29:  return VK_FORMAT_R8G8B8A8_SRGB;
        case 87:  return VK_FORMAT_B8G8R8A8_UNORM;
        case 91:  return VK_FORMAT_B8G8R8A8_SRGB;
        default:
            Log(L"[VK TextureMgr] Unknown DXGI format %u, defaulting to RGBA8\n", dxgiFormat);
            return VK_FORMAT_R8G8B8A8_UNORM;
    }
}

void TextureManager::CalculateMipOffsetAndSize(int width, int height, int mipLevel, VkFormat format,
                                                size_t& outOffset, size_t& outSize, int& outMipWidth, int& outMipHeight) {
    size_t blockSize = 0;
    switch (format) {
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
        case VK_FORMAT_BC4_UNORM_BLOCK:
        case VK_FORMAT_BC4_SNORM_BLOCK:
            blockSize = 8;
            break;
        case VK_FORMAT_BC2_UNORM_BLOCK:
        case VK_FORMAT_BC2_SRGB_BLOCK:
        case VK_FORMAT_BC3_UNORM_BLOCK:
        case VK_FORMAT_BC3_SRGB_BLOCK:
        case VK_FORMAT_BC5_UNORM_BLOCK:
        case VK_FORMAT_BC5_SNORM_BLOCK:
        case VK_FORMAT_BC6H_UFLOAT_BLOCK:
        case VK_FORMAT_BC6H_SFLOAT_BLOCK:
        case VK_FORMAT_BC7_UNORM_BLOCK:
        case VK_FORMAT_BC7_SRGB_BLOCK:
            blockSize = 16;
            break;
        default:
            blockSize = 4; // RGBA8 fallback
            break;
    }

    outOffset = 0;
    int currentWidth = width;
    int currentHeight = height;
    for (int i = 0; i < mipLevel; i++) {
        int blocksWide = (currentWidth + 3) / 4;
        int blocksHigh = (currentHeight + 3) / 4;
        if (blockSize <= 4) {
            // Uncompressed: per-pixel
            outOffset += (size_t)currentWidth * currentHeight * blockSize;
        } else {
            outOffset += (size_t)blocksWide * blocksHigh * blockSize;
        }
        currentWidth = currentWidth > 1 ? currentWidth / 2 : 1;
        currentHeight = currentHeight > 1 ? currentHeight / 2 : 1;
    }

    outMipWidth = currentWidth;
    outMipHeight = currentHeight;
    if (blockSize <= 4) {
        outSize = (size_t)currentWidth * currentHeight * blockSize;
    } else {
        int blocksWide = (currentWidth + 3) / 4;
        int blocksHigh = (currentHeight + 3) / 4;
        outSize = (size_t)blocksWide * blocksHigh * blockSize;
    }
}

bool TextureManager::Init(Context* context) {
    context_ = context;
    VkDevice device = context_->GetDevice();

    // Create shared sampler: linear, repeat, aniso 16x
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler_) != VK_SUCCESS) {
        Log(L"[VK TextureMgr] ERROR: Failed to create sampler\n");
        return false;
    }

    Log(L"[VK TextureMgr] Initialized\n");
    return true;
}

void TextureManager::Shutdown() {
    VkDevice device = context_ ? context_->GetDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) return;

    for (auto& tex : textures_) {
        if (tex.imageView) vkDestroyImageView(device, tex.imageView, nullptr);
        if (tex.image) vkDestroyImage(device, tex.image, nullptr);
        if (tex.memory) vkFreeMemory(device, tex.memory, nullptr);
        if (tex.stagingBuffer) vkDestroyBuffer(device, tex.stagingBuffer, nullptr);
        if (tex.stagingMemory) vkFreeMemory(device, tex.stagingMemory, nullptr);
    }
    textures_.clear();

    if (sampler_) { vkDestroySampler(device, sampler_, nullptr); sampler_ = VK_NULL_HANDLE; }

    Log(L"[VK TextureMgr] Shutdown\n");
}

int TextureManager::AddTexture(const KN5Texture& kn5Tex) {
    if (kn5Tex.data.empty()) {
        Log(L"[VK TextureMgr] Empty texture data for %S\n", kn5Tex.name.c_str());
        return -1;
    }

    // Try DDS first, then fall back to PNG/JPG via stb_image
    int width, height, mipCount;
    unsigned int dxgiFormat;
    bool isDDS = ParseDDSHeader(kn5Tex.data, width, height, mipCount, dxgiFormat);
    if (isDDS) {
        // Log DDS header details for debugging format issues
        const uint8_t* hdrDbg = kn5Tex.data.data();
        uint32_t pfFlags = *(uint32_t*)(hdrDbg + 80);
        uint32_t fccDbg = *(uint32_t*)(hdrDbg + 84);
        char fccStr[5] = { (char)(fccDbg & 0xFF), (char)((fccDbg >> 8) & 0xFF),
                           (char)((fccDbg >> 16) & 0xFF), (char)((fccDbg >> 24) & 0xFF), 0 };
        Log(L"[VK TextureMgr] DDS %S: %dx%d mips=%d dxgi=%u fourCC=%S(0x%08X) pfFlags=0x%X dataSize=%zu\n",
            kn5Tex.name.c_str(), width, height, mipCount, dxgiFormat,
            fccStr, fccDbg, pfFlags, kn5Tex.data.size());
    }
    bool isSTB = false;
    stbi_uc* stbPixels = nullptr;

    if (!isDDS) {
        // Try loading as PNG/JPG with stb_image
        int stbChannels = 0;
        stbPixels = stbi_load_from_memory(kn5Tex.data.data(), (int)kn5Tex.data.size(),
                                           &width, &height, &stbChannels, 4); // force RGBA
        if (stbPixels) {
            isSTB = true;
            mipCount = 1;
            dxgiFormat = 28; // RGBA8
            Log(L"[VK TextureMgr] Loaded %S as PNG/JPG (%dx%d, %d channels)\n",
                kn5Tex.name.c_str(), width, height, stbChannels);
        } else {
            const char* stbErr = stbi_failure_reason();
            Log(L"[VK TextureMgr] Failed to load %S: %S — inserting 1x1 flat dummy\n",
                kn5Tex.name.c_str(), stbErr ? stbErr : "unknown error");
            // Insert a 1x1 dummy texture to keep indices aligned.
            // Use flat normal blue (128,128,255) so failed normal maps don't distort shading.
            width = 1; height = 1; mipCount = 1; dxgiFormat = 28; // RGBA8
            stbPixels = (stbi_uc*)STBI_MALLOC(4);
            stbPixels[0] = 128; stbPixels[1] = 128; stbPixels[2] = 255; stbPixels[3] = 255; // flat normal blue
            isSTB = true;
        }
    }

    // Check for non-standard DDS formats that need conversion to RGBA8
    bool needsConversion = false;
    bool isLuminanceAlpha = false;
    unsigned int srcBitsPerPixel = 0;
    if (isDDS && (dxgiFormat & 0xFFFF0000) == 0xFFFF0000) {
        needsConversion = true;
        srcBitsPerPixel = dxgiFormat & 0xFF;
        isLuminanceAlpha = (dxgiFormat == 0xFFFF1010);
        dxgiFormat = 28;  // Will create image as RGBA8
    }

    VkFormat vkFormat = DXGIFormatToVulkan(dxgiFormat);
    VkDevice device = context_->GetDevice();

    // For uncompressed textures with only 1 mip, calculate full mip chain for GPU generation
    bool isBC = (vkFormat >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && vkFormat <= VK_FORMAT_BC7_SRGB_BLOCK);
    bool genMips = !isBC && (mipCount == 1) && (width > 1 || height > 1);
    if (genMips) {
        mipCount = (int)floor(log2((double)std::max(width, height))) + 1;
    }

    // For BC-compressed DDS: validate mipCount against available pixel data
    // If the DDS header claims more mips than the data contains, cap the mip count
    if (isDDS && isBC && !genMips && mipCount > 1) {
        const size_t ddsHdrSize = 128;
        bool dx10 = false;
        if (kn5Tex.data.size() >= 128) {
            uint32_t fcc = *(uint32_t*)(kn5Tex.data.data() + 84);
            if (fcc == 0x30315844) dx10 = true;
        }
        size_t hdrSize = ddsHdrSize + (dx10 ? 20 : 0);
        size_t availPixels = kn5Tex.data.size() > hdrSize ? kn5Tex.data.size() - hdrSize : 0;

        // Count how many mips actually fit in the available data
        size_t totalSize = 0;
        int validMips = 0;
        int mw = width, mh = height;
        for (int m = 0; m < mipCount; m++) {
            size_t off, sz;
            int mipW, mipH;
            CalculateMipOffsetAndSize(width, height, m, vkFormat, off, sz, mipW, mipH);
            if (off + sz > availPixels) break;
            validMips = m + 1;
            mw = mw > 1 ? mw / 2 : 1;
            mh = mh > 1 ? mh / 2 : 1;
        }

        if (validMips < mipCount) {
            Log(L"[VK TextureMgr] %S: DDS header claims %d mips but data only has %d, capping\n",
                kn5Tex.name.c_str(), mipCount, validMips);
            mipCount = validMips > 0 ? validMips : 1;
        }
    }

    GPUTexture tex{};
    tex.width = width;
    tex.height = height;
    tex.mipLevels = mipCount;
    tex.format = vkFormat;
    tex.generateMips = genMips;

    // Create VkImage
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = vkFormat;
    imageInfo.extent = { (uint32_t)width, (uint32_t)height, 1 };
    imageInfo.mipLevels = mipCount;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        | (genMips ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &imageInfo, nullptr, &tex.image) != VK_SUCCESS) {
        Log(L"[VK TextureMgr] Failed to create image for %S\n", kn5Tex.name.c_str());
        return -1;
    }

    // Allocate device-local memory for image
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, tex.image, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = context_->FindMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &tex.memory) != VK_SUCCESS) {
        Log(L"[VK TextureMgr] Failed to allocate image memory for %S\n", kn5Tex.name.c_str());
        vkDestroyImage(device, tex.image, nullptr);
        return -1;
    }
    vkBindImageMemory(device, tex.image, tex.memory, 0);

    // Determine source pixel data and staging buffer size
    VkDeviceSize srcDataSize = 0;
    const uint8_t* srcPixels = nullptr;

    if (isSTB) {
        // stb_image: already decoded to RGBA8
        srcDataSize = (VkDeviceSize)width * height * 4;
        srcPixels = stbPixels;
    } else {
        // DDS: skip header to get to pixel data
        const size_t ddsHeaderSize = 128;
        bool hasDX10Header = false;
        if (kn5Tex.data.size() >= 128) {
            const uint8_t* hdr = kn5Tex.data.data();
            uint32_t fourCC = *(uint32_t*)(hdr + 84);
            if (fourCC == 0x30315844) hasDX10Header = true;
        }
        const size_t headerSize = ddsHeaderSize + (hasDX10Header ? 20 : 0);

        if (kn5Tex.data.size() <= headerSize) {
            Log(L"[VK TextureMgr] DDS data too small for %S\n", kn5Tex.name.c_str());
            vkDestroyImage(device, tex.image, nullptr);
            vkFreeMemory(device, tex.memory, nullptr);
            return -1;
        }
        srcDataSize = kn5Tex.data.size() - headerSize;
        srcPixels = kn5Tex.data.data() + headerSize;
    }

    // Calculate staging buffer size (may differ from source if converting DDS formats)
    VkDeviceSize dataSize = srcDataSize;
    if (needsConversion) {
        dataSize = 0;
        int mw = width, mh = height;
        for (int m = 0; m < mipCount; m++) {
            dataSize += (VkDeviceSize)mw * mh * 4;
            mw = mw > 1 ? mw / 2 : 1;
            mh = mh > 1 ? mh / 2 : 1;
        }
    }

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = dataSize;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufInfo, nullptr, &tex.stagingBuffer) != VK_SUCCESS) {
        vkDestroyImage(device, tex.image, nullptr);
        vkFreeMemory(device, tex.memory, nullptr);
        return -1;
    }

    VkMemoryRequirements stagingReqs;
    vkGetBufferMemoryRequirements(device, tex.stagingBuffer, &stagingReqs);

    VkMemoryAllocateInfo stagingAlloc{};
    stagingAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingAlloc.allocationSize = stagingReqs.size;
    stagingAlloc.memoryTypeIndex = context_->FindMemoryType(stagingReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &stagingAlloc, nullptr, &tex.stagingMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device, tex.stagingBuffer, nullptr);
        vkDestroyImage(device, tex.image, nullptr);
        vkFreeMemory(device, tex.memory, nullptr);
        return -1;
    }
    vkBindBufferMemory(device, tex.stagingBuffer, tex.stagingMemory, 0);

    tex.stagingSize = dataSize;

    // Copy DDS pixel data to staging buffer (with optional format conversion)
    void* mapped;
    vkMapMemory(device, tex.stagingMemory, 0, dataSize, 0, &mapped);

    if (needsConversion) {
        // Convert source pixels to RGBA8
        uint8_t* dst = (uint8_t*)mapped;
        const uint8_t* src = srcPixels;
        int srcBpp = srcBitsPerPixel / 8;
        int mw = width, mh = height;
        for (int m = 0; m < mipCount; m++) {
            int pixelCount = mw * mh;
            // Check source has enough data for this mip
            size_t srcMipSize = (size_t)pixelCount * srcBpp;
            if ((size_t)(src - srcPixels) + srcMipSize > (size_t)srcDataSize) break;

            for (int p = 0; p < pixelCount; p++) {
                if (srcBpp == 3) {
                    // BGR24 or RGB24 → RGBA8
                    dst[0] = src[2]; dst[1] = src[1]; dst[2] = src[0]; dst[3] = 255;
                } else if (srcBpp == 2 && isLuminanceAlpha) {
                    // L8A8 → RGBA8
                    dst[0] = dst[1] = dst[2] = src[0];
                    dst[3] = src[1];
                } else if (srcBpp == 2) {
                    // R5G6B5 → RGBA8
                    uint16_t pixel = *(const uint16_t*)src;
                    dst[0] = (uint8_t)(((pixel >> 11) & 0x1F) * 255 / 31);
                    dst[1] = (uint8_t)(((pixel >> 5) & 0x3F) * 255 / 63);
                    dst[2] = (uint8_t)((pixel & 0x1F) * 255 / 31);
                    dst[3] = 255;
                } else if (srcBpp == 1) {
                    // L8 → RGBA8
                    dst[0] = dst[1] = dst[2] = src[0]; dst[3] = 255;
                } else {
                    dst[0] = dst[1] = dst[2] = 128; dst[3] = 255;
                }
                src += srcBpp;
                dst += 4;
            }
            mw = mw > 1 ? mw / 2 : 1;
            mh = mh > 1 ? mh / 2 : 1;
        }
    } else {
        memcpy(mapped, srcPixels, srcDataSize);
    }
    vkUnmapMemory(device, tex.stagingMemory);

    // Free stb_image pixels now that data is in the staging buffer
    if (stbPixels) {
        stbi_image_free(stbPixels);
        stbPixels = nullptr;
    }

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = tex.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = vkFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipCount;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &tex.imageView) != VK_SUCCESS) {
        vkDestroyBuffer(device, tex.stagingBuffer, nullptr);
        vkFreeMemory(device, tex.stagingMemory, nullptr);
        vkDestroyImage(device, tex.image, nullptr);
        vkFreeMemory(device, tex.memory, nullptr);
        return -1;
    }

    tex.uploaded = false;
    int index = (int)textures_.size();
    textures_.push_back(tex);
    return index;
}

bool TextureManager::UploadAll() {
    if (textures_.empty()) return true;

    int toUpload = 0;
    for (const auto& tex : textures_) {
        if (!tex.uploaded && tex.stagingBuffer != VK_NULL_HANDLE) toUpload++;
    }
    if (toUpload == 0) return true;

    Log(L"[VK TextureMgr] Uploading %d textures to GPU...\n", toUpload);

    VkDevice device = context_->GetDevice();
    const int BATCH_SIZE = 8;
    int totalUploaded = 0;

    while (totalUploaded < toUpload) {
        VkCommandBuffer cmd = context_->BeginSingleTimeCommands();

        int batchCount = 0;
        for (auto& tex : textures_) {
            if (tex.uploaded || tex.stagingBuffer == VK_NULL_HANDLE) continue;
            if (batchCount >= BATCH_SIZE) break;

            // Transition UNDEFINED -> TRANSFER_DST_OPTIMAL
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = tex.image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = tex.mipLevels;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            // Copy mip levels from staging buffer to image (with bounds validation)
            // For generateMips textures, only copy mip 0 (rest will be generated via blit)
            std::vector<VkBufferImageCopy> copyRegions;
            int mipsToUpload = tex.generateMips ? 1 : tex.mipLevels;
            int safeMipLevels = mipsToUpload;
            for (int mip = 0; mip < mipsToUpload; mip++) {
                size_t offset, size;
                int mipWidth, mipHeight;
                CalculateMipOffsetAndSize(tex.width, tex.height, mip, tex.format, offset, size, mipWidth, mipHeight);

                // Bounds check: ensure copy doesn't exceed staging buffer
                if (offset + size > (size_t)tex.stagingSize) {
                    Log(L"[VK TextureMgr] WARNING: Tex %dx%d fmt=%d mip %d/%d: offset+size=%zu > staging=%zu, capping mips\n",
                        tex.width, tex.height, (int)tex.format, mip, tex.mipLevels,
                        offset + size, (size_t)tex.stagingSize);
                    safeMipLevels = mip;
                    break;
                }

                VkBufferImageCopy region{};
                region.bufferOffset = offset;
                region.bufferRowLength = 0;
                region.bufferImageHeight = 0;
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = mip;
                region.imageSubresource.baseArrayLayer = 0;
                region.imageSubresource.layerCount = 1;
                region.imageOffset = {0, 0, 0};
                region.imageExtent = { (uint32_t)mipWidth, (uint32_t)mipHeight, 1 };

                copyRegions.push_back(region);
            }

            if (copyRegions.empty()) {
                Log(L"[VK TextureMgr] WARNING: Skipping texture %dx%d (no valid mip levels)\n", tex.width, tex.height);
                tex.uploaded = true;
                batchCount++;
                totalUploaded++;
                continue;
            }

            vkCmdCopyBufferToImage(cmd, tex.stagingBuffer, tex.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                (uint32_t)copyRegions.size(), copyRegions.data());

            if (tex.generateMips && tex.mipLevels > 1) {
                // Generate mipmaps via vkCmdBlitImage
                // Transition mip 0 to TRANSFER_SRC for reading
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.baseMipLevel = 0;
                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &barrier);

                int mipW = tex.width, mipH = tex.height;
                for (int mip = 1; mip < tex.mipLevels; mip++) {
                    int nextW = mipW > 1 ? mipW / 2 : 1;
                    int nextH = mipH > 1 ? mipH / 2 : 1;

                    // Blit from mip-1 (SRC) to mip (DST)
                    VkImageBlit blit{};
                    blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)(mip - 1), 0, 1 };
                    blit.srcOffsets[0] = { 0, 0, 0 };
                    blit.srcOffsets[1] = { mipW, mipH, 1 };
                    blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)mip, 0, 1 };
                    blit.dstOffsets[0] = { 0, 0, 0 };
                    blit.dstOffsets[1] = { nextW, nextH, 1 };

                    vkCmdBlitImage(cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit, VK_FILTER_LINEAR);

                    // Transition this mip to SRC for the next iteration
                    barrier.subresourceRange.baseMipLevel = mip;
                    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, nullptr, 0, nullptr, 1, &barrier);

                    mipW = nextW;
                    mipH = nextH;
                }

                // Transition ALL mips from TRANSFER_SRC to SHADER_READ_ONLY
                barrier.subresourceRange.baseMipLevel = 0;
                barrier.subresourceRange.levelCount = tex.mipLevels;
                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                    0, 0, nullptr, 0, nullptr, 1, &barrier);
            } else {
                // No mip generation — just transition to shader read
                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                    0, 0, nullptr, 0, nullptr, 1, &barrier);
            }

            tex.uploaded = true;
            batchCount++;
            totalUploaded++;
        }

        context_->EndSingleTimeCommands(cmd);

        // Free staging buffers after upload
        for (auto& tex : textures_) {
            if (tex.uploaded && tex.stagingBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, tex.stagingBuffer, nullptr);
                vkFreeMemory(device, tex.stagingMemory, nullptr);
                tex.stagingBuffer = VK_NULL_HANDLE;
                tex.stagingMemory = VK_NULL_HANDLE;
            }
        }

        Log(L"[VK TextureMgr] Batch uploaded: %d/%d textures\n", totalUploaded, toUpload);
    }

    Log(L"[VK TextureMgr] All %d textures uploaded to GPU\n", toUpload);
    return true;
}

VkImageView TextureManager::GetImageView(int index) const {
    if (index < 0 || index >= (int)textures_.size()) return VK_NULL_HANDLE;
    return textures_[index].imageView;
}

} // namespace vk
} // namespace acpt
