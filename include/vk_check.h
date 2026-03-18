#pragma once

#include <vulkan/vulkan.h>
#include "ignis_log.h"

// VK_CHECK: log and return false on Vulkan failure
// Usage: if (!VK_CHECK(vkCreateImage(...))) return false;
#define VK_CHECK(expr)                                                          \
    [&]() -> bool {                                                             \
        VkResult _vk_result = (expr);                                           \
        if (_vk_result != VK_SUCCESS) {                                         \
            Log(L"[VK_CHECK] %S failed: VkResult=%d (%S:%d)\n",                \
                #expr, (int)_vk_result, __FILE__, __LINE__);                    \
            return false;                                                       \
        }                                                                       \
        return true;                                                            \
    }()

// VK_CHECK_VOID: log and return (no value) on failure
#define VK_CHECK_VOID(expr)                                                     \
    do {                                                                         \
        VkResult _vk_result = (expr);                                           \
        if (_vk_result != VK_SUCCESS) {                                         \
            Log(L"[VK_CHECK] %S failed: VkResult=%d (%S:%d)\n",                \
                #expr, (int)_vk_result, __FILE__, __LINE__);                    \
            return;                                                             \
        }                                                                       \
    } while (0)

// VK_WARN: log but continue on failure (non-fatal)
#define VK_WARN(expr)                                                           \
    do {                                                                         \
        VkResult _vk_result = (expr);                                           \
        if (_vk_result != VK_SUCCESS) {                                         \
            Log(L"[VK_WARN] %S: VkResult=%d (%S:%d)\n",                        \
                #expr, (int)_vk_result, __FILE__, __LINE__);                    \
        }                                                                       \
    } while (0)
