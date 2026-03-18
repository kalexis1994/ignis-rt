#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include "vk_types.h"

namespace acpt {
namespace vk {

class Context;

// Pipeline management - creates graphics pipelines
class Pipeline {
public:
    bool Initialize(Context* context);
    void Shutdown();

    bool CreateGraphicsPipeline(const char* vertShaderPath, const char* fragShaderPath);

    VkPipeline GetPipeline() const { return pipeline_; }
    VkPipelineLayout GetPipelineLayout() const { return pipelineLayout_; }
    VkRenderPass GetRenderPass() const { return renderPass_; }
    VkDescriptorSetLayout GetDescriptorSetLayout() const { return descriptorSetLayout_; }

    // Public shader utilities (for shadow pipeline creation in Rasterizer)
    bool LoadShader(const char* filename, std::vector<char>& code);
    VkShaderModule CreateShaderModule(const std::vector<char>& code);

private:
    bool CreateRenderPass();
    bool CreateDescriptorSetLayout();

    Context* context_ = nullptr;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
};

} // namespace vk
} // namespace acpt
