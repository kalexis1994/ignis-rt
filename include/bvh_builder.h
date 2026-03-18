// ============================================================
// IGNIS RT - BVH BUILDER
// ============================================================
// CPU-side BVH construction for D3D11 compute shader ray tracing
// Uses Surface Area Heuristic (SAH) for optimal tree quality

#pragma once

#include <vector>
#include <cstdint>
#include <d3d11.h>
#include "ignis_texture.h"

namespace acpt {

// ============================================================
// BVH NODE STRUCTURE (GPU-compatible, 64 bytes aligned)
// ============================================================
// Layout optimized for cache-efficient traversal on GPU
// Each node is self-contained with AABB and child/primitive info

struct alignas(16) BVHNode {
    float aabbMin[3];           // 12 bytes - Minimum bounds of AABB
    uint32_t leftChild;         // 4 bytes - Left child index (0xFFFFFFFF = leaf node)

    float aabbMax[3];           // 12 bytes - Maximum bounds of AABB
    uint32_t rightChild;        // 4 bytes - Right child index

    uint32_t primitiveOffset;   // 4 bytes - First primitive index for leaf nodes
    uint32_t primitiveCount;    // 4 bytes - Number of primitives in leaf (0 for internal)
    uint32_t padding[2];        // 8 bytes - Alignment padding
    // Total: 48 bytes

    bool IsLeaf() const { return leftChild == 0xFFFFFFFF; }
};
static_assert(sizeof(BVHNode) == 48, "BVHNode must be 48 bytes to match HLSL struct");

// ============================================================
// BVH PRIMITIVE STRUCTURE (GPU-compatible, 128 bytes)
// ============================================================
// Pre-computed triangle data for efficient Moller-Trumbore intersection
// Includes vertex positions, edges, normals, UVs, and material info

struct alignas(16) BVHPrimitive {
    // Triangle vertex and edges (Moller-Trumbore format)
    float v0[3];                // 12 bytes - First vertex position
    float pad0;                 // 4 bytes
    float e1[3];                // 12 bytes - Edge 1: v1 - v0
    float pad1;                 // 4 bytes
    float e2[3];                // 12 bytes - Edge 2: v2 - v0
    float pad2;                 // 4 bytes

    // Shading normals (per-vertex, interpolated via barycentric)
    float n0[3];                // 12 bytes - Normal at v0
    float pad3;                 // 4 bytes
    float n1[3];                // 12 bytes - Normal at v1
    float pad4;                 // 4 bytes
    float n2[3];                // 12 bytes - Normal at v2
    float pad5;                 // 4 bytes

    // Texture coordinates + material info (packed)
    float uv0[2];               // 8 bytes - UV at v0
    float uv1[2];               // 8 bytes - UV at v1
    float uv2[2];               // 8 bytes - UV at v2
    uint32_t materialId;        // 4 bytes - Index into material buffer
    uint32_t geometryId;        // 4 bytes - Original mesh/geometry index
    // Total: 6*16 + 32 = 128 bytes
};
static_assert(sizeof(BVHPrimitive) == 128, "BVHPrimitive must be 128 bytes for GPU alignment");

// ============================================================
// BUILD CONFIGURATION
// ============================================================

// Filter mode for BuildFromCapturedGeometry
enum class BVHGeometryFilter {
    All,            // Include all draw calls (default, legacy behavior)
    StaticOnly,     // Only identity world matrix draw calls (track/scenery)
    DynamicOnly     // Only non-identity world matrix draw calls (car), stored in object space
};

struct BVHBuildConfig {
    uint32_t maxPrimitivesPerLeaf = 4;      // Max triangles in a leaf node
    uint32_t maxDepth = 32;                  // Maximum tree depth
    float traversalCost = 1.0f;              // Cost of traversing a node (SAH)
    float intersectionCost = 1.0f;           // Cost of ray-triangle test (SAH)
    bool useMultiThreading = true;           // Enable parallel build
    BVHGeometryFilter geometryFilter = BVHGeometryFilter::All;  // Filter for captured geometry
    bool verbose = true;                     // Enable detailed logging (disable for per-frame rebuilds)

    // If non-empty, only build from these draw call indices (overrides geometryFilter)
    const std::vector<uint32_t>* drawCallIndices = nullptr;
};

// ============================================================
// BVH BUILDER CLASS
// ============================================================

class BVHBuilder {
public:
    BVHBuilder();
    ~BVHBuilder();

    // ========== BUILD INTERFACE ==========

    // Build BVH from loaded KN5 model
    // Extracts mesh nodes recursively and builds acceleration structure
    // Returns true on success
    bool Build(const KN5Model& model,
               const BVHBuildConfig& config = BVHBuildConfig());

    // Build simple test scene (ground plane + boxes)
    bool BuildTestScene(const BVHBuildConfig& config = BVHBuildConfig());

    // Build BVH from captured AC geometry (hybrid rendering)
    // Uses GeometryCapture's draw calls to extract triangles
    bool BuildFromCapturedGeometry(class GeometryCapture* capture,
                                   const BVHBuildConfig& config = BVHBuildConfig());

    // Clear all data
    void Clear();

    // ========== GPU UPLOAD ==========

    // Upload BVH data to GPU buffers
    // Creates StructuredBuffers with SRVs for shader access
    bool UploadToGPU(ID3D11Device* device);

    // Release GPU resources
    void ReleaseGPU();

    // ========== ACCESSORS ==========

    // Get GPU buffers (valid after UploadToGPU)
    ID3D11Buffer* GetNodeBuffer() const { return m_nodeBuffer; }
    ID3D11Buffer* GetPrimitiveBuffer() const { return m_primitiveBuffer; }
    ID3D11ShaderResourceView* GetNodeSRV() const { return m_nodeSRV; }
    ID3D11ShaderResourceView* GetPrimitiveSRV() const { return m_primitiveSRV; }

    // Get CPU-side data (for debugging/visualization)
    const std::vector<BVHNode>& GetNodes() const { return m_nodes; }
    const std::vector<BVHPrimitive>& GetPrimitives() const { return m_primitives; }

    // Statistics
    uint32_t GetNodeCount() const { return static_cast<uint32_t>(m_nodes.size()); }
    uint32_t GetPrimitiveCount() const { return static_cast<uint32_t>(m_primitives.size()); }
    uint32_t GetTreeDepth() const { return m_treeDepth; }
    uint32_t GetLeafCount() const { return m_leafCount; }
    float GetBuildTimeMs() const { return m_buildTimeMs; }

private:
    // ========== BUILD HELPERS ==========

    // Compute AABB for a range of primitives
    void ComputeAABB(uint32_t start, uint32_t count,
                     float outMin[3], float outMax[3]) const;

    // Compute centroid AABB for SAH binning
    void ComputeCentroidAABB(uint32_t start, uint32_t count,
                              float outMin[3], float outMax[3]) const;

    // Recursive build with SAH
    uint32_t BuildRecursive(uint32_t start, uint32_t count, uint32_t depth);

    // Evaluate SAH cost for a split
    float EvaluateSAH(uint32_t start, uint32_t count,
                      int axis, float splitPos,
                      uint32_t& outLeftCount) const;

    // Surface area of AABB
    static float SurfaceArea(const float min[3], const float max[3]);

    // Partition primitives around split position
    uint32_t Partition(uint32_t start, uint32_t count, int axis, float splitPos);

    // Create leaf node
    uint32_t CreateLeaf(uint32_t start, uint32_t count);

    // ========== DATA ==========

    std::vector<BVHNode> m_nodes;
    std::vector<BVHPrimitive> m_primitives;
    std::vector<uint32_t> m_primitiveIndices;  // Sorted indices during build

    // Build configuration
    BVHBuildConfig m_config;

    // Statistics
    uint32_t m_treeDepth;
    uint32_t m_leafCount;
    float m_buildTimeMs;

    // GPU resources
    ID3D11Buffer* m_nodeBuffer;
    ID3D11Buffer* m_primitiveBuffer;
    ID3D11ShaderResourceView* m_nodeSRV;
    ID3D11ShaderResourceView* m_primitiveSRV;
};

} // namespace acpt
