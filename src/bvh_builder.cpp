// ============================================================
// IGNIS RT - BVH BUILDER IMPLEMENTATION
// ============================================================
// CPU-side BVH construction using Surface Area Heuristic (SAH)
// Reference: "On fast Construction of SAH-based Bounding Volume Hierarchies" (Wald 2007)

// Prevent Windows min/max macros from conflicting with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "bvh_builder.h"
#include "geometry_capture.h"
#include <algorithm>
#include <chrono>
#include <set>
#include <cmath>
#include <limits>
#include <thread>

namespace acpt {

// ============================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================

BVHBuilder::BVHBuilder()
    : m_treeDepth(0)
    , m_leafCount(0)
    , m_buildTimeMs(0.0f)
    , m_nodeBuffer(nullptr)
    , m_primitiveBuffer(nullptr)
    , m_nodeSRV(nullptr)
    , m_primitiveSRV(nullptr)
{
}

BVHBuilder::~BVHBuilder()
{
    ReleaseGPU();
}

// ============================================================
// BUILD INTERFACE
// ============================================================

// Helper to recursively extract triangles from node tree
static void ExtractTrianglesFromNode(const KN5Node& node, uint32_t& meshIdx, uint32_t& primIndex,
                                      std::vector<BVHPrimitive>& primitives,
                                      std::vector<uint32_t>& primitiveIndices)
{
    // Process this node if it's a mesh node with geometry
    if (node.nodeType == 2 && node.isRenderable && node.indexCount > 0) {
        uint32_t triangleCount = node.indexCount / 3;

        for (uint32_t tri = 0; tri < triangleCount; tri++) {
            BVHPrimitive prim = {};

            // Get triangle indices
            uint32_t i0 = node.indices[tri * 3 + 0];
            uint32_t i1 = node.indices[tri * 3 + 1];
            uint32_t i2 = node.indices[tri * 3 + 2];

            // Bounds check
            if (i0 >= static_cast<uint32_t>(node.vertexCount) ||
                i1 >= static_cast<uint32_t>(node.vertexCount) ||
                i2 >= static_cast<uint32_t>(node.vertexCount)) {
                continue;
            }

            // Vertex positions
            float v0[3] = {
                node.positions[i0 * 3 + 0],
                node.positions[i0 * 3 + 1],
                node.positions[i0 * 3 + 2]
            };
            float v1[3] = {
                node.positions[i1 * 3 + 0],
                node.positions[i1 * 3 + 1],
                node.positions[i1 * 3 + 2]
            };
            float v2[3] = {
                node.positions[i2 * 3 + 0],
                node.positions[i2 * 3 + 1],
                node.positions[i2 * 3 + 2]
            };

            // Store v0 and edges (Moller-Trumbore format)
            prim.v0[0] = v0[0]; prim.v0[1] = v0[1]; prim.v0[2] = v0[2];
            prim.e1[0] = v1[0] - v0[0]; prim.e1[1] = v1[1] - v0[1]; prim.e1[2] = v1[2] - v0[2];
            prim.e2[0] = v2[0] - v0[0]; prim.e2[1] = v2[1] - v0[1]; prim.e2[2] = v2[2] - v0[2];

            // Normals (if available)
            if (!node.normals.empty() && node.normals.size() >= static_cast<size_t>((i2 + 1) * 3)) {
                prim.n0[0] = node.normals[i0 * 3 + 0];
                prim.n0[1] = node.normals[i0 * 3 + 1];
                prim.n0[2] = node.normals[i0 * 3 + 2];

                prim.n1[0] = node.normals[i1 * 3 + 0];
                prim.n1[1] = node.normals[i1 * 3 + 1];
                prim.n1[2] = node.normals[i1 * 3 + 2];

                prim.n2[0] = node.normals[i2 * 3 + 0];
                prim.n2[1] = node.normals[i2 * 3 + 1];
                prim.n2[2] = node.normals[i2 * 3 + 2];
            } else {
                // Compute geometric normal
                float nx = prim.e1[1] * prim.e2[2] - prim.e1[2] * prim.e2[1];
                float ny = prim.e1[2] * prim.e2[0] - prim.e1[0] * prim.e2[2];
                float nz = prim.e1[0] * prim.e2[1] - prim.e1[1] * prim.e2[0];
                float len = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (len > 0.0001f) {
                    nx /= len; ny /= len; nz /= len;
                }
                prim.n0[0] = prim.n1[0] = prim.n2[0] = nx;
                prim.n0[1] = prim.n1[1] = prim.n2[1] = ny;
                prim.n0[2] = prim.n1[2] = prim.n2[2] = nz;
            }

            // UVs (if available)
            if (!node.uvs.empty() && node.uvs.size() >= static_cast<size_t>((i2 + 1) * 2)) {
                prim.uv0[0] = node.uvs[i0 * 2 + 0];
                prim.uv0[1] = node.uvs[i0 * 2 + 1];

                prim.uv1[0] = node.uvs[i1 * 2 + 0];
                prim.uv1[1] = node.uvs[i1 * 2 + 1];

                prim.uv2[0] = node.uvs[i2 * 2 + 0];
                prim.uv2[1] = node.uvs[i2 * 2 + 1];
            } else {
                prim.uv0[0] = prim.uv0[1] = 0.0f;
                prim.uv1[0] = prim.uv1[1] = 0.0f;
                prim.uv2[0] = prim.uv2[1] = 0.0f;
            }

            // Material and geometry IDs
            prim.materialId = (node.materialId >= 0) ? static_cast<uint32_t>(node.materialId) : 0;
            prim.geometryId = meshIdx;

            primitives.push_back(prim);
            primitiveIndices.push_back(primIndex++);
        }
        meshIdx++;
    }

    // Recursively process children
    for (const auto& child : node.children) {
        ExtractTrianglesFromNode(child, meshIdx, primIndex, primitives, primitiveIndices);
    }
}

bool BVHBuilder::Build(const KN5Model& model, const BVHBuildConfig& config)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    Clear();
    m_config = config;

    // Count total triangles first (for reservation)
    uint32_t totalTriangles = static_cast<uint32_t>(GetTotalTriangleCount(model));
    if (totalTriangles == 0) {
        return false;
    }

    // Reserve space
    m_primitives.reserve(totalTriangles);
    m_primitiveIndices.reserve(totalTriangles);
    m_nodes.reserve(totalTriangles * 2);  // Rough upper bound

    // Extract triangles from all mesh nodes
    uint32_t meshIdx = 0;
    uint32_t primIndex = 0;
    for (const auto& rootNode : model.rootNodes) {
        ExtractTrianglesFromNode(rootNode, meshIdx, primIndex, m_primitives, m_primitiveIndices);
    }

    if (m_primitives.empty()) {
        return false;
    }

    // Build BVH recursively using SAH
    m_treeDepth = 0;
    m_leafCount = 0;
    BuildRecursive(0, static_cast<uint32_t>(m_primitives.size()), 0);

    auto endTime = std::chrono::high_resolution_clock::now();
    m_buildTimeMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();

    return true;
}

bool BVHBuilder::BuildTestScene(const BVHBuildConfig& config)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    Clear();
    m_config = config;

    // Create a simple test scene: ground plane + a box
    m_primitives.reserve(14);  // 2 triangles for ground + 12 for box
    m_primitiveIndices.reserve(14);

    uint32_t primIdx = 0;

    // Ground plane (large, at y=0)
    float groundSize = 50.0f;
    {
        BVHPrimitive tri1 = {};
        tri1.v0[0] = -groundSize; tri1.v0[1] = 0.0f; tri1.v0[2] = -groundSize;
        tri1.e1[0] = 2*groundSize; tri1.e1[1] = 0.0f; tri1.e1[2] = 0.0f;  // v1 - v0
        tri1.e2[0] = 0.0f; tri1.e2[1] = 0.0f; tri1.e2[2] = 2*groundSize;  // v2 - v0
        tri1.n0[0] = 0; tri1.n0[1] = 1; tri1.n0[2] = 0;
        tri1.n1[0] = 0; tri1.n1[1] = 1; tri1.n1[2] = 0;
        tri1.n2[0] = 0; tri1.n2[1] = 1; tri1.n2[2] = 0;
        tri1.uv0[0] = 0; tri1.uv0[1] = 0;
        tri1.uv1[0] = 1; tri1.uv1[1] = 0;
        tri1.uv2[0] = 0; tri1.uv2[1] = 1;
        tri1.materialId = 0;
        tri1.geometryId = 0;
        m_primitives.push_back(tri1);
        m_primitiveIndices.push_back(primIdx++);

        BVHPrimitive tri2 = {};
        tri2.v0[0] = groundSize; tri2.v0[1] = 0.0f; tri2.v0[2] = groundSize;
        tri2.e1[0] = -2*groundSize; tri2.e1[1] = 0.0f; tri2.e1[2] = 0.0f;
        tri2.e2[0] = 0.0f; tri2.e2[1] = 0.0f; tri2.e2[2] = -2*groundSize;
        tri2.n0[0] = 0; tri2.n0[1] = 1; tri2.n0[2] = 0;
        tri2.n1[0] = 0; tri2.n1[1] = 1; tri2.n1[2] = 0;
        tri2.n2[0] = 0; tri2.n2[1] = 1; tri2.n2[2] = 0;
        tri2.uv0[0] = 1; tri2.uv0[1] = 1;
        tri2.uv1[0] = 0; tri2.uv1[1] = 1;
        tri2.uv2[0] = 1; tri2.uv2[1] = 0;
        tri2.materialId = 0;
        tri2.geometryId = 0;
        m_primitives.push_back(tri2);
        m_primitiveIndices.push_back(primIdx++);
    }

    // Box at center (1x1x1 at y=0.5)
    float boxMin[3] = {-0.5f, 0.0f, -0.5f};
    float boxMax[3] = {0.5f, 1.0f, 0.5f};

    // Helper lambda to add a quad (2 triangles)
    auto addQuad = [&](float v0[3], float v1[3], float v2[3], float v3[3], float n[3]) {
        BVHPrimitive t1 = {};
        t1.v0[0] = v0[0]; t1.v0[1] = v0[1]; t1.v0[2] = v0[2];
        t1.e1[0] = v1[0]-v0[0]; t1.e1[1] = v1[1]-v0[1]; t1.e1[2] = v1[2]-v0[2];
        t1.e2[0] = v2[0]-v0[0]; t1.e2[1] = v2[1]-v0[1]; t1.e2[2] = v2[2]-v0[2];
        t1.n0[0] = n[0]; t1.n0[1] = n[1]; t1.n0[2] = n[2];
        t1.n1[0] = n[0]; t1.n1[1] = n[1]; t1.n1[2] = n[2];
        t1.n2[0] = n[0]; t1.n2[1] = n[1]; t1.n2[2] = n[2];
        t1.materialId = 1;
        t1.geometryId = 1;
        m_primitives.push_back(t1);
        m_primitiveIndices.push_back(primIdx++);

        BVHPrimitive t2 = {};
        t2.v0[0] = v2[0]; t2.v0[1] = v2[1]; t2.v0[2] = v2[2];
        t2.e1[0] = v3[0]-v2[0]; t2.e1[1] = v3[1]-v2[1]; t2.e1[2] = v3[2]-v2[2];
        t2.e2[0] = v0[0]-v2[0]; t2.e2[1] = v0[1]-v2[1]; t2.e2[2] = v0[2]-v2[2];
        t2.n0[0] = n[0]; t2.n0[1] = n[1]; t2.n0[2] = n[2];
        t2.n1[0] = n[0]; t2.n1[1] = n[1]; t2.n1[2] = n[2];
        t2.n2[0] = n[0]; t2.n2[1] = n[1]; t2.n2[2] = n[2];
        t2.materialId = 1;
        t2.geometryId = 1;
        m_primitives.push_back(t2);
        m_primitiveIndices.push_back(primIdx++);
    };

    // Box faces (6 faces = 12 triangles)
    // Top face (+Y)
    { float v0[]={boxMin[0],boxMax[1],boxMin[2]}, v1[]={boxMax[0],boxMax[1],boxMin[2]},
           v2[]={boxMax[0],boxMax[1],boxMax[2]}, v3[]={boxMin[0],boxMax[1],boxMax[2]}, n[]={0,1,0};
      addQuad(v0,v1,v2,v3,n); }
    // Bottom face (-Y)
    { float v0[]={boxMin[0],boxMin[1],boxMax[2]}, v1[]={boxMax[0],boxMin[1],boxMax[2]},
           v2[]={boxMax[0],boxMin[1],boxMin[2]}, v3[]={boxMin[0],boxMin[1],boxMin[2]}, n[]={0,-1,0};
      addQuad(v0,v1,v2,v3,n); }
    // Front face (+Z)
    { float v0[]={boxMin[0],boxMin[1],boxMax[2]}, v1[]={boxMin[0],boxMax[1],boxMax[2]},
           v2[]={boxMax[0],boxMax[1],boxMax[2]}, v3[]={boxMax[0],boxMin[1],boxMax[2]}, n[]={0,0,1};
      addQuad(v0,v1,v2,v3,n); }
    // Back face (-Z)
    { float v0[]={boxMax[0],boxMin[1],boxMin[2]}, v1[]={boxMax[0],boxMax[1],boxMin[2]},
           v2[]={boxMin[0],boxMax[1],boxMin[2]}, v3[]={boxMin[0],boxMin[1],boxMin[2]}, n[]={0,0,-1};
      addQuad(v0,v1,v2,v3,n); }
    // Right face (+X)
    { float v0[]={boxMax[0],boxMin[1],boxMax[2]}, v1[]={boxMax[0],boxMax[1],boxMax[2]},
           v2[]={boxMax[0],boxMax[1],boxMin[2]}, v3[]={boxMax[0],boxMin[1],boxMin[2]}, n[]={1,0,0};
      addQuad(v0,v1,v2,v3,n); }
    // Left face (-X)
    { float v0[]={boxMin[0],boxMin[1],boxMin[2]}, v1[]={boxMin[0],boxMax[1],boxMin[2]},
           v2[]={boxMin[0],boxMax[1],boxMax[2]}, v3[]={boxMin[0],boxMin[1],boxMax[2]}, n[]={-1,0,0};
      addQuad(v0,v1,v2,v3,n); }

    // Build BVH
    m_treeDepth = 0;
    m_leafCount = 0;
    BuildRecursive(0, static_cast<uint32_t>(m_primitives.size()), 0);

    auto endTime = std::chrono::high_resolution_clock::now();
    m_buildTimeMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();

    return true;
}

// Forward declaration
extern void Log(const wchar_t* fmt, ...);

bool BVHBuilder::BuildFromCapturedGeometry(GeometryCapture* capture, const BVHBuildConfig& config)
{
    if (!capture) return false;

    auto startTime = std::chrono::high_resolution_clock::now();

    Clear();
    m_config = config;

    const auto& drawCalls = capture->GetDrawCalls();
    if (drawCalls.empty()) {
        Log(L"[BVH] No draw calls captured\n");
        return false;
    }

    const wchar_t* filterName = (config.geometryFilter == BVHGeometryFilter::StaticOnly) ? L"static" :
                                (config.geometryFilter == BVHGeometryFilter::DynamicOnly) ? L"dynamic" : L"all";
    if (config.verbose)
        Log(L"[BVH] Building from %u captured draw calls (filter=%ls)...\n", (uint32_t)drawCalls.size(), filterName);

    // Estimate triangle count for reservation
    uint32_t estimatedTriangles = 0;
    for (const auto& dc : drawCalls) {
        if (dc.isIndexed && dc.indexCount >= 3) {
            estimatedTriangles += dc.indexCount / 3;
        }
    }

    m_primitives.reserve(estimatedTriangles);
    m_primitiveIndices.reserve(estimatedTriangles);

    uint32_t primIdx = 0;
    uint32_t processedDrawCalls = 0;
    uint32_t skippedDrawCalls = 0;

    // Build a set of allowed DC indices for fast lookup (if filtering by index)
    std::set<uint32_t> allowedIndices;
    if (config.drawCallIndices) {
        for (uint32_t idx : *config.drawCallIndices)
            allowedIndices.insert(idx);
    }

    for (uint32_t dcIdx = 0; dcIdx < (uint32_t)drawCalls.size(); dcIdx++) {
        const auto& dc = drawCalls[dcIdx];

        // Only process indexed draw calls with triangles
        if (!dc.isIndexed || dc.indexCount < 3) {
            skippedDrawCalls++;
            continue;
        }

        // If filtering by specific indices, skip non-matching
        if (config.drawCallIndices && allowedIndices.find(dcIdx) == allowedIndices.end()) {
            skippedDrawCalls++;
            continue;
        }

        // Read vertex data - read ALL vertices from the buffer (indices may reference any of them)
        // First, figure out how many vertices are in the buffer
        D3D11_BUFFER_DESC vbDesc;
        dc.vertexBuffer->GetDesc(&vbDesc);
        uint32_t totalVerticesInBuffer = (dc.vertexStride > 0) ? (vbDesc.ByteWidth / dc.vertexStride) : 0;
        if (totalVerticesInBuffer == 0) {
            skippedDrawCalls++;
            continue;
        }

        std::vector<CapturedVertex> vertices;
        if (!capture->ReadVertexBuffer(dc.vertexBuffer, dc.vertexStride, dc.vertexOffset,
                                        totalVerticesInBuffer, vertices)) {
            skippedDrawCalls++;
            continue;
        }

        // Read index data
        // indexOffset is the byte offset from IASetIndexBuffer
        // startIndex is the element offset from DrawIndexed (startIndexLocation)
        uint32_t indexSize = (dc.indexFormat == DXGI_FORMAT_R32_UINT) ? 4 : 2;
        uint32_t actualIndexByteOffset = dc.indexOffset + dc.startIndex * indexSize;

        std::vector<uint32_t> indices;
        if (!capture->ReadIndexBuffer(dc.indexBuffer, dc.indexFormat, actualIndexByteOffset,
                                       dc.indexCount, indices)) {
            skippedDrawCalls++;
            continue;
        }

        // Apply baseVertex offset to indices
        if (dc.baseVertex != 0) {
            for (auto& idx : indices) {
                idx = static_cast<uint32_t>(static_cast<int32_t>(idx) + dc.baseVertex);
            }
        }

        // Load world matrix for this draw call
        const float* wm = &dc.worldMatrix._11;
        // Check if world matrix is identity (skip transform for static geometry)
        // Translation is at wm[3], wm[7], wm[11] (last column of each row)
        bool isIdentity = (fabsf(wm[0]-1.0f) < 1e-5f && fabsf(wm[5]-1.0f) < 1e-5f &&
                           fabsf(wm[10]-1.0f) < 1e-5f && fabsf(wm[15]-1.0f) < 1e-5f &&
                           fabsf(wm[3]) < 1e-5f && fabsf(wm[7]) < 1e-5f && fabsf(wm[11]) < 1e-5f);

        // Filter based on geometry type
        if (config.geometryFilter == BVHGeometryFilter::StaticOnly && !isIdentity) {
            skippedDrawCalls++;
            continue;
        }
        if (config.geometryFilter == BVHGeometryFilter::DynamicOnly && isIdentity) {
            skippedDrawCalls++;
            continue;
        }

        // Process triangles (all geometry transformed to world space)
        uint32_t triangleCount = dc.indexCount / 3;
        for (uint32_t tri = 0; tri < triangleCount; tri++) {
            uint32_t i0 = indices[tri * 3 + 0];
            uint32_t i1 = indices[tri * 3 + 1];
            uint32_t i2 = indices[tri * 3 + 2];

            // Bounds check
            if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size()) {
                continue;
            }

            const CapturedVertex& v0 = vertices[i0];
            const CapturedVertex& v1 = vertices[i1];
            const CapturedVertex& v2 = vertices[i2];

            // Transform positions by world matrix
            // AC uses row-major: pos * matrix, so row vector * matrix
            // XMFLOAT4X4 is stored row-major: _11,_12,_13,_14, _21,...
            float p0[3], p1[3], p2[3];
            if (isIdentity) {
                p0[0] = v0.position[0]; p0[1] = v0.position[1]; p0[2] = v0.position[2];
                p1[0] = v1.position[0]; p1[1] = v1.position[1]; p1[2] = v1.position[2];
                p2[0] = v2.position[0]; p2[1] = v2.position[1]; p2[2] = v2.position[2];
            } else {
                // AC stores matrices row-major with translation in last column:
                // Row 0: [r00, r01, r02, tx]  = wm[0..3]
                // Row 1: [r10, r11, r12, ty]  = wm[4..7]
                // Row 2: [r20, r21, r22, tz]  = wm[8..11]
                // Row 3: [0,   0,   0,   1]   = wm[12..15]
                // Transform: result = M * pos (matrix * column vector)
                for (int c = 0; c < 3; c++) {
                    const float* vp = (c == 0) ? v0.position : (c == 1) ? v1.position : v2.position;
                    float* pp = (c == 0) ? p0 : (c == 1) ? p1 : p2;
                    pp[0] = vp[0]*wm[0] + vp[1]*wm[1] + vp[2]*wm[2]  + wm[3];
                    pp[1] = vp[0]*wm[4] + vp[1]*wm[5] + vp[2]*wm[6]  + wm[7];
                    pp[2] = vp[0]*wm[8] + vp[1]*wm[9] + vp[2]*wm[10] + wm[11];
                }
            }

            // Skip degenerate triangles
            float e1x = p1[0] - p0[0];
            float e1y = p1[1] - p0[1];
            float e1z = p1[2] - p0[2];
            float e2x = p2[0] - p0[0];
            float e2y = p2[1] - p0[1];
            float e2z = p2[2] - p0[2];

            float crossX = e1y * e2z - e1z * e2y;
            float crossY = e1z * e2x - e1x * e2z;
            float crossZ = e1x * e2y - e1y * e2x;
            float area2 = crossX * crossX + crossY * crossY + crossZ * crossZ;
            if (area2 < 1e-10f) continue;

            // Create BVH primitive
            BVHPrimitive prim = {};

            // Position and edges (Moller-Trumbore format) - in world space
            prim.v0[0] = p0[0];
            prim.v0[1] = p0[1];
            prim.v0[2] = p0[2];

            prim.e1[0] = e1x;
            prim.e1[1] = e1y;
            prim.e1[2] = e1z;

            prim.e2[0] = e2x;
            prim.e2[1] = e2y;
            prim.e2[2] = e2z;

            // Transform normals by inverse transpose of upper-left 3x3
            // (handles non-uniform scaling in KN5 node hierarchies)
            float n0[3], n1[3], n2[3];
            if (isIdentity) {
                n0[0] = v0.normal[0]; n0[1] = v0.normal[1]; n0[2] = v0.normal[2];
                n1[0] = v1.normal[0]; n1[1] = v1.normal[1]; n1[2] = v1.normal[2];
                n2[0] = v2.normal[0]; n2[1] = v2.normal[1]; n2[2] = v2.normal[2];
            } else {
                // Compute inverse transpose of 3x3
                float a00 = wm[0], a01 = wm[1], a02 = wm[2];
                float a10 = wm[4], a11 = wm[5], a12 = wm[6];
                float a20 = wm[8], a21 = wm[9], a22 = wm[10];
                float det = a00*(a11*a22 - a12*a21) - a01*(a10*a22 - a12*a20) + a02*(a10*a21 - a11*a20);
                float c00, c01, c02, c10, c11, c12, c20, c21, c22;
                if (fabsf(det) < 1e-12f) {
                    // Degenerate — use direct transform
                    c00 = a00; c01 = a10; c02 = a20;
                    c10 = a01; c11 = a11; c12 = a21;
                    c20 = a02; c21 = a12; c22 = a22;
                } else {
                    float invDet = 1.0f / det;
                    c00 = (a11*a22 - a12*a21) * invDet;
                    c01 = (a12*a20 - a10*a22) * invDet;
                    c02 = (a10*a21 - a11*a20) * invDet;
                    c10 = (a02*a21 - a01*a22) * invDet;
                    c11 = (a00*a22 - a02*a20) * invDet;
                    c12 = (a01*a20 - a00*a21) * invDet;
                    c20 = (a01*a12 - a02*a11) * invDet;
                    c21 = (a02*a10 - a00*a12) * invDet;
                    c22 = (a00*a11 - a01*a10) * invDet;
                }
                for (int c = 0; c < 3; c++) {
                    const float* vn = (c == 0) ? v0.normal : (c == 1) ? v1.normal : v2.normal;
                    float* nn = (c == 0) ? n0 : (c == 1) ? n1 : n2;
                    nn[0] = vn[0]*c00 + vn[1]*c10 + vn[2]*c20;
                    nn[1] = vn[0]*c01 + vn[1]*c11 + vn[2]*c21;
                    nn[2] = vn[0]*c02 + vn[1]*c12 + vn[2]*c22;
                    float len = sqrtf(nn[0]*nn[0] + nn[1]*nn[1] + nn[2]*nn[2]);
                    if (len > 1e-6f) { nn[0] /= len; nn[1] /= len; nn[2] /= len; }
                }
            }
            prim.n0[0] = n0[0]; prim.n0[1] = n0[1]; prim.n0[2] = n0[2];
            prim.n1[0] = n1[0]; prim.n1[1] = n1[1]; prim.n1[2] = n1[2];
            prim.n2[0] = n2[0]; prim.n2[1] = n2[1]; prim.n2[2] = n2[2];

            // UVs
            prim.uv0[0] = v0.uv[0]; prim.uv0[1] = v0.uv[1];
            prim.uv1[0] = v1.uv[0]; prim.uv1[1] = v1.uv[1];
            prim.uv2[0] = v2.uv[0]; prim.uv2[1] = v2.uv[1];

            // Material/geometry IDs
            prim.materialId = dc.materialId;
            prim.geometryId = dc.drawCallId;

            m_primitives.push_back(prim);
            m_primitiveIndices.push_back(primIdx++);
        }

        processedDrawCalls++;
    }

    if (m_primitives.empty()) {
        Log(L"[BVH] No valid triangles extracted from captured geometry\n");
        return false;
    }

    if (config.verbose) {
        Log(L"[BVH] Extracted %u triangles from %u draw calls (skipped %u)\n",
            (uint32_t)m_primitives.size(), processedDrawCalls, skippedDrawCalls);

        // Debug: Check for vertices near origin (potential untransformed car geometry)
        uint32_t nearOriginCount = 0;
        uint32_t maxGeoId = 0;
        std::vector<uint32_t> geoIdCounts(1000, 0);

        for (size_t i = 0; i < m_primitives.size(); i++) {
            const BVHPrimitive& prim = m_primitives[i];
            float dist = sqrtf(prim.v0[0]*prim.v0[0] + prim.v0[1]*prim.v0[1] + prim.v0[2]*prim.v0[2]);
            if (dist < 10.0f) {
                nearOriginCount++;
                if (nearOriginCount <= 5) {
                    Log(L"[BVH] Near-origin tri %zu: v0=(%.2f, %.2f, %.2f) geoId=%u\n",
                        i, prim.v0[0], prim.v0[1], prim.v0[2], prim.geometryId);
                }
            }
            if (prim.geometryId < 1000) geoIdCounts[prim.geometryId]++;
            if (prim.geometryId > maxGeoId) maxGeoId = prim.geometryId;
        }

        Log(L"[BVH] Triangles near origin (dist<10): %u\n", nearOriginCount);
        Log(L"[BVH] Max geometryId: %u\n", maxGeoId);

        Log(L"[BVH] Triangle counts by geoId (first 20):\n");
        for (int i = 0; i < 20 && i < 1000; i++) {
            if (geoIdCounts[i] > 0)
                Log(L"[BVH]   geoId %d: %u triangles\n", i, geoIdCounts[i]);
        }

        // Compute and log bounds
        float globalMin[3] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        float globalMax[3] = { -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };

        for (size_t i = 0; i < m_primitives.size(); i++) {
            const BVHPrimitive& prim = m_primitives[i];
            float v0[3] = { prim.v0[0], prim.v0[1], prim.v0[2] };
            float v1[3] = { prim.v0[0] + prim.e1[0], prim.v0[1] + prim.e1[1], prim.v0[2] + prim.e1[2] };
            float v2[3] = { prim.v0[0] + prim.e2[0], prim.v0[1] + prim.e2[1], prim.v0[2] + prim.e2[2] };
            for (int j = 0; j < 3; j++) {
                globalMin[j] = std::min(globalMin[j], std::min(v0[j], std::min(v1[j], v2[j])));
                globalMax[j] = std::max(globalMax[j], std::max(v0[j], std::max(v1[j], v2[j])));
            }
        }

        Log(L"[BVH] Geometry bounds: min(%.2f, %.2f, %.2f) max(%.2f, %.2f, %.2f)\n",
            globalMin[0], globalMin[1], globalMin[2], globalMax[0], globalMax[1], globalMax[2]);
    }

    // Build BVH using SAH
    m_treeDepth = 0;
    m_leafCount = 0;
    m_nodes.reserve(m_primitives.size() * 2);  // Reserve to avoid reallocation
    m_primitiveIndices.resize(m_primitives.size());  // Make sure primitive indices are set up
    for (size_t i = 0; i < m_primitives.size(); i++) {
        m_primitiveIndices[i] = static_cast<uint32_t>(i);
    }
    BuildRecursive(0, static_cast<uint32_t>(m_primitives.size()), 0);

    auto endTime = std::chrono::high_resolution_clock::now();
    m_buildTimeMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();

    if (config.verbose)
        Log(L"[BVH] Built in %.2f ms: %u nodes, depth %u, %u leaves\n",
            m_buildTimeMs, (uint32_t)m_nodes.size(), m_treeDepth, m_leafCount);

    return true;
}

void BVHBuilder::Clear()
{
    m_nodes.clear();
    m_primitives.clear();
    m_primitiveIndices.clear();
    m_treeDepth = 0;
    m_leafCount = 0;
    m_buildTimeMs = 0.0f;
}

// ============================================================
// GPU UPLOAD
// ============================================================

bool BVHBuilder::UploadToGPU(ID3D11Device* device)
{
    if (m_nodes.empty() || m_primitives.empty()) {
        return false;
    }

    ReleaseGPU();

    // Reorder primitives according to primitiveIndices
    // This is necessary because the BVH was built with indirect indexing,
    // but the GPU expects direct access via primitiveOffset
    std::vector<BVHPrimitive> reorderedPrimitives(m_primitives.size());
    for (size_t i = 0; i < m_primitiveIndices.size(); i++) {
        reorderedPrimitives[i] = m_primitives[m_primitiveIndices[i]];
    }

    Log(L"[BVH] Reordered %u primitives for GPU upload\n", (uint32_t)reorderedPrimitives.size());

    HRESULT hr;

    // Create node buffer
    {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = static_cast<UINT>(m_nodes.size() * sizeof(BVHNode));
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(BVHNode);

        D3D11_SUBRESOURCE_DATA initData = {};
        initData.pSysMem = m_nodes.data();

        hr = device->CreateBuffer(&desc, &initData, &m_nodeBuffer);
        if (FAILED(hr)) {
            return false;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = static_cast<UINT>(m_nodes.size());

        hr = device->CreateShaderResourceView(m_nodeBuffer, &srvDesc, &m_nodeSRV);
        if (FAILED(hr)) {
            ReleaseGPU();
            return false;
        }
    }

    // Create primitive buffer (using reordered primitives)
    {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = static_cast<UINT>(reorderedPrimitives.size() * sizeof(BVHPrimitive));
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(BVHPrimitive);

        D3D11_SUBRESOURCE_DATA initData = {};
        initData.pSysMem = reorderedPrimitives.data();

        hr = device->CreateBuffer(&desc, &initData, &m_primitiveBuffer);
        if (FAILED(hr)) {
            ReleaseGPU();
            return false;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = static_cast<UINT>(reorderedPrimitives.size());

        hr = device->CreateShaderResourceView(m_primitiveBuffer, &srvDesc, &m_primitiveSRV);
        if (FAILED(hr)) {
            ReleaseGPU();
            return false;
        }
    }

    return true;
}

void BVHBuilder::ReleaseGPU()
{
    if (m_primitiveSRV) { m_primitiveSRV->Release(); m_primitiveSRV = nullptr; }
    if (m_nodeSRV) { m_nodeSRV->Release(); m_nodeSRV = nullptr; }
    if (m_primitiveBuffer) { m_primitiveBuffer->Release(); m_primitiveBuffer = nullptr; }
    if (m_nodeBuffer) { m_nodeBuffer->Release(); m_nodeBuffer = nullptr; }
}

// ============================================================
// BUILD HELPERS
// ============================================================

void BVHBuilder::ComputeAABB(uint32_t start, uint32_t count,
                              float outMin[3], float outMax[3]) const
{
    outMin[0] = outMin[1] = outMin[2] = std::numeric_limits<float>::max();
    outMax[0] = outMax[1] = outMax[2] = -std::numeric_limits<float>::max();

    for (uint32_t i = start; i < start + count; i++) {
        const BVHPrimitive& prim = m_primitives[m_primitiveIndices[i]];

        // Get all three vertices
        float v0[3] = { prim.v0[0], prim.v0[1], prim.v0[2] };
        float v1[3] = { prim.v0[0] + prim.e1[0], prim.v0[1] + prim.e1[1], prim.v0[2] + prim.e1[2] };
        float v2[3] = { prim.v0[0] + prim.e2[0], prim.v0[1] + prim.e2[1], prim.v0[2] + prim.e2[2] };

        for (int j = 0; j < 3; j++) {
            outMin[j] = std::min(outMin[j], std::min(v0[j], std::min(v1[j], v2[j])));
            outMax[j] = std::max(outMax[j], std::max(v0[j], std::max(v1[j], v2[j])));
        }
    }
}

void BVHBuilder::ComputeCentroidAABB(uint32_t start, uint32_t count,
                                      float outMin[3], float outMax[3]) const
{
    outMin[0] = outMin[1] = outMin[2] = std::numeric_limits<float>::max();
    outMax[0] = outMax[1] = outMax[2] = -std::numeric_limits<float>::max();

    for (uint32_t i = start; i < start + count; i++) {
        const BVHPrimitive& prim = m_primitives[m_primitiveIndices[i]];

        // Compute centroid
        float centroid[3] = {
            prim.v0[0] + (prim.e1[0] + prim.e2[0]) / 3.0f,
            prim.v0[1] + (prim.e1[1] + prim.e2[1]) / 3.0f,
            prim.v0[2] + (prim.e1[2] + prim.e2[2]) / 3.0f
        };

        for (int j = 0; j < 3; j++) {
            outMin[j] = std::min(outMin[j], centroid[j]);
            outMax[j] = std::max(outMax[j], centroid[j]);
        }
    }
}

float BVHBuilder::SurfaceArea(const float min[3], const float max[3])
{
    float dx = max[0] - min[0];
    float dy = max[1] - min[1];
    float dz = max[2] - min[2];
    return 2.0f * (dx * dy + dy * dz + dz * dx);
}

uint32_t BVHBuilder::BuildRecursive(uint32_t start, uint32_t count, uint32_t depth)
{
    m_treeDepth = std::max(m_treeDepth, depth);

    // Compute bounds for this node
    float nodeMin[3], nodeMax[3];
    ComputeAABB(start, count, nodeMin, nodeMax);

    // Create leaf if few enough primitives or max depth reached
    if (count <= m_config.maxPrimitivesPerLeaf || depth >= m_config.maxDepth) {
        return CreateLeaf(start, count);
    }

    // Compute centroid bounds for SAH binning
    float centroidMin[3], centroidMax[3];
    ComputeCentroidAABB(start, count, centroidMin, centroidMax);

    // Find best split using SAH
    int bestAxis = -1;
    float bestSplitPos = 0.0f;
    float bestCost = std::numeric_limits<float>::max();
    uint32_t bestLeftCount = 0;

    float nodeSA = SurfaceArea(nodeMin, nodeMax);
    float leafCost = m_config.intersectionCost * count;

    // Try each axis
    for (int axis = 0; axis < 3; axis++) {
        float extent = centroidMax[axis] - centroidMin[axis];
        if (extent < 0.0001f) continue;

        // Binned SAH: try several split positions
        const int numBins = 12;
        for (int bin = 1; bin < numBins; bin++) {
            float splitPos = centroidMin[axis] + extent * (float(bin) / float(numBins));

            uint32_t leftCount = 0;
            float cost = EvaluateSAH(start, count, axis, splitPos, leftCount);

            if (cost < bestCost && leftCount > 0 && leftCount < count) {
                bestCost = cost;
                bestAxis = axis;
                bestSplitPos = splitPos;
                bestLeftCount = leftCount;
            }
        }
    }

    // If no good split found or SAH says leaf is better, create leaf
    if (bestAxis < 0 || bestCost >= leafCost) {
        return CreateLeaf(start, count);
    }

    // Partition primitives around best split
    uint32_t mid = Partition(start, count, bestAxis, bestSplitPos);
    uint32_t leftCount = mid - start;
    uint32_t rightCount = count - leftCount;

    // Fallback if partition failed
    if (leftCount == 0 || rightCount == 0) {
        return CreateLeaf(start, count);
    }

    // Create internal node
    BVHNode node = {};
    node.aabbMin[0] = nodeMin[0]; node.aabbMin[1] = nodeMin[1]; node.aabbMin[2] = nodeMin[2];
    node.aabbMax[0] = nodeMax[0]; node.aabbMax[1] = nodeMax[1]; node.aabbMax[2] = nodeMax[2];
    node.primitiveOffset = 0;
    node.primitiveCount = 0;

    uint32_t nodeIndex = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back(node);

    // Build children recursively
    uint32_t leftChild = BuildRecursive(start, leftCount, depth + 1);
    uint32_t rightChild = BuildRecursive(mid, rightCount, depth + 1);

    // Update node with child indices
    m_nodes[nodeIndex].leftChild = leftChild;
    m_nodes[nodeIndex].rightChild = rightChild;

    return nodeIndex;
}

float BVHBuilder::EvaluateSAH(uint32_t start, uint32_t count,
                               int axis, float splitPos,
                               uint32_t& outLeftCount) const
{
    float leftMin[3] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
    float leftMax[3] = { -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };
    float rightMin[3] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
    float rightMax[3] = { -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };
    float parentMin[3] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
    float parentMax[3] = { -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };

    uint32_t leftCount = 0;
    uint32_t rightCount = 0;

    for (uint32_t i = start; i < start + count; i++) {
        const BVHPrimitive& prim = m_primitives[m_primitiveIndices[i]];

        // Compute centroid
        float centroid = prim.v0[axis] + (prim.e1[axis] + prim.e2[axis]) / 3.0f;

        // Get triangle bounds
        float v0[3] = { prim.v0[0], prim.v0[1], prim.v0[2] };
        float v1[3] = { prim.v0[0] + prim.e1[0], prim.v0[1] + prim.e1[1], prim.v0[2] + prim.e1[2] };
        float v2[3] = { prim.v0[0] + prim.e2[0], prim.v0[1] + prim.e2[1], prim.v0[2] + prim.e2[2] };

        // Update parent bounds
        for (int j = 0; j < 3; j++) {
            float minV = std::min(v0[j], std::min(v1[j], v2[j]));
            float maxV = std::max(v0[j], std::max(v1[j], v2[j]));
            parentMin[j] = std::min(parentMin[j], minV);
            parentMax[j] = std::max(parentMax[j], maxV);
        }

        if (centroid < splitPos) {
            leftCount++;
            for (int j = 0; j < 3; j++) {
                leftMin[j] = std::min(leftMin[j], std::min(v0[j], std::min(v1[j], v2[j])));
                leftMax[j] = std::max(leftMax[j], std::max(v0[j], std::max(v1[j], v2[j])));
            }
        } else {
            rightCount++;
            for (int j = 0; j < 3; j++) {
                rightMin[j] = std::min(rightMin[j], std::min(v0[j], std::min(v1[j], v2[j])));
                rightMax[j] = std::max(rightMax[j], std::max(v0[j], std::max(v1[j], v2[j])));
            }
        }
    }

    outLeftCount = leftCount;

    if (leftCount == 0 || rightCount == 0) {
        return std::numeric_limits<float>::max();
    }

    float parentSA = SurfaceArea(parentMin, parentMax);
    if (parentSA < 0.0001f) {
        return std::numeric_limits<float>::max();
    }

    float leftSA = SurfaceArea(leftMin, leftMax);
    float rightSA = SurfaceArea(rightMin, rightMax);

    // SAH cost: C = traversal + (SA_left/SA_parent * N_left + SA_right/SA_parent * N_right) * intersection
    // Properly normalized by parent surface area
    return m_config.traversalCost + m_config.intersectionCost *
           ((leftSA / parentSA) * leftCount + (rightSA / parentSA) * rightCount);
}

uint32_t BVHBuilder::Partition(uint32_t start, uint32_t count, int axis, float splitPos)
{
    uint32_t left = start;
    uint32_t right = start + count - 1;

    while (left <= right && right < start + count) {
        const BVHPrimitive& prim = m_primitives[m_primitiveIndices[left]];
        float centroid = prim.v0[axis] + (prim.e1[axis] + prim.e2[axis]) / 3.0f;

        if (centroid < splitPos) {
            left++;
        } else {
            std::swap(m_primitiveIndices[left], m_primitiveIndices[right]);
            right--;
        }
    }

    return left;
}

uint32_t BVHBuilder::CreateLeaf(uint32_t start, uint32_t count)
{
    BVHNode node = {};

    // Compute bounds
    float nodeMin[3], nodeMax[3];
    ComputeAABB(start, count, nodeMin, nodeMax);

    node.aabbMin[0] = nodeMin[0]; node.aabbMin[1] = nodeMin[1]; node.aabbMin[2] = nodeMin[2];
    node.aabbMax[0] = nodeMax[0]; node.aabbMax[1] = nodeMax[1]; node.aabbMax[2] = nodeMax[2];
    node.leftChild = 0xFFFFFFFF;  // Leaf marker
    node.rightChild = 0;
    node.primitiveOffset = start;
    node.primitiveCount = count;

    m_leafCount++;

    uint32_t nodeIndex = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back(node);

    return nodeIndex;
}

} // namespace acpt
