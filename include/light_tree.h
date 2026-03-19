#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

namespace acpt {

// GPU-side light tree node (32 bytes — fits in 2 vec4s)
struct LightTreeNode {
    float bboxMin[3];      // AABB min
    float energy;          // total energy of subtree
    float bboxMax[3];      // AABB max
    uint32_t childOrFirst; // inner: left child index, leaf: first emitter index
    float coneAxis[3];     // orientation cone axis (normalized)
    uint32_t countAndFlags;// bits [0:15] = emitter count (0 = inner), bits [16:31] = reserved
};

// CPU-side emitter for building the tree
struct LightEmitter {
    float position[3];
    float intensity;       // total power (R+G+B) * multiplier
    float color[3];
    float range;
    float direction[3];    // emission direction (for area/spot)
    float sizeX;           // area light width (0 for point)
    float sizeY;           // area light height
    float tangent[3];      // area light tangent
    uint32_t originalIndex;// index in the original lights array
};

// Build a light tree from emitters. Returns linearized node array.
// Emitters are reordered so leafs reference contiguous ranges.
std::vector<LightTreeNode> BuildLightTree(std::vector<LightEmitter>& emitters);

} // namespace acpt
