#include "light_tree.h"
#include <numeric>
#include <cfloat>

namespace acpt {

struct BuildNode {
    float bboxMin[3], bboxMax[3];
    float energy;
    float centroid[3];
    int firstEmitter, count;
    int leftChild, rightChild; // -1 = leaf
};

static void computeBounds(const std::vector<LightEmitter>& emitters, int first, int count,
                           float outMin[3], float outMax[3], float outCentroid[3], float& outEnergy) {
    outMin[0] = outMin[1] = outMin[2] = FLT_MAX;
    outMax[0] = outMax[1] = outMax[2] = -FLT_MAX;
    outEnergy = 0.0f;
    for (int i = first; i < first + count; i++) {
        const auto& e = emitters[i];
        for (int a = 0; a < 3; a++) {
            outMin[a] = std::min(outMin[a], e.position[a]);
            outMax[a] = std::max(outMax[a], e.position[a]);
        }
        outEnergy += e.intensity;
    }
    for (int a = 0; a < 3; a++)
        outCentroid[a] = (outMin[a] + outMax[a]) * 0.5f;
}

static int findSplitAxis(const std::vector<LightEmitter>& emitters, int first, int count) {
    float extents[3];
    float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (int i = first; i < first + count; i++) {
        for (int a = 0; a < 3; a++) {
            mn[a] = std::min(mn[a], emitters[i].position[a]);
            mx[a] = std::max(mx[a], emitters[i].position[a]);
        }
    }
    for (int a = 0; a < 3; a++) extents[a] = mx[a] - mn[a];
    int axis = 0;
    if (extents[1] > extents[axis]) axis = 1;
    if (extents[2] > extents[axis]) axis = 2;
    return axis;
}

static void buildRecursive(std::vector<LightEmitter>& emitters, int first, int count,
                            std::vector<BuildNode>& nodes, int nodeIdx) {
    BuildNode& node = nodes[nodeIdx];
    node.firstEmitter = first;
    node.count = count;
    node.leftChild = node.rightChild = -1;

    computeBounds(emitters, first, count, node.bboxMin, node.bboxMax, node.centroid, node.energy);

    // Leaf if small enough
    if (count <= 4) return;

    // Find split axis and sort
    int axis = findSplitAxis(emitters, first, count);
    std::sort(emitters.begin() + first, emitters.begin() + first + count,
              [axis](const LightEmitter& a, const LightEmitter& b) {
                  return a.position[axis] < b.position[axis];
              });

    // Find best split using SAH-like heuristic (energy-weighted)
    int bestSplit = count / 2;
    float bestCost = FLT_MAX;

    // Try 8 bucket splits
    int numBuckets = std::min(count, 8);
    for (int b = 1; b < numBuckets; b++) {
        int split = first + (count * b) / numBuckets;
        int leftCount = split - first;
        int rightCount = count - leftCount;
        if (leftCount == 0 || rightCount == 0) continue;

        float lMin[3], lMax[3], lC[3], lE;
        float rMin[3], rMax[3], rC[3], rE;
        computeBounds(emitters, first, leftCount, lMin, lMax, lC, lE);
        computeBounds(emitters, split, rightCount, rMin, rMax, rC, rE);

        // Cost = weighted energy imbalance
        float cost = lE * leftCount + rE * rightCount;
        if (cost < bestCost) {
            bestCost = cost;
            bestSplit = split - first;
        }
    }

    if (bestSplit <= 0 || bestSplit >= count) bestSplit = count / 2;

    int leftIdx = (int)nodes.size();
    nodes.push_back({});
    int rightIdx = (int)nodes.size();
    nodes.push_back({});

    node.leftChild = leftIdx;
    node.rightChild = rightIdx;

    buildRecursive(emitters, first, bestSplit, nodes, leftIdx);
    buildRecursive(emitters, first + bestSplit, count - bestSplit, nodes, rightIdx);
}

std::vector<LightTreeNode> BuildLightTree(std::vector<LightEmitter>& emitters) {
    if (emitters.empty()) return {};

    // Build
    std::vector<BuildNode> buildNodes;
    buildNodes.push_back({});
    buildRecursive(emitters, 0, (int)emitters.size(), buildNodes, 0);

    // Linearize to GPU-friendly array
    std::vector<LightTreeNode> gpuNodes(buildNodes.size());
    for (size_t i = 0; i < buildNodes.size(); i++) {
        const BuildNode& bn = buildNodes[i];
        LightTreeNode& gn = gpuNodes[i];

        gn.bboxMin[0] = bn.bboxMin[0]; gn.bboxMin[1] = bn.bboxMin[1]; gn.bboxMin[2] = bn.bboxMin[2];
        gn.bboxMax[0] = bn.bboxMax[0]; gn.bboxMax[1] = bn.bboxMax[1]; gn.bboxMax[2] = bn.bboxMax[2];
        gn.energy = bn.energy;

        // Compute orientation cone from emitter directions
        // The cone axis is the average emission direction, theta_o bounds the spread
        float axisSum[3] = {0, 0, 0};
        float maxAngle = 0.0f;
        bool hasDirectional = false;
        for (int e = bn.firstEmitter; e < bn.firstEmitter + bn.count; e++) {
            float dx = emitters[e].direction[0];
            float dy = emitters[e].direction[1];
            float dz = emitters[e].direction[2];
            float dLen = sqrtf(dx*dx + dy*dy + dz*dz);
            if (dLen > 0.001f) {
                axisSum[0] += dx / dLen;
                axisSum[1] += dy / dLen;
                axisSum[2] += dz / dLen;
                hasDirectional = true;
            }
        }

        float aLen = sqrtf(axisSum[0]*axisSum[0] + axisSum[1]*axisSum[1] + axisSum[2]*axisSum[2]);
        if (hasDirectional && aLen > 0.001f) {
            gn.coneAxis[0] = axisSum[0] / aLen;
            gn.coneAxis[1] = axisSum[1] / aLen;
            gn.coneAxis[2] = axisSum[2] / aLen;

            // theta_o = max angle between cone axis and any emitter direction
            for (int e = bn.firstEmitter; e < bn.firstEmitter + bn.count; e++) {
                float dx = emitters[e].direction[0];
                float dy = emitters[e].direction[1];
                float dz = emitters[e].direction[2];
                float dLen2 = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dLen2 > 0.001f) {
                    float cosA = (dx*gn.coneAxis[0] + dy*gn.coneAxis[1] + dz*gn.coneAxis[2]) / dLen2;
                    float angle = acosf(std::max(-1.0f, std::min(1.0f, cosA)));
                    maxAngle = std::max(maxAngle, angle);
                }
            }
            gn.theta_o = maxAngle;
            // theta_e: area/spot lights have limited emission angle, point lights are isotropic
            gn.theta_e = 3.14159f; // conservative: assume full hemisphere
        } else {
            // Isotropic emitters (point lights) — no preferred direction
            gn.coneAxis[0] = 0; gn.coneAxis[1] = 1; gn.coneAxis[2] = 0;
            gn.theta_o = 3.14159f;  // full sphere
            gn.theta_e = 3.14159f;
        }
        gn.pad[0] = gn.pad[1] = 0.0f;

        bool isLeaf = (bn.leftChild == -1);
        if (isLeaf) {
            gn.childOrFirst = (uint32_t)bn.firstEmitter;
            gn.countAndFlags = (uint32_t)bn.count;
        } else {
            gn.childOrFirst = (uint32_t)bn.leftChild;
            gn.countAndFlags = 0; // 0 count = inner node
        }
    }

    return gpuNodes;
}

} // namespace acpt
