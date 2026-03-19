// ============================================================
// light_tree.glsl — Light tree traversal for importance-based light sampling
// Requires: lightTreeNodes SSBO at binding 27, cam UBO
// ============================================================

#ifndef LIGHT_TREE_GLSL
#define LIGHT_TREE_GLSL

// GPU light tree node (32 bytes = 2 vec4s)
// Matches LightTreeNode in light_tree.h
struct LTNode {
    vec3 bboxMin;
    float energy;
    vec3 bboxMax;
    uint childOrFirst;  // inner: left child idx, leaf: first emitter idx
    vec3 coneAxis;
    uint countAndFlags; // 0 = inner node, >0 = leaf emitter count
};

// Read a node from the SSBO (8 floats per node)
LTNode readLTNode(uint idx) {
    LTNode n;
    uint base = idx * 8u;
    n.bboxMin = vec3(lightTreeData.d[base+0], lightTreeData.d[base+1], lightTreeData.d[base+2]);
    n.energy = lightTreeData.d[base+3];
    n.bboxMax = vec3(lightTreeData.d[base+4], lightTreeData.d[base+5], lightTreeData.d[base+6]);
    n.childOrFirst = floatBitsToUint(lightTreeData.d[base+7]);
    // coneAxis and countAndFlags in next 8 floats
    n.coneAxis = vec3(lightTreeData.d[base+8], lightTreeData.d[base+9], lightTreeData.d[base+10]);
    n.countAndFlags = floatBitsToUint(lightTreeData.d[base+11]);
    return n;
}

// Compute importance of a node for a shading point
float lightTreeImportance(LTNode node, vec3 worldPos, vec3 N) {
    if (node.energy <= 0.0) return 0.0;

    // Distance to closest point on AABB
    vec3 closest = clamp(worldPos, node.bboxMin, node.bboxMax);
    float dist = max(length(closest - worldPos), 0.01);

    // Geometric importance: energy / distance²
    float importance = node.energy / (dist * dist);

    // Angular importance: bias toward lights facing the surface
    vec3 toNode = normalize((node.bboxMin + node.bboxMax) * 0.5 - worldPos);
    float cosAngle = max(dot(N, toNode), 0.0);
    importance *= (cosAngle * 0.8 + 0.2);  // soft angular falloff, never zero

    return importance;
}

// Sample a light from the tree. Returns emitter index and PDF.
// maxEmitters = total number of emitters in the light array
int lightTreeSample(vec3 worldPos, vec3 N, uint nodeCount, out float pdf) {
    pdf = 1.0;
    if (nodeCount == 0u) return -1;

    uint nodeIdx = 0u;

    // Traverse tree top-down
    for (int depth = 0; depth < 32; depth++) {
        LTNode node = readLTNode(nodeIdx);

        // Leaf — select emitter within leaf
        if (node.countAndFlags > 0u) {
            uint count = node.countAndFlags;
            uint first = node.childOrFirst;

            // Uniform sampling within leaf (could use importance-weighted reservoir)
            uint selected = first + uint(rand01() * float(count));
            if (selected >= first + count) selected = first + count - 1u;
            pdf *= 1.0 / float(count);
            return int(selected);
        }

        // Inner node — compute child importances
        uint leftIdx = node.childOrFirst;
        uint rightIdx = leftIdx + 1u;

        LTNode leftNode = readLTNode(leftIdx);
        LTNode rightNode = readLTNode(rightIdx);

        float leftImp = lightTreeImportance(leftNode, worldPos, N);
        float rightImp = lightTreeImportance(rightNode, worldPos, N);

        float totalImp = leftImp + rightImp;
        if (totalImp <= 0.0) {
            // Fallback: 50/50
            totalImp = 1.0;
            leftImp = 0.5;
        }

        float leftProb = leftImp / totalImp;

        // Probabilistic descent
        if (rand01() < leftProb) {
            nodeIdx = leftIdx;
            pdf *= leftProb;
        } else {
            nodeIdx = rightIdx;
            pdf *= (1.0 - leftProb);
        }
    }

    return -1; // shouldn't reach here
}

#endif // LIGHT_TREE_GLSL
