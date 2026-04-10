// ============================================================
// light_tree.glsl — Light tree traversal for importance-based light sampling
// Binary BVH with orientation cones (Cycles/Estevez-Kulla matching)
// Requires: lightTreeNodes SSBO at binding 27, cam UBO
// ============================================================

#ifndef LIGHT_TREE_GLSL
#define LIGHT_TREE_GLSL

// GPU light tree node (64 bytes = 16 floats = 4 vec4s)
// Matches LightTreeNode in light_tree.h
struct LTNode {
    vec3 bboxMin;
    float energy;
    vec3 bboxMax;
    uint childOrFirst;  // inner: left child idx, leaf: first emitter idx
    vec3 coneAxis;      // orientation cone axis (normalized)
    uint countAndFlags; // 0 = inner node, >0 = leaf emitter count
    float theta_o;      // orientation cone half-angle
    float theta_e;      // emission spread angle
};

// Read a node from the SSBO (16 floats per node)
LTNode readLTNode(uint idx) {
    LTNode n;
    uint base = idx * 16u;
    n.bboxMin = vec3(lightTreeData.d[base+0], lightTreeData.d[base+1], lightTreeData.d[base+2]);
    n.energy = lightTreeData.d[base+3];
    n.bboxMax = vec3(lightTreeData.d[base+4], lightTreeData.d[base+5], lightTreeData.d[base+6]);
    n.childOrFirst = floatBitsToUint(lightTreeData.d[base+7]);
    n.coneAxis = vec3(lightTreeData.d[base+8], lightTreeData.d[base+9], lightTreeData.d[base+10]);
    n.countAndFlags = floatBitsToUint(lightTreeData.d[base+11]);
    n.theta_o = lightTreeData.d[base+12];
    n.theta_e = lightTreeData.d[base+13];
    return n;
}

// Compute importance of a node for a shading point (Cycles-matching)
// Uses distance, incidence angle bounds, and orientation cone visibility
float lightTreeImportance(LTNode node, vec3 P, vec3 N, bool hasTrans) {
    if (node.energy <= 0.0) return 0.0;

    // Distance to closest point on AABB
    vec3 closest = clamp(P, node.bboxMin, node.bboxMax);
    float dist = max(length(closest - P), 0.01);

    // Bounding sphere for angular extent
    vec3 centroid = 0.5 * (node.bboxMin + node.bboxMax);
    vec3 toCenter = centroid - P;
    float distCenterSq = dot(toCenter, toCenter);
    vec3 halfExt = node.bboxMax - centroid;
    float radiusSq = dot(halfExt, halfExt);

    // cos(theta_u): angular radius of bounding sphere from P
    float cosTheta_u = (distCenterSq <= radiusSq) ? -1.0
        : sqrt(max(1.0 - radiusSq / distCenterSq, 0.0));
    float sinTheta_u = sqrt(max(1.0 - cosTheta_u * cosTheta_u, 0.0));

    // Incidence angle (N dot direction-to-cluster)
    vec3 dir = normalize(toCenter);
    float cosTheta_i = hasTrans ? abs(dot(N, dir)) : dot(N, dir);
    float sinTheta_i = sqrt(max(1.0 - cosTheta_i * cosTheta_i, 0.0));

    // Minimum incidence angle (tighten by bounding sphere angular size)
    float cosMinIncidence;
    if (cosTheta_i >= cosTheta_u) {
        cosMinIncidence = 1.0;
    } else {
        cosMinIncidence = cosTheta_i * cosTheta_u + sinTheta_i * sinTheta_u;
    }
    if (!hasTrans && cosMinIncidence < 0.0) return 0.0;
    cosMinIncidence = abs(cosMinIncidence);

    // Outgoing angle (orientation cone visibility)
    float cosTheta = dot(node.coneAxis, -dir);
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    float cosThetaMinusU = cosTheta * cosTheta_u + sinTheta * sinTheta_u;

    float cosTheta_o = cos(node.theta_o);
    float sinTheta_o = sin(node.theta_o);

    float cosMinOutgoing;
    if (cosTheta >= cosTheta_u || cosThetaMinusU >= cosTheta_o) {
        cosMinOutgoing = 1.0;
    } else if (node.theta_o + node.theta_e > PI ||
               cosThetaMinusU > cos(node.theta_o + node.theta_e)) {
        cosMinOutgoing = cosThetaMinusU * cosTheta_o
            + sqrt(max(1.0 - cosThetaMinusU * cosThetaMinusU, 0.0)) * sinTheta_o;
        cosMinOutgoing = max(cosMinOutgoing, 0.0);
    } else {
        return 0.0; // cluster entirely facing away
    }

    return cosMinIncidence * node.energy * cosMinOutgoing / (dist * dist);
}

// Sample a light from the tree. Returns emitter index and PDF.
int lightTreeSample(vec3 P, vec3 N, bool hasTrans, uint nodeCount, out float pdf) {
    pdf = 1.0;
    if (nodeCount == 0u) return -1;

    uint nodeIdx = 0u;
    float rng = rand01();

    // Traverse tree top-down (max ~20 levels for 1M lights)
    for (int depth = 0; depth < 32; depth++) {
        LTNode node = readLTNode(nodeIdx);

        // Leaf — select emitter within leaf
        if (node.countAndFlags > 0u) {
            uint count = node.countAndFlags;
            uint first = node.childOrFirst;
            uint selected = first + uint(rng * float(count));
            if (selected >= first + count) selected = first + count - 1u;
            pdf *= 1.0 / float(count);
            return int(selected);
        }

        // Inner node — compute child importances
        uint leftIdx = node.childOrFirst;
        uint rightIdx = leftIdx + 1u;

        LTNode leftNode = readLTNode(leftIdx);
        LTNode rightNode = readLTNode(rightIdx);

        float leftImp = lightTreeImportance(leftNode, P, N, hasTrans);
        float rightImp = lightTreeImportance(rightNode, P, N, hasTrans);

        float totalImp = leftImp + rightImp;
        if (totalImp <= 0.0) return -1; // no visible lights

        float leftProb = leftImp / totalImp;

        // Probabilistic descent with random number reuse (Cycles pattern)
        if (rng < leftProb) {
            nodeIdx = leftIdx;
            pdf *= leftProb;
            rng = rng / max(leftProb, 1e-8);
        } else {
            nodeIdx = rightIdx;
            pdf *= (1.0 - leftProb);
            rng = (rng - leftProb) / max(1.0 - leftProb, 1e-8);
        }
    }

    return -1; // shouldn't reach here
}

#endif // LIGHT_TREE_GLSL
