#pragma once
#include <cstdint>

namespace acpt {

// Cluster Grid Configuration
// 16 * 120 = 1920
// 9 * 120 = 1080
// Z slices = 24 (logarithmic)
#define CLUSTER_GRID_X 16
#define CLUSTER_GRID_Y 9
#define CLUSTER_GRID_Z 24

// Maximum lights
#define MAX_LIGHTS_GLOBAL 1024
#define MAX_LIGHTS_PER_CLUSTER 256 // Limits bitmask size (256 bits = 8 uints)

// Data Structures (Must match GLSL std430 layout)

struct ClusterState {
    float viewMatrix[16];        // View matrix (World -> View)
    float inverseProjection[16]; // Inverse Projection (NDC -> View)
    float screenWidth;
    float screenHeight;
    float zNear;
    float zFar;
    float padding[12];           // Pad to 256 bytes (16 * 4 = 64 floats = 256 bytes)
};

struct GPULight {
    // Packed for std430/140 alignment safety
    float positionAndRange[4];    // xyz = position, w = range
    float colorAndIntensity[4];   // xyz = color, w = intensity
    float directionAndAngle[4];   // xyz = direction, w = spotAngleOuter
    
    // Packed params: 
    // x = spotAngleInner
    // y = type (0=Omni, 1=Spot, 2=Dir)
    // z, w = padding
    float params[4];
};

struct ClusterBufferHeader {
    uint32_t gridDimX;
    uint32_t gridDimY;
    uint32_t gridDimZ;
    uint32_t activeLightCount;
};

}