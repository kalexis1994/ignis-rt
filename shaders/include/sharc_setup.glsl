// SHARC integration for Ignis RT
// Sets up defines and includes NVIDIA's official SharcCommon.h
// Reference: https://github.com/NVIDIA-RTX/SHARC
// License: NVIDIA proprietary (see external/SHARC/License.md)

#ifndef SHARC_SETUP_GLSL
#define SHARC_SETUP_GLSL

// Configure SHARC for our single-pass approach
#define SHARC_ENABLE_GLSL 1
#define SHARC_ENABLE_64_BIT_ATOMICS 1
#define SHARC_PROPAGATION_DEPTH 4
#define SHARC_SAMPLE_NUM_THRESHOLD 0
#define SHARC_SAMPLE_NUM_BIT_NUM 16

// We compile with both UPDATE and QUERY in one pass
// (sparse pixels update, all pixels query)
#define SHARC_UPDATE 1
#define SHARC_QUERY 1

// Required GLSL extensions for SHARC
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_buffer_reference2 : require

// Include NVIDIA SHARC library
// Order: GLSL compat → HashGrid → Types → Main API
#include "../../external/SHARC/include/SharcGlsl.h"
#include "../../external/SHARC/include/HashGridCommon.h"
#include "../../external/SHARC/include/SharcTypes.h"

// SharcCommon.h uses #include which may conflict with already-included files.
// The headers above handle the dependencies, so SharcCommon.h's #includes are safe.
#include "../../external/SHARC/include/SharcCommon.h"

// Helper: construct SharcParameters from our CameraUBO
SharcParameters buildSharcParams() {
    SharcParameters params;

    // Grid parameters
    params.gridParameters.cameraPosition = (cam.view * vec4(0, 0, 0, 1)).xyz;
    // Camera position in world space = inverse view translation
    params.gridParameters.cameraPosition = vec3(
        cam.viewInverse[3][0],
        cam.viewInverse[3][1],
        cam.viewInverse[3][2]
    );
    params.gridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    params.gridParameters.sceneScale = cam.sharcSceneScale;
    params.gridParameters.levelBias = 0.0;

    // Hash map data (buffer references from device addresses)
    params.hashMapData.hashEntriesBuffer = RWStructuredBuffer_uint64_t(cam.sharcHashEntriesAddr);
    params.hashMapData.capacity = cam.sharcCapacity;

    // Radiance scale for integer accumulation
    params.radianceScale = cam.sharcRadianceScale;
    params.enableAntiFireflyFilter = true;

    // Buffer references
    params.accumulationBuffer = RWStructuredBuffer_SharcAccumulationData(cam.sharcAccumulationAddr);
    params.resolvedBuffer = RWStructuredBuffer_SharcPackedData(cam.sharcResolvedAddr);

    return params;
}

// Determine if this pixel should do SHARC update (sparse 4%)
bool isSharcUpdatePixel(uvec2 pixel, uint frameIndex) {
    // Select 1 pixel per 5×5 block, rotating each frame
    uint blockX = pixel.x / 5u;
    uint blockY = pixel.y / 5u;
    uint blockHash = (blockX * 73856093u ^ blockY * 19349663u ^ frameIndex * 83492791u);
    uint selectedX = blockX * 5u + (blockHash % 5u);
    uint selectedY = blockY * 5u + ((blockHash / 5u) % 5u);
    return (pixel.x == selectedX && pixel.y == selectedY);
}

#endif // SHARC_SETUP_GLSL
