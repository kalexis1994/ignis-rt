#pragma once

#include <cstdint>

// ============================================================================
// ignis_lights.h — polymorphic light descriptor (RTXDI DI port, Phase 5a).
//
// Single uniform representation for all light sources the path tracer
// sees: the distant sun, per-car / per-track analytic emitters (CSP
// point+spot), emissive triangles from authored geometry, and the HDRI
// environment. A tag-based polymorphic struct lets the RTXDI-style
// initial-sample + resampling passes iterate one buffer instead of
// three separate code paths.
//
// Layout matches shaders/include/wf_lights.glsl 1:1 (GLSL uses scalar
// layout). Every field is float / int32 / uint32 so C++ and GLSL see
// identical bytes; no pad unless explicitly named.
//
// Size: 96 bytes / light. A dense-city track (~6k point lights) needs
// < 1 MiB; emissive-triangle-rich scenes (a stadium with 100k tris)
// need ~10 MiB, still cheap on a 10 GB card.
// ============================================================================

namespace ignis {

enum class LightType : uint32_t {
    Directional = 0u,   // sun. direction = base; radiance in `emission`.
    Point       = 1u,   // isotropic. position = base; range in scalars[0].
    Spot        = 2u,   // position=base, axis=direction, cos(half-inner)=
                        //   scalars[0], cos(half-outer)=scalars[1],
                        //   range=scalars[2].
    Triangle    = 3u,   // area emitter. v0=base, edge01=edges0, edge02=edges1.
                        //   radiance = emission (×texture[0] if set).
    Environment = 4u,   // HDRI / procedural sky. No base; radiance from the
                        //   environment map indexed by texture[0].
};

struct LightInfo {
    uint32_t type;              // LightType
    uint32_t texture;           // emission-modulation tex index, 0xFFFFFFFF = none
    uint32_t stableId;          // stable identity for temporal reservoir reuse
    uint32_t pad1;

    // Base position / direction depending on type.
    //   Directional: unit direction (from light TO scene, i.e. sun dir).
    //   Point / Spot: world-space position.
    //   Triangle: vertex 0.
    //   Environment: unused.
    float base[3];
    float power;                // normalized cumulative light-pick CDF for DI
                                // initial sampling. Built from radiant power
                                // proxies on the CPU before upload.

    // Shape basis.
    //   Triangle: edge01 (v1 − v0).
    //   Spot: axis direction (unit).
    //   Point / Directional / Environment: unused.
    float edges0[3];
    float pad2;

    //   Triangle: edge02 (v2 − v0).
    //   Directional / Spot / Point / Environment: unused.
    float edges1[3];
    float pad3;

    // Emission radiance (RGB) — the physical radiance leaving the surface
    // in watts/sr/m² (or whatever our scene unit pretends to be). For Sun
    // this is the Solar disc radiance × atmospheric transmittance.
    float emission[3];
    float pad4;

    // Type-specific scalars (pack per-light extras without inflating the
    // struct per type).
    //   Point:    [0]=range, [1..3]=0.
    //   Spot:     [0]=cosInner, [1]=cosOuter, [2]=range, [3]=0.
    //   Triangle: all 0 (shape comes from edges + base).
    //   Environment: [0]=intensity mul, [1..3]=0.
    //   Directional: [0]=disc angular radius (rad), [1..3]=0.
    float scalars[4];
};

static_assert(sizeof(LightInfo) == 96, "LightInfo must be exactly 96 bytes");

} // namespace ignis
