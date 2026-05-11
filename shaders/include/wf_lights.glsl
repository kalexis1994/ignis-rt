// ============================================================
// wf_lights.glsl — GLSL mirror of ignis::LightInfo (C++ header).
//
// Polymorphic light descriptor for the RTXDI DI port. Single
// representation for directional / point / spot / triangle /
// environment lights. Use with scalar layout — C++ and GLSL see
// identical bytes.
//
// Size: 96 bytes / entry. See ignis_lights.h for the per-type
// field conventions.
// ============================================================

#ifndef WF_LIGHTS_GLSL
#define WF_LIGHTS_GLSL

// Type tags — match ignis::LightType in ignis_lights.h.
const uint WF_LIGHT_TYPE_DIRECTIONAL = 0u;
const uint WF_LIGHT_TYPE_POINT       = 1u;
const uint WF_LIGHT_TYPE_SPOT        = 2u;
const uint WF_LIGHT_TYPE_TRIANGLE    = 3u;
const uint WF_LIGHT_TYPE_ENVIRONMENT = 4u;

const uint WF_LIGHT_NO_TEXTURE = 0xFFFFFFFFu;

struct WFLightInfo {
    uint  type;        // WF_LIGHT_TYPE_*
    uint  texture;     // emission-modulation tex index, WF_LIGHT_NO_TEXTURE = none
    uint  stableId;
    uint  pad1;

    vec3  base;        // position (or direction for Directional)
    float power;       // normalized cumulative light-pick CDF

    vec3  edges0;      // triangle edge01 / spot axis
    float pad2;

    vec3  edges1;      // triangle edge02
    float pad3;

    vec3  emission;    // RGB radiance
    float pad4;

    vec4  scalars;     // type-specific: see header
};

#endif // WF_LIGHTS_GLSL
