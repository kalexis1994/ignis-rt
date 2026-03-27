#include "ignis_api.h"
#include "ignis_config.h"
#include "ignis_log.h"
#include "ignis_texture.h"
#include "vk/vk_renderer.h"
#include "vk/vk_rt_pipeline.h"
#include "vk/vk_accel_structure.h"
#include "vk/vk_texture_manager.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include "light_tree.h"

// ============================================================================
// Global state
// ============================================================================

namespace acpt {
    PathTracerConfig g_config;
}

static acpt::vk::Renderer* g_renderer = nullptr;
static HWND g_hiddenWindow = nullptr;
static bool g_hiddenWindowClassRegistered = false;
static std::string g_basePath;  // shader/resource root directory

static LRESULT CALLBACK HiddenWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static HWND CreateHiddenWindow() {
    HINSTANCE hInstance = GetModuleHandle(nullptr);

    if (!g_hiddenWindowClassRegistered) {
        WNDCLASSW wc{};
        wc.lpfnWndProc = HiddenWndProc;
        wc.hInstance = hInstance;
        wc.lpszClassName = L"IgnisRTHiddenWindow";
        if (!RegisterClassW(&wc)) {
            Log(L"[Ignis] ERROR: Failed to register hidden window class\n");
            return nullptr;
        }
        g_hiddenWindowClassRegistered = true;
    }

    HWND hwnd = CreateWindowExW(0, L"IgnisRTHiddenWindow", L"Ignis RT",
                                WS_OVERLAPPED, 0, 0, 1, 1,
                                nullptr, nullptr, hInstance, nullptr);
    if (!hwnd) {
        Log(L"[Ignis] ERROR: Failed to create hidden window\n");
    }
    return hwnd;
}

static void DestroyHiddenWindow() {
    if (g_hiddenWindow) {
        DestroyWindow(g_hiddenWindow);
        g_hiddenWindow = nullptr;
    }
}

// Expose config for NRD integration (extern in nrd_vulkan_integration.cpp)
namespace acpt {
    PathTracerConfig* VK_GetConfig() { return &g_config; }
}

// Resolve a relative path using the base path (for shader loading)
std::string IgnisResolvePath(const char* relativePath) {
    if (g_basePath.empty()) return relativePath;
    return g_basePath + relativePath;
}

// ============================================================================
// Helper: Halton sequence for DLSS jitter
// ============================================================================
static float HaltonSeq(int index, int base) {
    float f = 1.0f, r = 0.0f;
    while (index > 0) {
        f /= base;
        r += f * (index % base);
        index /= base;
    }
    return r;
}

// Convert azimuth/elevation (degrees) to normalized sun direction vector
static void ComputeSunDirection(float azimuthDeg, float elevationDeg, float out[3]) {
    float az = azimuthDeg * 3.14159265f / 180.0f;
    float el = elevationDeg * 3.14159265f / 180.0f;
    out[0] = sinf(az) * cosf(el);
    out[1] = sinf(el);
    out[2] = cosf(az) * cosf(el);
}

// Compute physically-plausible sun/ambient colors from sun elevation angle
static void ComputeAtmosphericColors(float elevationDeg, acpt::PathTracerConfig* cfg) {
    const float E_NIGHT = -2.0f, E_HORIZON = 0.0f, E_DAWN = 5.0f, E_MORNING = 15.0f, E_DAY = 45.0f;

    auto ss = [](float edge0, float edge1, float x) -> float {
        float t = (x - edge0) / (edge1 - edge0);
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        return t * t * (3.0f - 2.0f * t);
    };
    auto lerp3 = [](float out[3], const float a[3], const float b[3], float t) {
        out[0] = a[0] + (b[0] - a[0]) * t;
        out[1] = a[1] + (b[1] - a[1]) * t;
        out[2] = a[2] + (b[2] - a[2]) * t;
    };

    const float sunC[][3]  = {{0,0,0},{1.0f,0.3f,0.05f},{1.0f,0.55f,0.2f},{1.0f,0.85f,0.65f},{1.0f,0.96f,0.92f}};
    const float sunI[]     = {0.0f, 0.3f, 0.8f, 1.3f, 1.8f};
    const float ambC[][3]  = {{0.03f,0.03f,0.08f},{0.25f,0.15f,0.1f},{0.4f,0.3f,0.35f},{0.45f,0.5f,0.7f},{0.5f,0.6f,0.8f}};
    const float ambI[]     = {0.02f, 0.06f, 0.15f, 0.3f, 0.5f};

    float sc[3], ac[3], si, ai;
    float e = elevationDeg;

    if (e <= E_NIGHT) {
        sc[0]=sunC[0][0]; sc[1]=sunC[0][1]; sc[2]=sunC[0][2]; si=sunI[0];
        ac[0]=ambC[0][0]; ac[1]=ambC[0][1]; ac[2]=ambC[0][2]; ai=ambI[0];
    } else if (e <= E_HORIZON) {
        float t = ss(E_NIGHT, E_HORIZON, e);
        lerp3(sc, sunC[0], sunC[1], t); si = sunI[0] + (sunI[1]-sunI[0])*t;
        lerp3(ac, ambC[0], ambC[1], t); ai = ambI[0] + (ambI[1]-ambI[0])*t;
    } else if (e <= E_DAWN) {
        float t = ss(E_HORIZON, E_DAWN, e);
        lerp3(sc, sunC[1], sunC[2], t); si = sunI[1] + (sunI[2]-sunI[1])*t;
        lerp3(ac, ambC[1], ambC[2], t); ai = ambI[1] + (ambI[2]-ambI[1])*t;
    } else if (e <= E_MORNING) {
        float t = ss(E_DAWN, E_MORNING, e);
        lerp3(sc, sunC[2], sunC[3], t); si = sunI[2] + (sunI[3]-sunI[2])*t;
        lerp3(ac, ambC[2], ambC[3], t); ai = ambI[2] + (ambI[3]-ambI[2])*t;
    } else if (e <= E_DAY) {
        float t = ss(E_MORNING, E_DAY, e);
        lerp3(sc, sunC[3], sunC[4], t); si = sunI[3] + (sunI[4]-sunI[3])*t;
        lerp3(ac, ambC[3], ambC[4], t); ai = ambI[3] + (ambI[4]-ambI[3])*t;
    } else {
        sc[0]=sunC[4][0]; sc[1]=sunC[4][1]; sc[2]=sunC[4][2]; si=sunI[4];
        ac[0]=ambC[4][0]; ac[1]=ambC[4][1]; ac[2]=ambC[4][2]; ai=ambI[4];
    }

    cfg->sunColorR = sc[0]; cfg->sunColorG = sc[1]; cfg->sunColorB = sc[2];
    cfg->sunIntensity = si;
    cfg->ambientColorR = ac[0]; cfg->ambientColorG = ac[1]; cfg->ambientColorB = ac[2];
    cfg->ambientIntensity = ai;
}

// ============================================================================
// C API implementation
// ============================================================================

extern "C" {

IGNIS_API void ignis_set_base_path(const char* path) {
    if (path && path[0]) {
        g_basePath = path;
        // Ensure trailing slash
        if (g_basePath.back() != '/' && g_basePath.back() != '\\')
            g_basePath += '/';
        Log(L"[Ignis] Base path set: %S\n", g_basePath.c_str());
    } else {
        g_basePath.clear();
    }
}

IGNIS_API void ignis_set_log_path(const char* path) {
    SetLogPath(path);
    if (path && path[0])
        Log(L"[Ignis] Log path set: %S\n", path);
}

IGNIS_API bool ignis_create(uint32_t width, uint32_t height) {
    // Enforce minimum 1920px horizontal resolution — preserve aspect ratio
    const uint32_t MIN_WIDTH = 1920;
    if (width < MIN_WIDTH) {
        float aspect = (float)height / (float)width;
        width = MIN_WIDTH;
        height = (uint32_t)(MIN_WIDTH * aspect);
        // Align to 2 pixels (DLSS requirement)
        height = (height + 1) & ~1;
    }
    Log(L"[Ignis] ignis_create(%u, %u) called\n", width, height);
    if (g_renderer) {
        Log(L"[Ignis] WARNING: Renderer already created\n");
        return true;
    }

    // Create a hidden window for Vulkan surface (required by swapchain)
    g_hiddenWindow = CreateHiddenWindow();
    if (!g_hiddenWindow) {
        return false;
    }

    g_renderer = new acpt::vk::Renderer();
    if (!g_renderer->Initialize(g_hiddenWindow, width, height)) {
        delete g_renderer;
        g_renderer = nullptr;
        DestroyHiddenWindow();
        return false;
    }

    // Log NRD config at creation time
    acpt::PathTracerConfig* cfg = &acpt::g_config;
    Log(L"[Ignis] Created OK. NRD config: maxAccum=%.1f fastAccum=%.1f disocclusion=%.4f\n",
        cfg->nrdMaxAccumFrames, cfg->nrdFastAccumFrames, cfg->nrdDisocclusionThreshold);

    return true;
}

IGNIS_API const char* ignis_create_step(uint32_t width, uint32_t height) {
    const uint32_t MIN_WIDTH = 1920;
    if (width < MIN_WIDTH) {
        float aspect = (float)height / (float)width;
        width = MIN_WIDTH;
        height = (uint32_t)(MIN_WIDTH * aspect);
        height = (height + 1) & ~1;
    }

    if (!g_renderer) {
        g_hiddenWindow = CreateHiddenWindow();
        if (!g_hiddenWindow) return nullptr;
        g_renderer = new acpt::vk::Renderer();
        Log(L"[Ignis] ignis_create_step(%u, %u) — starting phased init\n", width, height);
    }

    const char* stepName = g_renderer->InitializeStep(g_hiddenWindow, width, height);
    if (stepName) {
        Log(L"[Ignis] Init step: %S\n", stepName);
        return stepName;
    }

    // nullptr = all steps complete
    if (g_renderer->GetInitStep() >= 5) {
        Log(L"[Ignis] Phased init complete\n");
        return nullptr;
    }

    // Error
    Log(L"[Ignis] Phased init failed at step %d\n", g_renderer->GetInitStep());
    delete g_renderer;
    g_renderer = nullptr;
    DestroyHiddenWindow();
    return nullptr;
}

void ignis_reset_prev_frame();  // forward declaration

IGNIS_API void ignis_destroy(void) {
    Log(L"[Ignis] ignis_destroy() called\n");
    if (g_renderer) {
        g_renderer->Shutdown();
        delete g_renderer;
        g_renderer = nullptr;
    }
    DestroyHiddenWindow();
    ignis_reset_prev_frame();
    Log(L"[Ignis] ignis_destroy() done\n");
}

IGNIS_API void ignis_clear_geometry() {
    if (g_renderer) g_renderer->ClearGeometry();
}

IGNIS_API int ignis_upload_mesh(const float* vertices, uint32_t vertexCount,
                                const uint32_t* indices, uint32_t indexCount) {
    if (!g_renderer) return -1;
    return g_renderer->BuildBLAS(vertices, vertexCount, indices, indexCount);
}

IGNIS_API bool ignis_refit_blas(int blasHandle, const float* vertices, uint32_t vertexCount,
                                const uint32_t* indices, uint32_t indexCount) {
    if (!g_renderer || !vertices || !indices) return false;
    return g_renderer->RefitBLAS(blasHandle, vertices, vertexCount, indices, indexCount);
}

IGNIS_API void* ignis_map_blas_deform_staging(int blasIndex, uint32_t vertexCount) {
    if (!g_renderer) return nullptr;
    return g_renderer->GetAccelBuilder()->MapBLASDeformStaging(blasIndex, vertexCount);
}

IGNIS_API bool ignis_commit_blas_deform(int blasIndex) {
    if (!g_renderer) return false;
    return g_renderer->GetAccelBuilder()->CommitBLASDeform(blasIndex);
}

IGNIS_API bool ignis_upload_mesh_attributes(int blasHandle,
                                            const float* normals, const float* uvs,
                                            uint32_t vertexCount,
                                            const float* colors) {
    if (!g_renderer) return false;
    return g_renderer->UploadBLASAttributes(blasHandle, normals, uvs, vertexCount, colors);
}

IGNIS_API bool ignis_upload_mesh_primitive_materials(int blasHandle,
                                                     const uint32_t* materialIds,
                                                     uint32_t primitiveCount) {
    if (!g_renderer) return false;
    return g_renderer->UploadBLASPrimitiveMaterials(blasHandle, materialIds, primitiveCount);
}

IGNIS_API void ignis_upload_materials(const void* data, uint32_t count) {
    if (g_renderer) g_renderer->UploadMaterialBuffer(data, count);
}

IGNIS_API bool ignis_build_tlas(const void* instances, uint32_t count) {
    if (!g_renderer || !instances || count == 0) return false;

    struct IgnisTLASInstance {
        int blasIndex;
        float transform[12];
        uint32_t customIndex;
        uint32_t mask;
    };

    auto* src = reinterpret_cast<const IgnisTLASInstance*>(instances);
    std::vector<acpt::vk::TLASInstance> vkInstances(count);
    for (uint32_t i = 0; i < count; i++) {
        vkInstances[i].blasIndex = src[i].blasIndex;
        memcpy(vkInstances[i].transform, src[i].transform, sizeof(float) * 12);
        vkInstances[i].customIndex = src[i].customIndex;
        vkInstances[i].mask = src[i].mask;
    }
    return g_renderer->BuildTLASInstanced(vkInstances);
}

IGNIS_API bool ignis_update_instance_transforms(const uint32_t* indices,
                                                 const float* transforms,
                                                 uint32_t count) {
    if (!g_renderer || !indices || !transforms || count == 0) return false;
    return g_renderer->UpdateInstanceTransforms(indices, transforms, count);
}

// ── GPU Hair Generation ──
IGNIS_API int ignis_generate_hair_gpu(const float* parentKeys, uint32_t nParents,
                                       uint32_t keysPerStrand, uint32_t childrenPerParent,
                                       const float* emitterVerts, uint32_t nEmitterVerts,
                                       const uint32_t* emitterTris, uint32_t nEmitterTris,
                                       const float* emitterCDF,
                                       float rootRadius, float tipFactor,
                                       float camX, float camY, float camZ,
                                       float avgSpacing,
                                       float kinkAmplitude, float kinkFrequency,
                                       float clumpFactor, float clumpShape,
                                       float rough1, float rough1Size,
                                       float rough2, float roughEnd) {
    if (!g_renderer || !parentKeys || nParents == 0) return -1;
    return g_renderer->GenerateHairGPU(parentKeys, nParents, keysPerStrand,
                                        childrenPerParent,
                                        emitterVerts, nEmitterVerts,
                                        emitterTris, nEmitterTris, emitterCDF,
                                        rootRadius, tipFactor,
                                        camX, camY, camZ, avgSpacing,
                                        kinkAmplitude, kinkFrequency,
                                        clumpFactor, clumpShape,
                                        rough1, rough1Size, rough2, roughEnd);
}

// Previous frame camera matrices for motion vectors
static float s_viewPrev[16] = {0};
static float s_projPrev[16] = {0};
static bool s_hasPrevFrame = false;

// Point/spot lights (expanded for light tree — up to 256 lights)
static float s_lightData[4096] = {0};  // 256 lights × 16 floats
static uint32_t s_lightCount = 0;

static std::vector<acpt::LightTreeNode> s_lightTreeNodes;
static std::vector<acpt::LightEmitter> s_lightEmitters;

// Emissive triangles for MIS
static float s_emissiveTriData[4096] = {0};  // 256 triangles × 16 floats
static uint32_t s_emissiveTriCount = 0;

void ignis_reset_prev_frame() {
    s_hasPrevFrame = false;
}

IGNIS_API void ignis_set_camera(const float* viewInverse, const float* projInverse,
                                const float* view, const float* proj,
                                uint32_t frameIndex) {
    if (!g_renderer) return;

    acpt::vk::CameraUBO cam{};
    if (viewInverse) memcpy(cam.viewInverse, viewInverse, 64);
    if (projInverse) memcpy(cam.projInverse, projInverse, 64);
    if (view) memcpy(cam.view, view, 64);
    if (proj) memcpy(cam.proj, proj, 64);

    // Previous frame matrices for motion vectors
    if (s_hasPrevFrame) {
        memcpy(cam.viewPrev, s_viewPrev, 64);
        memcpy(cam.projPrev, s_projPrev, 64);
    } else {
        if (view) memcpy(cam.viewPrev, view, 64);
        if (proj) memcpy(cam.projPrev, proj, 64);
    }

    // DLSS jitter
    if (g_renderer->IsDLSSActive()) {
        uint32_t rw, rh;
        g_renderer->GetRenderResolution(&rw, &rh);

        float jitterX = HaltonSeq((frameIndex % 256) + 1, 2) - 0.5f;
        float jitterY = HaltonSeq((frameIndex % 256) + 1, 3) - 0.5f;

        // Pass pixel-space jitter to renderer for DLSS SR/RR
        // Y is negated to match Vulkan's flipped Y convention (same as NDC transform)
        cam.jitterData[0] = jitterX;
        cam.jitterData[1] = -jitterY;

        float jitterNDC_X = 2.0f * jitterX / (float)rw;
        float jitterNDC_Y = -2.0f * jitterY / (float)rh;
        cam.projInverse[12] += cam.projInverse[0] * jitterNDC_X;
        cam.projInverse[13] += cam.projInverse[5] * jitterNDC_Y;
    }

    // Apply config lighting
    acpt::PathTracerConfig* cfg = &acpt::g_config;

    if (cfg->autoSkyColors) {
        // Save Blender's sun intensity/color before atmospheric override
        float savedSunIntensity = cfg->sunIntensity;
        float savedSunColorR = cfg->sunColorR;
        float savedSunColorG = cfg->sunColorG;
        float savedSunColorB = cfg->sunColorB;

        if (savedSunIntensity > 0.0f) {
            // Sun light exists — compute atmospheric sky/ambient from elevation
            ComputeAtmosphericColors(cfg->sunElevation, cfg);
            // Restore Blender's sun intensity/color
            cfg->sunIntensity = savedSunIntensity;
            cfg->sunColorR = savedSunColorR;
            cfg->sunColorG = savedSunColorG;
            cfg->sunColorB = savedSunColorB;
        } else {
            // No sun light in scene — zero out atmospheric ambient
            cfg->ambientColorR = 0.0f; cfg->ambientColorG = 0.0f; cfg->ambientColorB = 0.0f;
            cfg->ambientIntensity = 0.0f;
        }
    }

    cam.parameters[0] = frameIndex;
    cam.parameters[1] = static_cast<uint32_t>(cfg->debugView);
    cam.parameters[2] = static_cast<uint32_t>(cfg->maxBounces);
    cam.parameters[3] = g_renderer->IsDLSSRRActive() ? 2u :
                         (g_renderer->IsDLSSActive() ? 1u : 0u);  // 0=LDR, 1=HDR(SR), 2=RR

    float sunDir[3];
    ComputeSunDirection(cfg->sunAzimuth, cfg->sunElevation, sunDir);
    cam.sunLight[0] = sunDir[0]; cam.sunLight[1] = sunDir[1]; cam.sunLight[2] = sunDir[2];
    cam.sunLight[3] = cfg->sunIntensity;

    cam.ambientLight[0] = cfg->ambientColorR; cam.ambientLight[1] = cfg->ambientColorG;
    cam.ambientLight[2] = cfg->ambientColorB; cam.ambientLight[3] = cfg->ambientIntensity;

    cam.skyLight[0] = cfg->sunColorR; cam.skyLight[1] = cfg->sunColorG;
    cam.skyLight[2] = cfg->sunColorB; cam.skyLight[3] = cfg->cloudVisibility * 1000.0f;

    cam.ptParams[0] = cfg->ptExposure;
    cam.ptParams[1] = cfg->ptGIIntensity;
    cam.ptParams[2] = cfg->ptSaturation;
    cam.ptParams[3] = cfg->ptContrast;
    cam.ptParams[4] = cfg->ptSkyReflIntensity;
    cam.ptParams[5] = cfg->ptAmbientMax;
    cam.ptParams[6] = cfg->ptSunMinIntensity;
    cam.ptParams[7] = cfg->ptSkyBounceIntensity;

    cam.jitterData[2] = static_cast<float>(cfg->ptTonemapMode);
    cam.jitterData[3] = g_renderer->IsDLSSActive() ? 1.0f : 0.0f;

    // Wind params (default gentle breeze)
    static auto s_windStartTime = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(now - s_windStartTime).count();
    cam.windParams[0] = 0.707f;
    cam.windParams[1] = 0.707f;
    cam.windParams[2] = 0.8f;
    cam.windParams[3] = elapsed;

    // Rain parameters
    cam.rainParams[0] = cfg->rainEnabled ? cfg->rainWetness : 0.0f;
    cam.rainParams[1] = cfg->rainEnabled ? cfg->rainWaterLevel : 0.0f;
    cam.rainParams[2] = cfg->rainEnabled ? cfg->rainIntensity : 0.0f;
    cam.rainParams[3] = 0.0f;

    // Point/spot lights
    cam.lightCount = s_lightCount;
    cam.lightPad[0] = s_emissiveTriCount;
    cam.lightPad[1] = static_cast<uint32_t>(acpt::g_config.samplesPerPixel);
    cam.lightPad[2] = (cfg->backfaceCulling ? 1u : 0u) | (cfg->restirDI ? 2u : 0u);
    // HDRI environment map: pack index and strength into windParams
    cam.windParams[0] = static_cast<float>(cfg->hdriTexIndex);  // -1 = no HDRI
    cam.windParams[1] = cfg->hdriStrength;
    // World background color (from Blender's World → Surface → Background)
    // Pack as RGB into windParams[2] and rainParams[3]
    cam.windParams[2] = cfg->worldBgR;
    cam.windParams[3] = cfg->worldBgG;
    cam.rainParams[3] = cfg->worldBgB;
    if (s_lightCount > 0) {
        uint32_t uboLights = (s_lightCount > 32) ? 32 : s_lightCount;
        cam.lightCount = uboLights;
        memcpy(cam.lights, s_lightData, uboLights * 16 * sizeof(float));
    }

    // SHARC radiance cache: pass buffer device addresses + grid parameters
    if (g_renderer && g_renderer->GetRTPipeline() && g_renderer->GetRTPipeline()->HasSHARCBuffers()) {
        auto* rtp = g_renderer->GetRTPipeline();
        cam.sharcHashEntriesAddr = rtp->GetSHARCHashEntriesAddr();
        cam.sharcAccumulationAddr = rtp->GetSHARCAccumulationAddr();
        cam.sharcResolvedAddr = rtp->GetSHARCResolvedAddr();
        cam.sharcCapacity = rtp->SHARC_CAPACITY;
        cam.sharcSceneScale = 1.0f;     // meters — adjust for scene scale
        cam.sharcRadianceScale = 1e3f;   // quantization for uint accumulation
    }

    // Scene AABB for early sky-ray rejection (packed into jitterPattern[0..7])
    // jitterPattern[0].xyz = sceneMin, jitterPattern[1].xyz = sceneMax (as float bits in int32)
    memcpy(&cam.jitterPattern[0], cfg->sceneAABBMin, 3 * sizeof(float));
    memcpy(&cam.jitterPattern[4], cfg->sceneAABBMax, 3 * sizeof(float));

    // Sky Texture atmosphere properties (packed into jitterPattern[8..13] as float bits)
    // Shader reads via intBitsToFloat(cam.jitterPattern[2])
    float skyAtmo[6] = {
        cfg->sunSize,           // [2].x — sun angular diameter (radians)
        cfg->sunDiscIntensity,  // [2].y — sun disc brightness multiplier
        cfg->airDensity,        // [2].z — Rayleigh scattering
        cfg->dustDensity,       // [2].w — Mie scattering
        cfg->ozoneDensity,      // [3].x — ozone absorption
        cfg->altitude           // [3].y — camera altitude (meters)
    };
    memcpy(&cam.jitterPattern[8], skyAtmo, 6 * sizeof(float));

    g_renderer->UpdateCamera(cam);

    // Store current frame as prev for next frame
    if (view) memcpy(s_viewPrev, view, 64);
    if (proj) memcpy(s_projPrev, proj, 64);
    s_hasPrevFrame = true;
}

IGNIS_API void ignis_upload_lights(const float* lightData, uint32_t lightCount) {
    s_lightCount = (lightCount > 256) ? 256 : lightCount;
    if (lightData && s_lightCount > 0) {
        memcpy(s_lightData, lightData, s_lightCount * 16 * sizeof(float));
    }

    // Build light tree from uploaded lights
    s_lightEmitters.clear();
    s_lightEmitters.resize(s_lightCount);
    for (uint32_t i = 0; i < s_lightCount; i++) {
        const float* ld = lightData + i * 16;
        acpt::LightEmitter& e = s_lightEmitters[i];
        e.position[0] = ld[0]; e.position[1] = ld[1]; e.position[2] = ld[2];
        e.range = ld[3];
        e.color[0] = ld[4]; e.color[1] = ld[5]; e.color[2] = ld[6];
        e.intensity = ld[7] * (ld[4] + ld[5] + ld[6]); // power = intensity × sum(RGB)
        e.direction[0] = ld[8]; e.direction[1] = ld[9]; e.direction[2] = ld[10];
        e.sizeX = ld[11];
        e.tangent[0] = ld[12]; e.tangent[1] = ld[13]; e.tangent[2] = ld[14];
        e.sizeY = ld[15];
        e.originalIndex = i;
    }
    if (s_lightCount > 0) {
        s_lightTreeNodes = acpt::BuildLightTree(s_lightEmitters);
    } else {
        s_lightTreeNodes.clear();
    }

    // Upload tree to renderer
    if (g_renderer && !s_lightTreeNodes.empty()) {
        g_renderer->UploadLightTree(s_lightTreeNodes.data(),
                                     (uint32_t)s_lightTreeNodes.size(),
                                     s_lightEmitters.data(),
                                     (uint32_t)s_lightEmitters.size());
    }
}

IGNIS_API void ignis_upload_emissive_triangles(const float* data, uint32_t triangleCount) {
    s_emissiveTriCount = (triangleCount > 256) ? 256 : triangleCount;
    if (data && s_emissiveTriCount > 0) {
        memcpy(s_emissiveTriData, data, s_emissiveTriCount * 16 * sizeof(float));
    }
    if (g_renderer) {
        g_renderer->UploadEmissiveTriangles(s_emissiveTriData, s_emissiveTriCount);
    }
    Log(L"[Ignis] Uploaded %u emissive triangles\n", s_emissiveTriCount);
}

IGNIS_API void ignis_render_frame(void) {
    if (!g_renderer) return;
    if (!g_renderer->IsRTReady()) {
        static int s_notReadyCount = 0;
        if (s_notReadyCount < 3) {
            Log(L"[Ignis] render_frame called but RT not ready (no TLAS?)\n");
            s_notReadyCount++;
        }
        return;
    }
    g_renderer->RenderFrameRT();
}

IGNIS_API bool ignis_readback(void* outPixels, uint32_t bufferSize) {
    return g_renderer ? g_renderer->ReadbackPixels(outPixels, bufferSize) : false;
}

IGNIS_API bool ignis_readback_float(float* outPixels, uint32_t pixelCount) {
    if (!g_renderer || !outPixels || pixelCount == 0) return false;
    uint32_t byteSize = pixelCount * 4;
    // Read RGBA8 into a temp stack/heap buffer, then convert to float in-place
    // Use the persistent readback path (already mapped after RenderFrameRT)
    thread_local std::vector<uint8_t> tempBuf;
    if (tempBuf.size() < byteSize) tempBuf.resize(byteSize);
    if (!g_renderer->ReadbackPixels(tempBuf.data(), byteSize)) return false;

    // RGBA8 → float32, unrolled for auto-vectorization (SIMD)
    const float scale = 1.0f / 255.0f;
    const uint8_t* src = tempBuf.data();
    uint32_t totalFloats = pixelCount * 4;
    for (uint32_t i = 0; i < totalFloats; i++) {
        outPixels[i] = src[i] * scale;
    }
    return true;
}

IGNIS_API void ignis_set_float(const char* key, float value) {
    acpt::PathTracerConfig* cfg = &acpt::g_config;
    if (!key) return;

    if      (strcmp(key, "exposure") == 0)           cfg->ptExposure = value;
    else if (strcmp(key, "saturation") == 0)         cfg->ptSaturation = value;
    else if (strcmp(key, "contrast") == 0)           cfg->ptContrast = value;
    else if (strcmp(key, "gi_intensity") == 0)       cfg->ptGIIntensity = value;
    else if (strcmp(key, "sun_azimuth") == 0)        cfg->sunAzimuth = value;
    else if (strcmp(key, "sun_elevation") == 0)      cfg->sunElevation = value;
    else if (strcmp(key, "sun_intensity") == 0)      cfg->sunIntensity = value;
    else if (strcmp(key, "ambient_intensity") == 0)  cfg->ambientIntensity = value;
    else if (strcmp(key, "cloud_visibility") == 0)   cfg->cloudVisibility = value;
    else if (strcmp(key, "sun_color_r") == 0)        cfg->sunColorR = value;
    else if (strcmp(key, "sun_color_g") == 0)        cfg->sunColorG = value;
    else if (strcmp(key, "sun_color_b") == 0)        cfg->sunColorB = value;
    else if (strcmp(key, "ambient_color_r") == 0)    cfg->ambientColorR = value;
    else if (strcmp(key, "ambient_color_g") == 0)    cfg->ambientColorG = value;
    else if (strcmp(key, "ambient_color_b") == 0)    cfg->ambientColorB = value;
    else if (strcmp(key, "world_bg_r") == 0)          cfg->worldBgR = value;
    else if (strcmp(key, "world_bg_g") == 0)          cfg->worldBgG = value;
    else if (strcmp(key, "world_bg_b") == 0)          cfg->worldBgB = value;
    else if (strcmp(key, "sun_size") == 0)            cfg->sunSize = value;
    else if (strcmp(key, "sun_disc_intensity") == 0)  cfg->sunDiscIntensity = value;
    else if (strcmp(key, "air_density") == 0)         cfg->airDensity = value;
    else if (strcmp(key, "dust_density") == 0)        cfg->dustDensity = value;
    else if (strcmp(key, "ozone_density") == 0)       cfg->ozoneDensity = value;
    else if (strcmp(key, "altitude") == 0)            cfg->altitude = value;
    else if (strcmp(key, "scene_aabb_min_x") == 0)   cfg->sceneAABBMin[0] = value;
    else if (strcmp(key, "scene_aabb_min_y") == 0)   cfg->sceneAABBMin[1] = value;
    else if (strcmp(key, "scene_aabb_min_z") == 0)   cfg->sceneAABBMin[2] = value;
    else if (strcmp(key, "scene_aabb_max_x") == 0)   cfg->sceneAABBMax[0] = value;
    else if (strcmp(key, "scene_aabb_max_y") == 0)   cfg->sceneAABBMax[1] = value;
    else if (strcmp(key, "scene_aabb_max_z") == 0)   cfg->sceneAABBMax[2] = value;
    else if (strcmp(key, "sky_refl_intensity") == 0) cfg->ptSkyReflIntensity = value;
    else if (strcmp(key, "sky_bounce_intensity") == 0) cfg->ptSkyBounceIntensity = value;
    else if (strcmp(key, "auto_exposure_key") == 0)  cfg->ptAutoExposureKey = value;
    else if (strcmp(key, "auto_exposure_speed") == 0) cfg->ptAutoExposureSpeed = value;
    else if (strcmp(key, "auto_exposure_min") == 0)  cfg->ptAutoExposureMin = value;
    else if (strcmp(key, "auto_exposure_max") == 0)  cfg->ptAutoExposureMax = value;
    else if (strcmp(key, "nrd_max_accum") == 0)      cfg->nrdMaxAccumFrames = value;
    else if (strcmp(key, "nrd_fast_accum") == 0)      cfg->nrdFastAccumFrames = value;
    else if (strcmp(key, "nrd_disocclusion") == 0)   cfg->nrdDisocclusionThreshold = value;
    else if (strcmp(key, "nrd_diffuse_prepass_blur") == 0)  cfg->nrdDiffusePrepassBlur = value;
    else if (strcmp(key, "nrd_specular_prepass_blur") == 0) cfg->nrdSpecularPrepassBlur = value;
    else if (strcmp(key, "nrd_depth_threshold") == 0)       cfg->nrdDepthThreshold = value;
    else if (strcmp(key, "nrd_diffuse_phi_luminance") == 0) cfg->nrdDiffusePhiLuminance = value;
    else if (strcmp(key, "nrd_specular_phi_luminance") == 0) cfg->nrdSpecularPhiLuminance = value;
    else if (strcmp(key, "nrd_lobe_angle_fraction") == 0)   cfg->nrdLobeAngleFraction = value;
    else if (strcmp(key, "nrd_roughness_fraction") == 0)    cfg->nrdRoughnessFraction = value;
    else if (strcmp(key, "nrd_min_hit_dist_weight") == 0)   cfg->nrdMinHitDistanceWeight = value;
    else if (strcmp(key, "hdri_strength") == 0)             cfg->hdriStrength = value;
}

IGNIS_API void ignis_set_int(const char* key, int value) {
    acpt::PathTracerConfig* cfg = &acpt::g_config;
    if (!key) return;

    if      (strcmp(key, "tonemap_mode") == 0)     cfg->ptTonemapMode = value;
    else if (strcmp(key, "tonemap_lut") == 0)       cfg->tonemapLutId = value;
    else if (strcmp(key, "debug_view") == 0)        cfg->debugView = value;
    else if (strcmp(key, "auto_exposure") == 0)     cfg->ptAutoExposure = (value != 0);
    else if (strcmp(key, "auto_sky_colors") == 0)   cfg->autoSkyColors = (value != 0);
    else if (strcmp(key, "dlss_enabled") == 0)      cfg->dlssEnabled = (value != 0);
    else if (strcmp(key, "dlss_quality") == 0)      cfg->dlssQualityMode = value;
    else if (strcmp(key, "dlss_rr_enabled") == 0)   cfg->dlssRREnabled = (value != 0);
    else if (strcmp(key, "nrd_enabled") == 0)       cfg->nrdEnabled = (value != 0);
    else if (strcmp(key, "nrd_atrous_iterations") == 0) cfg->nrdAtrousIterations = (value < 2 ? 2 : (value > 8 ? 8 : value));
    else if (strcmp(key, "nrd_anti_firefly") == 0)  cfg->nrdAntiFirefly = (value != 0);
    else if (strcmp(key, "nrd_history_fix_frames") == 0) cfg->nrdHistoryFixFrameNum = (value < 0 ? 0 : (value > 6 ? 6 : value));
    else if (strcmp(key, "max_bounces") == 0)       cfg->maxBounces = (value < 1 ? 1 : (value > 8 ? 8 : value));
    else if (strcmp(key, "spp") == 0)              cfg->samplesPerPixel = (value < 1 ? 1 : (value > 128 ? 128 : value));
    else if (strcmp(key, "shader_mode") == 0)       cfg->shaderMode = value;
    else if (strcmp(key, "use_wavefront") == 0)    cfg->useWavefront = (value != 0);
    else if (strcmp(key, "backface_culling") == 0) cfg->backfaceCulling = (value != 0);
    else if (strcmp(key, "restir_di") == 0)      cfg->restirDI = (value != 0);
    else if (strcmp(key, "hdri_tex_index") == 0)  cfg->hdriTexIndex = value;
    else if (strcmp(key, "reset_history") == 0 && value != 0) {
        if (g_renderer) g_renderer->ResetFrameIndex();
    }
}

IGNIS_API int ignis_get_int(const char* key) {
    if (!key) return 0;

    // Renderer state queries
    if (strcmp(key, "dlss_active") == 0)      return g_renderer && g_renderer->IsDLSSActive() ? 1 : 0;
    if (strcmp(key, "dlss_rr_active") == 0)    return g_renderer && g_renderer->IsDLSSRRActive() ? 1 : 0;
    if (strcmp(key, "nrd_active") == 0)        return g_renderer && g_renderer->IsDLSSActive() && !g_renderer->IsDLSSRRActive() ? 1 : 0;

    // Config queries
    acpt::PathTracerConfig* cfg = &acpt::g_config;
    if (strcmp(key, "dlss_enabled") == 0)      return cfg->dlssEnabled ? 1 : 0;
    if (strcmp(key, "dlss_rr_enabled") == 0)   return cfg->dlssRREnabled ? 1 : 0;
    if (strcmp(key, "nrd_enabled") == 0)       return cfg->nrdEnabled ? 1 : 0;
    if (strcmp(key, "max_bounces") == 0)       return cfg->maxBounces;
    if (strcmp(key, "debug_view") == 0)        return cfg->debugView;
    if (strcmp(key, "shader_mode") == 0)       return cfg->shaderMode;
    if (strcmp(key, "use_wavefront") == 0)    return cfg->useWavefront ? 1 : 0;

    // Render size queries (actual Vulkan image size, may differ from viewport)
    if (strcmp(key, "render_width") == 0)  return g_renderer ? (int)g_renderer->GetRenderWidth() : 0;
    if (strcmp(key, "render_height") == 0) return g_renderer ? (int)g_renderer->GetRenderHeight() : 0;

    // Actual DLSS quality mode used (may differ from requested if GPU doesn't support it)
    if (strcmp(key, "dlss_quality_actual") == 0) return g_renderer ? g_renderer->GetActualDLSSQuality() : 0;

    return 0;
}

IGNIS_API void* ignis_create_texture_manager(void) {
    if (!g_renderer || !g_renderer->GetContext()) return nullptr;
    auto* mgr = new acpt::vk::TextureManager();
    if (!mgr->Init(g_renderer->GetContext())) {
        delete mgr;
        return nullptr;
    }
    return mgr;
}

IGNIS_API void ignis_destroy_texture_manager(void* mgr) {
    if (mgr) {
        auto* texMgr = static_cast<acpt::vk::TextureManager*>(mgr);
        texMgr->Shutdown();
        delete texMgr;
    }
}

IGNIS_API int ignis_texture_manager_add(void* mgr, const char* name,
                                         const uint8_t* data, uint32_t dataSize,
                                         int width, int height, int mipLevels,
                                         uint32_t dxgiFormat) {
    if (!mgr) return -1;
    acpt::KN5Texture tex;
    tex.name = name ? name : "";
    tex.data.assign(data, data + dataSize);
    tex.width = width;
    tex.height = height;
    tex.mipLevels = mipLevels;
    tex.dxgiFormat = dxgiFormat;
    return static_cast<acpt::vk::TextureManager*>(mgr)->AddTexture(tex);
}

IGNIS_API bool ignis_texture_manager_upload_all(void* mgr) {
    if (!mgr) return false;
    return static_cast<acpt::vk::TextureManager*>(mgr)->UploadAll();
}

IGNIS_API bool ignis_texture_manager_upload_one(void* mgr) {
    if (!mgr) return false;
    return static_cast<acpt::vk::TextureManager*>(mgr)->UploadOne();
}

IGNIS_API int ignis_texture_manager_pending_count(void* mgr) {
    if (!mgr) return 0;
    return static_cast<acpt::vk::TextureManager*>(mgr)->GetPendingUploadCount();
}

IGNIS_API void ignis_update_texture_descriptors(void* mgr) {
    if (g_renderer && mgr) g_renderer->UpdateTextureDescriptors(mgr);
}

IGNIS_API bool ignis_draw_gl(uint32_t viewportWidth, uint32_t viewportHeight) {
    if (!g_renderer) return false;
    if (!g_renderer->InitGLInterop()) return false;
    g_renderer->DrawGL(viewportWidth, viewportHeight);
    return true;
}

IGNIS_API bool ignis_read_pick_result(uint32_t* outCustomIndex,
                                       uint32_t* outPrimitiveId,
                                       uint32_t* outMaterialId) {
    if (!g_renderer || !outCustomIndex || !outPrimitiveId || !outMaterialId) return false;
    return g_renderer->ReadPickResult(*outCustomIndex, *outPrimitiveId, *outMaterialId);
}

IGNIS_API bool ignis_save_config(const char* path) {
    if (!path) return false;
    acpt::PathTracerConfig* cfg = &acpt::g_config;
    FILE* f = fopen(path, "w");
    if (!f) return false;

    fprintf(f, "[render]\n");
    fprintf(f, "exposure=%.4f\n", cfg->ptExposure);
    fprintf(f, "tonemap_mode=%d\n", cfg->ptTonemapMode);
    fprintf(f, "saturation=%.4f\n", cfg->ptSaturation);
    fprintf(f, "contrast=%.4f\n", cfg->ptContrast);
    fprintf(f, "max_bounces=%d\n", cfg->maxBounces);
    fprintf(f, "debug_view=%d\n", cfg->debugView);
    fprintf(f, "shader_mode=%d\n", cfg->shaderMode);

    fprintf(f, "\n[auto_exposure]\n");
    fprintf(f, "enabled=%d\n", cfg->ptAutoExposure ? 1 : 0);
    fprintf(f, "key=%.4f\n", cfg->ptAutoExposureKey);
    fprintf(f, "speed=%.4f\n", cfg->ptAutoExposureSpeed);
    fprintf(f, "min=%.4f\n", cfg->ptAutoExposureMin);
    fprintf(f, "max=%.4f\n", cfg->ptAutoExposureMax);

    fprintf(f, "\n[lighting]\n");
    fprintf(f, "gi_intensity=%.4f\n", cfg->ptGIIntensity);
    fprintf(f, "sky_refl_intensity=%.4f\n", cfg->ptSkyReflIntensity);
    fprintf(f, "ambient_max=%.4f\n", cfg->ptAmbientMax);
    fprintf(f, "sun_min_intensity=%.4f\n", cfg->ptSunMinIntensity);
    fprintf(f, "sky_bounce_intensity=%.4f\n", cfg->ptSkyBounceIntensity);

    fprintf(f, "\n[sun]\n");
    fprintf(f, "azimuth=%.4f\n", cfg->sunAzimuth);
    fprintf(f, "elevation=%.4f\n", cfg->sunElevation);
    fprintf(f, "intensity=%.4f\n", cfg->sunIntensity);
    fprintf(f, "color_r=%.4f\n", cfg->sunColorR);
    fprintf(f, "color_g=%.4f\n", cfg->sunColorG);
    fprintf(f, "color_b=%.4f\n", cfg->sunColorB);
    fprintf(f, "auto_sky_colors=%d\n", cfg->autoSkyColors ? 1 : 0);

    fprintf(f, "\n[ambient]\n");
    fprintf(f, "color_r=%.4f\n", cfg->ambientColorR);
    fprintf(f, "color_g=%.4f\n", cfg->ambientColorG);
    fprintf(f, "color_b=%.4f\n", cfg->ambientColorB);
    fprintf(f, "intensity=%.4f\n", cfg->ambientIntensity);
    fprintf(f, "visibility_km=%.4f\n", cfg->cloudVisibility);

    fprintf(f, "\n[nrd]\n");
    fprintf(f, "enabled=%d\n", cfg->nrdEnabled ? 1 : 0);
    fprintf(f, "max_accum_frames=%.1f\n", cfg->nrdMaxAccumFrames);
    fprintf(f, "fast_accum_frames=%.1f\n", cfg->nrdFastAccumFrames);
    fprintf(f, "lobe_angle_fraction=%.4f\n", cfg->nrdLobeAngleFraction);
    fprintf(f, "roughness_fraction=%.4f\n", cfg->nrdRoughnessFraction);
    fprintf(f, "min_hit_dist_weight=%.4f\n", cfg->nrdMinHitDistanceWeight);
    fprintf(f, "disocclusion_threshold=%.4f\n", cfg->nrdDisocclusionThreshold);
    fprintf(f, "diffuse_prepass_blur=%.1f\n", cfg->nrdDiffusePrepassBlur);
    fprintf(f, "specular_prepass_blur=%.1f\n", cfg->nrdSpecularPrepassBlur);
    fprintf(f, "atrous_iterations=%d\n", cfg->nrdAtrousIterations);
    fprintf(f, "anti_firefly=%d\n", cfg->nrdAntiFirefly ? 1 : 0);
    fprintf(f, "history_fix_frames=%d\n", cfg->nrdHistoryFixFrameNum);
    fprintf(f, "depth_threshold=%.4f\n", cfg->nrdDepthThreshold);
    fprintf(f, "diffuse_phi_luminance=%.4f\n", cfg->nrdDiffusePhiLuminance);
    fprintf(f, "specular_phi_luminance=%.4f\n", cfg->nrdSpecularPhiLuminance);

    fprintf(f, "\n[dlss]\n");
    fprintf(f, "enabled=%d\n", cfg->dlssEnabled ? 1 : 0);
    fprintf(f, "quality_mode=%d\n", cfg->dlssQualityMode);
    fprintf(f, "dlss_rr_enabled=%d\n", cfg->dlssRREnabled ? 1 : 0);

    fclose(f);
    Log(L"[Ignis] Config saved to %S\n", path);
    return true;
}

IGNIS_API bool ignis_load_config(const char* path) {
    if (!path) return false;
    FILE* f = fopen(path, "r");
    if (!f) return false;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        // Skip comments and section headers
        if (line[0] == '#' || line[0] == ';' || line[0] == '[' || line[0] == '\n') continue;

        char key[128];
        char val[128];
        if (sscanf(line, "%127[^=]=%127s", key, val) == 2) {
            // Try float first, then int
            float fval = (float)atof(val);
            int ival = atoi(val);

            // Route through existing set_float/set_int for consistency
            ignis_set_float(key, fval);
            ignis_set_int(key, ival);
        }
    }
    fclose(f);
    Log(L"[Ignis] Config loaded from %S\n", path);
    return true;
}

} // extern "C"
