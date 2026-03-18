#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef IGNIS_RT_EXPORTS
#define IGNIS_API __declspec(dllexport)
#else
#define IGNIS_API __declspec(dllimport)
#endif

// Configuration (call before ignis_create)
IGNIS_API void   ignis_set_base_path(const char* path);   // shader/resource root dir
IGNIS_API void   ignis_set_log_path(const char* path);    // log file output path

// Lifecycle
IGNIS_API bool   ignis_create(uint32_t width, uint32_t height);
IGNIS_API void   ignis_destroy(void);

// Geometry upload
IGNIS_API int    ignis_upload_mesh(const float* vertices, uint32_t vertexCount,
                                   const uint32_t* indices, uint32_t indexCount);
IGNIS_API bool   ignis_upload_mesh_attributes(int blasHandle,
                                               const float* normals, const float* uvs,
                                               uint32_t vertexCount);
IGNIS_API bool   ignis_upload_mesh_primitive_materials(int blasHandle,
                                                       const uint32_t* materialIds,
                                                       uint32_t primitiveCount);

// Materials
IGNIS_API void   ignis_upload_materials(const void* data, uint32_t count);

// Acceleration structures
IGNIS_API bool   ignis_build_tlas(const void* instances, uint32_t count);

// Camera
IGNIS_API void   ignis_set_camera(const float* viewInverse, const float* projInverse,
                                   const float* view, const float* proj,
                                   uint32_t frameIndex);

// Lights (point/spot, max 8)
// Each light: 8 floats [posX, posY, posZ, range, colorR, colorG, colorB, intensity]
IGNIS_API void   ignis_upload_lights(const float* lightData, uint32_t lightCount);

// Emissive triangles for MIS (max 256)
// Each triangle: 16 floats [v0.xyz+area, v1.xyz+cdf, v2.xyz+totalPower, emission.rgb+matIdx]
IGNIS_API void   ignis_upload_emissive_triangles(const float* data, uint32_t triangleCount);

// Rendering
IGNIS_API void   ignis_render_frame(void);
IGNIS_API bool   ignis_readback(void* outPixels, uint32_t bufferSize);
IGNIS_API bool   ignis_readback_float(float* outPixels, uint32_t pixelCount);
IGNIS_API bool   ignis_draw_gl(uint32_t viewportWidth, uint32_t viewportHeight);

// Configuration
IGNIS_API void   ignis_set_float(const char* key, float value);
IGNIS_API void   ignis_set_int(const char* key, int value);

// Pick buffer (GPU raycast selection)
IGNIS_API bool   ignis_read_pick_result(uint32_t* outCustomIndex,
                                         uint32_t* outPrimitiveId,
                                         uint32_t* outMaterialId);

// Config save/load (INI-style key=value)
IGNIS_API bool   ignis_save_config(const char* path);
IGNIS_API bool   ignis_load_config(const char* path);

// Texture management
IGNIS_API void*  ignis_create_texture_manager(void);
IGNIS_API void   ignis_destroy_texture_manager(void* mgr);
IGNIS_API int    ignis_texture_manager_add(void* mgr, const char* name,
                                            const uint8_t* data, uint32_t dataSize,
                                            int width, int height, int mipLevels,
                                            uint32_t dxgiFormat);
IGNIS_API bool   ignis_texture_manager_upload_all(void* mgr);
IGNIS_API void   ignis_update_texture_descriptors(void* mgr);

#ifdef __cplusplus
}
#endif
