cmake_minimum_required(VERSION 3.20)
project(ignis_rt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Options
# ============================================================================
option(IGNIS_USE_NRD   "Enable NRD denoiser"       OFF)
option(IGNIS_USE_DLSS  "Enable NVIDIA DLSS"         OFF)
option(IGNIS_ENABLE_VALIDATION "Enable Vulkan validation layers in debug builds" OFF)

set(IGNIS_NRD_ROOT "" CACHE PATH "Path to NVIDIA NRD root directory")
set(IGNIS_NGX_ROOT "" CACHE PATH "Path to NVIDIA NGX SDK (headers + lib)")

# ============================================================================
# Find Vulkan
# ============================================================================
find_package(Vulkan REQUIRED)
message(STATUS "Vulkan SDK: ${Vulkan_INCLUDE_DIRS}")

# ============================================================================
# GLM — try Vulkan SDK first, then FetchContent fallback
# ============================================================================
find_path(GLM_INCLUDE_DIR glm/glm.hpp
    HINTS
        "$ENV{VULKAN_SDK}/Include"
        "$ENV{VULKAN_SDK}/include"
        "${Vulkan_INCLUDE_DIRS}/.."
)

if(GLM_INCLUDE_DIR)
    message(STATUS "Found GLM: ${GLM_INCLUDE_DIR}")
else()
    message(STATUS "GLM not found locally — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.1
    )
    FetchContent_MakeAvailable(glm)
    set(GLM_INCLUDE_DIR ${glm_SOURCE_DIR})
endif()

# ============================================================================
# Core sources (always compiled)
# ============================================================================
set(IGNIS_CORE_SOURCES
    src/ignis_api.cpp
    src/ignis_log.cpp
    src/imgui_stub.cpp
    src/sky_nishita.cpp
    src/vk/vk_renderer.cpp
    src/vk/vk_context.cpp
    src/vk/vk_pipeline.cpp
    src/vk/vk_geometry.cpp
    src/vk/vk_rasterizer.cpp
    src/vk/vk_accel_structure.cpp
    src/vk/vk_rt_pipeline.cpp
    src/vk/vk_interop.cpp
    src/vk/vk_texture_manager.cpp
)

# ============================================================================
# NRD integration (always compiled — has built-in stubs when ACPT_HAVE_NRD is not defined)
# ============================================================================
list(APPEND IGNIS_CORE_SOURCES src/nrd_vulkan_integration.cpp)

# ============================================================================
# Optional: DLSS
# ============================================================================
if(IGNIS_USE_DLSS)
    list(APPEND IGNIS_CORE_SOURCES src/dlss_ngx.cpp)
else()
    list(APPEND IGNIS_CORE_SOURCES src/dlss_stub.cpp)
endif()


# ============================================================================
# Target: ignis_rt.dll
# ============================================================================
add_library(ignis_rt SHARED ${IGNIS_CORE_SOURCES})

target_include_directories(ignis_rt
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vk
        ${Vulkan_INCLUDE_DIRS}
        ${GLM_INCLUDE_DIR}
)

target_link_libraries(ignis_rt
    PRIVATE
        Vulkan::Vulkan
)

target_compile_definitions(ignis_rt
    PRIVATE
        IGNIS_RT_EXPORTS
        VK_USE_PLATFORM_WIN32_KHR
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
)

# Conditional compile definitions
if(IGNIS_USE_NRD)
    target_compile_definitions(ignis_rt PRIVATE ACPT_HAVE_NRD)
    if(EXISTS "${IGNIS_NRD_ROOT}/Include/NRD.h")
        message(STATUS "Using NRD at ${IGNIS_NRD_ROOT}")
        target_include_directories(ignis_rt PRIVATE "${IGNIS_NRD_ROOT}/Include")
        target_link_libraries(ignis_rt PRIVATE "${IGNIS_NRD_ROOT}/_Bin/Release/NRD.lib")
    else()
        message(WARNING "NRD not found at IGNIS_NRD_ROOT='${IGNIS_NRD_ROOT}' — set IGNIS_NRD_ROOT")
    endif()
endif()
if(IGNIS_USE_DLSS)
    target_compile_definitions(ignis_rt PRIVATE ACPT_HAVE_NGX)
    # Try NGX_ROOT
    set(_ngx_root "${IGNIS_NGX_ROOT}")
    if(NOT _ngx_root STREQUAL "" AND EXISTS "${_ngx_root}/include/ngx/nvsdk_ngx.h")
        message(STATUS "Using NGX SDK at ${_ngx_root}")
        target_include_directories(ignis_rt PRIVATE "${_ngx_root}/include")
        if(EXISTS "${_ngx_root}/lib/nvsdk_ngx_d.lib")
            target_link_libraries(ignis_rt PRIVATE "${_ngx_root}/lib/nvsdk_ngx_d.lib")
        endif()
    else()
        message(WARNING "NGX SDK not found — set IGNIS_NGX_ROOT")
    endif()
endif()
if(IGNIS_ENABLE_VALIDATION)
    target_compile_definitions(ignis_rt PRIVATE IGNIS_ENABLE_VALIDATION)
endif()

# ============================================================================
# Shader compilation (glslangValidator from Vulkan SDK)
# ============================================================================
if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")

    # Shader source:output pairs — output names must match C++ load paths exactly
    # Most shaders: source.ext → source.ext.spv
    # Exception: nrd_composite.comp → nrd_composite.spv (C++ loads without .comp)
    set(IGNIS_SHADER_SOURCES
        basic.vert:basic.vert.spv
        basic.frag:basic.frag.spv
        raygen.rgen:raygen.rgen.spv
        closesthit.rchit:closesthit.rchit.spv
        miss.rmiss:miss.rmiss.spv
        nrd_composite.comp:nrd_composite.spv
        exposure_resolve.comp:exposure_resolve.comp.spv
        tonemap.comp:tonemap.comp.spv
        sharc_resolve.comp:sharc_resolve.comp.spv
        cluster_culling.comp:cluster_culling.comp.spv
        hybrid.rgen:hybrid.rgen.spv
        raygen_blender.rgen:raygen_blender.rgen.spv
    )

    set(SPV_OUTPUTS "")
    foreach(SHADER_PAIR ${IGNIS_SHADER_SOURCES})
        string(REPLACE ":" ";" SHADER_LIST "${SHADER_PAIR}")
        list(GET SHADER_LIST 0 SHADER_SRC)
        list(GET SHADER_LIST 1 SHADER_SPV)

        add_custom_command(
            OUTPUT "${SHADER_DIR}/${SHADER_SPV}"
            COMMAND "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}" --target-env vulkan1.2 -V
                    -I"${SHADER_DIR}"
                    "${SHADER_DIR}/${SHADER_SRC}" -o "${SHADER_DIR}/${SHADER_SPV}"
            DEPENDS "${SHADER_DIR}/${SHADER_SRC}"
            COMMENT "Compiling ${SHADER_SRC} -> ${SHADER_SPV}"
        )
        list(APPEND SPV_OUTPUTS "${SHADER_DIR}/${SHADER_SPV}")
    endforeach()

    add_custom_target(ignis_shaders DEPENDS ${SPV_OUTPUTS})
    add_dependencies(ignis_rt ignis_shaders)

    message(STATUS "Shader compilation enabled (${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})")
else()
    message(WARNING "glslangValidator not found — shaders will NOT be compiled automatically")
endif()

# ============================================================================
# Install
# ============================================================================
install(TARGETS ignis_rt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    include/ignis_api.h
    include/ignis_config.h
    include/ignis_log.h
    include/ignis_texture.h
    DESTINATION include/ignis
)

# ============================================================================
# Test executable
# ============================================================================
add_executable(ignis_test tests/test_triangle.cpp)
target_link_libraries(ignis_test PRIVATE ignis_rt)
target_include_directories(ignis_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(ignis_test PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
)
