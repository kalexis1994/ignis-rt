@echo off
REM Ignis RT — Shader Compilation Script
REM Requires glslangValidator from the Vulkan SDK

setlocal

set GLSLC=glslangValidator
where %GLSLC% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: glslangValidator not found in PATH
    echo Install the Vulkan SDK or add it to PATH
    exit /b 1
)

set SHADER_DIR=%~dp0shaders
set ARGS=--target-env vulkan1.2 -V -Os -I"%SHADER_DIR%"

echo Compiling ray tracing shaders...
%GLSLC% %ARGS% "%SHADER_DIR%\raygen_blender.rgen"   -o "%SHADER_DIR%\raygen_blender.rgen.spv"
%GLSLC% %ARGS% "%SHADER_DIR%\hybrid.rgen"           -o "%SHADER_DIR%\hybrid.rgen.spv"
%GLSLC% %ARGS% "%SHADER_DIR%\closesthit.rchit"      -o "%SHADER_DIR%\closesthit.rchit.spv"
%GLSLC% %ARGS% "%SHADER_DIR%\miss.rmiss"            -o "%SHADER_DIR%\miss.rmiss.spv"

echo Compiling hybrid G-buffer shaders...
%GLSLC% %ARGS% "%SHADER_DIR%\gbuffer_hybrid.vert"   -o "%SHADER_DIR%\gbuffer_hybrid.vert.spv"
%GLSLC% %ARGS% "%SHADER_DIR%\gbuffer_hybrid.frag"   -o "%SHADER_DIR%\gbuffer_hybrid.frag.spv"

echo Compiling compute shaders...
%GLSLC% %ARGS% "%SHADER_DIR%\nrd_composite.comp"    -o "%SHADER_DIR%\nrd_composite.spv"
%GLSLC% %ARGS% "%SHADER_DIR%\tonemap.comp"          -o "%SHADER_DIR%\tonemap.comp.spv"
%GLSLC% %ARGS% "%SHADER_DIR%\exposure_resolve.comp" -o "%SHADER_DIR%\exposure_resolve.comp.spv"
%GLSLC% %ARGS% "%SHADER_DIR%\sharc_resolve.comp"    -o "%SHADER_DIR%\sharc_resolve.comp.spv"
%GLSLC% %ARGS% "%SHADER_DIR%\cluster_culling.comp"  -o "%SHADER_DIR%\cluster_culling.comp.spv"
%GLSLC% %ARGS% "%SHADER_DIR%\hair_generate.comp"    -o "%SHADER_DIR%\hair_generate.comp.spv"

echo Compiling rasterization shaders...
%GLSLC% %ARGS% "%SHADER_DIR%\basic.vert"            -o "%SHADER_DIR%\basic.vert.spv"
%GLSLC% %ARGS% "%SHADER_DIR%\basic.frag"            -o "%SHADER_DIR%\basic.frag.spv"

echo Done.
endlocal
