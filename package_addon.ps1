# package_addon.ps1 - Build ignis_rt.dll and create a portable addon zip
#
# Usage:
#   .\package_addon.ps1              # Build + package
#   .\package_addon.ps1 -NoBuild     # Skip build, just package from existing files
#   .\package_addon.ps1 -OutDir C:\tmp  # Output zip to specific directory
#
# The resulting zip can be installed directly in Blender:
#   Edit > Preferences > Add-ons > Install from Disk > select zip

param(
    [switch]$NoBuild,
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

$repoRoot   = $PSScriptRoot
$buildDir   = Join-Path $repoRoot "build"
$addonSrc   = Join-Path $repoRoot "blender\ignis_rt"
$dllRelease = Join-Path $buildDir "Release\ignis_rt.dll"

# Output location
if ($OutDir -eq "") { $OutDir = $repoRoot }
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zipName   = "ignis_rt_$timestamp.zip"
$zipPath   = Join-Path $OutDir $zipName

# Staging directory (temp)
$staging = Join-Path $env:TEMP "ignis_rt_package_$timestamp"
$stagingAddon = Join-Path $staging "ignis_rt"

Write-Host "=== Ignis RT Addon Packager ===" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# 1. Build (unless -NoBuild)
# ============================================================================
if (-not $NoBuild) {
    Write-Host "[1/4] Building ignis_rt.dll ..." -ForegroundColor Yellow

    if (-not (Test-Path $buildDir)) {
        Write-Host "  Configuring CMake ..."
        cmake -S $repoRoot -B $buildDir
        if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: CMake configure failed" -ForegroundColor Red; exit 1 }
    }

    cmake --build $buildDir --config Release
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Build OK" -ForegroundColor Green
} else {
    Write-Host "[1/4] Skipping build (-NoBuild)" -ForegroundColor DarkGray
}

# ============================================================================
# 2. Create staging directory
# ============================================================================
Write-Host "[2/4] Staging files ..." -ForegroundColor Yellow

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path $stagingAddon -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stagingAddon "lib") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stagingAddon "shaders") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stagingAddon "shaders\wavefront") -Force | Out-Null

# Python addon files
$pyCount = 0
Get-ChildItem $addonSrc -Filter "*.py" | ForEach-Object {
    Copy-Item $_.FullName -Destination $stagingAddon -Force; $pyCount++
}
Write-Host "  $pyCount Python files" -ForegroundColor Green

# Manifest
$manifest = Join-Path $addonSrc "blender_manifest.toml"
if (Test-Path $manifest) {
    Copy-Item $manifest -Destination $stagingAddon -Force
    Write-Host "  blender_manifest.toml" -ForegroundColor Green
}

# Icons
$icons = Join-Path $addonSrc "icons"
if (Test-Path $icons) {
    Copy-Item $icons -Destination (Join-Path $stagingAddon "icons") -Recurse -Force
    Write-Host "  icons/" -ForegroundColor Green
}

# ============================================================================
# 3. Copy DLLs
# ============================================================================
Write-Host "[3/4] Copying DLLs and shaders ..." -ForegroundColor Yellow
$libDest = Join-Path $stagingAddon "lib"

# Main DLL
if (Test-Path $dllRelease) {
    Copy-Item $dllRelease -Destination $libDest -Force
    $sz = [math]::Round((Get-Item $dllRelease).Length / 1MB, 1)
    Write-Host "  ignis_rt.dll ($sz MB)" -ForegroundColor Green
} else {
    Write-Host "  WARNING: ignis_rt.dll not found at $dllRelease" -ForegroundColor Red
}

# DLSS runtime DLLs (from addon lib/ — already deployed by previous builds)
$addonLib = Join-Path $addonSrc "lib"
foreach ($dll in @("nvngx_dlss.dll", "nvngx_dlssd.dll")) {
    $src = Join-Path $addonLib $dll
    if (Test-Path $src) {
        Copy-Item $src -Destination $libDest -Force
        Write-Host "  $dll" -ForegroundColor Green
    }
}

# Streamline DLLs (from SDK if available, then from addon lib/)
$slRoot = $env:IGNIS_STREAMLINE_ROOT
if (-not $slRoot) { $slRoot = "C:\Dev\streamline-sdk-v2.10.3" }
$slBin = Join-Path $slRoot "bin\x64"

foreach ($dll in @("sl.interposer.dll", "sl.common.dll", "sl.dlss_g.dll", "sl.reflex.dll", "sl.pcl.dll")) {
    $src = Join-Path $slBin $dll
    if (-not (Test-Path $src)) {
        # Fallback: check addon lib/
        $src = Join-Path $addonLib $dll
    }
    if (Test-Path $src) {
        Copy-Item $src -Destination $libDest -Force
        Write-Host "  $dll" -ForegroundColor Green
    } else {
        Write-Host "  $dll (not found - skipped)" -ForegroundColor DarkGray
    }
}

# NRD DLL (optional)
$nrdDll = ""
if ($env:IGNIS_NRD_ROOT -and (Test-Path "$env:IGNIS_NRD_ROOT\_Bin\Release\NRD.dll")) {
    $nrdDll = "$env:IGNIS_NRD_ROOT\_Bin\Release\NRD.dll"
} else {
    $nrdFallback = Join-Path (Split-Path $repoRoot -Parent) "NRD\_Bin\Release\NRD.dll"
    if (Test-Path $nrdFallback) { $nrdDll = $nrdFallback }
}
if ($nrdDll -and (Test-Path $nrdDll)) {
    Copy-Item $nrdDll -Destination $libDest -Force
    Write-Host "  NRD.dll" -ForegroundColor Green
}

# Compiled shaders
$shaderDest = Join-Path $stagingAddon "shaders"
$spvCount = 0
$shaderSrc = Join-Path $repoRoot "shaders"
Get-ChildItem $shaderSrc -Filter "*.spv" -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination $shaderDest -Force; $spvCount++
}
# LUT files
Get-ChildItem $shaderSrc -Filter "*.cube" -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName -Destination $shaderDest -Force
}
# Wavefront shaders
$wfSrc = Join-Path $shaderSrc "wavefront"
if (Test-Path $wfSrc) {
    $wfDest = Join-Path $shaderDest "wavefront"
    Get-ChildItem $wfSrc -Filter "*.spv" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName -Destination $wfDest -Force; $spvCount++
    }
}
Write-Host "  $spvCount shader SPVs" -ForegroundColor Green

# ============================================================================
# 4. Create zip
# ============================================================================
Write-Host "[4/4] Creating zip ..." -ForegroundColor Yellow

# List contents
$fileCount = (Get-ChildItem $stagingAddon -Recurse -File).Count
$totalSize = [math]::Round(((Get-ChildItem $stagingAddon -Recurse -File | Measure-Object -Property Length -Sum).Sum) / 1MB, 1)

# Use Compress-Archive (built-in)
Compress-Archive -Path $stagingAddon -DestinationPath $zipPath -Force
$zipSize = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)

# Cleanup staging
Remove-Item $staging -Recurse -Force

Write-Host ""
Write-Host "=== Package complete ===" -ForegroundColor Green
Write-Host "  Files     : $fileCount ($totalSize MB uncompressed)"
Write-Host "  Zip       : $zipPath ($zipSize MB)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Install in Blender:" -ForegroundColor Cyan
Write-Host "  Edit > Preferences > Add-ons > Install from Disk > select $zipName"
Write-Host ""
