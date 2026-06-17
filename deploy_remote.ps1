# deploy_remote.ps1 - Build ignis_rt.dll locally and deploy the addon to a REMOTE
#                     Blender machine (RTX) over an SMB share.
#
# Compile happens on THIS machine. The remote PC only runs Blender + the addon.
# Transfer is a plain SMB copy (robocopy) to a UNC path or mapped drive.
# Blender restart on the remote is MANUAL (close Blender there before a DLL change).
#
# Setup (one time):
#   1. On the RTX PC, share the Blender addon folder (or use the admin C$ share).
#   2. Copy deploy.config.example -> deploy.config and set REMOTE_ADDON_DIR.
#      Example (extensions, Blender 5.x):
#        REMOTE_ADDON_DIR=\\PC-RTX\C$\Users\you\AppData\Roaming\Blender Foundation\Blender\5.1\extensions\user_default\ignis_rt
#      Or via a mapped drive:
#        REMOTE_ADDON_DIR=Z:\extensions\user_default\ignis_rt
#   3. Make sure Windows has the share credentials cached
#      (net use \\PC-RTX\C$ /user:PC-RTX\you *  ->  saves them).
#
# Usage:
#   .\deploy_remote.ps1                 # Build (Rust/cargo) + deploy everything (DLL + spv + .py)
#   .\deploy_remote.ps1 -NoBuild        # Skip build, deploy existing artifacts
#   .\deploy_remote.ps1 -PyOnly         # Fast: only copy .py (no build, DLL untouched)
#   .\deploy_remote.ps1 -Dest '\\PC-RTX\...\ignis_rt'   # Override config destination
#   .\deploy_remote.ps1 -Cpp            # Legacy: build the C++ DLL (cmake) for A/B reference
#
# Default build is RUST. The first Rust deploy backs up the remote C++ DLL to
# lib\ignis_rt.dll.cpp-bak (restore with -Cpp or by copying that file back).

param(
    [string]$Dest = "",
    [switch]$NoBuild,
    [switch]$PyOnly,
    [switch]$Cpp,      # Legacy: build the C++ DLL via cmake instead of Rust (A/B reference)
    [switch]$Rust      # Deprecated no-op: Rust is now the default
)

$ErrorActionPreference = "Stop"

$repoRoot   = $PSScriptRoot
$buildDir   = Join-Path $repoRoot "build"
$addonSrc   = Join-Path $repoRoot "blender\ignis_rt"
$dllRelease = Join-Path $buildDir "Release\ignis_rt.dll"
$libDir     = Join-Path $addonSrc "lib"
$rustDir    = Join-Path $repoRoot "rust"
# Rust is the default build; the C++ DLL is opt-in via -Cpp.
if (-not $Cpp) { $dllRelease = Join-Path $rustDir "target\release\ignis_rt.dll" }

# ============================================================================
# 0. Resolve remote destination (param overrides deploy.config)
# ============================================================================
if ($Dest -eq "") {
    $cfg = Join-Path $repoRoot "deploy.config"
    if (Test-Path $cfg) {
        Get-Content $cfg | ForEach-Object {
            $line = $_.Trim()
            if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
                $k, $v = $line.Split("=", 2)
                if ($k.Trim() -eq "REMOTE_ADDON_DIR") { $Dest = $v.Trim() }
            }
        }
    }
}

if ($Dest -eq "") {
    Write-Host "ERROR: No remote destination set." -ForegroundColor Red
    Write-Host "  Set REMOTE_ADDON_DIR in deploy.config, or pass -Dest '\\PC-RTX\...\ignis_rt'" -ForegroundColor DarkGray
    exit 1
}

Write-Host "=== Ignis RT Remote Deploy (SMB) ===" -ForegroundColor Cyan
Write-Host "  Destination : $Dest"
Write-Host ""

# Verify the share is reachable. Prefer testing the dest itself; only fall back to
# the parent for a first-time deploy where the subfolder doesn't exist yet. Skip the
# parent test when it's a bare \\server root (not Test-Path-able for top-level shares).
$destParent = Split-Path $Dest -Parent
$reachable = Test-Path $Dest
if (-not $reachable -and $destParent -and ($destParent -notmatch '^\\\\[^\\]+\\?$')) {
    $reachable = Test-Path $destParent
}
if (-not $reachable) {
    Write-Host "ERROR: Remote path not reachable: $Dest" -ForegroundColor Red
    Write-Host "  - Is the RTX PC on and the share mounted?" -ForegroundColor DarkGray
    Write-Host "  - Are credentials cached? Try: net use \\ALEXIS-WORK\ignis_rt /user:ALEXIS-WORK\you *" -ForegroundColor DarkGray
    exit 1
}

# ============================================================================
# 1. Build (unless skipped)
# ============================================================================
if ($PyOnly) {
    Write-Host "[1/3] PyOnly mode - skipping build" -ForegroundColor DarkGray
} elseif ($NoBuild) {
    Write-Host "[1/3] Skipping build (-NoBuild)" -ForegroundColor DarkGray
} elseif ($Cpp) {
    Write-Host "[1/3] Building ignis_rt.dll (cmake, legacy C++) ..." -ForegroundColor Yellow
    if (-not (Test-Path $buildDir)) {
        Write-Host "  Configuring CMake ..."
        cmake -S $repoRoot -B $buildDir
        if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: CMake configure failed" -ForegroundColor Red; exit 1 }
    }
    cmake --build $buildDir --config Release
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Build failed" -ForegroundColor Red; exit 1 }
    Write-Host "  Build OK (C++)" -ForegroundColor Green
} else {
    Write-Host "[1/3] Building ignis_rt.dll (cargo, Rust) ..." -ForegroundColor Yellow
    cargo build --release --manifest-path (Join-Path $rustDir "Cargo.toml") -p ignis_rt
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: cargo build failed" -ForegroundColor Red; exit 1 }
    Write-Host "  Build OK (Rust)" -ForegroundColor Green
}

# ============================================================================
# 2. Stage artifacts into the local addon folder (DLL -> lib/, .spv -> shaders/)
#    Skipped in -PyOnly: we only push .py in that mode.
# ============================================================================
if (-not $PyOnly) {
    Write-Host "[2/3] Staging artifacts into addon ..." -ForegroundColor Yellow

    if (-not (Test-Path $dllRelease)) {
        Write-Host "ERROR: $dllRelease not found. Build first (drop -NoBuild)." -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Path $libDir)) { New-Item -ItemType Directory -Path $libDir -Force | Out-Null }
    Copy-Item $dllRelease -Destination $libDir -Force
    $dllSize = (Get-Item (Join-Path $libDir "ignis_rt.dll")).Length / 1KB
    Write-Host "  ignis_rt.dll -> lib/ ($([math]::Round($dllSize)) KB)" -ForegroundColor Green

    # Bundle compiled SPIR-V so the remote addon is standalone (no _deploy_root.txt needed)
    $shaderDest = Join-Path $addonSrc "shaders"
    if (-not (Test-Path $shaderDest)) { New-Item -ItemType Directory -Path $shaderDest -Force | Out-Null }
    $spvCount = 0
    Get-ChildItem (Join-Path $repoRoot "shaders") -Filter "*.spv" | ForEach-Object {
        Copy-Item $_.FullName -Destination $shaderDest -Force; $spvCount++
    }
    $wfSrc = Join-Path $repoRoot "shaders\wavefront"
    if (Test-Path $wfSrc) {
        $wfDest = Join-Path $shaderDest "wavefront"
        if (-not (Test-Path $wfDest)) { New-Item -ItemType Directory -Path $wfDest -Force | Out-Null }
        Get-ChildItem $wfSrc -Filter "*.spv" | ForEach-Object {
            Copy-Item $_.FullName -Destination $wfDest -Force; $spvCount++
        }
    }
    Write-Host "  $spvCount shader SPVs -> shaders/" -ForegroundColor Green

    # NRD.dll if present (sibling lib or NRD SDK)
    $nrdDll = ""
    if ($env:IGNIS_NRD_ROOT -and (Test-Path "$env:IGNIS_NRD_ROOT\_Bin\Release\NRD.dll")) {
        $nrdDll = "$env:IGNIS_NRD_ROOT\_Bin\Release\NRD.dll"
    } else {
        $nrdFallback = Join-Path (Split-Path $repoRoot -Parent) "NRD\_Bin\Release\NRD.dll"
        if (Test-Path $nrdFallback) { $nrdDll = $nrdFallback }
    }
    if ($nrdDll -and (Test-Path $nrdDll)) {
        Copy-Item $nrdDll -Destination $libDir -Force
        Write-Host "  NRD.dll -> lib/" -ForegroundColor Green
    }
} else {
    Write-Host "[2/3] PyOnly - skipping artifact staging" -ForegroundColor DarkGray
}

# Never ship the dev-machine breadcrumb: it points at THIS repo and would mislead
# the addon's shader resolver on the remote. Bundled shaders/ (step 2 in __init__.py)
# is what we want the remote to use.
$localBreadcrumb = Join-Path $addonSrc "_deploy_root.txt"
if (Test-Path $localBreadcrumb) { Remove-Item $localBreadcrumb -Force }

# ============================================================================
# 3. Mirror addon -> remote over SMB (robocopy: incremental, retry-light)
# ============================================================================
Write-Host "[3/3] Copying to remote over SMB ..." -ForegroundColor Yellow

if (-not (Test-Path $Dest)) { New-Item -ItemType Directory -Path $Dest -Force | Out-Null }

# Deploying Rust (the default): preserve the working C++ DLL once, so an early
# Rust build is reversible. Restore with -Cpp, or:
#   Copy-Item <dest>\lib\ignis_rt.dll.cpp-bak <dest>\lib\ignis_rt.dll -Force
if (-not $Cpp -and -not $PyOnly) {
    $remoteDll = Join-Path $Dest "lib\ignis_rt.dll"
    $bak       = Join-Path $Dest "lib\ignis_rt.dll.cpp-bak"
    if ((Test-Path $remoteDll) -and -not (Test-Path $bak)) {
        Copy-Item $remoteDll $bak -Force
        Write-Host "  Backed up C++ DLL -> lib\ignis_rt.dll.cpp-bak" -ForegroundColor DarkCyan
    }
}

# Always exclude pycache, the breadcrumb, and any stale logs the addon wrote.
$common = @("/R:2", "/W:2", "/NFL", "/NDL", "/NP", "/XF", "_deploy_root.txt", "/XD", "__pycache__")

if ($PyOnly) {
    # Only Python files at the top level — DLL/shaders untouched, no Blender restart needed.
    robocopy $addonSrc $Dest *.py @common | Out-Null
} else {
    # Full tree: .py + lib/ + shaders/. /E includes subdirs. No /MIR (don't purge remote logs).
    robocopy $addonSrc $Dest /E @common | Out-Null
}
$rc = $LASTEXITCODE

# robocopy: 0=nothing to copy, 1-7=success (files copied/extra/etc), >=8=error
if ($rc -ge 8) {
    Write-Host ""
    Write-Host "  robocopy reported errors (exit $rc)." -ForegroundColor Red
    Write-Host "  Most likely the DLL is LOCKED because Blender is OPEN on the RTX PC." -ForegroundColor Yellow
    Write-Host "  -> Close Blender there, then re-run. (.py files may have copied OK.)" -ForegroundColor Yellow
    exit 1
}

Write-Host "  Mirror OK (robocopy exit $rc)" -ForegroundColor Green

# ============================================================================
# Done
# ============================================================================
Write-Host ""
Write-Host "=== Remote deploy complete ===" -ForegroundColor Green
Write-Host ""
if ($PyOnly) {
    Write-Host "Python-only push. In Blender on the RTX PC:" -ForegroundColor Cyan
    Write-Host "  F3 > 'Reload Scripts'  (or toggle the addon off/on). No restart needed." -ForegroundColor Cyan
} else {
    Write-Host "DLL was updated. On the RTX PC:" -ForegroundColor Cyan
    Write-Host "  1. Close Blender BEFORE running this if you hit a lock error." -ForegroundColor Cyan
    Write-Host "  2. Reopen Blender, set engine to 'Ignis RT', viewport > Z > Rendered." -ForegroundColor Cyan
}

# robocopy leaves $LASTEXITCODE at 1-7 on success; normalize so callers see success.
exit 0
