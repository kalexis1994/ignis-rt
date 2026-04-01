# deploy_blender.ps1 - Build ignis_rt.dll and deploy the Blender addon
#
# Usage:
#   .\deploy_blender.ps1              # Build + deploy to latest Blender found
#   .\deploy_blender.ps1 -BlenderVer 5.1
#   .\deploy_blender.ps1 -NoBuild     # Skip cmake build, just copy files
#   .\deploy_blender.ps1 -Symlink     # Use directory junction instead of copy (dev mode)

param(
    [string]$BlenderVer = "",
    [switch]$NoBuild,
    [switch]$Symlink
)

$ErrorActionPreference = "Stop"

$repoRoot   = $PSScriptRoot
$buildDir   = Join-Path $repoRoot "build"
$addonSrc   = Join-Path $repoRoot "blender\ignis_rt"
$dllRelease = Join-Path $buildDir "Release\ignis_rt.dll"
$libDir     = Join-Path $addonSrc "lib"

# ============================================================================
# 1. Find Blender — search user config + Program Files installations
# ============================================================================
$blenderUserBase = "$env:APPDATA\Blender Foundation\Blender"
$blenderInstallBases = @(
    "$env:ProgramFiles\Blender Foundation",
    "${env:ProgramFiles(x86)}\Blender Foundation"
)

# Collect all known versions from user config dirs
# Blender 5.x uses extensions/user_default/ instead of scripts/addons/
$allVersions = @()
if (Test-Path $blenderUserBase) {
    Get-ChildItem $blenderUserBase -Directory | ForEach-Object {
        $ver = $_.Name
        # Blender 5.x: prefer extensions/user_default/ (extension system)
        $extDir = Join-Path $_.FullName "extensions\user_default"
        $legacyDir = Join-Path $_.FullName "scripts\addons"
        if ([version]$ver -ge [version]"5.0" -and (Test-Path (Split-Path $extDir -Parent))) {
            $allVersions += @{
                Name = $ver
                AddonsDir = $extDir
                Source = "user-ext"
            }
        } else {
            $allVersions += @{
                Name = $ver
                AddonsDir = $legacyDir
                Source = "user"
            }
        }
    }
}

# Collect versions from Program Files installations (e.g. "Blender 5.1" → version "5.1")
foreach ($base in $blenderInstallBases) {
    if (Test-Path $base) {
        Get-ChildItem $base -Directory | Where-Object { $_.Name -match '^Blender\s+(\d+\.\d+)' } | ForEach-Object {
            $ver = $Matches[1]
            $installAddons = Join-Path $_.FullName "$ver\scripts\addons"
            # Only add if not already found via user config
            if (-not ($allVersions | Where-Object { $_.Name -eq $ver })) {
                $allVersions += @{
                    Name = $ver
                    AddonsDir = $installAddons
                    Source = "install"
                }
            }
        }
    }
}

if ($allVersions.Count -eq 0) {
    Write-Host "ERROR: No Blender installations found." -ForegroundColor Red
    Write-Host "  Searched: $blenderUserBase" -ForegroundColor DarkGray
    foreach ($b in $blenderInstallBases) { Write-Host "  Searched: $b" -ForegroundColor DarkGray }
    exit 1
}

# Sort by version descending
$allVersions = $allVersions | Sort-Object { [version]$_.Name } -Descending

if ($BlenderVer -ne "") {
    $target = $allVersions | Where-Object { $_.Name -eq $BlenderVer } | Select-Object -First 1
    if (-not $target) {
        $available = ($allVersions | ForEach-Object { "$($_.Name) ($($_.Source))" }) -join ", "
        Write-Host "ERROR: Blender $BlenderVer not found. Available: $available" -ForegroundColor Red
        exit 1
    }
} else {
    $target = $allVersions[0]
}

$blenderVer  = $target.Name
$addonsDir   = $target.AddonsDir
$addonDest   = Join-Path $addonsDir "ignis_rt"

Write-Host "=== Ignis RT Blender Deploy ===" -ForegroundColor Cyan
Write-Host "  Blender version : $blenderVer ($($target.Source))"
Write-Host "  Addons dir      : $addonsDir"
Write-Host ""

# ============================================================================
# 2. Build (unless -NoBuild)
# ============================================================================
if (-not $NoBuild) {
    Write-Host "[1/3] Building ignis_rt.dll ..." -ForegroundColor Yellow

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
    Write-Host "[1/3] Skipping build (-NoBuild)" -ForegroundColor DarkGray
}

# ============================================================================
# 3. Copy DLL to addon lib/
# ============================================================================
Write-Host "[2/3] Copying DLL ..." -ForegroundColor Yellow

if (-not (Test-Path $dllRelease)) {
    Write-Host "ERROR: $dllRelease not found. Build first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $libDir)) {
    New-Item -ItemType Directory -Path $libDir -Force | Out-Null
}

Copy-Item $dllRelease -Destination $libDir -Force
$dllSize = (Get-Item (Join-Path $libDir "ignis_rt.dll")).Length / 1KB
Write-Host "  ignis_rt.dll -> lib/ ($([math]::Round($dllSize)) KB)" -ForegroundColor Green

# Copy compiled shaders (.spv) into addon for standalone zip installs
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

# Copy NRD.dll if NRD is enabled (check for it in NRD SDK path)
$nrdDll = ""
if ($env:IGNIS_NRD_ROOT -and (Test-Path "$env:IGNIS_NRD_ROOT\_Bin\Release\NRD.dll")) {
    $nrdDll = "$env:IGNIS_NRD_ROOT\_Bin\Release\NRD.dll"
} else {
    # Fallback: check sibling NRD directory
    $nrdFallback = Join-Path (Split-Path $repoRoot -Parent) "NRD\_Bin\Release\NRD.dll"
    if (Test-Path $nrdFallback) { $nrdDll = $nrdFallback }
}
if ($nrdDll -and (Test-Path $nrdDll)) {
    Copy-Item $nrdDll -Destination $libDir -Force
    $nrdSize = (Get-Item (Join-Path $libDir "NRD.dll")).Length / 1KB
    Write-Host "  NRD.dll -> lib/ ($([math]::Round($nrdSize)) KB)" -ForegroundColor Green
}

# Copy NRC DLLs (NRC_Vulkan.dll + CUDA runtime)
$nrcBin = Join-Path (Split-Path $repoRoot -Parent) "NRC\Bin"
if (Test-Path $nrcBin) {
    $nrcDlls = Get-ChildItem $nrcBin -Filter "*.dll"
    foreach ($dll in $nrcDlls) {
        Copy-Item $dll.FullName -Destination $libDir -Force
        $dllSize = $dll.Length / 1KB
        Write-Host "  $($dll.Name) -> lib/ ($([math]::Round($dllSize)) KB)" -ForegroundColor Green
    }
}

# ============================================================================
# 4. Deploy addon to Blender
# ============================================================================
Write-Host "[3/3] Deploying addon to Blender $blenderVer ..." -ForegroundColor Yellow

# Create scripts/addons if it doesn't exist
if (-not (Test-Path $addonsDir)) {
    New-Item -ItemType Directory -Path $addonsDir -Force | Out-Null
}

# Remove old deployment (junction, symlink, or folder)
$dllLocked = $false
if (Test-Path $addonDest) {
    $item = Get-Item $addonDest -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        # It's a junction/symlink - remove the link only
        cmd /c rmdir "$addonDest"
        Write-Host "  Removed old junction" -ForegroundColor DarkGray
    } else {
        try {
            Remove-Item $addonDest -Recurse -Force -ErrorAction Stop
            Write-Host "  Removed old copy" -ForegroundColor DarkGray
        } catch {
            # DLL probably locked by Blender - fall back to updating only non-locked files
            Write-Host "  DLL locked (Blender running?) - updating Python files only" -ForegroundColor Yellow
            $dllLocked = $true
        }
    }
}

if ($Symlink) {
    if ($dllLocked) {
        Write-Host "  Cannot create junction while DLL is locked. Close Blender first." -ForegroundColor Red
        exit 1
    }
    # Directory junction - Python files are always live, no re-deploy needed for .py changes
    cmd /c mklink /J "$addonDest" "$addonSrc"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create junction. Try running as admin?" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Junction: $addonDest -> $addonSrc" -ForegroundColor Green
    Write-Host "  (Dev mode: .py changes are live, only re-run for DLL rebuild)" -ForegroundColor DarkCyan
} elseif ($dllLocked) {
    # Copy only Python files (DLL is locked but still valid from previous deploy)
    Get-ChildItem $addonSrc -Filter "*.py" | ForEach-Object {
        Copy-Item $_.FullName -Destination (Join-Path $addonDest $_.Name) -Force
    }
    Write-Host "  Updated .py files in $addonDest" -ForegroundColor Green
    Write-Host "  NOTE: DLL NOT updated - close Blender and re-deploy for DLL changes" -ForegroundColor Yellow
} else {
    # Full copy - standalone, no dependency on repo
    Copy-Item $addonSrc -Destination $addonDest -Recurse -Force
    Write-Host "  Copied addon to $addonDest" -ForegroundColor Green
}

# Write repo root breadcrumb so the addon can find shaders
$repoRootNormalized = $repoRoot -replace '\\', '/'
Set-Content -Path (Join-Path $addonDest "_deploy_root.txt") -Value $repoRootNormalized -NoNewline
Write-Host "  Wrote _deploy_root.txt -> $repoRootNormalized" -ForegroundColor DarkGray

# ============================================================================
# Done
# ============================================================================
Write-Host ""
Write-Host "=== Deploy complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "In Blender:" -ForegroundColor Cyan
Write-Host "  1. Edit > Preferences > Add-ons > search 'Ignis' > Enable"
Write-Host "  2. Set render engine to 'Ignis RT'"
Write-Host "  3. Viewport > Z > Rendered"
Write-Host ""
if ($Symlink) {
    Write-Host "TIP: With -Symlink, only re-run this script when you rebuild the DLL." -ForegroundColor DarkCyan
    Write-Host "     Python changes are instant (just reload addon in Blender: F3 > 'Reload Scripts')." -ForegroundColor DarkCyan
}
