# Installation

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | NVIDIA RTX 2060 | RTX 3080 or newer |
| OS | Windows 10 (64-bit) | Windows 11 |
| Blender | 4.0+ | 5.0+ |
| NVIDIA Driver | 560+ | Latest Game Ready |
| VRAM | 6 GB | 8+ GB |

!!! warning "AMD/Intel GPUs"
    Ignis RT requires NVIDIA RTX hardware for ray tracing acceleration (VK_KHR_ray_query) and DLSS (Tensor Cores). AMD and Intel GPUs are not supported.

## Install from GitHub Release

1. Go to [Releases](https://github.com/kalexis1994/ignis-rt/releases)
2. Download `ignis_rt_addon.zip`
3. In Blender: **Edit → Preferences → Add-ons → Install from Disk**
4. Select the downloaded zip
5. Enable "Ignis RT" in the addon list

## Install from CI Build (Nightly)

1. Go to [Actions](https://github.com/kalexis1994/ignis-rt/actions)
2. Click the latest successful build
3. Download the `ignis-rt-blender-addon` artifact
4. The downloaded zip is directly installable in Blender

## Activate

1. In **Render Properties**, change the render engine to **Ignis RT**
2. Switch viewport to **Rendered** mode (Z → Rendered, or top-right shading buttons)
3. The first load takes 5-15 seconds (mesh upload, shader compilation, texture streaming)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Black viewport | Update NVIDIA drivers to 560+ |
| "DLL not found" | Ensure `lib/ignis_rt.dll` is in the addon directory |
| No DLSS | Verify RTX GPU; check that `nvngx_dlss.dll` and `nvngx_dlssd.dll` are in `lib/` |
| Low FPS | Reduce Samples/Pixel to 1, use DLSS Quality mode |
| Crash on load | Check Blender's system console for error messages |
