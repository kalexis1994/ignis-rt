# Architecture Overview

Ignis RT is structured in three layers: the **Blender Addon** (Python), the **Vulkan Renderer** (C++/DLL), and the **Shader System** (GLSL).

```mermaid
flowchart TB
    subgraph Blender["Blender Addon (Python)"]
        SE[Scene Export] --> |meshes, materials, lights| API[C API - ignis_api]
        ENG[Engine Sync] --> |camera, transforms, settings| API
        CM[Color Management] --> |OCIO LUT bake| API
    end

    subgraph DLL["Vulkan Renderer (C++ DLL)"]
        API --> VK[Vulkan Context]
        VK --> BLAS[BLAS Builder]
        VK --> TLAS[TLAS Builder]
        VK --> MAT[Material Buffer]
        VK --> TEX[Texture Manager]
        VK --> DLSS_M[DLSS Manager]
    end

    subgraph GPU["GPU Shaders (GLSL)"]
        RT[raygen_blender.rgen] --> VM[Node VM]
        RT --> BRDF[PBR BRDF]
        RT --> VOL[Volume March]
        RT --> SKY[Sky Model]
        TM[tonemap.comp] --> LUT[3D LUT Sample]
    end

    BLAS --> RT
    TLAS --> RT
    MAT --> RT
    TEX --> RT
    RT --> DLSS_M
    DLSS_M --> TM
    TM --> DISP[Display]

    style Blender fill:#4A6380,color:white
    style DLL fill:#2E2E30,color:white
    style GPU fill:#E06030,color:white
```

## Data Flow

### Initial Load (Staged)

| Stage | What | Duration |
|-------|------|----------|
| 1. Export | Python reads Blender scene → meshes, materials, textures | ~1-3s |
| 2. BLAS | Upload vertex/index buffers, build per-mesh BVH | ~2-5s |
| 3. Materials | Compile Node VM bytecode, upload material buffer | ~1-2s |
| 4. Textures | Decode images, upload to GPU (chunked) | ~2-10s |
| 5. TLAS | Build top-level acceleration structure (instances) | <0.1s |
| 6. Finalize | Sun extraction, emissive triangles, HDRI | <0.5s |

### Per-Frame Transform Sync

Three-tier system avoids expensive depsgraph iteration for most object movements:

```mermaid
flowchart TD
    VU[view_update] -->|Object changed| CO[_ignis_changed_objects]
    VU -->|Scene changed| OD[_ignis_objects_dirty]
    UNDO[undo_post / redo_post] --> OD

    CO --> CHECK{Object type?}
    CHECK -->|"Parent Empty<br>(in _ignis_parent_groups)"| HIER["Hierarchy path<br>1 API call + numpy batch multiply<br>O(K children)"]
    CHECK -->|"Direct mesh<br>(in _ignis_direct_objects)"| INST["Instant path<br>1 API call per object<br>O(1)"]
    CHECK -->|Unknown / new| FULL

    OD --> FULL["Full sync<br>iterate depsgraph<br>rebuild all caches"]

    HIER --> UPDATE[update_instance_transforms]
    INST --> UPDATE
    FULL --> BUILD[build_tlas]

    UPDATE -->|"TLAS refit<br>(VK_..._MODE_UPDATE)"| GPU[GPU]
    BUILD -->|"TLAS rebuild<br>(VK_..._MODE_BUILD)"| GPU

    IDLE["60 frames idle"] -->|safety net| FULL

    style HIER fill:#2E7D32,color:white
    style INST fill:#2E7D32,color:white
    style FULL fill:#C62828,color:white
```

**Hierarchy cache**: During initial load, collection instance parent-child relationships are detected via `inst.parent`. Each parent Empty stores its children's relative transforms (in Vulkan space). When the parent moves, child transforms are computed via numpy batch matrix multiply -- zero depsgraph iteration.

**TLAS refit**: When only transforms change (same instance count), the TLAS is updated in-place using `VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR` instead of a full rebuild. The C++ renderer caches the full instance array and patches only the changed entries.

**Safety net**: After 60 frames without any sync, a full sync is forced to catch any missed updates from undo, duplication, or other edge cases.

```mermaid
sequenceDiagram
    participant B as Blender
    participant P as Python Engine
    participant C as C++ DLL
    participant G as GPU

    B->>P: view_draw()
    P->>C: set_camera(matrices)
    P->>C: set_float(exposure, sun, etc.)
    alt Object moved (fast path)
        P->>P: numpy: parent_vk @ child_relative
        P->>C: update_instance_transforms(indices, xforms)
        C->>G: TLAS refit (UPDATE mode)
    else Scene changed (full sync)
        P->>P: iterate depsgraph
        P->>C: build_tlas(all instances)
        C->>G: TLAS rebuild (BUILD mode)
    end
    C->>G: dispatch raygen shader
    G->>G: path trace + denoise
    G->>C: output texture
    C->>P: readback RGBA
    P->>B: draw to viewport
```

## Key Design Decisions

### Node VM vs Native Shaders

Instead of generating a unique shader per material (like Cycles), Ignis RT uses a **bytecode interpreter** (Node VM) that runs inside the raygen shader. This avoids shader recompilation when materials change.

| Approach | Pros | Cons |
|----------|------|------|
| **Node VM** (ours) | No recompile on material change, hot-reload | Slightly slower per-pixel, 64 instruction limit |
| **Generated shaders** (Cycles) | Maximum performance, no instruction limit | Minutes to compile, can't hot-reload |

### DLSS Ray Reconstruction vs NRD

Ray Reconstruction replaces the traditional NRD denoiser with an AI model that understands ray-traced signals. Benefits:

- Better temporal stability
- Fewer ghosting artifacts
- Handles noisy path-traced input better (transformer architecture)
- Single DLL, no NRD build dependency

### Alpha Transparency

All rays use `gl_RayFlagsOpaqueEXT` for maximum hardware BVH performance. Alpha-tested materials (Mix Shader with Transparent BSDF) use stochastic pass-through in the bounce loop — `rand() > alpha` decides whether the ray continues through or shades the surface. Back faces of alpha-tested meshes are not culled, allowing visibility through 3D wireframe/foliage meshes.

### Performance Optimizations

| Optimization | Impact | Where |
|---|---|---|
| **Hierarchy transform sync** | O(1) API calls + numpy batch multiply instead of O(N) depsgraph iteration per frame | CPU |
| **TLAS refit (UPDATE mode)** | In-place BVH refit instead of full TLAS rebuild when only transforms change | GPU |
| **Shader Execution Reordering (SER)** | Reduces thread divergence by grouping threads with same material | RTX 40+ HW, 30 SW |
| **VM skip for empty programs** | Avoids function call + 20-field struct init when instrCount=0 | GPU |
| **Output opcode fast-path** | Opcodes >= 0xE0 branch directly to output handlers, skipping 50+ intermediate checks | GPU |
| **Opaque ray queries** | Hardware BVH finds closest hit natively, no per-candidate processing | GPU |
| **Runtime OCIO LUT bake** | Single 3D texture lookup for any view transform, no per-pixel OCIO evaluation | GPU |
