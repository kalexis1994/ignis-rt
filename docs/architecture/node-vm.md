# Node VM

The Node VM is a bytecode interpreter that runs inside the raygen shader, evaluating Blender's shader node tree per-pixel. It replaces the need for generating unique shaders per material.

## Architecture

```mermaid
flowchart TB
    VM((Node VM<br/>86 Opcodes<br/>32 Registers<br/>64 Instructions))

    VM --- COMP[Compilation<br/>Python _NodeVmCompiler]
    VM --- EXEC[Execution<br/>GLSL executeNodeVm]

    VM --- TEX & COL & MATH & INPUT & OUT

    subgraph TEX[Texture - 10 ops]
        direction LR
        T1[SAMPLE_TEX]
        T2[TEX_NOISE]
        T3[TEX_VORONOI]
        T4[TEX_GRADIENT]
        T5[TEX_CHECKER / WAVE]
        T6[TEX_MAGIC / BRICK]
    end

    subgraph COL[Color - 19 ops]
        direction LR
        C1[MIX / MIX_REG]
        C2[13 blend modes]
        C3[HUE_SAT / COLORRAMP]
        C4[INVERT / GAMMA]
        C5[BRIGHT_CONTRAST]
    end

    subgraph MATH[Math - 40 ops]
        direction LR
        M1[21 scalar math]
        M2[18 vector math]
        M3[MAP_RANGE / CLAMP]
    end

    subgraph INPUT[Input - 11 ops]
        direction LR
        I1[WORLD_POS / LOCAL_POS]
        I2[VIEW_DIR / NORMAL]
        I3[VERTEX_COLOR]
        I4[OBJECT_RANDOM]
        I5[LAYER_WEIGHT / FRESNEL]
    end

    subgraph OUT[Output - 10 ops]
        direction LR
        O1[COLOR / ROUGH / METAL]
        O2[ALPHA / EMISSION]
        O3[TRANSMISSION / IOR]
        O4[UV / BUMP / NORMAL]
    end

    style VM fill:#E06030,color:white
    style TEX fill:#4A6380,color:white
    style COL fill:#37474F,color:white
    style MATH fill:#2E2E30,color:white
    style INPUT fill:#B34A24,color:white
    style OUT fill:#00695C,color:white
```

## Instruction Format

Each instruction is a `uvec4` (16 bytes):

```
┌───────────────────────┬─────────┬─────────┬─────────┐
│         .x            │  .y     │  .z     │  .w     │
├───────────────────────┼─────────┼─────────┼─────────┤
│ opcode  (bits 0-7)    │ imm_y   │ imm_z   │ imm_w   │
│ dst     (bits 8-12)   │ (32bit) │ (32bit) │ (32bit) │
│ srcA    (bits 16-20)  │         │         │         │
│ srcB    (bits 24-28)  │         │         │         │
└───────────────────────┴─────────┴─────────┴─────────┘
```

- **opcode** (bits 0-7): Operation to perform
- **dst** (bits 8-12): Destination register (0-31, 5-bit encoding)
- **srcA** (bits 16-20): Source register A
- **srcB** (bits 24-28): Source register B
- **imm_y/z/w**: Immediate values (float or uint via `uintBitsToFloat`)

## Register Allocation

- **R[0]**: UV coordinates (default, shared by all texture lookups)
- **R[1-31]**: General purpose, allocated sequentially by the compiler
- **Cache**: `node_reg_cache` maps `(node_type, node_name, socket_id)` → register, preventing re-compilation of nodes with multiple outputs

## Compilation Flow

```mermaid
flowchart TD
    TREE[Blender Node Tree] --> FIND[Find Principled BSDF]
    FIND --> MIXCHECK{Mix Shader?}
    MIXCHECK -->|Yes| MIXCOMPILE[Compile both branches + blend]
    MIXCHECK -->|No| SINGLE[Compile single Principled]
    SINGLE --> UV[1. Compile UV chain]
    UV --> COLOR[2. Compile Base Color]
    COLOR --> ROUGH[3. Compile Roughness + Metallic]
    ROUGH --> EMIT[4. Compile Emission]
    EMIT --> TRANS[5. Compile Transmission]
    TRANS --> ALPHA[6. Compile Alpha]
    ALPHA --> BUMP[7. Compile Bump]
    BUMP --> BYTECODE[VM Bytecode - up to 64 instrs]
```

## Limitations

| Aspect | Limit | Notes |
|--------|-------|-------|
| Instructions | 64 max | Complex materials may exceed; compilation stops silently |
| Registers | 32 max | Shared across all node chains |
| RGB Curves | Passthrough only | Baked LUT produces blue tint; needs OCIO pipeline investigation |
| Combine RGB | Not compiled | Handled as passthrough or special-case UV optimization |
| Bump node | TEX_NOISE only | Other procedural textures not supported as bump height source |
| Texture sampling | 1 UV set | All textures share the same compiled UV transform |

## Noise Implementation

Uses Cycles' exact algorithms for matching procedural textures:

- **Hash**: Jenkins Lookup3 (`hash_uint3` from Cycles `util/hash.h`)
- **Gradient**: Perlin improved noise branchless `grad3` (from Cycles `svm/noise.h`)
- **Fade**: Quintic Hermite `t³(6t²-15t+10)`
- **FBM**: Multi-octave with configurable detail, roughness, lacunarity
