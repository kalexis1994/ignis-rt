# Node VM

The Node VM is a bytecode interpreter that runs inside the raygen shader, evaluating Blender's shader node tree per-pixel. It replaces the need for generating unique shaders per material.

## Architecture

```mermaid
mindmap
  root((Node VM))
    Compilation
      Python _NodeVmCompiler
      Walks node tree
      Emits bytecode
      64 instruction limit
      32 registers
    Execution
      GLSL executeNodeVm
      Per-pixel evaluation
      All material properties
      Texture sampling
    Opcodes - 86 total
      Texture Sampling
        SAMPLE_TEX
        TEX_NOISE
        TEX_VORONOI
        TEX_GRADIENT
        TEX_CHECKER
        TEX_WAVE
        TEX_MAGIC
        TEX_BRICK
        TEX_WHITE_NOISE
        NOISE_BUMP
      Color Blend - 13 modes
        MIX / MIX_REG
        MULTIPLY
        ADD / SUBTRACT
        SCREEN / OVERLAY
        DARKEN / LIGHTEN
        COLOR_DODGE / COLOR_BURN
        SOFT_LIGHT / LINEAR_LIGHT
      Color Adjust
        HUE_SAT_VAL
        COLORRAMP + RAMP_DATA
        INVERT
        GAMMA
        BRIGHT_CONTRAST
        LUMINANCE
      Math - 21 operations
        ADD MUL DIV POWER
        MIN MAX SUB ABS
        SQRT MOD FLOOR CEIL
        FRACT SIN COS TAN
        LESS GREATER ROUND
        SIGN SMOOTH_MIN
      Vector Math - 18 operations
        ADD SUB MUL DIV
        CROSS DOT LENGTH DIST
        NORMALIZE SCALE REFLECT
        ABS MIN MAX FLOOR
        FRACT MOD SIGN
      Channel
        SEPARATE_RGB
        COMBINE_RGB
      Input
        LOAD_WORLD_POS
        LOAD_LOCAL_POS
        LOAD_VIEW_DIR
        LOAD_NORMAL
        LOAD_INCOMING
        LOAD_VERTEX_COLOR
        OBJECT_RANDOM
        BACKFACING
        LAYER_WEIGHT
        FRESNEL
      UV
        UV_TRANSFORM
        UV_ROTATE
        UV_VFLIP
      Output
        OUTPUT_COLOR
        OUTPUT_ROUGH
        OUTPUT_METAL
        OUTPUT_ALPHA
        OUTPUT_EMISSION
        OUTPUT_TRANSMISSION
        OUTPUT_IOR
        OUTPUT_NORMAL
        OUTPUT_UV
        OUTPUT_BUMP
      Constants
        LOAD_CONST
        LOAD_SCALAR
        MATH_CLAMP
        MAP_RANGE_FULL
        VEC_MATH
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
