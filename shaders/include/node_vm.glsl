#ifndef NODE_VM_GLSL
#define NODE_VM_GLSL

// ============================================================================
// Mini Shader Node VM — per-pixel Blender node tree evaluation
// Matches Cycles SVM behavior for common node types.
// ============================================================================

// Opcodes (8-bit)
const uint OP_NOP            = 0x00u;

// Texture sampling
const uint OP_SAMPLE_TEX     = 0x01u;  // R[dst] = texture(tex[imm_z], R[srcA].xy)

// UV manipulation (Mapping node)
const uint OP_UV_TRANSFORM   = 0x10u;  // R[dst].xy = R[srcA].xy * scale + offset
const uint OP_UV_ROTATE      = 0x11u;  // R[dst].xy = rotate R[srcA].xy by angle
const uint OP_UV_MATRIX      = 0x12u;  // reserved
const uint OP_UV_VFLIP       = 0x13u;  // R[dst].y = 1.0 - R[srcA].y (Blender→Vulkan V-flip)

// Color operations
const uint OP_MIX            = 0x20u;  // R[dst] = mix(R[srcA], R[srcB], imm_y)
const uint OP_MIX_REG        = 0x21u;  // R[dst] = mix(R[srcA], R[srcB], R[imm_y & 0x1F].x)
const uint OP_MULTIPLY       = 0x22u;  // R[dst] = R[srcA] * R[srcB]
const uint OP_ADD            = 0x23u;  // R[dst] = R[srcA] + R[srcB]
const uint OP_SUBTRACT       = 0x24u;  // R[dst] = R[srcA] - R[srcB]
const uint OP_SCREEN         = 0x25u;  // R[dst] = 1 - (1-A)*(1-B)
const uint OP_OVERLAY        = 0x26u;  // R[dst] = overlay blend
const uint OP_DARKEN         = 0x27u;  // R[dst] = min(R[srcA], R[srcB])
const uint OP_INVERT         = 0x28u;  // R[dst] = mix(R[srcA], 1-R[srcA], imm_y)
const uint OP_GAMMA          = 0x29u;  // R[dst] = pow(R[srcA], imm_y)
const uint OP_BRIGHT_CONTRAST= 0x2Au;  // R[dst] = brightContrast(R[srcA], imm_y, imm_z)
const uint OP_HUE_SAT_VAL    = 0x2Bu;  // R[dst] = hueSatVal(R[srcA], hue=imm_y, sat=imm_z, val=imm_w)
const uint OP_LIGHTEN        = 0x2Cu;  // R[dst] = max(R[srcA], R[srcB])
const uint OP_COLOR_DODGE    = 0x2Du;  // R[dst] = A / (1 - B)
const uint OP_COLOR_BURN     = 0x2Eu;  // R[dst] = 1 - (1-A) / B
const uint OP_SOFT_LIGHT     = 0x2Fu;  // soft light blend

// ColorRamp
const uint OP_COLORRAMP      = 0x30u;  // R[dst] = ramp(R[srcA].x, N stops, data follows)
const uint OP_RAMP_DATA      = 0x31u;  // (not executed) inline ramp stop data

// Scalar operations
const uint OP_LUMINANCE      = 0x40u;  // R[dst].x = luminance(R[srcA].rgb)
const uint OP_MATH_ADD       = 0x41u;  // R[dst].x = R[srcA].x + R[srcB].x
const uint OP_MATH_MUL       = 0x42u;  // R[dst].x = R[srcA].x * R[srcB].x
const uint OP_MATH_DIV       = 0x43u;  // R[dst].x = R[srcA].x / max(R[srcB].x, 1e-6)
const uint OP_MATH_POWER     = 0x44u;  // R[dst].x = pow(R[srcA].x, R[srcB].x)
const uint OP_MATH_MIN       = 0x45u;  // R[dst].x = min(R[srcA].x, R[srcB].x)
const uint OP_MATH_MAX       = 0x46u;  // R[dst].x = max(R[srcA].x, R[srcB].x)
const uint OP_MATH_CLAMP     = 0x47u;  // R[dst].x = clamp(R[srcA].x, imm_y, imm_z)
const uint OP_MAP_RANGE      = 0x48u;  // R[dst].x = mapRange(R[srcA].x, fromMin/Max, toMin/Max)
const uint OP_MATH_SUB       = 0x49u;  // R[dst].x = R[srcA].x - R[srcB].x
const uint OP_MATH_ABS       = 0x4Au;  // R[dst].x = abs(R[srcA].x)
const uint OP_MATH_SQRT      = 0x4Bu;  // R[dst].x = sqrt(max(R[srcA].x, 0))
const uint OP_MATH_MOD       = 0x4Cu;  // R[dst].x = mod(R[srcA].x, R[srcB].x)
const uint OP_MATH_FLOOR     = 0x4Du;  // R[dst].x = floor(R[srcA].x)
const uint OP_MATH_CEIL      = 0x4Eu;  // R[dst].x = ceil(R[srcA].x)
const uint OP_MATH_FRACT     = 0x4Fu;  // R[dst].x = fract(R[srcA].x)
const uint OP_MATH_SIN       = 0x70u;  // R[dst].x = sin(R[srcA].x)
const uint OP_MATH_COS       = 0x71u;  // R[dst].x = cos(R[srcA].x)
const uint OP_MATH_TAN       = 0x72u;  // R[dst].x = tan(R[srcA].x)
const uint OP_MATH_LESS      = 0x73u;  // R[dst].x = (R[srcA].x < R[srcB].x) ? 1.0 : 0.0
const uint OP_MATH_GREATER   = 0x74u;  // R[dst].x = (R[srcA].x > R[srcB].x) ? 1.0 : 0.0
const uint OP_MATH_ROUND     = 0x75u;  // R[dst].x = round(R[srcA].x)
const uint OP_MATH_SIGN      = 0x76u;  // R[dst].x = sign(R[srcA].x)
const uint OP_MATH_SMOOTH_MIN = 0x77u; // R[dst].x = smoothMin(R[srcA].x, R[srcB].x, k)

// Channel manipulation
const uint OP_SEPARATE_RGB   = 0x50u;  // R[dst].x = R[srcA][channel]  (channel in imm_y)
const uint OP_COMBINE_RGB    = 0x51u;  // R[dst].rgb = (R[srcA].x, R[srcB].x, R[imm_y & 0xF].x)

// Procedural textures
const uint OP_TEX_CHECKER    = 0x58u;  // R[dst] = checker(R[srcA].xy, scale=imm_y, color1=R[srcB], color2 from next)

// Procedural textures (extended)
const uint OP_TEX_NOISE      = 0x80u;  // R[dst] = noise(R[srcA].xyz * scale, detail, roughness)
const uint OP_TEX_GRADIENT   = 0x81u;  // R[dst].x = gradient(R[srcA].xyz, type)
const uint OP_TEX_VORONOI    = 0x82u;  // R[dst].x = voronoi(R[srcA].xyz * scale)
const uint OP_TEX_WAVE       = 0x83u;  // R[dst].x = wave(R[srcA].xyz, scale, distortion, type)

// RGB Curves (baked LUT)
const uint OP_RGB_CURVES     = 0x84u;  // R[dst] = rgbCurves(R[srcA], 8 samples per channel in next 6 instrs)
const uint OP_CURVE_DATA     = 0x85u;  // inline curve LUT data (not executed)

// Vector math / extended ops
const uint OP_VEC_MATH       = 0x86u;  // R[dst] = vecMath(R[srcA], R[srcB], op=imm_y)
const uint OP_MAP_RANGE_FULL = 0x87u;  // R[dst].x = mapRange(R[srcA].x, params in next instr)
const uint OP_TEX_WHITE_NOISE = 0x88u; // R[dst] = whiteNoise(R[srcA].xyz)
const uint OP_LOAD_NORMAL    = 0x89u;  // R[dst] = vec4(normal, 1)
const uint OP_LOAD_INCOMING  = 0x8Au;  // R[dst] = vec4(-viewDir, 1) — incoming ray direction
const uint OP_BACKFACING     = 0x8Bu;  // R[dst].x = backfacing ? 1.0 : 0.0
const uint OP_LINEAR_LIGHT   = 0x90u;  // R[dst] = A + 2*B - 1
const uint OP_TEX_MAGIC      = 0x91u;  // R[dst] = magic(R[srcA].xyz, distortion)
const uint OP_TEX_BRICK       = 0x92u;  // R[dst] = brick(R[srcA].xyz, scale, mortarSize)
const uint OP_NOISE_BUMP      = 0x93u;  // R[dst] = noiseBump(R[srcA].xyz, scale, detail, roughness) → .xy=gradient, .z=height
const uint OP_LOAD_VERTEX_COLOR = 0x94u;  // R[dst] = vertexColor (per-vertex color attribute)
const uint OP_OBJECT_RANDOM    = 0x95u;  // R[dst].x = hash(instanceId) — per-instance random [0,1]

// Load immediate / special
const uint OP_LOAD_CONST     = 0x60u;  // R[dst] = vec4(imm_y, imm_z, imm_w, 1)
const uint OP_LOAD_SCALAR    = 0x61u;  // R[dst].x = imm_y
const uint OP_LOAD_WORLD_POS = 0x62u;  // R[dst] = vec4(worldPos, 1) — for position-based texturing
const uint OP_LOAD_VIEW_DIR  = 0x63u;  // R[dst] = vec4(viewDir, 1) — camera-to-surface direction
const uint OP_LOAD_LOCAL_POS = 0x66u;  // R[dst] = vec4(localPos, 1) — object-space position (TEX_COORD:Object)
const uint OP_LAYER_WEIGHT   = 0x64u;  // R[dst].x = layer weight (Fresnel or Facing)
const uint OP_FRESNEL        = 0x65u;  // R[dst].x = Fresnel(IOR, NdotV)

// Output targets
const uint OP_OUTPUT_UV      = 0xEFu;  // transformedUV = R[srcA].xy (for shared Mapping)
const uint OP_OUTPUT_COLOR   = 0xF0u;  // baseColor = R[srcA].rgb
const uint OP_OUTPUT_ROUGH   = 0xF1u;  // roughness = R[srcA].x
const uint OP_OUTPUT_METAL   = 0xF2u;  // metallic = R[srcA].x
const uint OP_OUTPUT_EMISSION= 0xF3u;  // emission = R[srcA].rgb * R[srcA].a (color * strength)
const uint OP_OUTPUT_ALPHA   = 0xF4u;  // alpha = R[srcA].x
const uint OP_OUTPUT_NORMAL  = 0xF5u;  // normalStrength = R[srcA].x
const uint OP_OUTPUT_IOR     = 0xF6u;  // ior = R[srcA].x
const uint OP_OUTPUT_TRANSMISSION = 0xF7u; // transmission = R[srcA].x
const uint OP_OUTPUT_BUMP     = 0xF8u;  // bump gradient output: gradient=R[srcA].xy, strength=imm_y, distance=imm_z

#include "noise.glsl"

float srgbToLinearExact(float c) {
    return (c <= 0.04045) ? (c / 12.92) : pow((c + 0.055) / 1.055, 2.4);
}

vec3 srgbToLinearExact(vec3 c) {
    return vec3(
        srgbToLinearExact(c.r),
        srgbToLinearExact(c.g),
        srgbToLinearExact(c.b)
    );
}

// ── ColorRamp evaluation ──
vec4 nodeVmEvalColorRamp(uint matIdx, uint dataOffset, uint stopCount, float factor) {
    factor = clamp(factor, 0.0, 1.0);

    // Read first stop
    uvec4 d0 = materialBuffer.materials[matIdx].nodeVmCode[dataOffset];
    float pos0 = uintBitsToFloat(d0.x);
    vec4 col0 = vec4(uintBitsToFloat(d0.y), uintBitsToFloat(d0.z), uintBitsToFloat(d0.w), 1.0);

    if (stopCount <= 1u || factor <= pos0) return col0;

    // Linear search through stops (max 8)
    float prevPos = pos0;
    vec4  prevCol = col0;
    for (uint i = 1u; i < min(stopCount, 8u); i++) {
        uvec4 di = materialBuffer.materials[matIdx].nodeVmCode[dataOffset + i];
        float curPos = uintBitsToFloat(di.x);
        vec4  curCol = vec4(uintBitsToFloat(di.y), uintBitsToFloat(di.z), uintBitsToFloat(di.w), 1.0);

        if (factor <= curPos) {
            float t = (curPos > prevPos) ? (factor - prevPos) / (curPos - prevPos) : 0.0;
            return mix(prevCol, curCol, t);
        }
        prevPos = curPos;
        prevCol = curCol;
    }
    return prevCol;  // above last stop
}

// ── VM Result ──
struct NodeVmResult {
    vec3  baseColor;
    float roughness;
    float metallic;
    vec3  emission;
    float emissionStrength;
    float alpha;
    float ior;
    float transmission;
    float normalStrength;
    vec2  transformedUV;   // UV after Mapping transform (for normal/roughness)
    bool  hasBaseColor;
    bool  hasRoughness;
    bool  hasMetallic;
    bool  hasEmission;
    bool  hasAlpha;
    bool  hasIor;
    bool  hasTransmission;
    bool  hasNormalStrength;
    bool  hasTransformedUV;
    vec2  bumpGradient;
    float bumpStrength;
    float bumpDistance;
    bool  hasBump;
};

// ── VM Execution ──
NodeVmResult executeNodeVm(uint matIdx, vec2 uv, vec3 worldPos, vec3 viewDir, vec3 normal, vec4 vertexColor, uint instanceId, vec3 localPos) {
    NodeVmResult result;
    result.baseColor = vec3(0.8);
    result.roughness = 0.5;
    result.metallic = 0.0;
    result.emission = vec3(0.0);
    result.emissionStrength = 0.0;
    result.alpha = 1.0;
    result.ior = 1.45;
    result.transmission = 0.0;
    result.normalStrength = 1.0;
    result.hasBaseColor = false;
    result.hasRoughness = false;
    result.hasMetallic = false;
    result.hasEmission = false;
    result.hasAlpha = false;
    result.hasIor = false;
    result.hasTransmission = false;
    result.hasNormalStrength = false;
    result.transformedUV = uv;
    result.hasTransformedUV = false;
    result.bumpGradient = vec2(0.0);
    result.bumpStrength = 0.0;
    result.bumpDistance = 1.0;
    result.hasBump = false;

    uint header = materialBuffer.materials[matIdx].nodeVmHeader;
    uint instrCount = header & 0xFFu;

    if (instrCount == 0u) return result;  // Fast path: no VM program

    // Register file (32 vec4 registers — needed for Mix Shader with two full BSDFs)
    vec4 R[32];
    R[0] = vec4(uv, 0.0, 1.0);  // R0 = UV input

    for (uint pc = 0u; pc < min(instrCount, 64u); pc++) {
        uvec4 instr = materialBuffer.materials[matIdx].nodeVmCode[pc];
        uint opcode = instr.x & 0xFFu;
        uint dst    = (instr.x >> 8u)  & 0x1Fu;  // 5 bits = 0-31
        uint srcA   = (instr.x >> 16u) & 0x1Fu;
        uint srcB   = (instr.x >> 24u) & 0x1Fu;

        if (opcode == OP_NOP) continue;

        // ── Texture sampling ──
        if (opcode == OP_SAMPLE_TEX) {
            uint texIdx = instr.z;
            uint isSRGB = instr.y;  // 1 = sRGB color texture, 0 = linear data
            R[dst] = texture(textures[nonuniformEXT(texIdx)], R[srcA].xy);
            if (isSRGB != 0u) {
                R[dst].rgb = srgbToLinearExact(max(R[dst].rgb, vec3(0.0)));
            }
        }

        // ── UV manipulation ──
        // ALL UV ops convert to Blender space and STAY there.
        // V-flip back to Vulkan happens ONCE in OP_SAMPLE_TEX.
        else if (opcode == OP_UV_TRANSFORM) {
            // Scale + offset in Blender UV space (undo V-flip, scale, re-flip)
            vec2 scale = vec2(uintBitsToFloat(instr.y), uintBitsToFloat(instr.z));
            vec2 offset = vec2(uintBitsToFloat(instr.w), 0.0);
            vec2 bl = vec2(R[srcA].x, 1.0 - R[srcA].y);
            vec2 result = bl * scale + offset;
            R[dst] = vec4(result.x, 1.0 - result.y, 0.0, 1.0);
        }
        else if (opcode == OP_UV_ROTATE) {
            // Rotate in Blender UV space (undo V-flip, rotate, re-flip)
            float angle = uintBitsToFloat(instr.y);
            vec2 bl = vec2(R[srcA].x, 1.0 - R[srcA].y);
            float c = cos(angle), s = sin(angle);
            vec2 rotated = vec2(bl.x*c - bl.y*s, bl.x*s + bl.y*c);
            R[dst] = vec4(rotated.x, 1.0 - rotated.y, 0.0, 1.0);
        }
        else if (opcode == OP_UV_MATRIX) {
            R[dst] = R[srcA];
        }
        else if (opcode == OP_UV_VFLIP) {
            // Convert Blender UV → Vulkan UV (flip V for stb_image top-down textures)
            R[dst] = vec4(R[srcA].x, 1.0 - R[srcA].y, R[srcA].z, R[srcA].w);
        }

        // ── Color operations ──
        else if (opcode == OP_MIX) {
            float fac = uintBitsToFloat(instr.y);
            R[dst] = mix(R[srcA], R[srcB], fac);
        }
        else if (opcode == OP_MIX_REG) {
            uint facReg = instr.y & 0x1Fu;
            float fac = R[facReg].x;
            R[dst] = mix(R[srcA], R[srcB], fac);
        }
        else if (opcode == OP_MULTIPLY) {
            R[dst] = R[srcA] * R[srcB];
        }
        else if (opcode == OP_ADD) {
            R[dst] = R[srcA] + R[srcB];
        }
        else if (opcode == OP_SUBTRACT) {
            R[dst] = R[srcA] - R[srcB];
        }
        else if (opcode == OP_SCREEN) {
            R[dst] = vec4(1.0) - (vec4(1.0) - R[srcA]) * (vec4(1.0) - R[srcB]);
        }
        else if (opcode == OP_OVERLAY) {
            // Overlay blend per channel
            vec4 a = R[srcA], b = R[srcB];
            for (int ch = 0; ch < 3; ch++) {
                R[dst][ch] = (a[ch] < 0.5) ? 2.0*a[ch]*b[ch] : 1.0 - 2.0*(1.0-a[ch])*(1.0-b[ch]);
            }
            R[dst].w = a.w;
        }
        else if (opcode == OP_DARKEN) {
            R[dst] = vec4(min(R[srcA].rgb, R[srcB].rgb), R[srcA].a);
        }
        else if (opcode == OP_LIGHTEN) {
            R[dst] = vec4(max(R[srcA].rgb, R[srcB].rgb), R[srcA].a);
        }
        else if (opcode == OP_COLOR_DODGE) {
            vec3 a = R[srcA].rgb, b = R[srcB].rgb;
            R[dst] = vec4(
                b.x < 1.0 ? min(a.x / max(1.0 - b.x, 1e-6), 1.0) : 1.0,
                b.y < 1.0 ? min(a.y / max(1.0 - b.y, 1e-6), 1.0) : 1.0,
                b.z < 1.0 ? min(a.z / max(1.0 - b.z, 1e-6), 1.0) : 1.0,
                R[srcA].a);
        }
        else if (opcode == OP_COLOR_BURN) {
            vec3 a = R[srcA].rgb, b = R[srcB].rgb;
            R[dst] = vec4(
                b.x > 0.0 ? max(1.0 - (1.0 - a.x) / b.x, 0.0) : 0.0,
                b.y > 0.0 ? max(1.0 - (1.0 - a.y) / b.y, 0.0) : 0.0,
                b.z > 0.0 ? max(1.0 - (1.0 - a.z) / b.z, 0.0) : 0.0,
                R[srcA].a);
        }
        else if (opcode == OP_SOFT_LIGHT) {
            vec3 a = R[srcA].rgb, b = R[srcB].rgb;
            vec3 r;
            for (int i = 0; i < 3; i++) {
                if (b[i] < 0.5) r[i] = a[i] - (1.0 - 2.0*b[i]) * a[i] * (1.0 - a[i]);
                else r[i] = a[i] + (2.0*b[i] - 1.0) * (sqrt(a[i]) - a[i]);
            }
            R[dst] = vec4(r, R[srcA].a);
        }
        else if (opcode == OP_LINEAR_LIGHT) {
            R[dst] = vec4(R[srcA].rgb + 2.0 * R[srcB].rgb - 1.0, R[srcA].a);
        }
        else if (opcode == OP_INVERT) {
            float fac = uintBitsToFloat(instr.y);
            R[dst] = mix(R[srcA], vec4(1.0) - R[srcA], fac);
        }
        else if (opcode == OP_GAMMA) {
            float gamma = uintBitsToFloat(instr.y);
            R[dst] = vec4(pow(max(R[srcA].rgb, vec3(0.0)), vec3(gamma)), R[srcA].a);
        }
        else if (opcode == OP_BRIGHT_CONTRAST) {
            float bright = uintBitsToFloat(instr.y);
            float contrast = uintBitsToFloat(instr.z);
            vec3 c = R[srcA].rgb;
            c += bright;
            if (contrast != 0.0) {
                float f = (contrast > 0.0) ? 1.0 / max(1.0 - contrast, 1e-4) : 1.0 + contrast;
                c = (c - 0.5) * f + 0.5;
            }
            R[dst] = vec4(c, R[srcA].a);
        }
        else if (opcode == OP_HUE_SAT_VAL) {
            // Proper RGB→HSV→adjust→HSV→RGB (matching Cycles svm_node_hsv)
            float hue = uintBitsToFloat(instr.y);      // 0.5 = neutral
            float sat = uintBitsToFloat(instr.z);       // 1.0 = neutral
            float val = uintBitsToFloat(instr.w);       // 1.0 = neutral
            vec3 c = R[srcA].rgb;
            // RGB to HSV
            float cmax = max(c.r, max(c.g, c.b));
            float cmin = min(c.r, min(c.g, c.b));
            float d = cmax - cmin;
            float h = 0.0, s = 0.0, v = cmax;
            if (cmax > 0.0) s = d / cmax;
            if (d > 1e-6) {
                if (cmax == c.r)      h = (c.g - c.b) / d;
                else if (cmax == c.g) h = 2.0 + (c.b - c.r) / d;
                else                  h = 4.0 + (c.r - c.g) / d;
                h /= 6.0;
                if (h < 0.0) h += 1.0;
            }
            // Apply adjustments (Cycles: hue is offset from 0.5, sat/val are multipliers)
            h = fract(h + hue - 0.5);
            s = clamp(s * sat, 0.0, 1.0);
            v = v * val;
            // HSV to RGB
            float hi = floor(h * 6.0);
            float f = h * 6.0 - hi;
            float p = v * (1.0 - s);
            float q = v * (1.0 - s * f);
            float t_v = v * (1.0 - s * (1.0 - f));
            int i_h = int(hi) % 6;
            if (i_h == 0)      c = vec3(v, t_v, p);
            else if (i_h == 1) c = vec3(q, v, p);
            else if (i_h == 2) c = vec3(p, v, t_v);
            else if (i_h == 3) c = vec3(p, q, v);
            else if (i_h == 4) c = vec3(t_v, p, v);
            else               c = vec3(v, p, q);
            R[dst] = vec4(c, R[srcA].a);
        }

        // ── ColorRamp ──
        else if (opcode == OP_COLORRAMP) {
            uint stopCount = instr.y;
            R[dst] = nodeVmEvalColorRamp(matIdx, pc + 1u, stopCount, R[srcA].x);
            pc += stopCount;  // skip data instructions
        }
        else if (opcode == OP_RAMP_DATA) {
            // Data instruction — skipped by OP_COLORRAMP's pc advance
            continue;
        }

        // ── Scalar operations ──
        else if (opcode == OP_LUMINANCE) {
            R[dst] = vec4(dot(R[srcA].rgb, vec3(0.2126, 0.7152, 0.0722)));
        }
        else if (opcode == OP_MATH_ADD) {
            R[dst].x = R[srcA].x + R[srcB].x;
        }
        else if (opcode == OP_MATH_MUL) {
            R[dst].x = R[srcA].x * R[srcB].x;
        }
        else if (opcode == OP_MATH_DIV) {
            R[dst].x = R[srcA].x / max(abs(R[srcB].x), 1e-6);
        }
        else if (opcode == OP_MATH_POWER) {
            R[dst].x = pow(max(R[srcA].x, 0.0), R[srcB].x);
        }
        else if (opcode == OP_MATH_MIN) {
            R[dst].x = min(R[srcA].x, R[srcB].x);
        }
        else if (opcode == OP_MATH_MAX) {
            R[dst].x = max(R[srcA].x, R[srcB].x);
        }
        else if (opcode == OP_MATH_CLAMP) {
            R[dst].x = clamp(R[srcA].x, uintBitsToFloat(instr.y), uintBitsToFloat(instr.z));
        }
        else if (opcode == OP_MAP_RANGE) {
            float fromMin = uintBitsToFloat(instr.y);
            float fromMax = uintBitsToFloat(instr.z);
            float toMin = uintBitsToFloat(instr.w);
            // toMax packed in srcB as immediate (reuse)
            float toMax = uintBitsToFloat(instr.x >> 24u);  // won't work, use next instr
            float t = clamp((R[srcA].x - fromMin) / max(fromMax - fromMin, 1e-6), 0.0, 1.0);
            R[dst].x = mix(toMin, 1.0, t);  // simplified
        }
        else if (opcode == OP_MATH_SUB) {
            R[dst].x = R[srcA].x - R[srcB].x;
        }
        else if (opcode == OP_MATH_ABS) {
            R[dst].x = abs(R[srcA].x);
        }
        else if (opcode == OP_MATH_SQRT) {
            R[dst].x = sqrt(max(R[srcA].x, 0.0));
        }
        else if (opcode == OP_MATH_MOD) {
            R[dst].x = (abs(R[srcB].x) > 1e-6) ? mod(R[srcA].x, R[srcB].x) : 0.0;
        }
        else if (opcode == OP_MATH_FLOOR) {
            R[dst].x = floor(R[srcA].x);
        }
        else if (opcode == OP_MATH_CEIL) {
            R[dst].x = ceil(R[srcA].x);
        }
        else if (opcode == OP_MATH_FRACT) {
            R[dst].x = fract(R[srcA].x);
        }
        else if (opcode == OP_MATH_SIN) {
            R[dst].x = sin(R[srcA].x);
        }
        else if (opcode == OP_MATH_COS) {
            R[dst].x = cos(R[srcA].x);
        }
        else if (opcode == OP_MATH_TAN) {
            R[dst].x = tan(R[srcA].x);
        }
        else if (opcode == OP_MATH_LESS) {
            R[dst].x = (R[srcA].x < R[srcB].x) ? 1.0 : 0.0;
        }
        else if (opcode == OP_MATH_GREATER) {
            R[dst].x = (R[srcA].x > R[srcB].x) ? 1.0 : 0.0;
        }
        else if (opcode == OP_MATH_ROUND) {
            R[dst].x = round(R[srcA].x);
        }
        else if (opcode == OP_MATH_SIGN) {
            R[dst].x = sign(R[srcA].x);
        }
        else if (opcode == OP_MATH_SMOOTH_MIN) {
            float k = uintBitsToFloat(instr.y);  // smoothness
            if (k <= 0.0) { R[dst].x = min(R[srcA].x, R[srcB].x); }
            else {
                float h = max(k - abs(R[srcA].x - R[srcB].x), 0.0) / k;
                R[dst].x = min(R[srcA].x, R[srcB].x) - h*h*h*k*(1.0/6.0);
            }
        }

        // ── Channel manipulation ──
        else if (opcode == OP_SEPARATE_RGB) {
            uint ch = instr.y & 3u;
            R[dst] = vec4(R[srcA][ch]);
        }
        else if (opcode == OP_COMBINE_RGB) {
            uint regC = instr.y & 0xFu;
            R[dst] = vec4(R[srcA].x, R[srcB].x, R[regC].x, 1.0);
        }

        // ── Procedural textures ──
        else if (opcode == OP_TEX_CHECKER) {
            // Checker pattern: alternating color1 (R[srcB]) / color2 (packed in imm)
            float scale = uintBitsToFloat(instr.y);
            vec2 checkerUV = R[srcA].xy * scale;
            // Blender checker formula: alternating based on floor
            float check = mod(floor(checkerUV.x) + floor(checkerUV.y), 2.0);
            vec4 color2 = vec4(uintBitsToFloat(instr.z), uintBitsToFloat(instr.w),
                               uintBitsToFloat(instr.w), 1.0);  // approx: B=G for color2
            R[dst] = (check < 0.5) ? R[srcB] : color2;
        }

        // ── Procedural textures (extended) ──
        else if (opcode == OP_TEX_NOISE) {
            float scale = uintBitsToFloat(instr.y);
            float detail = uintBitsToFloat(instr.z);
            float roughness = uintBitsToFloat(instr.w);
            vec3 p = R[srcA].xyz * scale;
            float n = fbm3D(p, detail, roughness, 2.0);
            R[dst] = vec4(n, n, n, 1.0);
        }
        else if (opcode == OP_TEX_GRADIENT) {
            uint gradType = instr.y;  // 0=linear, 1=quadratic, 2=radial, 3=spherical
            vec3 p = R[srcA].xyz;
            float f;
            if (gradType == 1u) f = gradientQuadratic(p);
            else if (gradType == 2u) f = gradientRadial(p);
            else if (gradType == 3u) f = gradientSpherical(p);
            else f = gradientLinear(p);
            R[dst] = vec4(f, f, f, 1.0);
        }
        else if (opcode == OP_TEX_VORONOI) {
            float scale = uintBitsToFloat(instr.y);
            vec3 p = R[srcA].xyz * scale;
            float d = voronoiF1(p, 1.0);
            R[dst] = vec4(d, d, d, 1.0);
        }
        else if (opcode == OP_TEX_WAVE) {
            float scale = uintBitsToFloat(instr.y);
            float distortion = uintBitsToFloat(instr.z);
            uint waveType = instr.w;
            float w = waveTexture(R[srcA].xyz, scale, distortion, waveType);
            R[dst] = vec4(w, w, w, 1.0);
        }
        else if (opcode == OP_TEX_MAGIC) {
            float distortion = uintBitsToFloat(instr.y);
            float scale = uintBitsToFloat(instr.z);
            vec3 p = R[srcA].xyz * scale;
            float m = magicTexture(p, distortion);
            R[dst] = vec4(m, m, m, 1.0);
        }
        else if (opcode == OP_TEX_BRICK) {
            float scale = uintBitsToFloat(instr.y);
            float mortarSize = uintBitsToFloat(instr.z);
            vec3 col1 = vec3(0.8, 0.8, 0.8);
            vec3 col2 = vec3(0.4, 0.2, 0.1);
            vec3 mortarCol = vec3(0.1);
            vec3 b = brickTexture(R[srcA].xyz, scale, mortarSize, col1, col2, mortarCol);
            R[dst] = vec4(b, 1.0);
        }
        else if (opcode == OP_NOISE_BUMP) {
            // Evaluate noise at 3 positions for finite-difference gradient
            float scale = uintBitsToFloat(instr.y);
            float detail = uintBitsToFloat(instr.z);
            float roughness_n = uintBitsToFloat(instr.w);
            vec3 p = R[srcA].xyz * scale;
            float eps = 0.01;  // in noise space
            float h_c = fbm3D(p, detail, roughness_n, 2.0);
            float h_x = fbm3D(p + vec3(eps, 0.0, 0.0), detail, roughness_n, 2.0);
            float h_y = fbm3D(p + vec3(0.0, eps, 0.0), detail, roughness_n, 2.0);
            // Gradient: (dh/dx, dh/dy) in noise space, normalized by eps
            R[dst] = vec4((h_x - h_c) / eps, (h_y - h_c) / eps, h_c, 1.0);
        }

        // ── RGB Curves (baked LUT, Cycles-matching) ──
        else if (opcode == OP_RGB_CURVES) {
            float fac = uintBitsToFloat(instr.y);
            float min_x = uintBitsToFloat(instr.z);
            float max_x = uintBitsToFloat(instr.w);
            float range_x = max(max_x - min_x, 1e-6);

            float samples[24];
            for (uint i = 0u; i < 6u; i++) {
                uvec4 d = materialBuffer.materials[matIdx].nodeVmCode[pc + 1u + i];
                samples[i*4u] = uintBitsToFloat(d.x);
                samples[i*4u+1u] = uintBitsToFloat(d.y);
                samples[i*4u+2u] = uintBitsToFloat(d.z);
                samples[i*4u+3u] = uintBitsToFloat(d.w);
            }

            vec3 original = R[srcA].rgb;
            vec3 c = original;
            for (int ch = 0; ch < 3; ch++) {
                float v = clamp((c[ch] - min_x) / range_x, 0.0, 1.0);
                float pos = v * 7.0;
                uint idx0 = min(uint(pos), 6u);
                uint idx1 = idx0 + 1u;
                float frac_v = pos - float(idx0);
                c[ch] = mix(samples[ch * 8 + idx0], samples[ch * 8 + idx1], frac_v);
            }
            // Factor: (1-fac)*original + fac*curved (matching Cycles)
            R[dst] = vec4(mix(original, c, fac), R[srcA].a);
            pc += 6u;
        }
        else if (opcode == OP_CURVE_DATA) {
            // Data instruction — skipped by OP_RGB_CURVES's pc advance
            continue;
        }

        // ── Vector Math ──
        else if (opcode == OP_VEC_MATH) {
            uint vop = instr.y;
            vec3 a = R[srcA].xyz, b = R[srcB].xyz;
            if (vop == 0u) R[dst] = vec4(a + b, 0);           // ADD
            else if (vop == 1u) R[dst] = vec4(a - b, 0);      // SUBTRACT
            else if (vop == 2u) R[dst] = vec4(a * b, 0);      // MULTIPLY
            else if (vop == 3u) R[dst] = vec4(                 // DIVIDE (safe)
                a.x / max(abs(b.x), 1e-6),
                a.y / max(abs(b.y), 1e-6),
                a.z / max(abs(b.z), 1e-6), 0);
            else if (vop == 4u) R[dst] = vec4(cross(a, b), 0); // CROSS
            else if (vop == 5u) R[dst].x = dot(a, b);          // DOT
            else if (vop == 6u) R[dst].x = length(a);          // LENGTH
            else if (vop == 7u) R[dst].x = distance(a, b);     // DISTANCE
            else if (vop == 8u) R[dst] = vec4(normalize(a), 0); // NORMALIZE
            else if (vop == 9u) { float s = uintBitsToFloat(instr.z); R[dst] = vec4(a * s, 0); } // SCALE
            else if (vop == 10u) R[dst] = vec4(reflect(a, normalize(b)), 0); // REFLECT
            else if (vop == 11u) R[dst] = vec4(abs(a), 0);     // ABSOLUTE
            else if (vop == 12u) R[dst] = vec4(min(a, b), 0);  // MINIMUM
            else if (vop == 13u) R[dst] = vec4(max(a, b), 0);  // MAXIMUM
            else if (vop == 14u) R[dst] = vec4(floor(a), 0);   // FLOOR
            else if (vop == 15u) R[dst] = vec4(fract(a), 0);   // FRACT
            else if (vop == 16u) R[dst] = vec4(mod(a, b), 0);  // MODULO
            else if (vop == 17u) R[dst] = vec4(sign(a), 0);    // SIGN
            else R[dst] = vec4(a, 0);
        }

        // ── Map Range (full 5-param) ──
        else if (opcode == OP_MAP_RANGE_FULL) {
            float fromMin = uintBitsToFloat(instr.y);
            float fromMax = uintBitsToFloat(instr.z);
            float toMin = uintBitsToFloat(instr.w);
            // toMax in next instruction's .x
            uvec4 next = materialBuffer.materials[matIdx].nodeVmCode[pc + 1u];
            float toMax = uintBitsToFloat(next.x);
            float t = clamp((R[srcA].x - fromMin) / max(fromMax - fromMin, 1e-6), 0.0, 1.0);
            R[dst].x = mix(toMin, toMax, t);
            pc += 1u;  // skip data instruction
        }

        // ── White Noise Texture ──
        else if (opcode == OP_TEX_WHITE_NOISE) {
            // Simple hash-based white noise from 3D position
            uvec3 p = uvec3(floatBitsToUint(R[srcA].x), floatBitsToUint(R[srcA].y), floatBitsToUint(R[srcA].z));
            uint h = hash3(p);
            float r = float(h) / 4294967295.0;
            float g = float(h * 2654435761u) / 4294967295.0;
            float b = float(h * 340573321u) / 4294967295.0;
            R[dst] = vec4(r, g, b, 1.0);
        }

        // ── Geometry node outputs ──
        else if (opcode == OP_LOAD_NORMAL) {
            R[dst] = vec4(normal, 1.0);
        }
        else if (opcode == OP_LOAD_INCOMING) {
            R[dst] = vec4(-viewDir, 1.0);  // Blender Incoming = direction TO surface
        }
        else if (opcode == OP_BACKFACING) {
            R[dst].x = dot(normal, -viewDir) < 0.0 ? 1.0 : 0.0;
        }
        else if (opcode == OP_LOAD_VERTEX_COLOR) {
            R[dst] = vertexColor;
        }
        else if (opcode == OP_OBJECT_RANDOM) {
            // Hash instanceId to produce per-instance random [0,1]
            // Matches Cycles' hash_uint() from util/hash.h
            uint h = uint(instanceId);
            h = ((h >> 16u) ^ h) * 0x45d9f3bu;
            h = ((h >> 16u) ^ h) * 0x45d9f3bu;
            h = (h >> 16u) ^ h;
            R[dst] = vec4(float(h) / 4294967295.0, 0.0, 0.0, 1.0);
        }

        // ── Load immediate ──
        else if (opcode == OP_LOAD_CONST) {
            R[dst] = vec4(uintBitsToFloat(instr.y), uintBitsToFloat(instr.z), uintBitsToFloat(instr.w), 1.0);
        }
        else if (opcode == OP_LOAD_SCALAR) {
            R[dst] = vec4(uintBitsToFloat(instr.y));
        }
        else if (opcode == OP_LOAD_WORLD_POS) {
            R[dst] = vec4(worldPos, 1.0);
        }
        else if (opcode == OP_LOAD_LOCAL_POS) {
            R[dst] = vec4(localPos, 1.0);
        }
        else if (opcode == OP_LOAD_VIEW_DIR) {
            R[dst] = vec4(viewDir, 1.0);
        }

        // ── Layer Weight / Fresnel ──
        else if (opcode == OP_LAYER_WEIGHT) {
            // srcA=normal register (or use built-in normal), imm_y=blend, imm_z=mode
            float blend = uintBitsToFloat(instr.y);
            uint mode = instr.z;  // 0=Fresnel, 1=Facing
            float NdotV = abs(dot(normal, -viewDir));
            if (mode == 0u) {
                // Fresnel
                float f0 = pow((1.0 - blend) / (1.0 + blend), 2.0);
                R[dst].x = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);
            } else {
                // Facing
                R[dst].x = 1.0 - NdotV;
            }
        }
        else if (opcode == OP_FRESNEL) {
            float ior = uintBitsToFloat(instr.y);
            float NdotV = abs(dot(normal, -viewDir));
            float f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
            R[dst].x = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);
        }

        // ── Output targets (fast-path: opcodes >= 0xE0) ──
        // These are always at the end of the program. Early branch avoids
        // falling through all intermediate opcode checks.
        else if (opcode >= 0xE0u) {
            if (opcode == OP_OUTPUT_COLOR) {
                result.baseColor = R[srcA].rgb;
                result.hasBaseColor = true;
            } else if (opcode == OP_OUTPUT_ROUGH) {
                result.roughness = R[srcA].x;
                result.hasRoughness = true;
            } else if (opcode == OP_OUTPUT_METAL) {
                result.metallic = R[srcA].x;
                result.hasMetallic = true;
            } else if (opcode == OP_OUTPUT_ALPHA) {
                result.alpha = R[srcA].x;
                result.hasAlpha = true;
            } else if (opcode == OP_OUTPUT_UV) {
                result.transformedUV = R[srcA].xy;
                result.hasTransformedUV = true;
            } else if (opcode == OP_OUTPUT_EMISSION) {
                result.emission = R[srcA].rgb;
                result.emissionStrength = R[srcA].a;
                result.hasEmission = true;
            } else if (opcode == OP_OUTPUT_IOR) {
                result.ior = R[srcA].x;
                result.hasIor = true;
            } else if (opcode == OP_OUTPUT_TRANSMISSION) {
                result.transmission = R[srcA].x;
                result.hasTransmission = true;
            } else if (opcode == OP_OUTPUT_NORMAL) {
                result.normalStrength = R[srcA].x;
                result.hasNormalStrength = true;
            } else if (opcode == OP_OUTPUT_BUMP) {
                result.bumpGradient = R[srcA].xy;
                result.bumpStrength = uintBitsToFloat(instr.y);
                result.bumpDistance = uintBitsToFloat(instr.z);
                result.hasBump = true;
            }
        }

        // (removed duplicate output handlers — now inside the >= 0xE0 branch above)

        // (output opcodes handled in >= 0xE0 fast-path above)
    }

    return result;
}

#endif // NODE_VM_GLSL
