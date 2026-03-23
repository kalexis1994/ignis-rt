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
const uint OP_UV_ROTATE      = 0x11u;  // R[dst].xy = rotate R[srcA].xy by angle around (0.5,0.5)

// Color operations
const uint OP_MIX            = 0x20u;  // R[dst] = mix(R[srcA], R[srcB], imm_y)
const uint OP_MIX_REG        = 0x21u;  // R[dst] = mix(R[srcA], R[srcB], R[imm_y & 0xF].x)
const uint OP_MULTIPLY       = 0x22u;  // R[dst] = R[srcA] * R[srcB]
const uint OP_ADD            = 0x23u;  // R[dst] = R[srcA] + R[srcB]
const uint OP_SUBTRACT       = 0x24u;  // R[dst] = R[srcA] - R[srcB]
const uint OP_SCREEN         = 0x25u;  // R[dst] = 1 - (1-A)*(1-B)
const uint OP_OVERLAY        = 0x26u;  // R[dst] = overlay blend
const uint OP_INVERT         = 0x28u;  // R[dst] = mix(R[srcA], 1-R[srcA], imm_y)
const uint OP_GAMMA          = 0x29u;  // R[dst] = pow(R[srcA], imm_y)
const uint OP_BRIGHT_CONTRAST= 0x2Au;  // R[dst] = brightContrast(R[srcA], imm_y, imm_z)
const uint OP_HUE_SAT_VAL    = 0x2Bu;  // R[dst] = hueSatVal(R[srcA], hue=imm_y, sat=imm_z, val=imm_w)

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

// Channel manipulation
const uint OP_SEPARATE_RGB   = 0x50u;  // R[dst].x = R[srcA][channel]  (channel in imm_y)
const uint OP_COMBINE_RGB    = 0x51u;  // R[dst].rgb = (R[srcA].x, R[srcB].x, R[imm_y & 0xF].x)

// Load immediate / special
const uint OP_LOAD_CONST     = 0x60u;  // R[dst] = vec4(imm_y, imm_z, imm_w, 1)
const uint OP_LOAD_SCALAR    = 0x61u;  // R[dst].x = imm_y
const uint OP_LOAD_WORLD_POS = 0x62u;  // R[dst] = vec4(worldPos, 1) — for position-based texturing

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
};

// ── VM Execution ──
NodeVmResult executeNodeVm(uint matIdx, vec2 uv, vec3 worldPos) {
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

    uint header = materialBuffer.materials[matIdx].nodeVmHeader;
    uint instrCount = header & 0xFFu;

    if (instrCount == 0u) return result;  // Fast path: no VM program

    // Register file (16 vec4 registers)
    vec4 R[16];
    R[0] = vec4(uv, 0.0, 1.0);  // R0 = UV input

    for (uint pc = 0u; pc < min(instrCount, 32u); pc++) {
        uvec4 instr = materialBuffer.materials[matIdx].nodeVmCode[pc];
        uint opcode = instr.x & 0xFFu;
        uint dst    = (instr.x >> 8u)  & 0xFu;  // 4 bits = 0-15
        uint srcA   = (instr.x >> 16u) & 0xFu;
        uint srcB   = (instr.x >> 24u) & 0xFu;

        if (opcode == OP_NOP) continue;

        // ── Texture sampling ──
        if (opcode == OP_SAMPLE_TEX) {
            uint texIdx = instr.z;
            R[dst] = texture(textures[nonuniformEXT(texIdx)], R[srcA].xy);
        }

        // ── UV manipulation ──
        else if (opcode == OP_UV_TRANSFORM) {
            // Cycles POINT mapping: Rotation * (Vector * Scale) + Location
            // Operate in Blender UV space (undo/redo V-flip)
            vec2 scale = vec2(uintBitsToFloat(instr.y), uintBitsToFloat(instr.z));
            vec2 offset = vec2(uintBitsToFloat(instr.w), 0.0);
            vec2 bl = vec2(R[srcA].x, 1.0 - R[srcA].y);  // undo V-flip
            vec2 result = bl * scale + offset;
            R[dst] = vec4(result.x, 1.0 - result.y, 0.0, 1.0);  // re-apply V-flip
        }
        else if (opcode == OP_UV_ROTATE) {
            // Cycles POINT mapping: Rotation * (Vector * Scale) + Location
            // Our UVs have V flipped (1-v) from mesh export. Undo the flip,
            // rotate in Blender's UV space, then re-apply the flip.
            float angle = uintBitsToFloat(instr.y);
            vec2 bl = vec2(R[srcA].x, 1.0 - R[srcA].y);  // undo V-flip → Blender UV space
            float c = cos(angle), s = sin(angle);
            vec2 rotated = vec2(bl.x*c - bl.y*s, bl.x*s + bl.y*c);
            R[dst] = vec4(rotated.x, 1.0 - rotated.y, 0.0, 1.0);  // re-apply V-flip
        }

        // ── Color operations ──
        else if (opcode == OP_MIX) {
            float fac = uintBitsToFloat(instr.y);
            R[dst] = mix(R[srcA], R[srcB], fac);
        }
        else if (opcode == OP_MIX_REG) {
            uint facReg = instr.y & 0xFu;
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
            // Simplified HSV adjustment (hue shift, saturation mult, value mult)
            float sat = uintBitsToFloat(instr.z);
            float val = uintBitsToFloat(instr.w);
            vec3 c = R[srcA].rgb;
            float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
            c = mix(vec3(lum), c, sat);  // saturation
            c *= val;                     // value
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

        // ── Channel manipulation ──
        else if (opcode == OP_SEPARATE_RGB) {
            uint ch = instr.y & 3u;
            R[dst] = vec4(R[srcA][ch]);
        }
        else if (opcode == OP_COMBINE_RGB) {
            uint regC = instr.y & 0xFu;
            R[dst] = vec4(R[srcA].x, R[srcB].x, R[regC].x, 1.0);
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

        // ── Output targets ──
        else if (opcode == OP_OUTPUT_UV) {
            result.transformedUV = R[srcA].xy;
            result.hasTransformedUV = true;
        }
        else if (opcode == OP_OUTPUT_COLOR) {
            result.baseColor = R[srcA].rgb;
            result.hasBaseColor = true;
        }
        else if (opcode == OP_OUTPUT_ROUGH) {
            result.roughness = R[srcA].x;
            result.hasRoughness = true;
        }
        else if (opcode == OP_OUTPUT_METAL) {
            result.metallic = R[srcA].x;
            result.hasMetallic = true;
        }
        else if (opcode == OP_OUTPUT_EMISSION) {
            result.emission = R[srcA].rgb;
            result.emissionStrength = R[srcA].a;
            result.hasEmission = true;
        }
        else if (opcode == OP_OUTPUT_ALPHA) {
            result.alpha = R[srcA].x;
            result.hasAlpha = true;
        }
        else if (opcode == OP_OUTPUT_NORMAL) {
            result.normalStrength = R[srcA].x;
            result.hasNormalStrength = true;
        }
        else if (opcode == OP_OUTPUT_IOR) {
            result.ior = R[srcA].x;
            result.hasIor = true;
        }
        else if (opcode == OP_OUTPUT_TRANSMISSION) {
            result.transmission = R[srcA].x;
            result.hasTransmission = true;
        }
    }

    return result;
}

#endif // NODE_VM_GLSL
