// nirc_mlp.glsl — Tiny MLP for Neural Incident Radiance Cache
// Architecture: inp[39] → hidden[64] → hidden[64] → hidden[64] → output[3]
//
// Input layout (39D total):
//   - Hash grid features: 24D (12 levels × 2 features)
//   - Spherical harmonics: 9D (3 bands: 1 + 3 + 5 coefficients)
//   - Surface params: 6D (albedo RGB + normal θφ + roughness... padded)
//
// Simplified to inp[32] for alignment:
//   - Hash grid: 24D
//   - Direction SH: 4D (band 0 + band 1)
//   - Roughness: 1D
//   - Pad: 3D → total 32D (aligned to 16)
//
// All weights stored in a single SSBO, offsets computed at compile time.

#ifndef NIRC_MLP_GLSL
#define NIRC_MLP_GLSL

#define NIRC_MLP_INPUT   32   // padded input dimension
#define NIRC_MLP_HIDDEN  64   // neurons per hidden layer
#define NIRC_MLP_OUTPUT  3    // RGB radiance
#define NIRC_MLP_LAYERS  4    // total layers (3 hidden + 1 output)

// Weight layout in SSBO (contiguous floats):
// Layer 1: NIRC_MLP_WEIGHTS_BUFFER[INPUT × HIDDEN] + bias[HIDDEN]   = 32×64 + 64 = 2112
// Layer 2: NIRC_MLP_WEIGHTS_BUFFER[HIDDEN × HIDDEN] + bias[HIDDEN]   = 64×64 + 64 = 4160
// Layer 3: NIRC_MLP_WEIGHTS_BUFFER[HIDDEN × HIDDEN] + bias[HIDDEN]   = 64×64 + 64 = 4160
// Layer 4: NIRC_MLP_WEIGHTS_BUFFER[HIDDEN × OUTPUT] + bias[OUTPUT]    = 64×3  + 3  = 195
// Total: 10627 floats ≈ 42 KB

#define NIRC_W1_OFFSET   0
#define NIRC_B1_OFFSET   (NIRC_MLP_INPUT * NIRC_MLP_HIDDEN)           // 2048
#define NIRC_W2_OFFSET   (NIRC_B1_OFFSET + NIRC_MLP_HIDDEN)           // 2112
#define NIRC_B2_OFFSET   (NIRC_W2_OFFSET + NIRC_MLP_HIDDEN * NIRC_MLP_HIDDEN) // 6208
#define NIRC_W3_OFFSET   (NIRC_B2_OFFSET + NIRC_MLP_HIDDEN)           // 6272
#define NIRC_B3_OFFSET   (NIRC_W3_OFFSET + NIRC_MLP_HIDDEN * NIRC_MLP_HIDDEN) // 10368
#define NIRC_W4_OFFSET   (NIRC_B3_OFFSET + NIRC_MLP_HIDDEN)           // 10432
#define NIRC_B4_OFFSET   (NIRC_W4_OFFSET + NIRC_MLP_HIDDEN * NIRC_MLP_OUTPUT) // 10624
#define NIRC_TOTAL_WEIGHTS (NIRC_B4_OFFSET + NIRC_MLP_OUTPUT)          // 10627

// ReLU activation
float nircRelu(float x) { return max(x, 0.0); }

// MLP forward pass — fully fused (no global memory between layers)
// Requires NIRC_MLP_WEIGHTS_BUFFER macro defined to the SSBO accessor
// input: 32D input vector
// returns: RGB predicted incident radiance
#ifdef NIRC_MLP_WEIGHTS_BUFFER
vec3 nircMlpForward(in float inp[NIRC_MLP_INPUT]) {
    // inp = 32D input vector
    // Layer 1: inp[32] → hidden1[64] + ReLU
    float hidden1[NIRC_MLP_HIDDEN];
    for (int j = 0; j < NIRC_MLP_HIDDEN; j++) {
        float sum = NIRC_MLP_WEIGHTS_BUFFER[NIRC_B1_OFFSET + j];  // bias
        for (int i = 0; i < NIRC_MLP_INPUT; i++) {
            sum += inp[i] * NIRC_MLP_WEIGHTS_BUFFER[NIRC_W1_OFFSET + i * NIRC_MLP_HIDDEN + j];
        }
        hidden1[j] = nircRelu(sum);
    }

    // Layer 2: hidden1[64] → hidden2[64] + ReLU
    float hidden2[NIRC_MLP_HIDDEN];
    for (int j = 0; j < NIRC_MLP_HIDDEN; j++) {
        float sum = NIRC_MLP_WEIGHTS_BUFFER[NIRC_B2_OFFSET + j];
        for (int i = 0; i < NIRC_MLP_HIDDEN; i++) {
            sum += hidden1[i] * NIRC_MLP_WEIGHTS_BUFFER[NIRC_W2_OFFSET + i * NIRC_MLP_HIDDEN + j];
        }
        hidden2[j] = nircRelu(sum);
    }

    // Layer 3: hidden2[64] → hidden3[64] + ReLU
    float hidden3[NIRC_MLP_HIDDEN];
    for (int j = 0; j < NIRC_MLP_HIDDEN; j++) {
        float sum = NIRC_MLP_WEIGHTS_BUFFER[NIRC_B3_OFFSET + j];
        for (int i = 0; i < NIRC_MLP_HIDDEN; i++) {
            sum += hidden2[i] * NIRC_MLP_WEIGHTS_BUFFER[NIRC_W3_OFFSET + i * NIRC_MLP_HIDDEN + j];
        }
        hidden3[j] = nircRelu(sum);
    }

    // Layer 4: hidden3[64] → output[3] (linear, no activation)
    vec3 outp;
    for (int c = 0; c < 3; c++) {
        float sum = NIRC_MLP_WEIGHTS_BUFFER[NIRC_B4_OFFSET + c];
        for (int i = 0; i < NIRC_MLP_HIDDEN; i++) {
            sum += hidden3[i] * NIRC_MLP_WEIGHTS_BUFFER[NIRC_W4_OFFSET + i * NIRC_MLP_OUTPUT + c];
        }
        outp[c] = sum;
    }

    // Softplus output to ensure positive radiance
    outp = log(vec3(1.0) + exp(outp));
    return max(outp, vec3(0.0));
}
#endif // NIRC_MLP_WEIGHTS_BUFFER

// Spherical harmonics encoding for direction (bands 0-1, 4 coefficients)
// Returns 4 floats: Y00, Y1-1, Y10, Y11
void nircDirectionSH(in vec3 dir, out float sh[4]) {
    // Band 0
    sh[0] = 0.28209479;  // 1/(2*sqrt(pi))
    // Band 1
    sh[1] = 0.48860251 * dir.y;   // sqrt(3)/(2*sqrt(pi)) * y
    sh[2] = 0.48860251 * dir.z;   // sqrt(3)/(2*sqrt(pi)) * z
    sh[3] = 0.48860251 * dir.x;   // sqrt(3)/(2*sqrt(pi)) * x
}

// Build MLP input from hash grid features + direction + surface params
void nircBuildInput(
    in float hashFeatures[NIRC_TOTAL_FEATURES],  // 24D from hash grid
    in vec3 direction,                             // incident direction
    in float roughness,                            // surface roughness
    out float mlpInp[NIRC_MLP_INPUT])            // 32D output
{
    // Hash grid features (24D)
    for (int i = 0; i < NIRC_TOTAL_FEATURES; i++) {
        mlpInp[i] = hashFeatures[i];
    }

    // Direction SH (4D)
    float sh[4];
    nircDirectionSH(direction, sh);
    mlpInp[24] = sh[0];
    mlpInp[25] = sh[1];
    mlpInp[26] = sh[2];
    mlpInp[27] = sh[3];

    // Surface params
    mlpInp[28] = roughness;

    // Padding to 32
    mlpInp[29] = 0.0;
    mlpInp[30] = 0.0;
    mlpInp[31] = 0.0;
}

#endif // NIRC_MLP_GLSL
