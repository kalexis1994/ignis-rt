// nirc_mlp.glsl — Tiny MLP for NIRC (inline raygen inference)
// Architecture: input[32] → hidden[16] → output[3]
// Only 2 layers for inline speed (~600 ops vs ~10K for 4×64)

#ifndef NIRC_MLP_GLSL
#define NIRC_MLP_GLSL

#define NIRC_MLP_INPUT   32
#define NIRC_MLP_HIDDEN  16
#define NIRC_MLP_OUTPUT  3

// Weight layout: Layer1 W[32×16] + B[16] + Layer2 W[16×3] + B[3] = 579 floats
#define NIRC_W1_OFFSET   0                                              // 0
#define NIRC_B1_OFFSET   (NIRC_MLP_INPUT * NIRC_MLP_HIDDEN)            // 512
#define NIRC_W2_OFFSET   (NIRC_B1_OFFSET + NIRC_MLP_HIDDEN)            // 528
#define NIRC_B2_OFFSET   (NIRC_W2_OFFSET + NIRC_MLP_HIDDEN * NIRC_MLP_OUTPUT) // 576
#define NIRC_TOTAL_WEIGHTS (NIRC_B2_OFFSET + NIRC_MLP_OUTPUT)           // 579

float nircRelu(float x) { return max(x, 0.0); }

// Forward pass: 2 layers, ~600 multiply-adds
#ifdef NIRC_MLP_WEIGHTS_BUFFER
vec3 nircMlpForward(in float inp[NIRC_MLP_INPUT]) {
    // Layer 1: input[32] → hidden[16] + ReLU
    float hidden[NIRC_MLP_HIDDEN];
    for (int j = 0; j < NIRC_MLP_HIDDEN; j++) {
        float sum = NIRC_MLP_WEIGHTS_BUFFER[NIRC_B1_OFFSET + j];
        for (int i = 0; i < NIRC_MLP_INPUT; i++) {
            sum += inp[i] * NIRC_MLP_WEIGHTS_BUFFER[NIRC_W1_OFFSET + i * NIRC_MLP_HIDDEN + j];
        }
        hidden[j] = nircRelu(sum);
    }

    // Layer 2: hidden[16] → output[3] (softplus)
    vec3 outp;
    for (int c = 0; c < 3; c++) {
        float sum = NIRC_MLP_WEIGHTS_BUFFER[NIRC_B2_OFFSET + c];
        for (int i = 0; i < NIRC_MLP_HIDDEN; i++) {
            sum += hidden[i] * NIRC_MLP_WEIGHTS_BUFFER[NIRC_W2_OFFSET + i * NIRC_MLP_OUTPUT + c];
        }
        outp[c] = log(1.0 + exp(sum));  // softplus
    }
    return max(outp, vec3(0.0));
}
#endif

// Spherical harmonics encoding (bands 0-1, 4 coefficients)
void nircDirectionSH(in vec3 dir, out float sh[4]) {
    sh[0] = 0.28209479;
    sh[1] = 0.48860251 * dir.y;
    sh[2] = 0.48860251 * dir.z;
    sh[3] = 0.48860251 * dir.x;
}

// Build MLP input from features
void nircBuildInput(
    in float hashFeatures[NIRC_TOTAL_FEATURES],
    in vec3 direction,
    in float roughness,
    out float mlpInp[NIRC_MLP_INPUT])
{
    for (int i = 0; i < NIRC_TOTAL_FEATURES; i++) mlpInp[i] = hashFeatures[i];
    float sh[4];
    nircDirectionSH(direction, sh);
    mlpInp[24] = sh[0];
    mlpInp[25] = sh[1];
    mlpInp[26] = sh[2];
    mlpInp[27] = sh[3];
    mlpInp[28] = roughness;
    mlpInp[29] = 0.0;
    mlpInp[30] = 0.0;
    mlpInp[31] = 0.0;
}

#endif
