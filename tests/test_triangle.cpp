// Ignis RT — Minimal test: create renderer, upload triangle, render, readback, save BMP
//
// Build:  cmake --build build --config Release --target ignis_test
// Run:    cd ignis-rt && build\Release\ignis_test.exe
//         (must run from repo root so shaders/ are found)

#include "ignis_api.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>

// ============================================================================
// BMP writer (no dependencies)
// ============================================================================

static bool WriteBMP(const char* path, const uint8_t* rgba, uint32_t w, uint32_t h) {
    // BMP stores rows bottom-to-top, 3 bytes per pixel (BGR), rows padded to 4-byte boundary
    uint32_t rowBytes = w * 3;
    uint32_t rowPad = (4 - (rowBytes % 4)) % 4;
    uint32_t dataSize = (rowBytes + rowPad) * h;
    uint32_t fileSize = 54 + dataSize;

    FILE* f = fopen(path, "wb");
    if (!f) return false;

    // BMP file header (14 bytes)
    uint8_t header[54] = {};
    header[0] = 'B'; header[1] = 'M';
    memcpy(&header[2], &fileSize, 4);
    uint32_t offset = 54;
    memcpy(&header[10], &offset, 4);

    // DIB header (40 bytes — BITMAPINFOHEADER)
    uint32_t dibSize = 40;
    memcpy(&header[14], &dibSize, 4);
    int32_t sw = (int32_t)w, sh = (int32_t)h;
    memcpy(&header[18], &sw, 4);
    memcpy(&header[22], &sh, 4);
    uint16_t planes = 1; memcpy(&header[26], &planes, 2);
    uint16_t bpp = 24;   memcpy(&header[28], &bpp, 2);
    memcpy(&header[34], &dataSize, 4);

    fwrite(header, 1, 54, f);

    // Pixel data: BMP is bottom-to-top, BGR
    std::vector<uint8_t> row(rowBytes + rowPad, 0);
    for (int32_t y = (int32_t)h - 1; y >= 0; y--) {
        for (uint32_t x = 0; x < w; x++) {
            const uint8_t* px = &rgba[(y * w + x) * 4];
            row[x * 3 + 0] = px[2]; // B
            row[x * 3 + 1] = px[1]; // G
            row[x * 3 + 2] = px[0]; // R
        }
        fwrite(row.data(), 1, rowBytes + rowPad, f);
    }

    fclose(f);
    return true;
}

// ============================================================================
// Simple 4x4 matrix helpers (column-major, matching GLM/Vulkan conventions)
// ============================================================================

static void Mat4Identity(float m[16]) {
    memset(m, 0, 64);
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void Mat4LookAt(float out[16], float eyeX, float eyeY, float eyeZ,
                        float centerX, float centerY, float centerZ,
                        float upX, float upY, float upZ) {
    float fx = centerX - eyeX, fy = centerY - eyeY, fz = centerZ - eyeZ;
    float flen = sqrtf(fx*fx + fy*fy + fz*fz);
    fx /= flen; fy /= flen; fz /= flen;

    // s = f x up
    float sx = fy*upZ - fz*upY, sy = fz*upX - fx*upZ, sz = fx*upY - fy*upX;
    float slen = sqrtf(sx*sx + sy*sy + sz*sz);
    sx /= slen; sy /= slen; sz /= slen;

    // u = s x f
    float ux = sy*fz - sz*fy, uy = sz*fx - sx*fz, uz = sx*fy - sy*fx;

    memset(out, 0, 64);
    out[0]  = sx;  out[4]  = sy;  out[8]  = sz;  out[12] = -(sx*eyeX + sy*eyeY + sz*eyeZ);
    out[1]  = ux;  out[5]  = uy;  out[9]  = uz;  out[13] = -(ux*eyeX + uy*eyeY + uz*eyeZ);
    out[2]  = -fx; out[6]  = -fy; out[10] = -fz; out[14] = (fx*eyeX + fy*eyeY + fz*eyeZ);
    out[3]  = 0;   out[7]  = 0;   out[11] = 0;   out[15] = 1.0f;
}

static void Mat4Perspective(float out[16], float fovYRad, float aspect, float near, float far) {
    float tanHalf = tanf(fovYRad / 2.0f);
    memset(out, 0, 64);
    out[0]  = 1.0f / (aspect * tanHalf);
    out[5]  = 1.0f / tanHalf;
    out[10] = -(far + near) / (far - near);
    out[11] = -1.0f;
    out[14] = -(2.0f * far * near) / (far - near);
}

static void Mat4Inverse(float inv[16], const float m[16]) {
    // Generic 4x4 inverse via cofactors
    float s0 = m[0]*m[5] - m[4]*m[1];
    float s1 = m[0]*m[6] - m[4]*m[2];
    float s2 = m[0]*m[7] - m[4]*m[3];
    float s3 = m[1]*m[6] - m[5]*m[2];
    float s4 = m[1]*m[7] - m[5]*m[3];
    float s5 = m[2]*m[7] - m[6]*m[3];

    float c5 = m[10]*m[15] - m[14]*m[11];
    float c4 = m[9]*m[15]  - m[13]*m[11];
    float c3 = m[9]*m[14]  - m[13]*m[10];
    float c2 = m[8]*m[15]  - m[12]*m[11];
    float c1 = m[8]*m[14]  - m[12]*m[10];
    float c0 = m[8]*m[13]  - m[12]*m[9];

    float det = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
    if (fabsf(det) < 1e-12f) { Mat4Identity(inv); return; }
    float id = 1.0f / det;

    inv[0]  = ( m[5]*c5 - m[6]*c4 + m[7]*c3) * id;
    inv[1]  = (-m[1]*c5 + m[2]*c4 - m[3]*c3) * id;
    inv[2]  = ( m[13]*s5 - m[14]*s4 + m[15]*s3) * id;
    inv[3]  = (-m[9]*s5 + m[10]*s4 - m[11]*s3) * id;
    inv[4]  = (-m[4]*c5 + m[6]*c2 - m[7]*c1) * id;
    inv[5]  = ( m[0]*c5 - m[2]*c2 + m[3]*c1) * id;
    inv[6]  = (-m[12]*s5 + m[14]*s2 - m[15]*s1) * id;
    inv[7]  = ( m[8]*s5 - m[10]*s2 + m[11]*s1) * id;
    inv[8]  = ( m[4]*c4 - m[5]*c2 + m[7]*c0) * id;
    inv[9]  = (-m[0]*c4 + m[1]*c2 - m[3]*c0) * id;
    inv[10] = ( m[12]*s4 - m[13]*s2 + m[15]*s0) * id;
    inv[11] = (-m[8]*s4 + m[9]*s2 - m[11]*s0) * id;
    inv[12] = (-m[4]*c3 + m[5]*c1 - m[6]*c0) * id;
    inv[13] = ( m[0]*c3 - m[1]*c1 + m[2]*c0) * id;
    inv[14] = (-m[12]*s3 + m[13]*s1 - m[14]*s0) * id;
    inv[15] = ( m[8]*s3 - m[9]*s1 + m[10]*s0) * id;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    const uint32_t WIDTH  = 800;
    const uint32_t HEIGHT = 600;

    printf("=== Ignis RT Test: Triangle Render ===\n\n");

    // ---- Step 1: Create renderer ----
    printf("[1/7] Creating renderer (%ux%u)...\n", WIDTH, HEIGHT);
    if (!ignis_create(WIDTH, HEIGHT)) {
        printf("FAIL: ignis_create() returned false.\n");
        printf("      (Missing compiled shaders? Run from repo root.)\n");
        return 1;
    }
    printf("  OK\n");

    // ---- Step 2: Upload triangle mesh ----
    printf("[2/7] Uploading triangle mesh...\n");

    // 3 vertices: position only (x, y, z)
    //   v0 = ( 0.0,  0.5, 0.0)  top      — red
    //   v1 = (-0.5, -0.5, 0.0)  bottom-left  — green
    //   v2 = ( 0.5, -0.5, 0.0)  bottom-right — blue
    float vertices[] = {
         0.0f,  0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
    };
    uint32_t indices[] = { 0, 1, 2 };

    int blasHandle = ignis_upload_mesh(vertices, 3, indices, 3);
    if (blasHandle < 0) {
        printf("FAIL: ignis_upload_mesh() returned %d\n", blasHandle);
        ignis_destroy();
        return 1;
    }
    printf("  OK (BLAS handle = %d)\n", blasHandle);

    // ---- Step 3: Upload vertex attributes (normals + UVs) ----
    printf("[3/7] Uploading vertex attributes...\n");

    // Normals: all pointing toward camera (+Z)
    float normals[] = {
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
    };
    // UVs: simple mapping
    float uvs[] = {
        0.5f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };

    if (!ignis_upload_mesh_attributes(blasHandle, normals, uvs, 3, nullptr)) {
        printf("WARN: ignis_upload_mesh_attributes() failed (non-fatal)\n");
    } else {
        printf("  OK\n");
    }

    // ---- Step 4: Build TLAS with 1 instance (identity transform) ----
    printf("[4/7] Building TLAS...\n");

    // Must match the struct layout in ignis_api.cpp (IgnisTLASInstance)
    struct TLASInstance {
        int blasIndex;
        float transform[12]; // 3x4 row-major
        uint32_t customIndex;
        uint32_t mask;
    };

    TLASInstance instance{};
    instance.blasIndex = blasHandle;
    // Identity 3x4: row 0 = (1,0,0,0), row 1 = (0,1,0,0), row 2 = (0,0,1,0)
    instance.transform[0]  = 1.0f; // m[0][0]
    instance.transform[5]  = 1.0f; // m[1][1]
    instance.transform[10] = 1.0f; // m[2][2]
    instance.customIndex = 0;
    instance.mask = 0xFF;

    if (!ignis_build_tlas(&instance, 1)) {
        printf("FAIL: ignis_build_tlas() returned false\n");
        ignis_destroy();
        return 1;
    }
    printf("  OK\n");

    // ---- Step 5: Set camera ----
    printf("[5/7] Setting camera...\n");

    // Camera at (0, 0, 2) looking at origin
    float view[16], proj[16], viewInv[16], projInv[16];
    Mat4LookAt(view, 0.0f, 0.0f, 2.0f,  0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f);
    Mat4Perspective(proj, 60.0f * 3.14159265f / 180.0f, (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);
    Mat4Inverse(viewInv, view);
    Mat4Inverse(projInv, proj);

    // Set some reasonable lighting defaults
    ignis_set_float("sun_elevation", 45.0f);
    ignis_set_float("sun_azimuth", 120.0f);
    ignis_set_float("sun_intensity", 1.5f);
    ignis_set_float("exposure", 0.8f);
    ignis_set_int("auto_sky_colors", 1);

    ignis_set_camera(viewInv, projInv, view, proj, 0);
    printf("  OK\n");

    // ---- Step 6: Render frames ----
    const int NUM_FRAMES = 4;
    printf("[6/7] Rendering %d frames...\n", NUM_FRAMES);
    for (int i = 0; i < NUM_FRAMES; i++) {
        ignis_set_camera(viewInv, projInv, view, proj, (uint32_t)i);
        ignis_render_frame();
    }
    printf("  OK\n");

    // ---- Step 7: Readback and save ----
    printf("[7/7] Reading back pixels...\n");

    uint32_t bufferSize = WIDTH * HEIGHT * 4; // RGBA8
    std::vector<uint8_t> pixels(bufferSize, 0);

    if (!ignis_readback(pixels.data(), bufferSize)) {
        printf("FAIL: ignis_readback() returned false\n");
        ignis_destroy();
        return 1;
    }

    // Count non-zero pixels
    uint32_t nonZero = 0;
    for (uint32_t i = 0; i < bufferSize; i += 4) {
        if (pixels[i] != 0 || pixels[i+1] != 0 || pixels[i+2] != 0) {
            nonZero++;
        }
    }
    printf("  Non-zero pixels: %u / %u\n", nonZero, WIDTH * HEIGHT);

    // Save BMP
    const char* outputPath = "test_output.bmp";
    if (WriteBMP(outputPath, pixels.data(), WIDTH, HEIGHT)) {
        printf("  Saved: %s\n", outputPath);
    } else {
        printf("  WARNING: Failed to save %s\n", outputPath);
    }

    // ---- Cleanup ----
    ignis_destroy();

    // ---- Result ----
    if (nonZero > 0) {
        printf("\nPASS: Rendered %u non-zero pixels.\n", nonZero);
        return 0;
    } else {
        printf("\nFAIL: All pixels are black (readback returned zeroes).\n");
        return 1;
    }
}
