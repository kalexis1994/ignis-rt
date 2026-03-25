// ============================================================
// noise.glsl — Perlin noise, FBM, Voronoi, wave & gradient textures
// Procedural texture evaluation for the Node VM (Cycles-compatible)
// ============================================================

#ifndef NOISE_GLSL
#define NOISE_GLSL

// ============================================================
// Integer hash — Jenkins Lookup3 (exact Cycles match)
// Source: intern/cycles/util/hash.h, hash_uint3()
// License: Apache 2.0 (Blender Foundation)
// ============================================================

uint hash3(uvec3 k) {
    // Jenkins Lookup3 "final" mixing
    uint a = 0xdeadbf0eu + k.x;  // 0xdeadbeef + (3 << 2) + 13
    uint b = 0xdeadbf0eu + k.y;
    uint c = 0xdeadbf0eu + k.z;

    c ^= b; c -= (b << 14u) | (b >> 18u);
    a ^= c; a -= (c << 11u) | (c >> 21u);
    b ^= a; b -= (a << 25u) | (a >> 7u);
    c ^= b; c -= (b << 16u) | (b >> 16u);
    a ^= c; a -= (c << 4u)  | (c >> 28u);
    b ^= a; b -= (a << 14u) | (a >> 18u);
    c ^= b; c -= (b << 24u) | (b >> 8u);

    return c;
}

// ============================================================
// Gradient dot — Perlin improved noise (exact Cycles grad3)
// Source: intern/cycles/kernel/svm/noise.h, grad3()
// ============================================================

float gradientDot3(uint hash, float x, float y, float z) {
    uint h = hash & 15u;
    float u = h < 8u ? x : y;
    float vt = (h == 12u || h == 14u) ? x : z;
    float v = h < 4u ? y : vt;
    return ((h & 1u) != 0u ? -u : u) + ((h & 2u) != 0u ? -v : v);
}

// ============================================================
// Perlin noise 3D — quintic fade, trilinear gradient interpolation
// Returns approximately [-1, 1]
// ============================================================

float perlinNoise3D(vec3 p) {
    vec3 fl = floor(p);
    ivec3 i = ivec3(fl);
    vec3  f = p - fl;

    // Quintic fade curves (Perlin improved noise)
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    // Hash the 8 corner positions and compute gradient dots
    float g000 = gradientDot3(hash3(uvec3(i)),                    f.x,       f.y,       f.z);
    float g100 = gradientDot3(hash3(uvec3(i + ivec3(1, 0, 0))),   f.x - 1.0, f.y,       f.z);
    float g010 = gradientDot3(hash3(uvec3(i + ivec3(0, 1, 0))),   f.x,       f.y - 1.0, f.z);
    float g110 = gradientDot3(hash3(uvec3(i + ivec3(1, 1, 0))),   f.x - 1.0, f.y - 1.0, f.z);
    float g001 = gradientDot3(hash3(uvec3(i + ivec3(0, 0, 1))),   f.x,       f.y,       f.z - 1.0);
    float g101 = gradientDot3(hash3(uvec3(i + ivec3(1, 0, 1))),   f.x - 1.0, f.y,       f.z - 1.0);
    float g011 = gradientDot3(hash3(uvec3(i + ivec3(0, 1, 1))),   f.x,       f.y - 1.0, f.z - 1.0);
    float g111 = gradientDot3(hash3(uvec3(i + ivec3(1, 1, 1))),   f.x - 1.0, f.y - 1.0, f.z - 1.0);

    // Trilinear interpolation
    float x00 = mix(g000, g100, u.x);
    float x10 = mix(g010, g110, u.x);
    float x01 = mix(g001, g101, u.x);
    float x11 = mix(g011, g111, u.x);

    float y0 = mix(x00, x10, u.y);
    float y1 = mix(x01, x11, u.y);

    return mix(y0, y1, u.z);
}

// ============================================================
// FBM — Fractal Brownian Motion (Cycles-compatible)
// detail:     number of octaves (fractional allowed)
// roughness:  amplitude decay per octave (typically 0.5)
// lacunarity: frequency multiplier per octave (typically 2.0)
// Returns [0, 1]
// ============================================================

float fbm3D(vec3 p, float detail, float roughness, float lacunarity) {
    float value = 0.0;
    float amplitude = 1.0;
    float maxAmplitude = 0.0;
    float frequency = 1.0;

    int nOctaves = int(detail);
    for (int i = 0; i < nOctaves; i++) {
        value += amplitude * perlinNoise3D(p * frequency);
        maxAmplitude += amplitude;
        amplitude *= roughness;
        frequency *= lacunarity;
    }

    // Fractional octave blend
    float remainder = detail - float(nOctaves);
    if (remainder > 0.0) {
        value += remainder * amplitude * perlinNoise3D(p * frequency);
        maxAmplitude += remainder * amplitude;
    }

    // Normalize from [-1,1] to [0,1]
    if (maxAmplitude > 0.0) {
        value /= maxAmplitude;
    }
    return value * 0.5 + 0.5;
}

// ============================================================
// Gradient texture helpers
// ============================================================

float gradientLinear(vec3 p)    { return clamp(p.x, 0.0, 1.0); }
float gradientQuadratic(vec3 p) { float r = max(p.x, 0.0); return clamp(r * r, 0.0, 1.0); }
float gradientRadial(vec3 p)    { return atan(p.y, p.x) / (2.0 * 3.14159265) + 0.5; }
float gradientSpherical(vec3 p) { return max(1.0 - length(p), 0.0); }

// ============================================================
// Voronoi F1 — distance to closest cell point (Euclidean)
// randomness: 0 = regular grid, 1 = fully jittered
// Returns approximately [0, 1]
// ============================================================

float voronoiF1(vec3 p, float randomness) {
    vec3 fl = floor(p);
    vec3 f  = fract(p);

    float minDist = 1e10;

    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                vec3 offset = vec3(float(i), float(j), float(k));
                uvec3 cellId = uvec3(ivec3(fl) + ivec3(i, j, k));

                // Hash cell to get pseudo-random jitter per axis
                uint h = hash3(cellId);
                vec3 jitter = vec3(
                    float(h & 0xFFFFu) / 65535.0,
                    float((h >> 8u) & 0xFFFFu) / 65535.0,
                    float((h >> 16u) & 0xFFFFu) / 65535.0
                );
                vec3 cellPoint = offset + randomness * jitter;
                vec3 diff = cellPoint - f;
                float d = dot(diff, diff);
                minDist = min(minDist, d);
            }
        }
    }

    return sqrt(minDist);
}

// ============================================================
// Wave texture — sine or sawtooth with optional noise distortion
// waveType: 0 = sine, 1 = sawtooth
// ============================================================

float waveTexture(vec3 p, float scale, float distortion, uint waveType) {
    float coord = p.x * scale;

    if (distortion != 0.0) {
        coord += distortion * perlinNoise3D(p);
    }

    if (waveType == 0u) {
        return 0.5 + 0.5 * sin(coord);
    } else {
        return fract(coord);
    }
}

// ============================================================
// Magic texture — iterative sine distortion (Cycles-compatible)
// ============================================================

float magicTexture(vec3 p, float distortion) {
    // Cycles magic texture: iterative sine distortion
    float x = sin((p.x + p.y + p.z) * 5.0);
    float y = sin((p.x - p.y + p.z) * 5.0);
    float z = sin((p.x + p.y - p.z) * 5.0);
    if (distortion > 0.0) {
        x = sin(x * distortion);
        y = sin(y * distortion);
        z = sin(z * distortion);
    }
    return (x + y + z) / 3.0 * 0.5 + 0.5;
}

// ============================================================
// Brick texture — procedural brick pattern with mortar
// ============================================================

vec3 brickTexture(vec3 p, float scale, float mortarSize, vec3 color1, vec3 color2, vec3 mortarColor) {
    vec2 uv = p.xy * scale;
    // Offset every other row
    float row = floor(uv.y);
    if (mod(row, 2.0) > 0.5) {
        uv.x += 0.5;
    }
    vec2 brick = fract(uv);
    // Mortar
    float mx = step(mortarSize, brick.x) * step(brick.x, 1.0 - mortarSize);
    float my = step(mortarSize, brick.y) * step(brick.y, 1.0 - mortarSize);
    float isBrick = mx * my;
    // Alternate colors based on brick hash
    vec2 brickId = floor(uv);
    float h = fract(sin(dot(brickId, vec2(127.1, 311.7))) * 43758.5453);
    vec3 brickColor = mix(color1, color2, step(0.5, h));
    return mix(mortarColor, brickColor, isBrick);
}

#endif // NOISE_GLSL
