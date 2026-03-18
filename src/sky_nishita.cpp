// Nishita Sky Model - Ported from Blender
// Based on "Display of The Earth Taking Into Account Atmospheric Scattering" by Tomoyuki Nishita et al.
// Original: Blender intern/sky (Apache 2.0 License)
// Adapted for Ignis RT

#include "sky_nishita.h"
#include "ignis_log.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace acpt {

// ============================================================
// Physical Constants
// ============================================================

static constexpr float RAYLEIGH_SCALE = 8e3f;       // Rayleigh scale height (m)
static constexpr float MIE_SCALE = 1.2e3f;          // Mie scale height (m)
static constexpr float MIE_COEFF = 2e-5f;           // Mie scattering coefficient (m^-1)
static constexpr float MIE_G = 0.76f;               // Aerosols anisotropy
static constexpr float SQR_G = MIE_G * MIE_G;       // Squared aerosols anisotropy
static constexpr float EARTH_RADIUS = 6360e3f;      // Radius of Earth (m)
static constexpr float ATMOSPHERE_RADIUS = 6420e3f; // Radius of atmosphere (m)
static constexpr int STEPS = 32;                    // Segments of primary ray
static constexpr int NUM_WAVELENGTHS = 21;          // Number of wavelengths
static constexpr int MIN_WAVELENGTH = 380;          // Lowest sampled wavelength (nm)
static constexpr int MAX_WAVELENGTH = 780;          // Highest sampled wavelength (nm)
static constexpr float STEP_LAMBDA = (MAX_WAVELENGTH - MIN_WAVELENGTH) / (NUM_WAVELENGTHS - 1);

static constexpr float M_PI_F = 3.1415926535897932f;
static constexpr float M_PI_2_F = 1.5707963267948966f;
static constexpr float M_2PI_F = 6.2831853071795864f;
static constexpr float M_1_PI_F = 0.3183098861837067f;

// Sun irradiance on top of the atmosphere (W*m^-2*nm^-1)
static constexpr float IRRADIANCE[NUM_WAVELENGTHS] = {
    1.45756829855592995315f, 1.56596305559738380175f, 1.65148449067670455293f,
    1.71496242737209314555f, 1.75797983805020541226f, 1.78256407885924539336f,
    1.79095108475838560302f, 1.78541550133410664714f, 1.76815554864306845317f,
    1.74122069647250410362f, 1.70647127164943679389f, 1.66556087452739887134f,
    1.61993437242451854274f, 1.57083597368892080581f, 1.51932335059305478886f,
    1.46628494965214395407f, 1.41245852740172450623f, 1.35844961970384092709f,
    1.30474913844739281998f, 1.25174963272610817455f, 1.19975998755420620867f
};

// Rayleigh scattering coefficient (m^-1)
static constexpr float RAYLEIGH_COEFF[NUM_WAVELENGTHS] = {
    0.00005424820087636473f, 0.00004418549866505454f, 0.00003635151910165377f,
    0.00003017929012024763f, 0.00002526320226989157f, 0.00002130859310621843f,
    0.00001809838025320633f, 0.00001547057129129042f, 0.00001330284977336850f,
    0.00001150184784075764f, 0.00000999557429990163f, 0.00000872799973630707f,
    0.00000765513700977967f, 0.00000674217203751443f, 0.00000596134125832052f,
    0.00000529034598065810f, 0.00000471115687557433f, 0.00000420910481110487f,
    0.00000377218381260133f, 0.00000339051255477280f, 0.00000305591531679811f
};

// Ozone absorption coefficient (m^-1)
static constexpr float OZONE_COEFF[NUM_WAVELENGTHS] = {
    0.00000000325126849861f, 0.00000000585395365047f, 0.00000001977191155085f,
    0.00000007309568762914f, 0.00000020084561514287f, 0.00000040383958096161f,
    0.00000063551335912363f, 0.00000096707041180970f, 0.00000154797400424410f,
    0.00000209038647223331f, 0.00000246128056164565f, 0.00000273551299461512f,
    0.00000215125863128643f, 0.00000159051840791988f, 0.00000112356197979857f,
    0.00000073527551487574f, 0.00000046450130357806f, 0.00000033096079921048f,
    0.00000022512612292678f, 0.00000014879129266490f, 0.00000016828623364192f
};

// CIE XYZ color matching functions
static constexpr float CMF_XYZ[NUM_WAVELENGTHS][3] = {
    {0.00136800000f, 0.00003900000f, 0.00645000100f},
    {0.01431000000f, 0.00039600000f, 0.06785001000f},
    {0.13438000000f, 0.00400000000f, 0.64560000000f},
    {0.34828000000f, 0.02300000000f, 1.74706000000f},
    {0.29080000000f, 0.06000000000f, 1.66920000000f},
    {0.09564000000f, 0.13902000000f, 0.81295010000f},
    {0.00490000000f, 0.32300000000f, 0.27200000000f},
    {0.06327000000f, 0.71000000000f, 0.07824999000f},
    {0.29040000000f, 0.95400000000f, 0.02030000000f},
    {0.59450000000f, 0.99500000000f, 0.00390000000f},
    {0.91630000000f, 0.87000000000f, 0.00165000100f},
    {1.06220000000f, 0.63100000000f, 0.00080000000f},
    {0.85444990000f, 0.38100000000f, 0.00019000000f},
    {0.44790000000f, 0.17500000000f, 0.00002000000f},
    {0.16490000000f, 0.06100000000f, 0.00000000000f},
    {0.04677000000f, 0.01700000000f, 0.00000000000f},
    {0.01135916000f, 0.00410200000f, 0.00000000000f},
    {0.00289932700f, 0.00104700000f, 0.00000000000f},
    {0.00069007860f, 0.00024920000f, 0.00000000000f},
    {0.00016615050f, 0.00006000000f, 0.00000000000f},
    {0.00004150994f, 0.00001499000f, 0.00000000000f}
};

// Gauss-Laguerre quadrature for optical depth integration
static constexpr int QUADRATURE_STEPS = 8;
static constexpr float QUADRATURE_NODES[QUADRATURE_STEPS] = {
    0.006811185292f, 0.03614807107f, 0.09004346519f, 0.1706680068f,
    0.2818362161f, 0.4303406404f, 0.6296271457f, 0.9145252695f
};
static constexpr float QUADRATURE_WEIGHTS[QUADRATURE_STEPS] = {
    0.01750893642f, 0.04135477391f, 0.06678839063f, 0.09507698807f,
    0.1283416365f, 0.1707430204f, 0.2327233347f, 0.3562490486f
};

// ============================================================
// Vector Math Helpers
// ============================================================

struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }

    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    float length_squared() const { return dot(*this); }
    float length() const { return sqrtf(length_squared()); }
};

static inline float clamp(float x, float min, float max) {
    return x < min ? min : (x > max ? max : x);
}

static inline float sqr(float x) {
    return x * x;
}

// ============================================================
// Atmospheric Density Functions
// ============================================================

static float density_rayleigh(float height) {
    return expf(-height / RAYLEIGH_SCALE);
}

static float density_mie(float height) {
    return expf(-height / MIE_SCALE);
}

static float density_ozone(float height) {
    return fmaxf(0.0f, 1.0f - (fabsf(height - 25000.0f) / 15000.0f));
}

// ============================================================
// Phase Functions
// ============================================================

static float phase_rayleigh(float mu) {
    return (0.1875f * M_1_PI_F) * (1.0f + sqr(mu));
}

static float phase_mie(float mu) {
    return (3.0f * (1.0f - SQR_G) * (1.0f + sqr(mu))) /
           (8.0f * M_PI_F * (2.0f + SQR_G) * powf((1.0f + SQR_G - 2.0f * MIE_G * mu), 1.5f));
}

// ============================================================
// Ray-Sphere Intersection
// ============================================================

static bool surface_intersection(const Vec3& pos, const Vec3& dir) {
    if (dir.z >= 0) return false;

    float b = 2.0f * pos.dot(dir);
    float c = pos.length_squared() - sqr(EARTH_RADIUS);
    float d = b * b - 4.0f * c;
    return d >= 0.0f;
}

static Vec3 atmosphere_intersection(const Vec3& pos, const Vec3& dir) {
    float b = 2.0f * pos.dot(dir);
    float c = pos.length_squared() - sqr(ATMOSPHERE_RADIUS);
    float t = (-b + sqrtf(b * b - 4.0f * c)) / 2.0f;
    return pos + dir * t;
}

// ============================================================
// Optical Depth Calculation
// ============================================================

static Vec3 ray_optical_depth(const Vec3& ray_origin, const Vec3& ray_dir) {
    Vec3 ray_end = atmosphere_intersection(ray_origin, ray_dir);
    float ray_length = (ray_end - ray_origin).length();
    Vec3 segment = ray_dir * ray_length;

    Vec3 optical_depth(0, 0, 0);

    for (int i = 0; i < QUADRATURE_STEPS; i++) {
        Vec3 P = ray_origin + segment * QUADRATURE_NODES[i];
        float height = P.length() - EARTH_RADIUS;

        Vec3 density(density_rayleigh(height), density_mie(height), density_ozone(height));
        optical_depth = optical_depth + density * QUADRATURE_WEIGHTS[i];
    }

    return optical_depth * ray_length;
}

// ============================================================
// Geographic to Direction Conversion
// ============================================================

static Vec3 geographical_to_direction(float lat, float lon) {
    return Vec3(cosf(lat) * cosf(lon), cosf(lat) * sinf(lon), sinf(lat));
}

// ============================================================
// Spectrum to XYZ Conversion
// ============================================================

static Vec3 spec_to_xyz(const float* spectrum) {
    Vec3 xyz(0, 0, 0);
    for (int i = 0; i < NUM_WAVELENGTHS; i++) {
        xyz.x += CMF_XYZ[i][0] * spectrum[i];
        xyz.y += CMF_XYZ[i][1] * spectrum[i];
        xyz.z += CMF_XYZ[i][2] * spectrum[i];
    }
    return xyz * STEP_LAMBDA;
}

// ============================================================
// Single Scattering Computation
// ============================================================

static void single_scattering(const Vec3& ray_dir,
                              const Vec3& sun_dir,
                              const Vec3& ray_origin,
                              float air_density,
                              float aerosol_density,
                              float ozone_density,
                              float* r_spectrum)
{
    Vec3 ray_end = atmosphere_intersection(ray_origin, ray_dir);
    float ray_length = (ray_end - ray_origin).length();

    float segment_length = ray_length / STEPS;
    Vec3 segment = ray_dir * segment_length;
    Vec3 optical_depth(0, 0, 0);

    // Initialize spectrum to zero
    for (int wl = 0; wl < NUM_WAVELENGTHS; wl++) {
        r_spectrum[wl] = 0.0f;
    }

    // Phase functions and density scaling
    float mu = ray_dir.dot(sun_dir);
    Vec3 phase_function(phase_rayleigh(mu), phase_mie(mu), 0.0f);
    Vec3 density_scale(air_density, aerosol_density, ozone_density);

    Vec3 P = ray_origin + segment * 0.5f;

    for (int i = 0; i < STEPS; i++) {
        float height = P.length() - EARTH_RADIUS;

        Vec3 density = density_scale * Vec3(density_rayleigh(height),
                                           density_mie(height),
                                           density_ozone(height));
        optical_depth = optical_depth + density * segment_length;

        // Check if sun is visible
        if (!surface_intersection(P, sun_dir)) {
            Vec3 light_optical_depth = density_scale * ray_optical_depth(P, sun_dir);
            Vec3 total_optical_depth = optical_depth + light_optical_depth;

            // Compute inscattering for each wavelength
            for (int wl = 0; wl < NUM_WAVELENGTHS; wl++) {
                Vec3 extinction_density = total_optical_depth * Vec3(RAYLEIGH_COEFF[wl],
                                                                     1.11f * MIE_COEFF,
                                                                     OZONE_COEFF[wl]);
                float attenuation = expf(-(extinction_density.x + extinction_density.y + extinction_density.z));

                Vec3 scattering_density = density * Vec3(RAYLEIGH_COEFF[wl], MIE_COEFF, 0.0f);
                float phase_scatter = phase_function.x * scattering_density.x +
                                     phase_function.y * scattering_density.y;

                r_spectrum[wl] += attenuation * phase_scatter * IRRADIANCE[wl] * segment_length;
            }
        }

        P = P + segment;
    }
}

// ============================================================
// Public API Implementation
// ============================================================

void SKY_nishita_precompute_single_scattering(
    float* pixels,
    int width,
    int height,
    float sun_elevation,
    float altitude,
    float air_density,
    float dust_density,
    float ozone_density)
{
    Log(L"[SKY-Nishita] Generating single scattering LUT (%dx%d)...\n", width, height);

    altitude = clamp(altitude, 1.0f, 59999.0f);

    const int half_width = width / 2;
    const int half_height = height / 2;
    const Vec3 cam_pos(0, 0, EARTH_RADIUS + altitude);
    const Vec3 sun_dir = geographical_to_direction(sun_elevation, 0.0f);
    const float longitude_step = M_2PI_F / width;

    // Track min/max values for debugging
    float minXYZ = FLT_MAX, maxXYZ = -FLT_MAX;

    // Compute upper hemisphere
    for (int y = half_height; y < height; y++) {
        float latitude = M_PI_2_F * sqr(float(y) / half_height - 1.0f);

        for (int x = 0; x < half_width; x++) {
            float longitude = longitude_step * x - M_PI_F;
            Vec3 dir = geographical_to_direction(latitude, longitude);

            float spectrum[NUM_WAVELENGTHS];
            single_scattering(dir, sun_dir, cam_pos, air_density, dust_density, ozone_density, spectrum);
            Vec3 xyz = spec_to_xyz(spectrum);

            // Track min/max
            float maxComponent = fmaxf(xyz.x, fmaxf(xyz.y, xyz.z));
            minXYZ = fminf(minXYZ, maxComponent);
            maxXYZ = fmaxf(maxXYZ, maxComponent);

            // Store pixel (XYZ - raw values)
            int idx = (y * width + x) * 3;
            pixels[idx + 0] = xyz.x;
            pixels[idx + 1] = xyz.y;
            pixels[idx + 2] = xyz.z;

            // Mirror to right half
            int mirror_x = width - x - 1;
            int mirror_idx = (y * width + mirror_x) * 3;
            pixels[mirror_idx + 0] = xyz.x;
            pixels[mirror_idx + 1] = xyz.y;
            pixels[mirror_idx + 2] = xyz.z;
        }

        if (y % 32 == 0) {
            Log(L"[SKY-Nishita] Progress: %d%%\n", (y * 100) / height);
        }
    }

    Log(L"[SKY-Nishita] XYZ value range: [%.6f, %.6f]\n", minXYZ, maxXYZ);

    // Debug: Sample a few pixels to see actual values
    int debugY = height - 1;  // Top row (zenith)
    int debugX = width / 2;   // Middle column
    int debugIdx = (debugY * width + debugX) * 3;
    Log(L"[SKY-Nishita] Sample zenith XYZ: (%.6f, %.6f, %.6f)\n",
        pixels[debugIdx], pixels[debugIdx+1], pixels[debugIdx+2]);

    debugY = half_height;  // Horizon
    debugIdx = (debugY * width + debugX) * 3;
    Log(L"[SKY-Nishita] Sample horizon XYZ: (%.6f, %.6f, %.6f)\n",
        pixels[debugIdx], pixels[debugIdx+1], pixels[debugIdx+2]);

    // Fill lower hemisphere with horizon fade
    for (int y = 0; y < half_height; y++) {
        float latitude = M_PI_2_F * sqr(float(y) / half_height - 1.0f);
        Vec3 dir = geographical_to_direction(latitude, 0.0f);

        float fade = 0.0f;
        if (dir.z < 0.4f) {
            fade = 1.0f - dir.z * 2.5f;
            fade = sqr(fade) * fade;
        }

        for (int x = 0; x < width; x++) {
            int src_idx = (height - 1 - y) * width * 3 + x * 3;
            int dst_idx = y * width * 3 + x * 3;

            pixels[dst_idx + 0] = pixels[src_idx + 0] * fade;
            pixels[dst_idx + 1] = pixels[src_idx + 1] * fade;
            pixels[dst_idx + 2] = pixels[src_idx + 2] * fade;
        }
    }

    Log(L"[SKY-Nishita] Single scattering LUT generated successfully\n");
}

void SKY_nishita_precompute_multiple_scattering(
    float* pixels,
    int width,
    int height,
    float sun_elevation,
    float altitude,
    float air_density,
    float dust_density,
    float ozone_density)
{
    Log(L"[SKY-Nishita] Multiple scattering not yet implemented, filling with zeros\n");
    memset(pixels, 0, width * height * 3 * sizeof(float));
}

void SKY_nishita_precompute_sun(
    float sun_elevation,
    float angular_diameter,
    float altitude,
    float air_density,
    float dust_density,
    float r_pixel_bottom[3],
    float r_pixel_top[3])
{
    // Simplified sun disk - just use constant values for now
    r_pixel_bottom[0] = 1.0f;
    r_pixel_bottom[1] = 0.9f;
    r_pixel_bottom[2] = 0.7f;

    r_pixel_top[0] = 1.0f;
    r_pixel_top[1] = 0.95f;
    r_pixel_top[2] = 0.85f;
}

float SKY_nishita_earth_intersection_angle(float altitude) {
    return acosf(EARTH_RADIUS / (EARTH_RADIUS + altitude));
}

} // namespace acpt
