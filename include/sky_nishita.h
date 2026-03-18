#pragma once

// Nishita Sky Model - Ported from Blender
// Based on "Display of The Earth Taking Into Account Atmospheric Scattering" by Tomoyuki Nishita et al.
// Original: Blender intern/sky (Apache 2.0 License)
// Adapted for Ignis RT

#include <cstdint>

namespace acpt {

// ============================================================
// Sky Nishita - Physically Based Atmospheric Scattering
// ============================================================

// Precompute single scattering lookup table
// This generates a 2D texture (width x height) with RGB values
// representing atmospheric scattering as a function of view angle
void SKY_nishita_precompute_single_scattering(
    float* pixels,           // Output: RGB texture (width * height * 3 floats)
    int width,               // Texture width (recommended: 512)
    int height,              // Texture height (recommended: 256)
    float sun_elevation,     // Sun elevation angle in radians [-π/2, π/2]
    float altitude,          // Observer altitude in meters (0 = sea level)
    float air_density,       // Air density factor [0, 1] (1 = normal)
    float dust_density,      // Aerosol/dust density [0, 1] (0 = clear, 1 = hazy)
    float ozone_density      // Ozone density [0, 1] (1 = normal)
);

// Precompute multiple scattering lookup table
// Multiple scattering adds realism for thick atmospheres
void SKY_nishita_precompute_multiple_scattering(
    float* pixels,           // Output: RGB texture (width * height * 3 floats)
    int width,
    int height,
    float sun_elevation,
    float altitude,
    float air_density,
    float dust_density,
    float ozone_density
);

// Precompute sun disk color (single scattering)
// Returns two colors: bottom and top of sun disk
void SKY_nishita_precompute_sun(
    float sun_elevation,
    float angular_diameter,  // Sun angular diameter in radians (default: 0.00935 for realistic sun)
    float altitude,
    float air_density,
    float dust_density,
    float r_pixel_bottom[3], // Output: RGB for bottom of sun
    float r_pixel_top[3]     // Output: RGB for top of sun
);

// Utility: Calculate horizon angle based on altitude
// Used to adjust view angles for observers above sea level
float SKY_nishita_earth_intersection_angle(float altitude);

} // namespace acpt
