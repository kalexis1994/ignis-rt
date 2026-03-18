#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace acpt {

struct IgnisTexture {
    std::string name;
    std::vector<uint8_t> data;
    int width       = 0;
    int height      = 0;
    int mipLevels   = 1;
    unsigned int dxgiFormat = 0;   // DXGI format code (0 = raw RGBA8)
};

// Backward-compat alias
using KN5Texture = IgnisTexture;

} // namespace acpt

// Global-scope alias for code that uses KN5Texture without namespace
using acpt::KN5Texture;
