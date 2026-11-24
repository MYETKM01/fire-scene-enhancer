#pragma once

#include <array>

#include "fse/params.hpp"

namespace fse {

// Mask-aware CLAHE on the Y plane (limited range 16..235).
//
// Differences from the textbook / OpenCV version:
//  * Pixels flagged as fire are left out of every tile histogram, so a bright
//    flame does not pull the mapping of the surrounding scene.
//  * The clip limit depends on the tile's mean brightness and on the overall
//    hazard level, instead of being one global number.
//  * The per-tile CDFs are low-pass filtered across neighbouring tiles
//    (two passes) before interpolation, which removes visible tile borders.
//  * Pixels are mapped with bilinear interpolation between the four nearest
//    tile centres.
class MaskedClahe {
public:
    static constexpr int kTilesX = 8;
    static constexpr int kTilesY = 8;
    static constexpr int kBins = 256;

    explicit MaskedClahe(const Params& p = Params()) : p_(p) {}

    // Build one CDF per tile. y_plane: w*h ints. fire_flag: w*h bytes.
    void build(const int* y_plane, const unsigned char* fire_flag, int w, int h,
               int base_clip_limit);

    // Remap y_plane in place using the CDFs from the last build().
    void apply(int* y_plane, int w, int h) const;

private:
    Params p_;
    std::array<std::array<std::array<unsigned char, kBins>, kTilesX>, kTilesY> cdf_{};
};

}  // namespace fse
