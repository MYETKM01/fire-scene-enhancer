#include "fse/clahe.hpp"

#include <algorithm>
#include <cmath>

#include "fse/common.hpp"

using namespace std;

namespace fse {

namespace {
// Y in [16, 235] -> histogram bin in [0, 255].
inline int y_to_bin(int y) { return clamp_u8((y - 16) * 255 / 219); }
}  // namespace

void MaskedClahe::build(const int* y_plane, const unsigned char* fire_flag, int w, int h,
                        int base_clip_limit) {
    const int tile_w = w / kTilesX, tile_h = h / kTilesY;

#pragma omp parallel for collapse(2)
    for (int ty = 0; ty < kTilesY; ++ty) {
        for (int tx = 0; tx < kTilesX; ++tx) {
            int hist[kBins] = {0};
            const int x0 = tx * tile_w, x1 = (tx == kTilesX - 1) ? w : x0 + tile_w;
            const int y0 = ty * tile_h, y1 = (ty == kTilesY - 1) ? h : y0 + tile_h;

            int valid = 0;
            long long y_sum = 0;
            for (int y = y0; y < y1; ++y) {
                const int row = y * w;
                for (int x = x0; x < x1; ++x) {
                    if (fire_flag[row + x] != 0) continue;
                    const int yv = y_plane[row + x];
                    ++hist[y_to_bin(yv)];
                    ++valid;
                    y_sum += yv;
                }
            }

            auto& cdf = cdf_[ty][tx];

            // Tile is all fire: identity mapping.
            if (valid == 0) {
                for (int i = 0; i < kBins; ++i) cdf[i] = static_cast<unsigned char>(i);
                continue;
            }

            // Brighter tiles get a higher clip limit, very dark tiles a lower one.
            const int avg_y = static_cast<int>(y_sum / valid);
            int clip = base_clip_limit;
            if (avg_y > 20)
                clip += (avg_y - 20) / 8;
            else
                clip = max(1, base_clip_limit - 1);
            clip = min(clip, p_.clahe_clip_max);

            int clipped = 0;
            for (int i = 0; i < kBins; ++i) {
                if (hist[i] > clip) {
                    clipped += hist[i] - clip;
                    hist[i] = clip;
                }
            }
            const int redist = clipped / kBins;
            for (int i = 0; i < kBins; ++i) hist[i] += redist;

            int sum = 0;
            const float scale = 255.0f / valid;
            for (int i = 0; i < kBins; ++i) {
                sum += hist[i];
                cdf[i] = static_cast<unsigned char>(sum * scale);
            }
        }
    }

    // Smooth the CDFs across neighbouring tiles with a [1 2 1] / 4 filter,
    // horizontally then vertically, two passes. Done in place, border tiles
    // are not touched.
    for (int iter = 0; iter < 2; ++iter) {
        for (int ty = 0; ty < kTilesY; ++ty)
            for (int tx = 1; tx < kTilesX - 1; ++tx)
                for (int k = 0; k < kBins; ++k)
                    cdf_[ty][tx][k] = (cdf_[ty][tx - 1][k] + (cdf_[ty][tx][k] << 1) +
                                       cdf_[ty][tx + 1][k]) >> 2;

        for (int tx = 0; tx < kTilesX; ++tx)
            for (int ty = 1; ty < kTilesY - 1; ++ty)
                for (int k = 0; k < kBins; ++k)
                    cdf_[ty][tx][k] = (cdf_[ty - 1][tx][k] + (cdf_[ty][tx][k] << 1) +
                                       cdf_[ty + 1][tx][k]) >> 2;
    }
}

void MaskedClahe::apply(int* y_plane, int w, int h) const {
    const int tile_w = w / kTilesX, tile_h = h / kTilesY;

#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        // Position relative to tile centres.
        const float fy = static_cast<float>(y) / tile_h - 0.5f;
        const int y_floor = static_cast<int>(floor(fy));
        const float wy = fy - y_floor;
        const int ty1 = max(0, min(kTilesY - 1, y_floor));
        const int ty2 = max(0, min(kTilesY - 1, y_floor + 1));

        for (int x = 0; x < w; ++x) {
            const float fx = static_cast<float>(x) / tile_w - 0.5f;
            const int x_floor = static_cast<int>(floor(fx));
            const float wx = fx - x_floor;
            const int tx1 = max(0, min(kTilesX - 1, x_floor));
            const int tx2 = max(0, min(kTilesX - 1, x_floor + 1));

            const int bin = y_to_bin(y_plane[y * w + x]);
            const float top = cdf_[ty1][tx1][bin] * (1.0f - wx) + cdf_[ty1][tx2][bin] * wx;
            const float bot = cdf_[ty2][tx1][bin] * (1.0f - wx) + cdf_[ty2][tx2][bin] * wx;

            // The int cast before "* 219 / 255" is intentional: it is how the
            // thesis experiments were run.
            y_plane[y * w + x] =
                clamp_u8(static_cast<int>(top * (1.0f - wy) + bot * wy) * 219 / 255 + 16);
        }
    }
}

}  // namespace fse
