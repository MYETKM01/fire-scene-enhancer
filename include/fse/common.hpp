#pragma once

#include <opencv2/core.hpp>

namespace fse {

// Saturating conversions to 8 bit. The float overload truncates, which
// matches the behaviour of the original thesis code.
inline uchar clamp_u8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : static_cast<uchar>(v)); }
inline uchar clamp_u8(float v) { return v < 0.f ? 0 : (v > 255.f ? 255 : static_cast<uchar>(v)); }

// Integer BGR -> gray, weights (77, 150, 29) / 256.
void bgr_to_gray_fixed(const cv::Mat& src, cv::Mat& dst);

// Integer BT.601 limited-range conversions (Y in [16, 235]).
inline void rgb_to_yuv(uchar r, uchar g, uchar b, int& y, int& u, int& v) {
    y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
}

inline void yuv_to_rgb(int y, int u, int v, uchar& r, uchar& g, uchar& b) {
    const int c = y - 16, d = u - 128, e = v - 128;
    r = clamp_u8((298 * c + 409 * e + 128) >> 8);
    g = clamp_u8((298 * c - 100 * d - 208 * e + 128) >> 8);
    b = clamp_u8((298 * c + 516 * d + 128) >> 8);
}

}  // namespace fse
