#pragma once

#include <opencv2/core.hpp>

#include "fse/params.hpp"

namespace fse {

// Builds a fire mask and a smoke mask for each frame.
//
//  fire  : bright + warm-coloured pixels, cleaned with morphology and an
//          area filter.
//  smoke : bright, low-saturation, grey pixels with sky and vegetation
//          excluded, then smoothed over time so the mask does not flicker.
//
// The object keeps the previous smoke mask, so use one instance per video.
class MaskGenerator {
public:
    explicit MaskGenerator(const Params& p = Params()) : p_(p) {}

    // src: BGR 8-bit. Both outputs are CV_8UC1 with values 0 / 255.
    void generate(const cv::Mat& src, cv::Mat& fire_mask, cv::Mat& smoke_mask);

    void reset() { prev_smoke_.release(); }

private:
    Params p_;
    cv::Mat hsv_, gray_, prev_smoke_;
};

}  // namespace fse
