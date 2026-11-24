#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "fse/clahe.hpp"
#include "fse/mask.hpp"
#include "fse/params.hpp"

namespace fse {

// Full per-frame pipeline:
//
//   1. fire / smoke masks (temporal smoothing on smoke)
//   2. dark-channel dehazing inside the smoke mask (only if smoke covers
//      more than Params::smoke_ratio_trigger of the frame)
//   3. scene-adaptive gamma + shadow lift on Y
//   4. mask-aware CLAHE on Y
//   5. per-pixel blend between the original and the enhanced frame, with
//      the original kept on fire, bright halos and near-black pixels
//
// One instance holds state (previous smoke mask, work buffers) and should
// be used for one video stream, from a single thread.
class FireSceneEnhancer {
public:
    explicit FireSceneEnhancer(const Params& p = Params());

    // src: BGR 8-bit. dst is (re)allocated to the same size and type.
    // fire_flags: CV_8UC1, 1 on fire pixels and 0 elsewhere (useful for
    // masked metrics).
    void process(const cv::Mat& src, cv::Mat& dst, cv::Mat& fire_flags);

    // Gamma chosen for the last processed frame.
    float last_gamma() const { return gamma_; }

private:
    Params p_;
    MaskGenerator masks_;
    MaskedClahe clahe_;
    float gamma_ = 0.6f;

    std::vector<unsigned char> fire_flag_, smoke_flag_, dehazed_;
    std::vector<int> Y_, U_, V_, Y_orig_;
};

}  // namespace fse
