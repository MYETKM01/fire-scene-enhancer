#pragma once

#include <opencv2/core.hpp>

#include "fse/params.hpp"

namespace fse {

// Dark-channel-prior dehazing, applied only where the smoke mask is set.
//
// The transmission map is estimated on a 1/4-size image and refined with a
// guided-filter-style step to keep the cost low on CPU. Fire pixels are
// excluded when the airlight is picked, otherwise flames would be taken
// as the brightest "sky" region.
//
//   src         : BGR 8-bit frame (continuous)
//   dst         : BGR buffer, same size as src. Must already hold a copy of
//                 src; only smoke pixels are overwritten.
//   smoke_alpha : w*h bytes, 0..255 blend weight per pixel
//   fire_flag   : w*h bytes, non-zero on fire pixels
void dehaze_smoke_regions(const cv::Mat& src, unsigned char* dst,
                          const unsigned char* smoke_alpha,
                          const unsigned char* fire_flag, const Params& p);

}  // namespace fse
