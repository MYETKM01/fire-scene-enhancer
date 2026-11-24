#pragma once

#include <opencv2/core.hpp>

namespace fse {

// Both metrics ignore fire pixels (fire_flags != 0), because flames are
// supposed to stay untouched and would dominate the numbers otherwise.

// Shannon entropy of the gray histogram, in bits (0..8).
double masked_entropy(const cv::Mat& gray, const cv::Mat& fire_flags);

// Mean Sobel gradient magnitude ("average gradient").
double masked_avg_gradient(const cv::Mat& gray, const cv::Mat& fire_flags);

}  // namespace fse
