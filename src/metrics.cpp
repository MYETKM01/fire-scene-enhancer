#include "fse/metrics.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

namespace fse {

double masked_entropy(const Mat& gray, const Mat& fire_flags) {
    CV_Assert(gray.type() == CV_8UC1 && fire_flags.type() == CV_8UC1 &&
              gray.size() == fire_flags.size());

    int hist[256] = {0};
    int valid = 0;
    for (int r = 0; r < gray.rows; ++r) {
        const uchar* g = gray.ptr<uchar>(r);
        const uchar* m = fire_flags.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c) {
            if (m[c] == 0) {
                ++hist[g[c]];
                ++valid;
            }
        }
    }
    if (valid == 0) return 0.0;

    double entropy = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (hist[i] == 0) continue;
        const double p = static_cast<double>(hist[i]) / valid;
        entropy -= p * log2(p);
    }
    return entropy;
}

double masked_avg_gradient(const Mat& gray, const Mat& fire_flags) {
    CV_Assert(gray.type() == CV_8UC1 && fire_flags.type() == CV_8UC1 &&
              gray.size() == fire_flags.size());

    Mat gx, gy, grad;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);
    magnitude(gx, gy, grad);

    double sum = 0.0;
    int valid = 0;
    for (int r = 0; r < grad.rows; ++r) {
        const float* g = grad.ptr<float>(r);
        const uchar* m = fire_flags.ptr<uchar>(r);
        for (int c = 0; c < grad.cols; ++c) {
            if (m[c] == 0) {
                sum += g[c];
                ++valid;
            }
        }
    }
    return valid > 0 ? sum / valid : 0.0;
}

}  // namespace fse
