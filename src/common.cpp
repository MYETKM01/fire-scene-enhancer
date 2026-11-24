#include "fse/common.hpp"

using namespace cv;

namespace fse {

void bgr_to_gray_fixed(const Mat& src, Mat& dst) {
    if (dst.empty() || dst.size() != src.size() || dst.type() != CV_8UC1)
        dst.create(src.size(), CV_8UC1);

    int rows = src.rows, cols = src.cols;
    if (src.isContinuous() && dst.isContinuous()) {
        cols *= rows;
        rows = 1;
    }

#pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        const uchar* ps = src.ptr<uchar>(i);
        uchar* pd = dst.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j) {
            const int b = ps[3 * j], g = ps[3 * j + 1], r = ps[3 * j + 2];
            pd[j] = static_cast<uchar>((77 * r + 150 * g + 29 * b) >> 8);
        }
    }
}

}  // namespace fse
