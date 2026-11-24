#include "fse/mask.hpp"

#include <algorithm>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>

#include "fse/common.hpp"

using namespace std;
using namespace cv;

namespace fse {

namespace {

// Polygon area of a contour (shoelace formula). Same value as cv::contourArea
// for integer points, written out here because OpenCV 5 moved contourArea to
// a different header and this keeps the code building on both 4.x and 5.x.
double polygon_area(const vector<Point>& c) {
    double twice = 0.0;
    const size_t n = c.size();
    for (size_t i = 0; i < n; ++i) {
        const Point& a = c[i];
        const Point& b = c[(i + 1) % n];
        twice += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
    }
    return abs(twice) * 0.5;
}

// Keep only the external contours whose area is above min_area, filled.
Mat keep_large_blobs(const Mat& binary, double min_area) {
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat out = Mat::zeros(binary.size(), CV_8UC1);
    for (const auto& c : contours) {
        if (polygon_area(c) > min_area)
            drawContours(out, vector<vector<Point>>{c}, -1, 255, -1);
    }
    return out;
}

}  // namespace

void MaskGenerator::generate(const Mat& src, Mat& fire_mask, Mat& smoke_mask) {
    const int h = src.rows, w = src.cols;

    cvtColor(src, hsv_, COLOR_BGR2HSV);
    bgr_to_gray_fixed(src, gray_);

    // ---------------- fire ----------------
    // Warm hue (red-orange-yellow, both ends of the hue circle) and bright.
    Mat warm_lo, warm_hi;
    inRange(hsv_, Scalar(0, 40, 100), Scalar(45, 255, 255), warm_lo);
    inRange(hsv_, Scalar(150, 40, 100), Scalar(180, 255, 255), warm_hi);
    const Mat warm = warm_lo | warm_hi;

    Mat bright;
    threshold(gray_, bright, p_.fire_gray_threshold, 255, THRESH_BINARY);

    Mat fire_raw;
    bitwise_and(bright, warm, fire_raw);

    // Close small holes (dilate x2, erode x2), then drop tiny blobs.
    const Mat k5 = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(fire_raw, fire_raw, k5);
    dilate(fire_raw, fire_raw, k5);
    erode(fire_raw, fire_raw, k5);
    erode(fire_raw, fire_raw, k5);

    fire_mask = keep_large_blobs(fire_raw, p_.fire_min_area);
    blur(fire_mask, fire_mask, Size(3, 3));

    // ---------------- smoke ----------------
    // Bright, almost colourless pixels. Sky (blue-ish) and vegetation
    // (green-ish) are excluded, and so is the fire region.
    Mat smoke_raw = Mat::zeros(h, w, CV_8UC1);

#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        const Vec3b* hsv_row = hsv_.ptr<Vec3b>(y);
        const uchar* gray_row = gray_.ptr<uchar>(y);
        const uchar* src_row = src.ptr<uchar>(y);
        const uchar* fire_row = fire_mask.ptr<uchar>(y);
        uchar* out_row = smoke_raw.ptr<uchar>(y);

        for (int x = 0; x < w; ++x) {
            const int S = hsv_row[x][1];
            const int Y = gray_row[x];
            const int b = src_row[3 * x], g = src_row[3 * x + 1], r = src_row[3 * x + 2];
            const int diff = max({r, g, b}) - min({r, g, b});

            const bool grey = diff < 15;
            const bool bright_px = Y > 120;
            const bool low_sat = S < 30;
            const bool sky = b > r + 10;
            const bool vegetation = (g > r + 5) && (g > b + 5);
            const bool fire = fire_row[x] > 128;

            if (grey && bright_px && low_sat && !sky && !vegetation && !fire)
                out_row[x] = 255;
        }
    }

    const Mat k15 = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    morphologyEx(smoke_raw, smoke_raw, MORPH_OPEN, k15);
    morphologyEx(smoke_raw, smoke_raw, MORPH_CLOSE, k15);
    smoke_raw = keep_large_blobs(smoke_raw, p_.smoke_min_area);

    // ---------------- temporal smoothing ----------------
    if (prev_smoke_.empty() || prev_smoke_.size() != smoke_raw.size()) {
        smoke_raw.copyTo(prev_smoke_);
        smoke_raw.copyTo(smoke_mask);
    } else {
        Mat blended;
        const float a = p_.smoke_temporal_alpha;
        addWeighted(smoke_raw, a, prev_smoke_, 1.0f - a, 0.0, blended);
        threshold(blended, smoke_mask, p_.smoke_temporal_thresh, 255, THRESH_BINARY);
        smoke_mask.copyTo(prev_smoke_);
    }
}

}  // namespace fse
