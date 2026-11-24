#include "fse/dehaze.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "fse/common.hpp"

using namespace std;
using namespace cv;

namespace fse {

namespace {

// Per-pixel min over B, G, R, then a min filter over a patch x patch window.
void compute_dark_channel(const Mat& bgr, Mat& dark, int patch) {
    const int rows = bgr.rows, cols = bgr.cols;
    Mat min_bgr(rows, cols, CV_8UC1);

#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        const uchar* p = bgr.ptr<uchar>(y);
        uchar* q = min_bgr.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x)
            q[x] = min({p[3 * x], p[3 * x + 1], p[3 * x + 2]});
    }

    const Mat kernel = getStructuringElement(MORPH_RECT, Size(patch, patch));
    erode(min_bgr, dark, kernel);
}

// Airlight = mean colour of the brightest 0.1% dark-channel pixels,
// ignoring fire pixels.
Vec3f estimate_airlight(const Mat& bgr, const Mat& dark, const uchar* fire_flag) {
    const int total = bgr.rows * bgr.cols;

    vector<pair<int, int>> vals;  // (dark value, pixel index)
    vals.reserve(total);
    for (int i = 0; i < total; ++i)
        if (fire_flag[i] == 0) vals.emplace_back(dark.data[i], i);

    if (vals.empty()) return Vec3f(200, 200, 200);

    const int top_n = max(1, static_cast<int>(vals.size()) / 1000);
    partial_sort(vals.begin(), vals.begin() + top_n, vals.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    Vec3d sum(0, 0, 0);
    for (int k = 0; k < top_n; ++k) {
        const int idx = vals[k].second;
        sum[0] += bgr.data[idx * 3];
        sum[1] += bgr.data[idx * 3 + 1];
        sum[2] += bgr.data[idx * 3 + 2];
    }
    return Vec3f(static_cast<float>(sum[0] / top_n), static_cast<float>(sum[1] / top_n),
                 static_cast<float>(sum[2] / top_n));
}

}  // namespace

void dehaze_smoke_regions(const Mat& src, unsigned char* dst, const unsigned char* smoke_alpha,
                          const unsigned char* fire_flag, const Params& p) {
    const int w = src.cols, h = src.rows;
    const int scale = p.dehaze_downscale;
    const Size small_size(w / scale, h / scale);

    // ---- work at low resolution ----
    Mat small;
    resize(src, small, small_size, 0, 0, INTER_AREA);

    Mat fire_small;
    resize(Mat(h, w, CV_8UC1, const_cast<unsigned char*>(fire_flag)), fire_small,
               small_size, 0, 0, INTER_NEAREST);

    Mat dark;
    compute_dark_channel(small, dark, p.dehaze_patch);

    Vec3f A = estimate_airlight(small, dark, fire_small.data);
    for (int c = 0; c < 3; ++c) A[c] = min(A[c], p.airlight_cap);

    // ---- transmission: t = 1 - omega * dark_channel(I / A) ----
    Mat normalized(small.size(), CV_32FC3);
#pragma omp parallel for
    for (int y = 0; y < small.rows; ++y) {
        const uchar* ps = small.ptr<uchar>(y);
        float* pn = normalized.ptr<float>(y);
        for (int x = 0; x < small.cols; ++x)
            for (int c = 0; c < 3; ++c) pn[3 * x + c] = ps[3 * x + c] / max(1.0f, A[c]);
    }

    Mat norm_dark(small.size(), CV_32FC1);
#pragma omp parallel for
    for (int y = 0; y < small.rows; ++y) {
        const float* pn = normalized.ptr<float>(y);
        float* q = norm_dark.ptr<float>(y);
        for (int x = 0; x < small.cols; ++x)
            q[x] = min({pn[3 * x], pn[3 * x + 1], pn[3 * x + 2]});
    }

    Mat norm_dark_8u, eroded;
    norm_dark.convertTo(norm_dark_8u, CV_8UC1, 255.0f);
    erode(norm_dark_8u, eroded,
              getStructuringElement(MORPH_RECT, Size(p.dehaze_patch, p.dehaze_patch)));

    Mat t_small(small.size(), CV_32FC1);
#pragma omp parallel for
    for (int y = 0; y < small.rows; ++y) {
        const uchar* pe = eroded.ptr<uchar>(y);
        float* pt = t_small.ptr<float>(y);
        for (int x = 0; x < small.cols; ++x) pt[x] = 1.0f - p.dehaze_omega * (pe[x] / 255.0f);
    }

    // ---- refine t with a guided filter (guide = gray image, 17x17 window) ----
    const Size win(17, 17);
    Mat gray, guide;
    cvtColor(small, gray, COLOR_BGR2GRAY);
    gray.convertTo(guide, CV_32FC1, 1.0f / 255.0f);

    Mat mean_I, mean_p, mean_Ip, mean_II;
    boxFilter(guide, mean_I, -1, win);
    boxFilter(t_small, mean_p, -1, win);
    boxFilter(guide.mul(t_small), mean_Ip, -1, win);
    boxFilter(guide.mul(guide), mean_II, -1, win);

    const Mat a = (mean_Ip - mean_I.mul(mean_p)) / (mean_II - mean_I.mul(mean_I) + 1e-3f);
    const Mat b = mean_p - a.mul(mean_I);
    boxFilter(a, mean_I, -1, win);  // reuse as mean_a
    boxFilter(b, mean_p, -1, win);  // reuse as mean_b
    const Mat t_refined_small = mean_I.mul(guide) + mean_p;

    Mat t_full;
    resize(t_refined_small, t_full, Size(w, h), 0, 0, INTER_LINEAR);

    // ---- recover the scene at full resolution, only inside the smoke mask ----
    const float t_min = p.dehaze_t_min;
    const uchar* ps = src.data;

#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        const uchar* alpha_row = smoke_alpha + y * w;
        const float* t_row = t_full.ptr<float>(y);
        const int row_off = y * w * 3;

        for (int x = 0; x < w; ++x) {
            if (alpha_row[x] == 0) continue;

            const int idx = row_off + 3 * x;
            const float inv_t = 1.0f / max(t_row[x], t_min);
            const float alpha = alpha_row[x] / 255.0f;

            for (int c = 0; c < 3; ++c) {
                const float dehazed = (static_cast<float>(ps[idx + c]) - A[c]) * inv_t + A[c];
                dst[idx + c] = clamp_u8(ps[idx + c] * (1.0f - alpha) + dehazed * alpha);
            }
        }
    }
}

}  // namespace fse
