#include "fse/enhancer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <opencv2/imgproc.hpp>

#include "fse/common.hpp"
#include "fse/dehaze.hpp"

using namespace std;
using namespace cv;

namespace fse {

FireSceneEnhancer::FireSceneEnhancer(const Params& p) : p_(p), masks_(p), clahe_(p) {}

void FireSceneEnhancer::process(const Mat& src_in, Mat& dst, Mat& fire_flags) {
    CV_Assert(src_in.type() == CV_8UC3);

    // The code below indexes pixels linearly, so make sure the data is contiguous.
    const Mat src = src_in.isContinuous() ? src_in : src_in.clone();
    const int w = src.cols, h = src.rows;
    const int total = w * h;

    dst.create(src.size(), src.type());
    fire_flags.create(src.size(), CV_8UC1);

    // ---------- 1. masks ----------
    Mat fire_mask, smoke_mask;
    masks_.generate(src, fire_mask, smoke_mask);

    // Soft versions used for blending.
    Mat fire_alpha, smoke_alpha;
    GaussianBlur(fire_mask, fire_alpha, Size(25, 25), 0);
    GaussianBlur(smoke_mask, smoke_alpha, Size(35, 35), 0);

    if (fire_flag_.size() != static_cast<size_t>(total)) {
        fire_flag_.assign(total, 0);
        smoke_flag_.assign(total, 0);
        dehazed_.assign(static_cast<size_t>(total) * 3, 0);
        Y_.assign(total, 0);
        U_.assign(total, 0);
        V_.assign(total, 0);
        Y_orig_.assign(total, 0);
    }

    unsigned char* fire_flag = fire_flag_.data();
    unsigned char* smoke_flag = smoke_flag_.data();
    unsigned char* dehazed = dehazed_.data();
    int *Y = Y_.data(), *U = U_.data(), *V = V_.data(), *Y_orig = Y_orig_.data();
    const unsigned char* fire_ptr = fire_mask.data;
    const unsigned char* smoke_ptr = smoke_alpha.data;
    unsigned char* flags_out = fire_flags.data;

    int smoke_pixels = 0;
#pragma omp parallel for reduction(+ : smoke_pixels)
    for (int i = 0; i < total; ++i) {
        fire_flag[i] = fire_ptr[i] > 128 ? 1 : 0;
        smoke_flag[i] = smoke_ptr[i];
        flags_out[i] = fire_flag[i];
        if (smoke_ptr[i] > 10) ++smoke_pixels;
    }
    const float smoke_ratio = static_cast<float>(smoke_pixels) / total;

    // ---------- 2. dehaze smoke ----------
    memcpy(dehazed, src.data, static_cast<size_t>(total) * 3);
    if (smoke_ratio > p_.smoke_ratio_trigger)
        dehaze_smoke_regions(src, dehazed, smoke_flag, fire_flag, p_);

    // ---------- 3. to YUV, scene statistics ----------
    long long sum_y = 0;
    int non_fire = 0;
#pragma omp parallel for reduction(+ : sum_y, non_fire)
    for (int i = 0; i < total; ++i) {
        const unsigned char* px = smoke_flag[i] > 0 ? dehazed + i * 3 : src.data + i * 3;
        rgb_to_yuv(px[2], px[1], px[0], Y[i], U[i], V[i]);
        Y_orig[i] = Y[i];
        if (fire_flag[i] == 0) {
            sum_y += Y_orig[i];
            ++non_fire;
        }
    }

    const int fire_count = total - non_fire;
    const float fire_ratio = static_cast<float>(fire_count) / total;
    const float mean_y = static_cast<float>(sum_y) / max(1, non_fire);

    // hazard: 0..1, grows with smoke and fire coverage. Drives the CLAHE clip limit.
    const float hazard = min(1.0f, smoke_ratio * 4.0f + fire_ratio * 10.0f);
    // ambient: 0..1, how bright the non-fire part of the scene is.
    const float ambient = min(((mean_y - 16.0f) / 219.0f) * 1.5f + fire_ratio * 3.0f, 1.0f);

    // Darker scenes get a smaller gamma (stronger lift).
    gamma_ = p_.gamma_min + p_.gamma_range * ambient;

    // ---------- 4. tone curve ----------
    // smoothstep(b)^gamma plus a bump that lifts shadows, stronger in dark scenes.
    constexpr int kLut = 1024;
    float lut[kLut + 1];
    for (int i = 0; i <= kLut; ++i) {
        const float b = static_cast<float>(i) / kLut;
        const float smooth = b * b * (3.0f - 2.0f * b);
        const float base = pow(smooth, gamma_);
        const float shadow_boost = 1.5f * (1.0f - ambient) * b * exp(-6.0f * b);
        lut[i] = min(1.0f, base + shadow_boost);
    }

#pragma omp parallel for
    for (int i = 0; i < total; ++i) {
        if (fire_flag[i]) continue;
        const float norm = max(0.0f, min(1.0f, (Y[i] - 16) / 219.0f));
        const int li = min(kLut, max(0, static_cast<int>(norm * kLut)));
        Y[i] = clamp_u8(lut[li] * 219.0f + 16.0f);
    }

    // ---------- 5. CLAHE ----------
    const int clip = max(p_.clahe_clip_base, static_cast<int>(3.0f + 4.0f * hazard));
    clahe_.build(Y, fire_flag, w, h, clip);
    clahe_.apply(Y, w, h);

    // Fire keeps its original luminance.
#pragma omp parallel for
    for (int i = 0; i < total; ++i)
        if (fire_flag[i]) Y[i] = Y_orig[i];

    // ---------- 6. protected blend ----------
    // raw_weight is how much of the ORIGINAL pixel is kept (0..255). It is the
    // maximum of three protections:
    //   fire   : soft fire mask
    //   light  : bright pixels and the halo around light sources
    //   dark   : near-black pixels, so noise and tile borders are not amplified
    // Note: the weights sum to 255 and the result is shifted by 8, so a fully
    // protected pixel can come out one level darker. Kept as is to match the
    // thesis experiments.
    const unsigned char* fire_alpha_ptr = fire_alpha.data;
    const int hl_lo = p_.highlight_lo, hl_hi = p_.highlight_hi;
    const int dk_lo = p_.dark_lo, dk_hi = p_.dark_hi;
    const unsigned char* s = src.data;
    unsigned char* d = dst.data;

#pragma omp parallel for
    for (int i = 0; i < total; ++i) {
        unsigned char r_enh, g_enh, b_enh;
        yuv_to_rgb(Y[i], U[i], V[i], r_enh, g_enh, b_enh);

        const int y0 = Y_orig[i];

        const int fire_w = fire_alpha_ptr[i];

        int light_w = 0;
        if (y0 > hl_hi)
            light_w = 255;
        else if (y0 > hl_lo)
            light_w = ((y0 - hl_lo) * 255) / (hl_hi - hl_lo);

        int dark_w = 0;
        if (y0 <= dk_lo)
            dark_w = 255;
        else if (y0 < dk_hi)
            dark_w = ((dk_hi - y0) * 255) / (dk_hi - dk_lo);

        const int raw_w = max({fire_w, light_w, dark_w});
        const int enh_w = 255 - raw_w;

        d[i * 3] = static_cast<unsigned char>((s[i * 3] * raw_w + b_enh * enh_w) >> 8);
        d[i * 3 + 1] = static_cast<unsigned char>((s[i * 3 + 1] * raw_w + g_enh * enh_w) >> 8);
        d[i * 3 + 2] = static_cast<unsigned char>((s[i * 3 + 2] * raw_w + r_enh * enh_w) >> 8);
    }
}

}  // namespace fse
