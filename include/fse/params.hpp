#pragma once

namespace fse {

// All tunable constants of the pipeline in one place.
// Defaults are the values used for the thesis experiments.
struct Params {
    // --- fire mask ---
    int   fire_gray_threshold = 190;   // bright-pixel threshold on gray
    float fire_min_area       = 80.f;  // drop fire blobs smaller than this (px)

    // --- smoke mask ---
    float smoke_min_area        = 1000.f;  // drop smoke blobs smaller than this (px)
    float smoke_temporal_alpha  = 0.72f;   // weight of the current frame
    int   smoke_temporal_thresh = 80;      // re-binarise the blended mask

    // --- dark-channel dehazing (smoke regions only) ---
    float smoke_ratio_trigger = 0.02f;  // skip dehazing below this smoke coverage
    int   dehaze_downscale    = 4;      // estimate transmission at 1/4 size
    int   dehaze_patch        = 5;      // dark-channel patch size (on the small image)
    float airlight_cap        = 210.f;  // upper bound for the estimated airlight
    float dehaze_omega        = 0.90f;  // t = 1 - omega * dark(I / A)
    float dehaze_t_min        = 0.35f;  // lower bound on transmission

    // --- tone mapping ---
    float gamma_min   = 0.40f;  // gamma in the darkest scenes
    float gamma_range = 0.50f;  // gamma = gamma_min + gamma_range * ambient

    // --- CLAHE ---
    int clahe_clip_base = 3;
    int clahe_clip_max  = 16;

    // --- protection blend ---
    int highlight_lo = 100;  // linear ramp starts here ...
    int highlight_hi = 190;  // ... and reaches full protection here
    int dark_lo      = 30;   // pixels at or below this are left untouched
    int dark_hi      = 50;   // ramp back to full enhancement at this level
};

}  // namespace fse
