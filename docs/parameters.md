# Parameters

All values live in `include/fse/params.hpp` (`fse::Params`). Defaults are the
ones used in the thesis experiments. To change them, build a `Params` object
and pass it to `FireSceneEnhancer`:

```cpp
fse::Params p;
p.smoke_ratio_trigger = 0.05f;
fse::FireSceneEnhancer enhancer(p);
```

| Parameter | Default | Effect |
|---|---|---|
| `fire_gray_threshold` | 190 | Gray level above which a warm pixel counts as fire. Lower it for dimmer flames. |
| `fire_min_area` | 80 | Fire blobs smaller than this (pixels) are dropped. |
| `smoke_min_area` | 1000 | Smoke blobs smaller than this (pixels) are dropped. |
| `smoke_temporal_alpha` | 0.72 | Weight of the current frame in the smoke mask. Lower = smoother, but slower to follow. |
| `smoke_temporal_thresh` | 80 | Threshold used to binarise the blended smoke mask. |
| `smoke_ratio_trigger` | 0.02 | Dehazing is skipped below this smoke coverage. |
| `dehaze_downscale` | 4 | Transmission is estimated at 1/N size. Larger is faster and coarser. |
| `dehaze_patch` | 5 | Dark channel window on the downscaled image. |
| `airlight_cap` | 210 | Upper bound of the estimated airlight per channel. |
| `dehaze_omega` | 0.90 | Dehazing strength in `t = 1 - omega * dark(I/A)`. |
| `dehaze_t_min` | 0.35 | Lower bound of transmission. Higher = gentler, less noise amplification. |
| `gamma_min`, `gamma_range` | 0.40, 0.50 | `gamma = gamma_min + gamma_range * ambient`, where `ambient` is 0 for a dark scene and 1 for a bright one. |
| `clahe_clip_base` | 3 | Base clip limit; raised by tile brightness and hazard level. |
| `clahe_clip_max` | 16 | Hard upper bound of the clip limit. |
| `highlight_lo`, `highlight_hi` | 100, 190 | Y range of the linear ramp that protects bright pixels and halos. |
| `dark_lo`, `dark_hi` | 30, 50 | Pixels at or below `dark_lo` stay untouched; full enhancement from `dark_hi`. |

Not exposed as parameters: the blur sizes used for blending (25x25 for fire, 35x35 for smoke) and the 8x8 CLAHE tile grid. They are fixed in the code.
