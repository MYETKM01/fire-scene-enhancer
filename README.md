# Fire Scene Enhancer

即時火場影像增強（C++17 / OpenCV / OpenMP）。碩士論文實作，僅用 CPU。
Real-time image enhancement for fire scenes (C++17 / OpenCV / OpenMP). Master's thesis implementation, CPU only.

[中文](#中文) | [English](#english)

---

## 中文

### 這是什麼

這是我在成功大學（NCKU）資訊工程碩士論文《Real-Time Image Enhancement System for Extreme Fire Scenes》的程式碼。

火場影像有三個麻煩：煙霧讓畫面發灰、陰影處看不清楚、火焰本身太亮。如果直接做直方圖均衡化，火焰周圍會出現光暈，暗處的雜訊會被放大，畫面上還會看到格子狀的邊界。這個專案的做法是先把火和煙分出來，再針對不同區域用不同方式處理，最後把「不該動的像素」保護起來。

目標是在沒有 GPU 的電腦上即時跑完。在 1280×720 的影片上，增強處理約 30–45 FPS（只算增強本身，不含解碼和畫面顯示）。實際速度依 CPU 而定，專案會在跑完後印出平均 ms/frame，可以自己驗證。

### 處理流程

每一幀依序做以下步驟，對應 `src/enhancer.cpp`：

1. **火與煙的遮罩**（`src/mask.cpp`）
   - 火：暖色調（HSV）且夠亮的像素，經形態學處理後，去掉面積太小的區塊。
   - 煙：亮、低飽和度、近乎灰色的像素，排除偏藍（天空）和偏綠（植物）的區域，以及火的區域。再用前一幀做時間平滑，避免遮罩閃爍。
2. **去霧**（`src/dehaze.cpp`）：只在煙霧遮罩內做暗通道先驗（DCP）去霧。煙霧覆蓋率低於 2% 時直接跳過。透射率在 1/4 解析度估計，再用類似 guided filter 的方式細化，以節省時間。計算大氣光時排除火的像素。
3. **自適應 gamma**：依場景亮度決定 gamma（0.40–0.90），暗的場景提亮得比較多，另外加上一段專門拉起陰影的曲線。
4. **遮罩感知的 CLAHE**（`src/clahe.cpp`）：8×8 分塊。火的像素不計入直方圖；clip limit 依每塊亮度和整體危險程度調整；各塊的 CDF 先跨相鄰塊平滑兩次再雙線性內插，避免格子狀邊界。
5. **保護性融合**：每個像素決定要保留多少原圖。火焰、高亮度與光暈（Y 約 100 以上線性過渡、190 以上完全保留）、近乎全黑的像素（Y ≤ 30）都保留原圖，其餘才使用增強結果。

為了在 CPU 上達到即時，用了 OpenMP 平行化、整數運算的色彩轉換、gamma 查表、在低解析度估計去霧透射率，並用獨立執行緒讀取影片（生產者-消費者佇列），讓解碼和處理同時進行（`src/frame_reader.cpp`）。

### 評估方式

沒有火場影像的標準答案可以比對，所以用兩個不需要參考影像的指標，並且**排除火的像素**（火本來就不該被改變）：

- 資訊熵（Information Entropy）：灰階直方圖的熵，越高代表層次越豐富。
- 平均梯度（Average Gradient）：Sobel 梯度強度的平均，越高代表細節越清楚。

論文中在 5 段消防員第一視角的真實影片上，和 HE、標準 CLAHE、固定 gamma 比較，並用消融實驗（ablation）量化每個模組的貢獻。程式執行時會逐幀印出原圖與增強後的指標。

### 專案結構

```
include/fse/    標頭檔（公開介面與參數）
src/            實作與命令列程式
  main.cpp          命令列：讀影片、處理、顯示、寫出、印指標
  enhancer.cpp      整條處理流程
  mask.cpp          火 / 煙遮罩
  dehaze.cpp        暗通道去霧
  clahe.cpp         遮罩感知 CLAHE
  metrics.cpp       資訊熵與平均梯度
  frame_reader.cpp  背景執行緒讀取影片
docs/           補充說明
```

所有可調整的數值都放在 `include/fse/params.hpp`，預設值就是論文實驗用的設定。

### 編譯

需要 CMake 3.16 以上、支援 C++17 與 OpenMP 的編譯器、OpenCV 5（含 highgui、videoio、geometry）。

Linux / macOS：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Windows（本專案已在 MSYS2 UCRT64 + Clang + OpenCV 5 測試）：

請在 **MSYS2 UCRT64** 終端機執行（需先安裝 Clang、CMake、Ninja、OpenCV 5 和 LLVM OpenMP runtime）：

```bash
cmake -S . -B build-clang -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-clang
```

執行時也建議留在 UCRT64 終端機，讓 OpenCV、Qt 和 OpenMP 的 runtime DLL 能從 `/ucrt64/bin` 載入：

```bash
./build-clang/fire_enhance.exe your_video.mp4 -o result.mp4
```

若使用 Visual Studio，請提供與目前 OpenCV 5 安裝相容的 `OpenCV_DIR`，並確認 OpenMP 支援已啟用。

macOS 的 Apple clang 預設沒有 OpenMP，可先 `brew install libomp`。沒有 OpenMP 時需要改用 GCC。

### 使用

```bash
./build/fire_enhance <影片路徑或相機編號> [選項]

  -o <檔案>       輸出影片，預設 result.mp4
  --no-write      不輸出影片
  --no-display    不開預覽視窗（沒有螢幕的環境用）
```

範例：

```bash
./build/fire_enhance fire.mp4 -o out.mp4
./build/fire_enhance 0                      # 第 0 號相機
./build/fire_enhance fire.mp4 --no-display  # 只看數據
```

輸出影片左邊是原圖，右邊是增強結果，並標出兩者的資訊熵、平均梯度和處理時間。預覽視窗按 ESC 結束。

寬度超過 1280 的影片會先縮成 1280×720 再處理。

### 當作函式庫使用

```cpp
#include "fse/enhancer.hpp"

fse::FireSceneEnhancer enhancer;   // 一支影片用一個物件（內部保存前一幀的煙霧遮罩）
cv::Mat result, fire_flags;
enhancer.process(frame_bgr, result, fire_flags);
```

### 資料與限制

- 論文使用的消防員影片不在這個儲存庫裡，因為有使用限制。請用自己的影片測試。
- 煙霧遮罩是用顏色規則（低飽和度的亮灰色）判斷的，所以煙霧偏黃、偏黑，或場景中有大片灰白色牆面、雲時，可能判錯。
- 沒有使用深度學習，也沒有 GPU 版本。
- 處理時是一幀一幀獨立做 gamma 與 CLAHE，只有煙霧遮罩有跨幀平滑，亮度快速變化時畫面可能有輕微閃動。
- 最後融合時權重總和為 255 但用右移 8 位元，完全保留原圖的像素可能比原圖暗 1 階。這是論文實驗當時的行為，為了讓結果可重現，沒有改動。

### 授權與引用

MIT License，見 `LICENSE`。如果這份程式碼對你的研究有幫助，可以參考 `CITATION.cff`。

---

## English

### What this is

Code for my master's thesis at National Cheng Kung University (NCKU), *Real-Time Image Enhancement System for Extreme Fire Scenes*.

Fire-scene video has a few problems at once: smoke washes out the image, shadows hide detail, and the flames are far brighter than everything else. Plain histogram equalisation gives halos around the flames, amplifies noise in dark areas and leaves visible tile borders. This project first separates fire and smoke, treats each region differently, and then protects the pixels that should not be touched.

It is meant to run in real time on a CPU, with no GPU. On 1280×720 video the enhancement step runs at about 30–45 FPS (enhancement only, not decoding or display). The actual number depends on your CPU. The program prints the average ms/frame at the end, so you can check it on your own machine.

### Pipeline

Each frame goes through these steps, implemented in `src/enhancer.cpp`:

1. **Fire and smoke masks** (`src/mask.cpp`)
   - Fire: bright pixels with a warm hue (HSV), cleaned with morphology, small blobs removed.
   - Smoke: bright, low-saturation, nearly grey pixels. Bluish (sky) and greenish (vegetation) pixels and the fire region are excluded. The mask is smoothed with the previous frame so it does not flicker.
2. **Dehazing** (`src/dehaze.cpp`): dark channel prior, applied only inside the smoke mask and skipped when smoke covers less than 2% of the frame. The transmission map is estimated at 1/4 resolution and refined with a guided-filter-style step to save time. Fire pixels are ignored when estimating the airlight.
3. **Adaptive gamma**: gamma (0.40–0.90) follows the brightness of the scene, so darker scenes are lifted more. An extra curve lifts the shadows.
4. **Mask-aware CLAHE** (`src/clahe.cpp`): 8×8 tiles. Fire pixels are left out of the histograms. The clip limit depends on each tile's brightness and on the overall hazard level. Tile CDFs are smoothed across neighbouring tiles (two passes) before bilinear interpolation, which removes the grid look.
5. **Protected blend**: for every pixel, decide how much of the original to keep. Fire, bright pixels and halos (linear ramp from Y≈100, full protection above 190) and near-black pixels (Y ≤ 30) keep the original. Everything else uses the enhanced result.

To reach real time on a CPU the code uses OpenMP, integer colour conversions, a gamma lookup table, low-resolution transmission estimation, and a separate thread that reads frames through a producer-consumer queue so decoding overlaps with processing (`src/frame_reader.cpp`).

### Evaluation

There is no ground truth for fire-scene footage, so I use two no-reference metrics, both computed **without the fire pixels** (the flames are not supposed to change):

- Information entropy of the gray histogram: higher means richer tonal detail.
- Average gradient (mean Sobel magnitude): higher means clearer detail.

In the thesis the method is compared with HE, standard CLAHE and fixed gamma on five real firefighter point-of-view videos, and an ablation study measures what each module contributes. The program prints both metrics for the original and the enhanced frame as it runs.

### Layout

```
include/fse/    public headers and parameters
src/            implementation and command line tool
  main.cpp          CLI: read video, process, show, write, print metrics
  enhancer.cpp      the full pipeline
  mask.cpp          fire / smoke masks
  dehaze.cpp        dark channel dehazing
  clahe.cpp         mask-aware CLAHE
  metrics.cpp       entropy and average gradient
  frame_reader.cpp  background video reader
docs/           extra notes
```

Every tunable number is in `include/fse/params.hpp`. The defaults are the values used for the thesis experiments.

### Build

You need CMake 3.16+, a compiler with C++17 and OpenMP, and OpenCV 5 (including highgui, videoio and geometry).

Linux / macOS:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Windows (tested with MSYS2 UCRT64, Clang, OpenCV 5 and LLVM OpenMP):

Run these commands in the **MSYS2 UCRT64** terminal. Clang, CMake, Ninja, OpenCV 5 and the LLVM OpenMP runtime must be installed first.

```bash
cmake -S . -B build-clang -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-clang
```

Run the executable from UCRT64 so the OpenCV, Qt and OpenMP runtime DLLs can be found through `/ucrt64/bin`:

```bash
./build-clang/fire_enhance.exe your_video.mp4 -o result.mp4
```

For Visual Studio, point `OpenCV_DIR` to a compatible OpenCV 5 build and make sure OpenMP support is available.

Apple clang has no OpenMP by default. Run `brew install libomp`, or use GCC.

### Usage

```bash
./build/fire_enhance <video file or camera index> [options]

  -o <file>       output video, default result.mp4
  --no-write      do not write an output video
  --no-display    no preview window (for machines without a display)
```

Examples:

```bash
./build/fire_enhance fire.mp4 -o out.mp4
./build/fire_enhance 0                      # camera 0
./build/fire_enhance fire.mp4 --no-display  # numbers only
```

The output video shows the original on the left and the result on the right, with entropy, average gradient and timing printed on top. Press ESC to close the preview.

Frames wider than 1280 pixels are resized to 1280×720 before processing.

### Using it as a library

```cpp
#include "fse/enhancer.hpp"

fse::FireSceneEnhancer enhancer;   // one object per video (it keeps the previous smoke mask)
cv::Mat result, fire_flags;
enhancer.process(frame_bgr, result, fire_flags);
```

### Data and limitations

- The firefighter videos used in the thesis are not in this repository because of usage restrictions. Please test with your own footage.
- The smoke mask is a colour rule (bright, low-saturation grey). Yellowish or black smoke, or large white or grey walls and clouds in the scene, can be misclassified.
- No deep learning and no GPU version.
- Gamma and CLAHE are computed per frame. Only the smoke mask is smoothed over time, so fast brightness changes can cause slight flicker.
- The final blend uses weights that sum to 255 but shifts right by 8 bits, so a fully protected pixel can end up one level darker than the original. This is how the thesis experiments were run, and I left it unchanged so results stay reproducible.

### License and citation

MIT License, see `LICENSE`. If this code helps your research, see `CITATION.cff`.
