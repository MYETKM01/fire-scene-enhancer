// Command line demo: run the enhancer on a video file or a camera, show the
// original and enhanced frames side by side, and print per-frame timing and
// image-quality metrics.

#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include "fse/enhancer.hpp"
#include "fse/frame_reader.hpp"
#include "fse/metrics.hpp"

using namespace std;
using namespace cv;

namespace {

struct Options {
    string input;
    string output = "result.mp4";
    bool display = true;
    bool write = true;
};

void print_usage(const char* exe) {
    cout << "Usage: " << exe << " <video file | camera index> [options]\n"
              << "  -o <file>      output video (default: result.mp4)\n"
              << "  --no-write     do not write an output video\n"
              << "  --no-display   do not open a preview window (ESC closes it otherwise)\n";
}

bool parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const string a = argv[i];
        if (a == "-o" && i + 1 < argc)
            opt.output = argv[++i];
        else if (a == "--no-write")
            opt.write = false;
        else if (a == "--no-display")
            opt.display = false;
        else if (a == "-h" || a == "--help")
            return false;
        else if (opt.input.empty())
            opt.input = a;
        else
            return false;
    }
    return !opt.input.empty();
}

string fmt(double v, int digits) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*f", digits, v);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    Options opt;
    cout << unitbuf;
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }

    fse::FrameReader reader(opt.input);
    if (!reader.is_opened()) {
        cerr << "Cannot open input: " << opt.input << "\n";
        return 1;
    }

    // Frames wider than 1280 are scaled to 1280x720 before processing.
    const int in_w = reader.get(CAP_PROP_FRAME_WIDTH);
    const int in_h = reader.get(CAP_PROP_FRAME_HEIGHT);
    const bool downscale = in_w > 1280;
    const Size proc_size = downscale ? Size(1280, 720) : Size(in_w, in_h);

    VideoWriter writer;
    if (opt.write) {
        const Size out_size(proc_size.width * 2, proc_size.height);
        writer.open(opt.output, VideoWriter::fourcc('a', 'v', 'c', '1'), 30.0, out_size);
        if (!writer.isOpened())
            writer.open(opt.output, VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, out_size);
        if (!writer.isOpened())
            cerr << "Warning: cannot open " << opt.output << ", running without output.\n";
    }

    cout << "| Frame |  ms  | FPS | Gamma | OEnt | EEnt | +Ent | OGrad | EGrad | +Grad |\n";

    fse::FireSceneEnhancer enhancer;

    Mat raw, frame, result, fire_flags, combined, gray_in, gray_out;
    int idx = 0;
    double total_ms = 0.0;

    while (reader.next(raw)) {
        if (downscale)
            resize(raw, frame, proc_size);
        else
            frame = raw;

        // Only the enhancement itself is timed (no decoding, metrics or drawing).
        const double t0 = static_cast<double>(getTickCount());
        enhancer.process(frame, result, fire_flags);
        const double ms = (static_cast<double>(getTickCount()) - t0) /
                          getTickFrequency() * 1000.0;
        total_ms += ms;
        const double fps = 1000.0 / ms;

        cvtColor(frame, gray_in, COLOR_BGR2GRAY);
        cvtColor(result, gray_out, COLOR_BGR2GRAY);
        const double ent_in = fse::masked_entropy(gray_in, fire_flags);
        const double ent_out = fse::masked_entropy(gray_out, fire_flags);
        const double grad_in = fse::masked_avg_gradient(gray_in, fire_flags);
        const double grad_out = fse::masked_avg_gradient(gray_out, fire_flags);

        if (idx % 10 == 0) {
            cout << "| " << setw(5) << idx << " | " << setw(4) << fmt(ms, 1)
                      << " | " << setw(3) << static_cast<int>(fps) << " | " << setw(5)
                      << fmt(enhancer.last_gamma(), 2) << " | " << setw(4) << fmt(ent_in, 2)
                      << " | " << setw(4) << fmt(ent_out, 2) << " | " << setw(4)
                      << fmt(ent_out - ent_in, 2) << " | " << setw(5) << fmt(grad_in, 1)
                      << " | " << setw(5) << fmt(grad_out, 1) << " | " << setw(5)
                      << fmt(grad_out - grad_in, 1) << " |\n";
        }

        if (writer.isOpened() || opt.display) {
            hconcat(frame, result, combined);
            const int ox = frame.cols + 30;
            putText(combined, "Original", {30, 40}, FONT_HERSHEY_SIMPLEX, 1,
                        Scalar(255, 255, 255), 2);
            putText(combined, "Ent " + fmt(ent_in, 2) + "  Grad " + fmt(grad_in, 0),
                        {30, 80}, FONT_HERSHEY_SIMPLEX, 0.65, Scalar(180, 180, 180), 1);
            putText(combined, "Enhanced", {ox, 40}, FONT_HERSHEY_SIMPLEX, 0.9,
                        Scalar(0, 255, 0), 2);
            putText(combined,
                        "Ent " + fmt(ent_out, 2) + " (" + fmt(ent_out - ent_in, 2) + ")  Grad " +
                            fmt(grad_out, 0) + " (" + fmt(grad_out - grad_in, 0) + ")",
                        {ox, 80}, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 220, 180), 1);
            putText(combined,
                        "Time " + fmt(ms, 1) + " ms  FPS " + to_string(static_cast<int>(fps)) +
                            "  Gamma " + fmt(enhancer.last_gamma(), 2),
                        {ox, 115}, FONT_HERSHEY_SIMPLEX, 0.65, Scalar(0, 100, 255), 2);

            if (writer.isOpened()) writer.write(combined);
            if (opt.display) {
                imshow("Fire scene enhancement", combined);
                if (waitKey(1) == 27) break;
            }
        }
        ++idx;
    }

    if (idx > 0) {
        const double avg = total_ms / idx;
        cout << "\n" << idx << " frames, average " << fmt(avg, 1) << " ms/frame ("
                  << fmt(1000.0 / avg, 1) << " FPS, enhancement only)\n";
    }

    if (opt.display) destroyAllWindows();
    return 0;
}
