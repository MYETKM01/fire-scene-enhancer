#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <opencv2/videoio.hpp>

namespace fse {

// Reads frames on a background thread and hands them to the consumer through
// a bounded queue (producer-consumer). Video decoding then overlaps with
// processing instead of adding to the per-frame time.
//
// source: a video path, or a camera index such as "0".
class FrameReader {
public:
    explicit FrameReader(const std::string& source, int queue_size = 60);
    ~FrameReader();

    FrameReader(const FrameReader&) = delete;
    FrameReader& operator=(const FrameReader&) = delete;

    bool is_opened() const { return cap_.isOpened(); }
    int get(int prop_id) const { return static_cast<int>(cap_.get(prop_id)); }

    // Blocks until a frame is available. Returns false at end of stream.
    bool next(cv::Mat& out);

private:
    void run();

    cv::VideoCapture cap_;
    std::queue<cv::Mat> queue_;
    std::mutex mtx_;
    std::condition_variable not_full_, not_empty_;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    bool finished_ = false;  // guarded by mtx_
    size_t max_size_;
};

}  // namespace fse
