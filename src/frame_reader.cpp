#include "fse/frame_reader.hpp"

#include <cctype>
#include <utility>

using namespace std;
using namespace cv;

namespace fse {

FrameReader::FrameReader(const string& source, int queue_size)
    : max_size_(static_cast<size_t>(queue_size > 0 ? queue_size : 1)) {
    if (!source.empty() && isdigit(static_cast<unsigned char>(source[0])))
        cap_.open(stoi(source));
    else
        cap_.open(source);

    if (cap_.isOpened()) worker_ = thread(&FrameReader::run, this);
}

FrameReader::~FrameReader() {
    {
        // Set the flag under the lock so a waiting thread cannot miss it.
        lock_guard<mutex> lk(mtx_);
        stop_ = true;
    }
    not_full_.notify_all();
    not_empty_.notify_all();
    if (worker_.joinable()) worker_.join();
    cap_.release();
}

void FrameReader::run() {
    while (!stop_) {
        Mat frame;
        if (!cap_.read(frame)) {
            {
                lock_guard<mutex> lk(mtx_);
                finished_ = true;
            }
            not_empty_.notify_all();
            break;
        }

        unique_lock<mutex> lk(mtx_);
        not_full_.wait(lk, [this] { return queue_.size() < max_size_ || stop_.load(); });
        if (stop_) break;
        queue_.push(move(frame));
        lk.unlock();
        not_empty_.notify_one();
    }
}

bool FrameReader::next(Mat& out) {
    unique_lock<mutex> lk(mtx_);
    not_empty_.wait(lk, [this] { return !queue_.empty() || finished_ || stop_.load(); });
    if (queue_.empty()) return false;
    out = move(queue_.front());
    queue_.pop();
    lk.unlock();
    not_full_.notify_one();
    return true;
}

}  // namespace fse
