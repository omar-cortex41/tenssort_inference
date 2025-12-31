#include "trt_detector/async_pipeline.hpp"
#include <chrono>
#include <iostream>

namespace trt_detector {

AsyncPipeline::AsyncPipeline() = default;

AsyncPipeline::~AsyncPipeline() {
    stop();
}

bool AsyncPipeline::init(const ModelConfig& config) {
    return engine_.load(config);
}

bool AsyncPipeline::start(const std::string& video_path) {
    if (running_) return false;
    
    cap_.open(video_path);
    if (!cap_.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return false;
    }
    
    running_ = true;
    capture_done_ = false;
    frame_counter_ = 0;
    
    // Clear queues
    {
        std::lock_guard<std::mutex> lock(capture_mutex_);
        std::queue<CaptureFrame> empty;
        std::swap(capture_queue_, empty);
    }
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        std::queue<FrameResult> empty;
        std::swap(result_queue_, empty);
    }
    
    capture_thread_ = std::thread(&AsyncPipeline::captureThread, this);
    inference_thread_ = std::thread(&AsyncPipeline::inferenceThread, this);
    
    return true;
}

bool AsyncPipeline::start(int camera_id) {
    if (running_) return false;
    
    cap_.open(camera_id);
    if (!cap_.isOpened()) {
        std::cerr << "Failed to open camera: " << camera_id << std::endl;
        return false;
    }
    
    running_ = true;
    capture_done_ = false;
    frame_counter_ = 0;
    
    capture_thread_ = std::thread(&AsyncPipeline::captureThread, this);
    inference_thread_ = std::thread(&AsyncPipeline::inferenceThread, this);
    
    return true;
}

void AsyncPipeline::stop() {
    running_ = false;
    
    capture_cv_.notify_all();
    result_cv_.notify_all();
    
    if (capture_thread_.joinable()) capture_thread_.join();
    if (inference_thread_.joinable()) inference_thread_.join();
    
    cap_.release();
}

void AsyncPipeline::captureThread() {
    while (running_) {
        cv::Mat frame;
        if (!cap_.read(frame)) {
            capture_done_ = true;
            capture_cv_.notify_all();
            result_cv_.notify_all();
            break;
        }

        CaptureFrame cf;
        cf.frame = std::move(frame);  // Move instead of clone when possible
        cf.frame_id = frame_counter_++;

        {
            std::unique_lock<std::mutex> lock(capture_mutex_);
            // Wait if queue is full (with timeout to check running_)
            if (!capture_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return capture_queue_.size() < max_capture_queue_ || !running_;
            })) {
                continue;  // Timeout, check running_ again
            }

            if (!running_) break;

            capture_queue_.push(std::move(cf));
        }
        capture_cv_.notify_one();
    }
    capture_done_ = true;
    capture_cv_.notify_all();
    result_cv_.notify_all();
}

void AsyncPipeline::inferenceThread() {
    while (true) {
        CaptureFrame cf;

        {
            std::unique_lock<std::mutex> lock(capture_mutex_);

            // Wait with timeout
            if (!capture_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !capture_queue_.empty() || capture_done_ || !running_;
            })) {
                if (!running_) break;
                continue;
            }

            if (capture_queue_.empty()) {
                if (capture_done_ || !running_) break;
                continue;
            }

            cf = std::move(capture_queue_.front());
            capture_queue_.pop();
        }
        capture_cv_.notify_one();

        // Run inference
        auto start = std::chrono::high_resolution_clock::now();
        auto detections = engine_.detect(cf.frame);
        auto end = std::chrono::high_resolution_clock::now();

        double inference_ms = std::chrono::duration<double, std::milli>(end - start).count();

        FrameResult result;
        result.frame = std::move(cf.frame);
        result.detections = std::move(detections);
        result.frame_id = cf.frame_id;
        result.inference_time_ms = inference_ms;

        {
            std::unique_lock<std::mutex> lock(result_mutex_);

            // Wait with timeout
            if (!result_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return result_queue_.size() < max_result_queue_ || !running_;
            })) {
                if (!running_) break;
                continue;  // Drop frame if can't push in time
            }

            if (!running_) break;

            result_queue_.push(std::move(result));
        }
        result_cv_.notify_one();
    }

    // Signal end
    result_cv_.notify_all();
}

bool AsyncPipeline::getResult(FrameResult& result) {
    std::unique_lock<std::mutex> lock(result_mutex_);
    result_cv_.wait(lock, [this] {
        return !result_queue_.empty() || (!running_ && capture_done_);
    });

    if (result_queue_.empty()) return false;

    result = std::move(result_queue_.front());
    result_queue_.pop();
    result_cv_.notify_one();
    return true;
}

bool AsyncPipeline::tryGetResult(FrameResult& result) {
    std::lock_guard<std::mutex> lock(result_mutex_);

    if (result_queue_.empty()) return false;

    result = std::move(result_queue_.front());
    result_queue_.pop();
    result_cv_.notify_one();
    return true;
}

size_t AsyncPipeline::getCaptureQueueSize() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(capture_mutex_));
    return capture_queue_.size();
}

size_t AsyncPipeline::getResultQueueSize() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(result_mutex_));
    return result_queue_.size();
}

} // namespace trt_detector

