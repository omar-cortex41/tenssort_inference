#pragma once

#include "trt_engine.hpp"
#include "detection.hpp"
#include "model_config.hpp"

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <functional>

namespace trt_detector {

struct FrameResult {
    cv::Mat frame;
    std::vector<Detection> detections;
    int frame_id;
    double inference_time_ms;
};

class AsyncPipeline {
public:
    AsyncPipeline();
    ~AsyncPipeline();
    
    // Non-copyable
    AsyncPipeline(const AsyncPipeline&) = delete;
    AsyncPipeline& operator=(const AsyncPipeline&) = delete;
    
    // Initialize with model config
    bool init(const ModelConfig& config);
    
    // Start processing video
    bool start(const std::string& video_path);
    bool start(int camera_id);
    
    // Stop processing
    void stop();
    
    // Get next result (blocks if not ready, returns false if stopped)
    bool getResult(FrameResult& result);
    
    // Try to get result without blocking
    bool tryGetResult(FrameResult& result);
    
    // Check if running
    bool isRunning() const { return running_; }
    
    // Get queue sizes for monitoring
    size_t getCaptureQueueSize() const;
    size_t getResultQueueSize() const;
    
    // Configuration
    void setMaxCaptureQueueSize(size_t size) { max_capture_queue_ = size; }
    void setMaxResultQueueSize(size_t size) { max_result_queue_ = size; }

private:
    void captureThread();
    void inferenceThread();
    
    TRTEngine engine_;
    cv::VideoCapture cap_;
    
    std::thread capture_thread_;
    std::thread inference_thread_;
    
    // Frame queue (capture -> inference)
    struct CaptureFrame {
        cv::Mat frame;
        int frame_id;
    };
    std::queue<CaptureFrame> capture_queue_;
    std::mutex capture_mutex_;
    std::condition_variable capture_cv_;
    
    // Result queue (inference -> consumer)
    std::queue<FrameResult> result_queue_;
    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    
    std::atomic<bool> running_{false};
    std::atomic<bool> capture_done_{false};
    std::atomic<int> frame_counter_{0};
    
    size_t max_capture_queue_ = 4;
    size_t max_result_queue_ = 4;
};

} // namespace trt_detector

