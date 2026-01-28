#pragma once

#include "trt_engine.hpp"
#include "model_config.hpp"
#include "detection.hpp"

#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>
#include <vector>

namespace trt_detector {

class DetectorService {
public:
    DetectorService();
    ~DetectorService();

    bool loadModel(const ModelConfig& config);
    void unloadModel();
    bool isLoaded() const;

    // Single frame detection
    std::vector<Detection> detect(const cv::Mat& frame);

    // Batched detection - process multiple frames in one inference call
    std::vector<std::vector<Detection>> detectBatch(const std::vector<cv::Mat>& frames);

    // Get max supported batch size
    int getMaxBatchSize() const;

private:
    std::unique_ptr<TRTEngine> engine_;
    mutable std::mutex mutex_;
};

} // namespace trt_detector

