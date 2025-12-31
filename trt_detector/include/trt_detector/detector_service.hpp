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
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    std::unique_ptr<TRTEngine> engine_;
    mutable std::mutex mutex_;
};

} // namespace trt_detector

