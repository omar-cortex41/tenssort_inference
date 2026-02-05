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

    // Zero-copy batched detection from GPU NV12 frames (RTSPModule integration)
    std::vector<std::vector<Detection>> detectBatchGpuNV12(
        const std::vector<uint64_t>& gpu_ptrs,
        const std::vector<int>& widths,
        const std::vector<int>& heights
    );

    // Batched detection from CPU NV12 frames (skips CPU color conversion)
    std::vector<std::vector<Detection>> detectBatchNV12(
        const std::vector<const uint8_t*>& nv12_data,
        const std::vector<size_t>& data_sizes,
        const std::vector<int>& widths,
        const std::vector<int>& heights
    );

    // Get max supported batch size
    int getMaxBatchSize() const;

private:
    std::unique_ptr<TRTEngine> engine_;
    mutable std::mutex mutex_;
};

} // namespace trt_detector

