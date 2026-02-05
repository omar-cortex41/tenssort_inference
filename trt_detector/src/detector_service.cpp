#include "trt_detector/detector_service.hpp"

namespace trt_detector {

DetectorService::DetectorService() = default;
DetectorService::~DetectorService() = default;

bool DetectorService::loadModel(const ModelConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (engine_) return false;  // Already loaded

    auto engine = std::make_unique<TRTEngine>();
    if (!engine->load(config)) return false;

    engine_ = std::move(engine);
    return true;
}

void DetectorService::unloadModel() {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.reset();
}

bool DetectorService::isLoaded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_ != nullptr;
}

std::vector<Detection> DetectorService::detect(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!engine_) return {};
    return engine_->detect(frame);
}

std::vector<std::vector<Detection>> DetectorService::detectBatch(const std::vector<cv::Mat>& frames) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!engine_) return {};
    return engine_->detectBatch(frames);
}

std::vector<std::vector<Detection>> DetectorService::detectBatchGpuNV12(
    const std::vector<uint64_t>& gpu_ptrs,
    const std::vector<int>& widths,
    const std::vector<int>& heights
) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!engine_) return {};
    return engine_->detectBatchGpuNV12(gpu_ptrs, widths, heights);
}

std::vector<std::vector<Detection>> DetectorService::detectBatchNV12(
    const std::vector<const uint8_t*>& nv12_data,
    const std::vector<size_t>& data_sizes,
    const std::vector<int>& widths,
    const std::vector<int>& heights
) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!engine_) return {};
    return engine_->detectBatchNV12(nv12_data, data_sizes, widths, heights);
}

int DetectorService::getMaxBatchSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!engine_) return 0;
    return engine_->getMaxBatchSize();
}

} // namespace trt_detector

