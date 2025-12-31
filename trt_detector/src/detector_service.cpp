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

} // namespace trt_detector

