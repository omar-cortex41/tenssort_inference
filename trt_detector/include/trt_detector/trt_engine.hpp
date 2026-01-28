#pragma once

#include "model_config.hpp"
#include "detection.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <memory>

namespace trt_detector {

class TRTEngine {
public:
    TRTEngine();
    ~TRTEngine();

    TRTEngine(const TRTEngine&) = delete;
    TRTEngine& operator=(const TRTEngine&) = delete;
    TRTEngine(TRTEngine&&) noexcept;
    TRTEngine& operator=(TRTEngine&&) noexcept;

    bool load(const ModelConfig& config);

    // Single frame detection
    std::vector<Detection> detect(const cv::Mat& frame);

    // Batched detection - returns vector of detections per frame
    std::vector<std::vector<Detection>> detectBatch(const std::vector<cv::Mat>& frames);

    bool isLoaded() const { return engine_ != nullptr; }
    const ModelConfig& getConfig() const { return config_; }
    int getMaxBatchSize() const { return max_batch_size_; }

private:
    void allocateBuffers();
    void freeBuffers();

    ModelConfig config_;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;

    // Buffers (sized for max batch)
    float* h_output_ = nullptr;
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    uint8_t* d_src_ = nullptr;  // Device source image for CUDA preprocess

    size_t input_size_per_batch_ = 0;   // Size for single image
    size_t output_size_per_batch_ = 0;  // Size for single image output
    size_t input_size_ = 0;             // Total allocated (max_batch * per_batch)
    size_t output_size_ = 0;            // Total allocated
    size_t src_size_ = 0;
    int num_detections_ = 0;
    int max_batch_size_ = 1;

    std::string input_name_;
    std::string output_name_;
    bool use_cuda_preprocess_ = true;
};

} // namespace trt_detector

