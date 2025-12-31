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
    std::vector<Detection> detect(const cv::Mat& frame);
    bool isLoaded() const { return engine_ != nullptr; }
    const ModelConfig& getConfig() const { return config_; }

private:
    void allocateBuffers();
    void freeBuffers();
    
    ModelConfig config_;
    
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    
    // Buffers
    float* h_output_ = nullptr;
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    uint8_t* d_src_ = nullptr;  // Device source image for CUDA preprocess
    
    size_t input_size_ = 0;
    size_t output_size_ = 0;
    size_t src_size_ = 0;
    int num_detections_ = 0;
    
    std::string input_name_;
    std::string output_name_;
    bool use_cuda_preprocess_ = true;
};

} // namespace trt_detector

