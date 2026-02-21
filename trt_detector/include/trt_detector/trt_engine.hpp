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

    /**
     * Zero-copy batched detection from GPU NV12 frames
     * For RTSPModule integration - frames stay on GPU the entire time
     *
     * @param gpu_ptrs Vector of CUDA device pointers to NV12 frames
     * @param widths   Width of each frame
     * @param heights  Height of each frame
     * @return Vector of detections per frame
     */
    std::vector<std::vector<Detection>> detectBatchGpuNV12(
        const std::vector<uint64_t>& gpu_ptrs,
        const std::vector<int>& widths,
        const std::vector<int>& heights
    );

    /**
     * Batched detection from CPU NV12 frames
     * Skips CPU color conversion - uploads NV12 directly to GPU and converts there
     * Much faster than cv2.cvtColor + detectBatch
     *
     * @param nv12_frames Vector of NV12 frame data (H*1.5 x W bytes each)
     * @param widths      Width of each frame
     * @param heights     Height of each frame (Y plane height, not total)
     * @return Vector of detections per frame
     */
    std::vector<std::vector<Detection>> detectBatchNV12(
        const std::vector<const uint8_t*>& nv12_data,
        const std::vector<size_t>& data_sizes,
        const std::vector<int>& widths,
        const std::vector<int>& heights
    );

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

    // CUDA Graphs for low-latency inference
    cudaGraph_t cuda_graph_ = nullptr;
    cudaGraphExec_t cuda_graph_exec_ = nullptr;
    cudaGraphNode_t memcpy_node_ = nullptr;  // Node for updating D2H copy size
    int graph_batch_size_ = 0;  // Batch size the graph was recorded for
    bool use_cuda_graphs_ = false;  // Disabled: Fixed batch alone gives 637 FPS!
    int warmup_count_ = 0;
    static constexpr int WARMUP_ITERATIONS = 5;  // Warmup before recording graph

    // Buffers (sized for max batch)
    float* h_output_ = nullptr;
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    std::vector<uint8_t*> d_src_batch_;  // Per-batch source buffers for parallel preprocess
    std::vector<uint8_t*> h_src_batch_;  // Pinned host staging buffers for faster H2D transfer
    std::vector<cudaStream_t> preprocess_streams_;  // Streams for parallel preprocessing
    std::vector<cudaEvent_t> preprocess_events_;    // Events for efficient multi-stream sync

    size_t input_size_per_batch_ = 0;   // Size for single image
    size_t output_size_per_batch_ = 0;  // Size for single image output
    size_t input_size_ = 0;             // Total allocated (max_batch * per_batch)
    size_t output_size_ = 0;            // Total allocated
    size_t src_size_per_batch_ = 0;     // Size for single source image
    int num_detections_ = 0;
    int max_batch_size_ = 1;

    std::string input_name_;
    std::string output_name_;
    bool use_cuda_preprocess_ = true;
};

} // namespace trt_detector

