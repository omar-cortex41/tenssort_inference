#include "trt_detector/trt_engine.hpp"
#include <fstream>
#include <iostream>
#include <cstring>

extern "C" void cudaPreprocess(
    const uint8_t* d_src, float* d_dst,
    int src_w, int src_h, int dst_w, int dst_h,
    float* out_scale, int* out_pad_x, int* out_pad_y,
    cudaStream_t stream
);

extern "C" void cudaPreprocessNV12(
    const uint8_t* d_nv12, float* d_dst,
    int src_w, int src_h, int dst_w, int dst_h,
    float* out_scale, int* out_pad_x, int* out_pad_y,
    cudaStream_t stream
);

namespace trt_detector {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
    }
} gLogger;

TRTEngine::TRTEngine() = default;

TRTEngine::~TRTEngine() {
    freeBuffers();
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
    if (stream_) cudaStreamDestroy(stream_);
}

TRTEngine::TRTEngine(TRTEngine&& other) noexcept
    : config_(std::move(other.config_))
    , runtime_(other.runtime_), engine_(other.engine_), context_(other.context_)
    , stream_(other.stream_), h_output_(other.h_output_)
    , d_input_(other.d_input_), d_output_(other.d_output_)
    , d_src_batch_(std::move(other.d_src_batch_))
    , h_src_batch_(std::move(other.h_src_batch_))
    , preprocess_streams_(std::move(other.preprocess_streams_))
    , preprocess_events_(std::move(other.preprocess_events_))
    , input_size_per_batch_(other.input_size_per_batch_)
    , output_size_per_batch_(other.output_size_per_batch_)
    , input_size_(other.input_size_), output_size_(other.output_size_)
    , src_size_per_batch_(other.src_size_per_batch_), num_detections_(other.num_detections_)
    , max_batch_size_(other.max_batch_size_)
    , input_name_(std::move(other.input_name_)), output_name_(std::move(other.output_name_))
    , use_cuda_preprocess_(other.use_cuda_preprocess_)
{
    other.runtime_ = nullptr;
    other.engine_ = nullptr;
    other.context_ = nullptr;
    other.stream_ = nullptr;
    other.h_output_ = nullptr;
    other.d_input_ = other.d_output_ = nullptr;
}

TRTEngine& TRTEngine::operator=(TRTEngine&& other) noexcept {
    if (this != &other) {
        freeBuffers();
        if (context_) delete context_;
        if (engine_) delete engine_;
        if (runtime_) delete runtime_;
        if (stream_) cudaStreamDestroy(stream_);

        config_ = std::move(other.config_);
        runtime_ = other.runtime_; engine_ = other.engine_; context_ = other.context_;
        stream_ = other.stream_; h_output_ = other.h_output_;
        d_input_ = other.d_input_; d_output_ = other.d_output_;
        d_src_batch_ = std::move(other.d_src_batch_);
        h_src_batch_ = std::move(other.h_src_batch_);
        preprocess_streams_ = std::move(other.preprocess_streams_);
        input_size_per_batch_ = other.input_size_per_batch_;
        output_size_per_batch_ = other.output_size_per_batch_;
        input_size_ = other.input_size_; output_size_ = other.output_size_;
        src_size_per_batch_ = other.src_size_per_batch_; num_detections_ = other.num_detections_;
        max_batch_size_ = other.max_batch_size_;
        input_name_ = std::move(other.input_name_); output_name_ = std::move(other.output_name_);
        use_cuda_preprocess_ = other.use_cuda_preprocess_;

        other.runtime_ = nullptr;
        other.engine_ = nullptr;
        other.context_ = nullptr;
        other.stream_ = nullptr;
        other.h_output_ = nullptr;
        other.d_input_ = other.d_output_ = nullptr;
    }
    return *this;
}

bool TRTEngine::load(const ModelConfig& config) {
    config_ = config;
    
    std::ifstream file(config.engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine: " << config.engine_path << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) return false;
    
    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!engine_) return false;
    
    context_ = engine_->createExecutionContext();
    if (!context_) return false;
    
    cudaStreamCreate(&stream_);
    allocateBuffers();
    return true;
}

void TRTEngine::allocateBuffers() {
    // Check for dynamic batch support via optimization profiles
    max_batch_size_ = 1;

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;

            // Check if engine has optimization profiles (dynamic shapes)
            if (engine_->getNbOptimizationProfiles() > 0) {
                auto max_dims = engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
                max_batch_size_ = max_dims.d[0];

                // Calculate per-batch size (C * H * W)
                input_size_per_batch_ = 1;
                for (int d = 1; d < max_dims.nbDims; ++d) {
                    input_size_per_batch_ *= max_dims.d[d];
                }
            } else {
                // Fixed batch size engine
                auto dims = engine_->getTensorShape(name);
                max_batch_size_ = dims.d[0];
                input_size_per_batch_ = 1;
                for (int d = 1; d < dims.nbDims; ++d) {
                    input_size_per_batch_ *= dims.d[d];
                }
            }

            input_size_ = input_size_per_batch_ * max_batch_size_;
            cudaMalloc(&d_input_, input_size_ * sizeof(float));

        } else {
            output_name_ = name;
            auto dims = engine_->getTensorShape(name);

            // Output shape is [batch, num_outputs, num_detections] for YOLO
            // num_detections is the last dimension
            num_detections_ = dims.d[dims.nbDims - 1];

            // Calculate per-batch output size
            output_size_per_batch_ = 1;
            for (int d = 1; d < dims.nbDims; ++d) {
                output_size_per_batch_ *= dims.d[d];
            }

            output_size_ = output_size_per_batch_ * max_batch_size_;
            cudaMallocHost(&h_output_, output_size_ * sizeof(float));
            cudaMalloc(&d_output_, output_size_ * sizeof(float));
        }
    }

    // Allocate per-batch source buffers and streams for parallel preprocessing
    src_size_per_batch_ = 3840 * 2160 * 3;  // 4K image (NV12 is 1.5x, but BGR is 3x)
    d_src_batch_.resize(max_batch_size_);
    h_src_batch_.resize(max_batch_size_);  // Pinned staging buffers (Fix #4)
    preprocess_streams_.resize(max_batch_size_);
    preprocess_events_.resize(max_batch_size_);

    for (int i = 0; i < max_batch_size_; ++i) {
        cudaMalloc(&d_src_batch_[i], src_size_per_batch_);
        cudaMallocHost(&h_src_batch_[i], src_size_per_batch_);  // Pinned memory for faster H2D
        cudaStreamCreate(&preprocess_streams_[i]);
        cudaEventCreateWithFlags(&preprocess_events_[i], cudaEventDisableTiming);
    }

    std::cout << "[TRTEngine] Max batch size: " << max_batch_size_ << std::endl;
    std::cout << "[TRTEngine] Parallel preprocessing streams: " << max_batch_size_ << std::endl;
    std::cout << "[TRTEngine] Pinned staging buffers: " << max_batch_size_ << " x "
              << (src_size_per_batch_ / 1024 / 1024) << " MB" << std::endl;
}

void TRTEngine::freeBuffers() {
    if (h_output_) { cudaFreeHost(h_output_); h_output_ = nullptr; }
    if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }

    for (auto& ptr : d_src_batch_) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    }
    d_src_batch_.clear();

    // Free pinned staging buffers (Fix #4)
    for (auto& ptr : h_src_batch_) {
        if (ptr) { cudaFreeHost(ptr); ptr = nullptr; }
    }
    h_src_batch_.clear();

    for (auto& s : preprocess_streams_) {
        if (s) { cudaStreamDestroy(s); s = nullptr; }
    }
    preprocess_streams_.clear();

    for (auto& e : preprocess_events_) {
        if (e) { cudaEventDestroy(e); e = nullptr; }
    }
    preprocess_events_.clear();
}

std::vector<Detection> TRTEngine::detect(const cv::Mat& frame) {
    // Single frame detection - use batch of 1
    auto results = detectBatch({frame});
    return results.empty() ? std::vector<Detection>{} : results[0];
}

std::vector<std::vector<Detection>> TRTEngine::detectBatch(const std::vector<cv::Mat>& frames) {
    if (!isLoaded() || frames.empty()) return {};

    int batch_size = static_cast<int>(frames.size());
    if (batch_size > max_batch_size_) {
        std::cerr << "[TRTEngine] Batch size " << batch_size
                  << " exceeds max " << max_batch_size_ << std::endl;
        batch_size = max_batch_size_;
    }

    // Store preprocessing info for each frame
    std::vector<float> scales(batch_size);
    std::vector<int> pad_xs(batch_size);
    std::vector<int> pad_ys(batch_size);
    std::vector<int> frame_widths(batch_size);
    std::vector<int> frame_heights(batch_size);

    // Set dynamic input shape for this batch
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = batch_size;
    input_dims.d[1] = 3;
    input_dims.d[2] = config_.input_height;
    input_dims.d[3] = config_.input_width;
    context_->setInputShape(input_name_.c_str(), input_dims);

    if (use_cuda_preprocess_) {
        // Preprocess all frames in PARALLEL using separate streams
        for (int i = 0; i < batch_size; ++i) {
            const cv::Mat& frame = frames[i];
            size_t frame_size = frame.total() * frame.elemSize();

            frame_widths[i] = frame.cols;
            frame_heights[i] = frame.rows;

            // Each frame uses its own stream and source buffer
            cudaStream_t s = preprocess_streams_[i];
            uint8_t* d_src = d_src_batch_[i];

            // Copy frame to device (async, parallel)
            cudaMemcpyAsync(d_src, frame.data, frame_size, cudaMemcpyHostToDevice, s);

            // Preprocess to correct position in input buffer (async, parallel)
            float* d_input_offset = static_cast<float*>(d_input_) + i * input_size_per_batch_;
            cudaPreprocess(
                d_src, d_input_offset,
                frame.cols, frame.rows,
                config_.input_width, config_.input_height,
                &scales[i], &pad_xs[i], &pad_ys[i], s
            );
        }

        // Use events for efficient GPU-side synchronization (no host blocking)
        for (int i = 0; i < batch_size; ++i) {
            cudaEventRecord(preprocess_events_[i], preprocess_streams_[i]);
        }
        // Make inference stream wait for all preprocess streams via events
        for (int i = 0; i < batch_size; ++i) {
            cudaStreamWaitEvent(stream_, preprocess_events_[i], 0);
        }
    } else {
        // CPU preprocessing
        thread_local std::vector<float> h_input_buf;
        size_t total_input = input_size_per_batch_ * batch_size;
        if (h_input_buf.size() < total_input) h_input_buf.resize(total_input);

        for (int i = 0; i < batch_size; ++i) {
            const cv::Mat& frame = frames[i];
            frame_widths[i] = frame.cols;
            frame_heights[i] = frame.rows;

            float* buf_offset = h_input_buf.data() + i * input_size_per_batch_;
            LetterboxInfo info = Preprocessor::process(
                frame, buf_offset, config_.input_width, config_.input_height
            );
            scales[i] = info.scale;
            pad_xs[i] = info.pad_x;
            pad_ys[i] = info.pad_y;
        }

        cudaMemcpyAsync(d_input_, h_input_buf.data(), total_input * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
    }

    // Run inference
    context_->setTensorAddress(input_name_.c_str(), d_input_);
    context_->setTensorAddress(output_name_.c_str(), d_output_);
    context_->enqueueV3(stream_);

    // Copy output back
    size_t output_bytes = output_size_per_batch_ * batch_size * sizeof(float);
    cudaMemcpyAsync(h_output_, d_output_, output_bytes, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Postprocess each frame's output
    std::vector<std::vector<Detection>> results(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        const float* output_offset = h_output_ + i * output_size_per_batch_;
        results[i] = Postprocessor::process(
            output_offset, num_detections_,
            static_cast<int>(config_.class_names.size()),
            config_.conf_threshold, config_.nms_threshold,
            scales[i], static_cast<float>(pad_xs[i]), static_cast<float>(pad_ys[i]),
            frame_widths[i], frame_heights[i], config_.class_names
        );
    }

    return results;
}

std::vector<std::vector<Detection>> TRTEngine::detectBatchGpuNV12(
    const std::vector<uint64_t>& gpu_ptrs,
    const std::vector<int>& widths,
    const std::vector<int>& heights
) {
    if (!isLoaded() || gpu_ptrs.empty()) return {};

    int batch_size = static_cast<int>(gpu_ptrs.size());
    if (batch_size > max_batch_size_) {
        std::cerr << "[TRTEngine] Batch size " << batch_size
                  << " exceeds max " << max_batch_size_ << std::endl;
        batch_size = max_batch_size_;
    }

    // Store preprocessing info for each frame
    std::vector<float> scales(batch_size);
    std::vector<int> pad_xs(batch_size);
    std::vector<int> pad_ys(batch_size);

    // Set dynamic input shape for this batch
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = batch_size;
    input_dims.d[1] = 3;
    input_dims.d[2] = config_.input_height;
    input_dims.d[3] = config_.input_width;
    context_->setInputShape(input_name_.c_str(), input_dims);

    // Preprocess all NV12 frames in PARALLEL using separate streams
    // Zero-copy: frames are already on GPU, no H2D transfer needed!
    for (int i = 0; i < batch_size; ++i) {
        cudaStream_t s = preprocess_streams_[i];

        // Get GPU pointer from uint64_t
        const uint8_t* d_nv12 = reinterpret_cast<const uint8_t*>(gpu_ptrs[i]);

        // Preprocess NV12 directly to input buffer (async, parallel)
        float* d_input_offset = static_cast<float*>(d_input_) + i * input_size_per_batch_;
        cudaPreprocessNV12(
            d_nv12, d_input_offset,
            widths[i], heights[i],
            config_.input_width, config_.input_height,
            &scales[i], &pad_xs[i], &pad_ys[i], s
        );
    }

    // Use events for efficient GPU-side synchronization
    for (int i = 0; i < batch_size; ++i) {
        cudaEventRecord(preprocess_events_[i], preprocess_streams_[i]);
    }
    for (int i = 0; i < batch_size; ++i) {
        cudaStreamWaitEvent(stream_, preprocess_events_[i], 0);
    }

    // Run inference
    context_->setTensorAddress(input_name_.c_str(), d_input_);
    context_->setTensorAddress(output_name_.c_str(), d_output_);
    context_->enqueueV3(stream_);

    // Copy output back
    size_t output_bytes = output_size_per_batch_ * batch_size * sizeof(float);
    cudaMemcpyAsync(h_output_, d_output_, output_bytes, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Postprocess each frame's output
    std::vector<std::vector<Detection>> results(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        const float* output_offset = h_output_ + i * output_size_per_batch_;
        results[i] = Postprocessor::process(
            output_offset, num_detections_,
            static_cast<int>(config_.class_names.size()),
            config_.conf_threshold, config_.nms_threshold,
            scales[i], static_cast<float>(pad_xs[i]), static_cast<float>(pad_ys[i]),
            widths[i], heights[i], config_.class_names
        );
    }

    return results;
}

std::vector<std::vector<Detection>> TRTEngine::detectBatchNV12(
    const std::vector<const uint8_t*>& nv12_data,
    const std::vector<size_t>& data_sizes,
    const std::vector<int>& widths,
    const std::vector<int>& heights
) {
    if (!isLoaded() || nv12_data.empty()) return {};

    int batch_size = static_cast<int>(nv12_data.size());
    if (batch_size > max_batch_size_) {
        std::cerr << "[TRTEngine] Batch size " << batch_size
                  << " exceeds max " << max_batch_size_ << std::endl;
        batch_size = max_batch_size_;
    }

    // Store preprocessing info for each frame
    std::vector<float> scales(batch_size);
    std::vector<int> pad_xs(batch_size);
    std::vector<int> pad_ys(batch_size);

    // Set dynamic input shape for this batch
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = batch_size;
    input_dims.d[1] = 3;
    input_dims.d[2] = config_.input_height;
    input_dims.d[3] = config_.input_width;
    context_->setInputShape(input_name_.c_str(), input_dims);

    // Upload NV12 frames to GPU and preprocess in PARALLEL
    // Fix #4: Use pinned staging buffers for faster H2D transfer
    for (int i = 0; i < batch_size; ++i) {
        cudaStream_t s = preprocess_streams_[i];
        uint8_t* d_src = d_src_batch_[i];
        uint8_t* h_src = h_src_batch_[i];  // Pinned staging buffer

        // NV12 size = width * height * 1.5
        size_t nv12_size = data_sizes[i];

        // Copy to pinned staging buffer first (fast memcpy)
        // Then async copy from pinned to device (DMA, doesn't block CPU)
        std::memcpy(h_src, nv12_data[i], nv12_size);
        cudaMemcpyAsync(d_src, h_src, nv12_size, cudaMemcpyHostToDevice, s);

        // Preprocess NV12 directly to input buffer (async, parallel)
        float* d_input_offset = static_cast<float*>(d_input_) + i * input_size_per_batch_;
        cudaPreprocessNV12(
            d_src, d_input_offset,
            widths[i], heights[i],
            config_.input_width, config_.input_height,
            &scales[i], &pad_xs[i], &pad_ys[i], s
        );
    }

    // Use events for efficient GPU-side synchronization
    for (int i = 0; i < batch_size; ++i) {
        cudaEventRecord(preprocess_events_[i], preprocess_streams_[i]);
    }
    for (int i = 0; i < batch_size; ++i) {
        cudaStreamWaitEvent(stream_, preprocess_events_[i], 0);
    }

    // Run inference
    context_->setTensorAddress(input_name_.c_str(), d_input_);
    context_->setTensorAddress(output_name_.c_str(), d_output_);
    context_->enqueueV3(stream_);

    // Copy output back
    size_t output_bytes = output_size_per_batch_ * batch_size * sizeof(float);
    cudaMemcpyAsync(h_output_, d_output_, output_bytes, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Postprocess each frame's output
    std::vector<std::vector<Detection>> results(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        const float* output_offset = h_output_ + i * output_size_per_batch_;
        results[i] = Postprocessor::process(
            output_offset, num_detections_,
            static_cast<int>(config_.class_names.size()),
            config_.conf_threshold, config_.nms_threshold,
            scales[i], static_cast<float>(pad_xs[i]), static_cast<float>(pad_ys[i]),
            widths[i], heights[i], config_.class_names
        );
    }

    return results;
}

} // namespace trt_detector
