#include "trt_detector/trt_engine.hpp"
#include <fstream>
#include <iostream>

extern "C" void cudaPreprocess(
    const uint8_t* d_src, float* d_dst,
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
    , d_input_(other.d_input_), d_output_(other.d_output_), d_src_(other.d_src_)
    , input_size_(other.input_size_), output_size_(other.output_size_)
    , src_size_(other.src_size_), num_detections_(other.num_detections_)
    , input_name_(std::move(other.input_name_)), output_name_(std::move(other.output_name_))
    , use_cuda_preprocess_(other.use_cuda_preprocess_)
{
    other.runtime_ = nullptr;
    other.engine_ = nullptr;
    other.context_ = nullptr;
    other.stream_ = nullptr;
    other.h_output_ = nullptr;
    other.d_input_ = other.d_output_ = nullptr;
    other.d_src_ = nullptr;
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
        d_input_ = other.d_input_; d_output_ = other.d_output_; d_src_ = other.d_src_;
        input_size_ = other.input_size_; output_size_ = other.output_size_;
        src_size_ = other.src_size_; num_detections_ = other.num_detections_;
        input_name_ = std::move(other.input_name_); output_name_ = std::move(other.output_name_);
        use_cuda_preprocess_ = other.use_cuda_preprocess_;
        
        other.runtime_ = nullptr;
        other.engine_ = nullptr;
        other.context_ = nullptr;
        other.stream_ = nullptr;
        other.h_output_ = nullptr;
        other.d_input_ = other.d_output_ = nullptr; other.d_src_ = nullptr;
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
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dims = engine_->getTensorShape(name);
        
        size_t size = 1;
        for (int d = 0; d < dims.nbDims; ++d) size *= dims.d[d];
        
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
            input_size_ = size;
            cudaMalloc(&d_input_, size * sizeof(float));
        } else {
            output_name_ = name;
            output_size_ = size;
            num_detections_ = dims.d[dims.nbDims - 1];
            cudaMallocHost(&h_output_, size * sizeof(float));
            cudaMalloc(&d_output_, size * sizeof(float));
        }
    }
    src_size_ = 1920 * 1080 * 3;
    cudaMalloc(&d_src_, src_size_);
}

void TRTEngine::freeBuffers() {
    if (h_output_) { cudaFreeHost(h_output_); h_output_ = nullptr; }
    if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
    if (d_src_) { cudaFree(d_src_); d_src_ = nullptr; }
}

std::vector<Detection> TRTEngine::detect(const cv::Mat& frame) {
    if (!isLoaded()) return {};

    float scale;
    int pad_x, pad_y;

    if (use_cuda_preprocess_) {
        size_t frame_size = frame.total() * frame.elemSize();
        if (frame_size > src_size_) {
            cudaFree(d_src_);
            src_size_ = frame_size;
            cudaMalloc(&d_src_, src_size_);
        }

        cudaMemcpyAsync(d_src_, frame.data, frame_size, cudaMemcpyHostToDevice, stream_);

        cudaPreprocess(
            d_src_, static_cast<float*>(d_input_),
            frame.cols, frame.rows,
            config_.input_width, config_.input_height,
            &scale, &pad_x, &pad_y, stream_
        );
    } else {
        thread_local std::vector<float> h_input_buf;
        if (h_input_buf.size() != input_size_) h_input_buf.resize(input_size_);

        LetterboxInfo info = Preprocessor::process(
            frame, h_input_buf.data(), config_.input_width, config_.input_height
        );
        scale = info.scale;
        pad_x = info.pad_x;
        pad_y = info.pad_y;

        cudaMemcpyAsync(d_input_, h_input_buf.data(), input_size_ * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
    }

    context_->setTensorAddress(input_name_.c_str(), d_input_);
    context_->setTensorAddress(output_name_.c_str(), d_output_);
    context_->enqueueV3(stream_);

    cudaMemcpyAsync(h_output_, d_output_, output_size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    return Postprocessor::process(
        h_output_, num_detections_,
        static_cast<int>(config_.class_names.size()),
        config_.conf_threshold, config_.nms_threshold,
        scale, static_cast<float>(pad_x), static_cast<float>(pad_y),
        frame.cols, frame.rows, config_.class_names
    );
}

} // namespace trt_detector

