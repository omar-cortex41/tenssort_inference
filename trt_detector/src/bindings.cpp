#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "trt_detector/detector_service.hpp"
#include "trt_detector/async_pipeline.hpp"
#include "trt_detector/preprocessor.hpp"
#include "trt_detector/postprocessor.hpp"
#include "trt_detector/cuda_preprocess.hpp"
#include <cuda_runtime.h>

namespace py = pybind11;
using namespace trt_detector;

cv::Mat numpy_to_mat(py::array_t<uint8_t>& arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 3) throw std::runtime_error("Expected 3D array (H, W, C)");
    if (buf.shape[2] != 3) throw std::runtime_error("Expected 3 channels (BGR)");
    return cv::Mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
}

// Preprocessing result structure
struct PreprocessResult {
    py::array_t<float> tensor;  // (N, 3, H, W) preprocessed tensor
    std::vector<float> scales;
    std::vector<int> pad_xs;
    std::vector<int> pad_ys;
    std::vector<int> orig_widths;
    std::vector<int> orig_heights;
};

// CUDA preprocessing helper class
class CudaPreprocessor {
public:
    CudaPreprocessor(int input_width = 640, int input_height = 640, int max_batch = 8)
        : input_width_(input_width), input_height_(input_height), max_batch_(max_batch) {

        cudaStreamCreate(&stream_);

        // Allocate device buffers
        size_t input_size = input_width * input_height * 3 * max_batch * sizeof(float);
        cudaMalloc(&d_input_, input_size);

        // Allocate per-batch source buffers (4K max)
        src_size_per_batch_ = 3840 * 2160 * 3;
        d_src_batch_.resize(max_batch);
        for (int i = 0; i < max_batch; ++i) {
            cudaMalloc(&d_src_batch_[i], src_size_per_batch_);
        }

        // Host output buffer (pinned memory for faster transfer)
        cudaMallocHost(&h_output_, input_size);
    }

    ~CudaPreprocessor() {
        if (h_output_) cudaFreeHost(h_output_);
        if (d_input_) cudaFree(d_input_);
        for (auto& ptr : d_src_batch_) {
            if (ptr) cudaFree(ptr);
        }
        if (stream_) cudaStreamDestroy(stream_);
    }

    PreprocessResult preprocess_batch(py::list frames_list) {
        std::vector<cv::Mat> frames;
        frames.reserve(py::len(frames_list));
        for (auto item : frames_list) {
            py::array_t<uint8_t> arr = item.cast<py::array_t<uint8_t>>();
            frames.push_back(numpy_to_mat(arr));
        }

        int batch_size = static_cast<int>(frames.size());
        if (batch_size > max_batch_) {
            throw std::runtime_error("Batch size exceeds max: " + std::to_string(max_batch_));
        }

        PreprocessResult result;
        result.scales.resize(batch_size);
        result.pad_xs.resize(batch_size);
        result.pad_ys.resize(batch_size);
        result.orig_widths.resize(batch_size);
        result.orig_heights.resize(batch_size);

        size_t input_size_per_batch = input_width_ * input_height_ * 3;

        {
            py::gil_scoped_release release;

            // Parallel CUDA preprocessing
            for (int i = 0; i < batch_size; ++i) {
                const cv::Mat& frame = frames[i];
                size_t frame_size = frame.total() * frame.elemSize();

                result.orig_widths[i] = frame.cols;
                result.orig_heights[i] = frame.rows;

                // Copy to device
                cudaMemcpyAsync(d_src_batch_[i], frame.data, frame_size,
                               cudaMemcpyHostToDevice, stream_);

                // Preprocess
                float* d_input_offset = static_cast<float*>(d_input_) + i * input_size_per_batch;
                cudaPreprocess(
                    d_src_batch_[i], d_input_offset,
                    frame.cols, frame.rows,
                    input_width_, input_height_,
                    &result.scales[i], &result.pad_xs[i], &result.pad_ys[i], stream_
                );
            }

            // Copy result back to host
            size_t total_size = input_size_per_batch * batch_size * sizeof(float);
            cudaMemcpyAsync(h_output_, d_input_, total_size, cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);
        }

        // Create numpy array (N, 3, H, W)
        result.tensor = py::array_t<float>({batch_size, 3, input_height_, input_width_});
        auto buf = result.tensor.request();
        std::memcpy(buf.ptr, h_output_, batch_size * input_size_per_batch * sizeof(float));

        return result;
    }

private:
    int input_width_;
    int input_height_;
    int max_batch_;
    size_t src_size_per_batch_;

    cudaStream_t stream_ = nullptr;
    void* d_input_ = nullptr;
    float* h_output_ = nullptr;
    std::vector<uint8_t*> d_src_batch_;
};

PYBIND11_MODULE(trt_detector, m) {
    m.doc() = "TensorRT Object Detection Module";
    
    py::class_<Detection>(m, "Detection")
        .def(py::init<>())
        .def_readonly("x", &Detection::x)
        .def_readonly("y", &Detection::y)
        .def_readonly("width", &Detection::width)
        .def_readonly("height", &Detection::height)
        .def_readonly("class_id", &Detection::class_id)
        .def_readonly("confidence", &Detection::confidence)
        .def_readonly("label", &Detection::label)
        .def("__repr__", [](const Detection& d) {
            return "<Detection " + d.label + " conf=" + std::to_string(d.confidence) + ">";
        });
    
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def(py::init<const std::string&, const std::vector<std::string>&, float, float, int, int>(),
             py::arg("engine_path"), py::arg("class_names"),
             py::arg("conf_threshold") = 0.5f, py::arg("nms_threshold") = 0.45f,
             py::arg("input_width") = 640, py::arg("input_height") = 640)
        .def_readwrite("engine_path", &ModelConfig::engine_path)
        .def_readwrite("input_width", &ModelConfig::input_width)
        .def_readwrite("input_height", &ModelConfig::input_height)
        .def_readwrite("conf_threshold", &ModelConfig::conf_threshold)
        .def_readwrite("nms_threshold", &ModelConfig::nms_threshold)
        .def_readwrite("class_names", &ModelConfig::class_names);
    
    py::class_<DetectorService>(m, "DetectorService")
        .def(py::init<>())
        .def("load_model", &DetectorService::loadModel)
        .def("unload_model", &DetectorService::unloadModel)
        .def("is_loaded", &DetectorService::isLoaded)
        .def("get_max_batch_size", &DetectorService::getMaxBatchSize)
        .def("detect", [](DetectorService& self, py::array_t<uint8_t>& frame) {
            cv::Mat mat = numpy_to_mat(frame);
            py::gil_scoped_release release;
            return self.detect(mat);
        })
        .def("detect_batch", [](DetectorService& self, py::list frames_list) {
            // Convert list of numpy arrays to vector of cv::Mat
            std::vector<cv::Mat> frames;
            frames.reserve(py::len(frames_list));
            for (auto item : frames_list) {
                py::array_t<uint8_t> arr = item.cast<py::array_t<uint8_t>>();
                frames.push_back(numpy_to_mat(arr));
            }
            py::gil_scoped_release release;
            return self.detectBatch(frames);
        })
        .def("detect_batch_gpu_nv12", [](DetectorService& self,
                                          py::list gpu_ptrs_list,
                                          py::list widths_list,
                                          py::list heights_list) {
            // Zero-copy detection from GPU NV12 frames (RTSPModule integration)
            std::vector<uint64_t> gpu_ptrs;
            std::vector<int> widths;
            std::vector<int> heights;

            gpu_ptrs.reserve(py::len(gpu_ptrs_list));
            widths.reserve(py::len(widths_list));
            heights.reserve(py::len(heights_list));

            for (auto item : gpu_ptrs_list) {
                gpu_ptrs.push_back(item.cast<uint64_t>());
            }
            for (auto item : widths_list) {
                widths.push_back(item.cast<int>());
            }
            for (auto item : heights_list) {
                heights.push_back(item.cast<int>());
            }

            py::gil_scoped_release release;
            return self.detectBatchGpuNV12(gpu_ptrs, widths, heights);
        }, py::arg("gpu_ptrs"), py::arg("widths"), py::arg("heights"),
           "Zero-copy batched detection from GPU NV12 frames.\n\n"
           "For RTSPModule integration - frames stay on GPU the entire time.\n\n"
           "Args:\n"
           "    gpu_ptrs (list[int]): CUDA device pointers to NV12 frames\n"
           "    widths (list[int]): Width of each frame\n"
           "    heights (list[int]): Height of each frame\n\n"
           "Returns:\n"
           "    list[list[Detection]]: Detections per frame")
        .def("detect_batch_nv12", [](DetectorService& self,
                                     py::list frames_list,
                                     int width,
                                     int height) {
            // Fast NV12 detection - skips CPU color conversion
            // Uploads NV12 directly to GPU and converts there
            std::vector<const uint8_t*> nv12_data;
            std::vector<size_t> data_sizes;
            std::vector<int> widths;
            std::vector<int> heights;

            size_t batch_size = py::len(frames_list);
            nv12_data.reserve(batch_size);
            data_sizes.reserve(batch_size);
            widths.reserve(batch_size);
            heights.reserve(batch_size);

            // NV12 size = width * height * 1.5
            size_t nv12_size = static_cast<size_t>(width * height * 3 / 2);

            for (auto item : frames_list) {
                py::array_t<uint8_t> arr = item.cast<py::array_t<uint8_t>>();
                py::buffer_info buf = arr.request();
                nv12_data.push_back(static_cast<const uint8_t*>(buf.ptr));
                data_sizes.push_back(nv12_size);
                widths.push_back(width);
                heights.push_back(height);
            }

            py::gil_scoped_release release;
            return self.detectBatchNV12(nv12_data, data_sizes, widths, heights);
        }, py::arg("frames"), py::arg("width"), py::arg("height"),
           "Fast batched detection from CPU NV12 frames.\n\n"
           "Skips CPU color conversion - uploads NV12 directly to GPU.\n"
           "Much faster than cv2.cvtColor + detect_batch.\n\n"
           "Args:\n"
           "    frames (list[numpy.ndarray]): List of NV12 frames (H*1.5, W) uint8\n"
           "    width (int): Frame width\n"
           "    height (int): Frame height (Y plane height)\n\n"
           "Returns:\n"
           "    list[list[Detection]]: Detections per frame");

    // FrameResult for async pipeline
    py::class_<FrameResult>(m, "FrameResult")
        .def(py::init<>())
        .def_readonly("detections", &FrameResult::detections)
        .def_readonly("frame_id", &FrameResult::frame_id)
        .def_readonly("inference_time_ms", &FrameResult::inference_time_ms)
        .def("get_frame", [](FrameResult& self) {
            // Convert cv::Mat to numpy array
            py::array_t<uint8_t> arr({self.frame.rows, self.frame.cols, 3});
            auto buf = arr.request();
            std::memcpy(buf.ptr, self.frame.data, self.frame.total() * self.frame.elemSize());
            return arr;
        });

    // Async Pipeline
    py::class_<AsyncPipeline>(m, "AsyncPipeline")
        .def(py::init<>())
        .def("init", &AsyncPipeline::init)
        .def("start", py::overload_cast<const std::string&>(&AsyncPipeline::start))
        .def("start_camera", py::overload_cast<int>(&AsyncPipeline::start))
        .def("stop", &AsyncPipeline::stop)
        .def("is_running", &AsyncPipeline::isRunning)
        .def("get_capture_queue_size", &AsyncPipeline::getCaptureQueueSize)
        .def("get_result_queue_size", &AsyncPipeline::getResultQueueSize)
        .def("set_max_capture_queue_size", &AsyncPipeline::setMaxCaptureQueueSize)
        .def("set_max_result_queue_size", &AsyncPipeline::setMaxResultQueueSize)
        .def("get_result", [](AsyncPipeline& self) -> py::object {
            FrameResult result;
            bool got;
            {
                py::gil_scoped_release release;
                got = self.getResult(result);
            }
            if (!got) return py::none();
            return py::cast(result);
        })
        .def("try_get_result", [](AsyncPipeline& self) -> py::object {
            FrameResult result;
            if (!self.tryGetResult(result)) return py::none();
            return py::cast(result);
        });

    // PreprocessResult for Triton integration
    py::class_<PreprocessResult>(m, "PreprocessResult")
        .def(py::init<>())
        .def_readonly("tensor", &PreprocessResult::tensor)
        .def_readonly("scales", &PreprocessResult::scales)
        .def_readonly("pad_xs", &PreprocessResult::pad_xs)
        .def_readonly("pad_ys", &PreprocessResult::pad_ys)
        .def_readonly("orig_widths", &PreprocessResult::orig_widths)
        .def_readonly("orig_heights", &PreprocessResult::orig_heights);

    // CudaPreprocessor for fast GPU preprocessing
    py::class_<CudaPreprocessor>(m, "CudaPreprocessor")
        .def(py::init<int, int, int>(),
             py::arg("input_width") = 640, py::arg("input_height") = 640, py::arg("max_batch") = 8)
        .def("preprocess_batch", &CudaPreprocessor::preprocess_batch);

    // Postprocess function for Triton output
    m.def("postprocess_batch", [](
        py::array_t<float> output,
        const std::vector<float>& scales,
        const std::vector<int>& pad_xs,
        const std::vector<int>& pad_ys,
        const std::vector<int>& orig_widths,
        const std::vector<int>& orig_heights,
        const std::vector<std::string>& class_names,
        float conf_threshold,
        float nms_threshold
    ) {
        auto buf = output.request();
        if (buf.ndim != 3) {
            throw std::runtime_error("Expected 3D output (batch, num_dets, 6)");
        }

        int batch_size = buf.shape[0];
        int num_detections = buf.shape[1];
        const float* data = static_cast<const float*>(buf.ptr);

        std::vector<std::vector<Detection>> results(batch_size);
        size_t output_size_per_batch = num_detections * buf.shape[2];

        for (int i = 0; i < batch_size; ++i) {
            const float* output_offset = data + i * output_size_per_batch;
            results[i] = Postprocessor::process(
                output_offset, num_detections,
                static_cast<int>(class_names.size()),
                conf_threshold, nms_threshold,
                scales[i], static_cast<float>(pad_xs[i]), static_cast<float>(pad_ys[i]),
                orig_widths[i], orig_heights[i], class_names
            );
        }
        return results;
    }, py::arg("output"), py::arg("scales"), py::arg("pad_xs"), py::arg("pad_ys"),
       py::arg("orig_widths"), py::arg("orig_heights"), py::arg("class_names"),
       py::arg("conf_threshold") = 0.5f, py::arg("nms_threshold") = 0.45f);
}

