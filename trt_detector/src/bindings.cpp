#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "trt_detector/detector_service.hpp"
#include "trt_detector/async_pipeline.hpp"

namespace py = pybind11;
using namespace trt_detector;

cv::Mat numpy_to_mat(py::array_t<uint8_t>& arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 3) throw std::runtime_error("Expected 3D array (H, W, C)");
    if (buf.shape[2] != 3) throw std::runtime_error("Expected 3 channels (BGR)");
    return cv::Mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
}

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
        .def("detect", [](DetectorService& self, py::array_t<uint8_t>& frame) {
            cv::Mat mat = numpy_to_mat(frame);
            py::gil_scoped_release release;
            return self.detect(mat);
        });

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
}

