# TensorRT Detector C++ Service - Architecture Plan

## Executive Summary

Convert Python TensorRT inference script into a high-performance C++ service supporting multiple concurrent models, designed for integration into a 14,000+ line AI engine.

---

## 1. Current State Analysis

### Python Implementation (main.py)
```
Components:
├── CUDA Context Management (pycuda)
├── TensorRT Engine Loading
├── GPU Memory Allocation
├── Preprocessing (letterbox + normalization)
├── Inference Execution
├── Postprocessing (NMS + coordinate mapping)
└── Video Loop (OpenCV)
```

### Limitations of Python Version
- Single model only
- Global state (not thread-safe)
- Python GIL blocks parallelism
- Memory managed by garbage collector
- ~10-15% overhead from Python wrappers

---

## 2. Target Architecture

### High-Level Design
```
┌─────────────────────────────────────────────────────────────────┐
│                    TRT Detector Service                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Model A    │  │  Model B    │  │  Model N    │             │
│  │  (PPE)      │  │  (Vehicles) │  │  (Custom)   │             │
│  │  Engine     │  │  Engine     │  │  Engine     │             │
│  │  Context    │  │  Context    │  │  Context    │             │
│  │  Buffers    │  │  Buffers    │  │  Buffers    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│                ┌─────────▼─────────┐                           │
│                │   CUDA Stream     │                           │
│                │   Manager         │                           │
│                └─────────┬─────────┘                           │
│                          │                                      │
│                ┌─────────▼─────────┐                           │
│                │   Shared GPU      │                           │
│                │   Memory Pool     │                           │
│                └───────────────────┘                           │
├─────────────────────────────────────────────────────────────────┤
│                     Python Bindings (pybind11)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Breakdown

### 3.1 Core Classes

#### `TRTEngine` - Single Model Handler
```cpp
class TRTEngine {
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;

    // Pre-allocated buffers (no runtime allocation)
    void* d_input;      // GPU input buffer
    void* d_output;     // GPU output buffer
    float* h_input;     // Pinned host input
    float* h_output;    // Pinned host output

    cudaStream_t stream;

public:
    bool load(const std::string& engine_path);
    void infer(const float* input, float* output);
    void inferAsync(const float* input, float* output, cudaStream_t stream);
};
```

#### `ModelConfig` - Per-Model Configuration
```cpp
struct ModelConfig {
    std::string engine_path;
    std::string model_name;
    int input_width = 640;
    int input_height = 640;
    float conf_threshold = 0.5f;
    float nms_threshold = 0.45f;
    std::vector<std::string> class_names;
};
```

#### `DetectorService` - Multi-Model Manager
```cpp
class DetectorService {
    std::unordered_map<std::string, std::unique_ptr<TRTEngine>> engines;
    std::shared_ptr<CUDAStreamPool> stream_pool;

public:
    bool loadModel(const std::string& name, const ModelConfig& config);
    void unloadModel(const std::string& name);

    std::vector<Detection> detect(
        const std::string& model_name,
        const cv::Mat& frame
    );

    // Batch inference across multiple models
    std::vector<std::vector<Detection>> detectMulti(
        const std::vector<std::string>& model_names,
        const cv::Mat& frame
    );
};
```

#### `Detection` - Output Structure
```cpp
struct Detection {
    int x, y, width, height;  // Bounding box
    int class_id;             // Class index
    float confidence;         // Detection score
    std::string label;        // Class name
};
```

### 3.2 Preprocessing Module
```cpp
class Preprocessor {
public:
    // CPU letterbox (current approach)
    static void letterbox(
        const cv::Mat& src,
        cv::Mat& dst,
        int target_w,
        int target_h,
        float& scale,
        int& pad_x,
        int& pad_y
    );

    // GPU letterbox (CUDA kernel - optional optimization)
    static void letterboxGPU(
        const uint8_t* d_src,
        float* d_dst,
        int src_w, int src_h,
        int dst_w, int dst_h,
        cudaStream_t stream
    );
};
```

### 3.3 Postprocessing Module
```cpp
class Postprocessor {
public:
    static std::vector<Detection> process(
        const float* raw_output,
        int num_detections,      // 8400 for YOLO
        int num_classes,
        float conf_threshold,
        float nms_threshold,
        float scale,
        float pad_x,
        float pad_y,
        int frame_w,
        int frame_h,
        const std::vector<std::string>& class_names
    );
};
```

---

## 4. Memory Management Strategy

### 4.1 Pre-allocation (Zero Runtime Allocation)
```cpp
// At model load time - allocate everything
void TRTEngine::allocateBuffers() {
    // Pinned host memory (faster DMA transfers)
    cudaMallocHost(&h_input, input_size * sizeof(float));
    cudaMallocHost(&h_output, output_size * sizeof(float));

    // Device memory
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
}

// At inference time - zero allocations
void TRTEngine::infer(const float* input, float* output) {
    // Just copy and execute - no new/delete
    cudaMemcpyAsync(d_input, input, ...);
    context->enqueueV3(stream);
    cudaMemcpyAsync(output, d_output, ...);
}
```

### 4.2 Memory Pool for Temporary Buffers
```cpp
class BufferPool {
    std::vector<void*> available;
    std::mutex mutex;

public:
    void* acquire(size_t size);
    void release(void* ptr);
};
```

---

## 5. Thread Safety Design

### 5.1 Model-Level Isolation
- Each model has its own execution context
- Each model has its own CUDA stream
- No shared mutable state between models

### 5.2 Concurrent Inference
```cpp
// Safe: Different models can run in parallel
std::future<Results> f1 = pool.submit([&]{ return detector.detect("ppe", frame); });
std::future<Results> f2 = pool.submit([&]{ return detector.detect("vehicle", frame); });

// Both run concurrently on GPU via separate streams
```

### 5.3 Same-Model Concurrency (Optional)
```cpp
// For high throughput: multiple contexts per engine
class TRTEngine {
    nvinfer1::ICudaEngine* engine;  // Shared
    std::vector<ContextInstance> contexts;  // Pool of contexts

    ContextInstance* acquireContext();
    void releaseContext(ContextInstance*);
};
```

---

## 6. Python Bindings (pybind11)

### 6.1 Interface Design
```python
# Python usage (after C++ conversion)
from trt_detector import DetectorService, ModelConfig

# Initialize service
detector = DetectorService()

# Load multiple models
detector.load_model("ppe", ModelConfig(
    engine_path="models/ppe.engine",
    class_names=["helmet", "vest", "boots", ...],
    conf_threshold=0.5
))

detector.load_model("vehicles", ModelConfig(
    engine_path="models/vehicles.engine",
    class_names=["car", "truck", "bus", ...],
    conf_threshold=0.6
))

# Run inference
frame = cv2.imread("image.jpg")
ppe_detections = detector.detect("ppe", frame)
vehicle_detections = detector.detect("vehicles", frame)

# Batch inference (runs in parallel on GPU)
all_detections = detector.detect_multi(["ppe", "vehicles"], frame)
```

### 6.2 Binding Implementation
```cpp
PYBIND11_MODULE(trt_detector, m) {
    py::class_<Detection>(m, "Detection")
        .def_readonly("x", &Detection::x)
        .def_readonly("y", &Detection::y)
        .def_readonly("width", &Detection::width)
        .def_readonly("height", &Detection::height)
        .def_readonly("class_id", &Detection::class_id)
        .def_readonly("confidence", &Detection::confidence)
        .def_readonly("label", &Detection::label);

    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("engine_path", &ModelConfig::engine_path)
        .def_readwrite("conf_threshold", &ModelConfig::conf_threshold)
        .def_readwrite("nms_threshold", &ModelConfig::nms_threshold)
        .def_readwrite("class_names", &ModelConfig::class_names);

    py::class_<DetectorService>(m, "DetectorService")
        .def(py::init<>())
        .def("load_model", &DetectorService::loadModel)
        .def("unload_model", &DetectorService::unloadModel)
        .def("detect", &DetectorService::detect,
             py::arg("model_name"),
             py::arg("frame").noconvert(),  // Accept numpy array
             py::call_guard<py::gil_scoped_release>())  // Release GIL!
        .def("detect_multi", &DetectorService::detectMulti,
             py::call_guard<py::gil_scoped_release>());
}
```

---

## 7. Directory Structure

```
trt_detector/
├── CMakeLists.txt
├── setup.py                    # pip install support
│
├── include/
│   ├── trt_detector/
│   │   ├── detector_service.hpp
│   │   ├── trt_engine.hpp
│   │   ├── preprocessor.hpp
│   │   ├── postprocessor.hpp
│   │   ├── detection.hpp
│   │   ├── model_config.hpp
│   │   └── cuda_utils.hpp
│
├── src/
│   ├── detector_service.cpp
│   ├── trt_engine.cpp
│   ├── preprocessor.cpp
│   ├── postprocessor.cpp
│   └── bindings.cpp            # pybind11
│
├── cuda/
│   └── preprocess_kernel.cu    # Optional GPU preprocessing
│
└── tests/
    ├── test_engine.cpp
    └── test_detector.py
```

---

## 8. Build System (CMake)

```cmake
cmake_minimum_required(VERSION 3.18)
project(trt_detector LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find dependencies
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

# TensorRT (manual path)
set(TENSORRT_ROOT "/usr/local/tensorrt")
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

# Build shared library
pybind11_add_module(trt_detector
    src/detector_service.cpp
    src/trt_engine.cpp
    src/preprocessor.cpp
    src/postprocessor.cpp
    src/bindings.cpp
)

target_link_libraries(trt_detector PRIVATE
    nvinfer
    nvinfer_plugin
    cudart
    ${OpenCV_LIBS}
)

target_include_directories(trt_detector PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
```

---

## 9. Implementation Phases

### Phase 1: Core Engine (Week 1)
- [ ] `TRTEngine` class - load engine, allocate buffers, run inference
- [ ] `Preprocessor` - letterbox implementation in C++
- [ ] `Postprocessor` - NMS and coordinate mapping
- [ ] Basic single-model test

### Phase 2: Multi-Model Support (Week 2)
- [ ] `DetectorService` - model registry and management
- [ ] `ModelConfig` - configuration handling
- [ ] CUDA stream per model
- [ ] Concurrent inference testing

### Phase 3: Python Bindings (Week 3)
- [ ] pybind11 bindings for all classes
- [ ] NumPy array handling (zero-copy where possible)
- [ ] GIL release for inference calls
- [ ] pip-installable package

### Phase 4: Optimization (Week 4)
- [ ] GPU preprocessing kernel (optional)
- [ ] Memory pool for temporary allocations
- [ ] Benchmark and profile
- [ ] Documentation

---

## 10. Performance Targets

| Metric | Python (Current) | C++ (Target) |
|--------|-----------------|--------------|
| Single model latency | ~12ms | ~10ms |
| Multi-model (2) latency | ~24ms (serial) | ~12ms (parallel) |
| Memory overhead | ~200MB | ~50MB |
| CPU usage | 15-20% | 5-10% |
| Max concurrent models | 1 | 8+ |

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| TensorRT version compatibility | Pin TensorRT version, document requirements |
| CUDA context issues | One context per process, proper cleanup |
| Memory leaks | RAII wrappers, valgrind testing |
| Python binding complexity | Start simple, iterate |

---

## 12. Dependencies

```
Required:
- CUDA Toolkit >= 11.8
- TensorRT >= 8.6
- OpenCV >= 4.5 (with CUDA support optional)
- pybind11 >= 2.10
- CMake >= 3.18
- GCC >= 9 or Clang >= 10

Optional:
- NVIDIA Nsight (profiling)
- Google Test (unit tests)
```

---

## Next Steps

1. Review and approve this plan
2. Set up C++ project structure
3. Implement Phase 1 (Core Engine)
4. Test against Python baseline
