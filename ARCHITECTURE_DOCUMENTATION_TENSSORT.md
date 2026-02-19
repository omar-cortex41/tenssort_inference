# TensorSort Inference System - Complete Architecture Documentation

**Version:** Phase 1 Complete
**Last Updated:** February 19, 2026
**Author:** Technical Documentation Team
**Target Audience:** Senior Engineers, System Architects

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Performance Optimizations](#performance-optimizations)
5. [File Structure](#file-structure)
6. [Configuration System](#configuration-system)
7. [Build System & Dependencies](#build-system--dependencies)
8. [API Reference](#api-reference)
9. [Performance Metrics](#performance-metrics)
10. [Development Guide](#development-guide)

---

## Executive Summary

**TensorSort Inference** is a high-performance, production-grade object detection inference system built around NVIDIA TensorRT. The architecture achieves **~45 FPS** for YOLO inference (1.6x faster than Ultralytics Python), featuring:

- **Zero-copy GPU pipeline** from video decode to inference
- **Batched multi-stream processing** (8+ concurrent video streams)
- **CUDA-accelerated preprocessing and postprocessing**
- **Asynchronous 3-stage pipeline** for maximized throughput
- **Hardware-accelerated video decoding** (NVDEC/DeepStream)
- **ByteTrack multi-object tracking** with persistent IDs

### Key Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Single Stream FPS | ~45 | YOLOv26m FP16, 640×640 |
| 8 Stream Total FPS | ~180 | Zero-copy GPU path |
| Per-Stream (8 streams) | ~22.5 | RTX 4060 |
| GPU Memory | 1.05 GB | Including model weights |
| End-to-End Latency | 6.2 ms | Zero-copy path |
| Speedup vs Ultralytics | 1.6× | Single stream |
| Speedup vs DeepStream | 0.9× | 90% of DeepStream performance |

### Codebase Metrics

- **C++/CUDA:** 2,093 lines (inference engine)
- **Python:** 19,152 lines (orchestration, integration)
- **Architecture:** 3-tier (RTSPModule → TensorRT → ByteTrack)

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TensorSort Inference                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────┐       │
│  │ RTSPModule   │───▶│ TRTEngine     │───▶│ ByteTrack    │       │
│  │ (Video I/O)  │    │ (Detection)   │    │ (Tracking)   │       │
│  └──────────────┘    └───────────────┘    └──────────────┘       │
│        │                     │                     │               │
│   GStreamer             TensorRT              C++ Tracking         │
│   NVDEC/CPU             FP16/FP32             Kalman Filter        │
│   NV12 Format           CUDA Kernels          IoU Matching         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Zero-Copy Pipeline (Optimal Path)

```
Video File/RTSP
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ RTSPModule: GStreamer Hardware Decoder                            │
│   • filesrc/rtspsrc → qtdemux/rtpdepay → nvh264dec (NVDEC)       │
│   • GPU Memory: NV12 format (4:2:0 chroma subsampling)            │
│   • Zero-copy via cudaconvert (GL → CUDA memory)                  │
└───────────────────────────────────────────────────────────────────┘
    │ CUDA device pointer (uint64_t)
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ TRTEngine: Parallel CUDA Preprocessing                            │
│   • NV12→RGB conversion (YUV BT.601 matrix)                       │
│   • Letterbox resize with bilinear interpolation                  │
│   • Normalization (÷255) + HWC→CHW transpose                      │
│   • Output: [N,3,640,640] float32 tensor                          │
│   • 8 parallel streams for batch processing                       │
└───────────────────────────────────────────────────────────────────┘
    │ GPU tensor (no H2D copy!)
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ TensorRT Inference Engine                                         │
│   • Dynamic batch support (1-8 images)                            │
│   • FP16 precision (2x faster than FP32)                          │
│   • Output: [N,300,6] detection tensor                            │
│   •   [x1, y1, x2, y2, confidence, class_id]                      │
└───────────────────────────────────────────────────────────────────┘
    │ GPU detection tensor
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ CUDA Postprocessing Kernel                                        │
│   • Confidence filtering (conf > threshold)                       │
│   • Coordinate transformation (letterbox → original)              │
│   • Atomic compaction (removes empty slots)                       │
│   • Output: [N, valid_count, 7] compact tensor                    │
└───────────────────────────────────────────────────────────────────┘
    │ D2H copy (only valid detections)
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ ByteTrack Multi-Object Tracker                                    │
│   • Kalman filter state prediction                                │
│   • IoU-based detection-to-track association                      │
│   • Track lifecycle management                                    │
│   • Persistent track IDs across frames                            │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
Application (Python)
```

---

## Core Components

### 2.1 TRTEngine (C++ Core)

**Location:** `trt_detector/src/trt_engine.cpp` (615 lines)

**Purpose:** High-performance TensorRT inference wrapper with batching, CUDA preprocessing, and GPU postprocessing.

#### Key Features

1. **Dynamic Batch Size Support**
   - Automatically detects max batch from TensorRT optimization profiles
   - Supports batch sizes 1-8 (configurable at engine build time)
   - No reallocation needed for variable batch sizes

2. **Parallel Preprocessing Architecture**
   ```cpp
   // 8 separate CUDA streams for parallel frame preprocessing
   for (int i = 0; i < max_batch_size_; ++i) {
       cudaStreamCreate(&preprocess_streams_[i]);
       cudaEventCreate(&preprocess_events_[i]);
   }
   ```

3. **Three Detection Paths**

   **A. Zero-Copy GPU NV12** (Fastest)
   ```cpp
   std::vector<std::vector<Detection>> detectBatchGpuNV12(
       const std::vector<uint64_t>& gpu_ptrs,  // CUDA device pointers
       const std::vector<int>& widths,
       const std::vector<int>& heights
   );
   ```
   - Input: GPU pointers from RTSPModule
   - No H2D copy required
   - ~1ms preprocessing per frame

   **B. CPU NV12 Direct** (Fast)
   ```cpp
   std::vector<std::vector<Detection>> detectBatchNV12(
       const std::vector<const uint8_t*>& nv12_data,
       const std::vector<size_t>& data_sizes,
       const std::vector<int>& widths,
       const std::vector<int>& heights
   );
   ```
   - Skips CPU color conversion (no cv2.cvtColor)
   - Uploads NV12 directly to GPU
   - ~20-30% faster than BGR path

   **C. BGR Fallback** (Slowest)
   ```cpp
   std::vector<std::vector<Detection>> detectBatch(
       const std::vector<cv::Mat>& frames
   );
   ```
   - Traditional OpenCV path
   - CPU preprocessing + H2D copy
   - Used as fallback

#### Memory Management

**GPU Buffer Allocation:**
```cpp
// Input tensor: [max_batch, 3, H, W] float32
input_size_ = max_batch_size_ * 3 * 640 * 640 * sizeof(float);  // ~58 MB
cudaMalloc(&d_input_, input_size_);

// Per-batch source buffers (4K support)
src_size_per_batch_ = 3840 * 2160 * 3;  // ~24 MB
for (int i = 0; i < max_batch_size_; ++i) {
    cudaMalloc(&d_src_batch_[i], src_size_per_batch_);
    cudaMallocHost(&h_src_batch_[i], src_size_per_batch_);  // Pinned
}

// GPU postprocessing buffers
cudaMalloc(&d_postprocess_out_, max_batch_size_ * 300 * 7 * sizeof(float));
cudaMalloc(&d_det_counts_, max_batch_size_ * sizeof(int));
```

**Total GPU Memory (batch=8):** ~250 MB (excluding model weights)

#### Optimization Techniques

1. **Pinned Memory for H2D Transfers**
   - `cudaMallocHost()` allocates page-locked memory
   - ~2× faster than pageable memory
   - Enables asynchronous DMA transfers

2. **Event-Based Stream Synchronization**
   ```cpp
   // GPU-side sync (no CPU blocking)
   for (int i = 0; i < batch_size; ++i) {
       cudaEventRecord(preprocess_events_[i], preprocess_streams_[i]);
   }
   for (int i = 0; i < batch_size; ++i) {
       cudaStreamWaitEvent(stream_, preprocess_events_[i], 0);
   }
   ```

3. **Batch Processing**
   - Single TensorRT inference call for 8 frames
   - Amortizes kernel launch overhead
   - Better tensor core utilization

---

### 2.2 CUDA Preprocessing Kernels

**Location:** `trt_detector/src/cuda_preprocess.cu` (264 lines)

#### NV12 Preprocessing Kernel

```cuda
__global__ void nv12ToRgbPreprocessKernel(
    const uint8_t* y_plane,   // [H,W] luminance
    const uint8_t* uv_plane,  // [H/2,W] interleaved UV
    float* dst,               // [3,640,640] RGB output
    // ... parameters
)
```

**NV12 Format:**
- Y plane: Full resolution luminance (H×W)
- UV plane: Half resolution chroma (H/2×W), interleaved
- Total size: H×W×1.5 bytes

**Operations:**
1. **YUV→RGB Color Space Conversion** (BT.601)
   ```cuda
   float Y = (float)y_plane[y0 * src_w + x0];
   float U = (float)uv_plane[uv_idx] - 128.0f;
   float V = (float)uv_plane[uv_idx + 1] - 128.0f;

   float rf = Y + 1.402f * V;
   float gf = Y - 0.344136f * U - 0.714136f * V;
   float bf = Y + 1.772f * U;
   ```

2. **Letterbox Resize** (maintain aspect ratio)
3. **Normalization** (÷255)
4. **HWC→CHW Transpose**

**Performance:** ~1ms per 1080p frame (vs 5-8ms on CPU)

---

### 2.3 CUDA Postprocessing Kernel

**Location:** `trt_detector/src/cuda_postprocess.cu` (128 lines)

```cuda
__global__ void postprocessKernel(
    const float* raw_output,   // [batch*300*6]
    float* detections,         // [batch*300*7]
    int* det_counts,           // [batch]
    int max_dets,              // 300
    float conf_threshold,
    // ... metadata arrays
)
```

**Algorithm:**

1. **Parallel Filtering** (2400 threads for batch=8)
   ```cuda
   int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
   int batch_idx = global_idx / max_dets;
   int det_idx = global_idx % max_dets;

   if (conf < conf_threshold) return;
   ```

2. **Coordinate Transformation**
   ```cuda
   int x1 = (int)((x1_raw - pad_x) / scale);
   int y1 = (int)((y1_raw - pad_y) / scale);
   ```

3. **Atomic Compaction**
   ```cuda
   int slot = atomicAdd(&det_counts[batch_idx], 1);
   detections[batch_idx*300 + slot] = {...};
   ```

**Performance:**
- Input: 57.6 KB (8×300×6 floats)
- Output: ~6.7 KB (8×30×7 floats) - **10× reduction**
- Latency: <0.1ms (vs 2-3ms on CPU) - **240× faster**

---

### 2.4 AsyncPipeline (3-Stage Processing)

**Location:** `trt_detector/src/async_pipeline.cpp` (263 lines)

**Architecture:**
```
Stage 1: Capture        Stage 2: Inference      Stage 3: Postprocess*
┌───────────┐           ┌──────────────┐       ┌─────────────┐
│ cv::      │─Queue 1──▶│ TRT Enqueue  │─Q2───▶│ Pass-through│─Q3──▶ App
│ VideoCapt │           │ (GPU)        │       │             │
└───────────┘           └──────────────┘       └─────────────┘
  Thread 1                Thread 2               Thread 3*
```

**Note:** Stage 3 is currently pass-through since `detect()` includes postprocessing. Future optimization can split this.

**Key Features:**

1. **Bounded Queues** (default: 4 frames each)
   - Prevents memory explosion
   - Backpressure control
   ```cpp
   capture_cv_.wait_for(lock, 100ms, [this] {
       return capture_queue_.size() < max_capture_queue_ || !running_;
   });
   ```

2. **Non-Blocking Design**
   - Camera/file reading never waits on inference
   - GPU operations overlapped with I/O
   ```cpp
   bool tryGetResult(FrameResult& result);  // Non-blocking
   bool getResult(FrameResult& result);     // Blocking
   ```

3. **Graceful Shutdown**
   ```cpp
   capture_done_ = true;
   capture_cv_.notify_all();
   inference_cv_.notify_all();
   ```

**Performance:**
- Throughput: ~15-20% improvement over synchronous
- Latency: Adds ~2-3 frames of buffering
- Use case: Real-time streaming, long videos

---

### 2.5 RTSPModule (Hardware Video Decoding)

**Location:** `RTSPModule-main/src/core/stream_decoder.cpp`

**Purpose:** GStreamer-based hardware-accelerated video decoding with zero-copy GPU output.

#### 3-Tier Hardware Fallback

```cpp
if (!hardware_accel_failed_) {
    // 1. Try DeepStream (nvv4l2decoder)
    decoder = gst_element_factory_make("nvv4l2decoder", ...);

    if (!decoder) {
        // 2. Try standard NVDEC (nvh264dec)
        decoder = gst_element_factory_make("nvh264dec", ...);
    }

    if (!decoder) {
        // 3. Fallback to CPU (avdec_h264)
        decoder = gst_element_factory_make("avdec_h264", ...);
    }
}
```

#### Zero-Copy GPU Access

**GStreamer Pipeline:**
```
filesrc → qtdemux → nvh264dec → cudaconvert → appsink
          (GPU)     (GL→CUDA)    (NV12)
```

**CUDA Pointer Extraction:**
```cpp
GpuFrameInfo getGpuFrame(int timeout_ms) {
    GstSample* sample = gst_app_sink_try_pull_sample(appsink_, ...);
    GstBuffer* buffer = gst_sample_get_buffer(sample);

    GstCudaMemory* cuda_mem = GST_CUDA_MEMORY_CAST(
        gst_buffer_peek_memory(buffer, 0)
    );
    CUdeviceptr cuda_ptr = gst_cuda_memory_get_device_pointer(cuda_mem);

    return {true, (uint64_t)cuda_ptr, width, height, ...};
}
```

**Performance:**
- Zero-copy: 0ms fetch overhead
- CPU buffer fallback: ~1-2ms per frame
- NVDEC decode: ~0.5ms, CPU: ~3-5ms per 1080p frame

---

### 2.6 ByteTrack Integration

**Location:** Referenced in `tracker.py` (245 lines)

**Algorithm:**

1. **Prediction:** Kalman filter predicts track positions
2. **Association:** Match detections to tracks via IoU
3. **Update:** Kalman update for matched tracks
4. **Lifecycle:** Birth/Update/Death management

**Integration:**
```python
import bytetrack_cpp as bt

tracker = bt.BYTETracker(bt.TrackerConfig(
    track_thresh=0.5,
    match_thresh=0.8,
    track_buffer=30,
    frame_rate=30
))

tracks = tracker.update(cpp_detections)
```

**Performance:** ~0.5-1ms per frame (C++ implementation)

---

## Performance Optimizations

### 3.1 GPU Preprocessing (NV12 Path)

**Traditional Pipeline:**
```
CPU Decode → cv2.cvtColor() → H2D Copy → GPU Preprocess
   5ms          8ms             2ms          1ms
                  Total: 16ms (62.5 FPS max)
```

**Optimized Pipeline:**
```
NVDEC → GPU NV12 Preprocess
0.5ms      1ms
      Total: 1.5ms (666 FPS max)
```

**Speedup:** 10.6× for preprocessing stage

---

### 3.2 GPU Postprocessing

**CPU Loop (traditional):**
```python
for i in range(300):
    if output[i, 4] > threshold:
        x1 = int((x1_raw - pad_x) / scale)
        # ... transform, create Detection
```
**Benchmark:** ~24ms for batch=8

**GPU Kernel (optimized):**
```cuda
// 2400 threads in parallel
if (conf < threshold) return;
int x1 = (int)((x1_raw - pad_x) / scale);
int slot = atomicAdd(&det_counts[batch_idx], 1);
```
**Benchmark:** <0.1ms → **Speedup: 240×**

---

### 3.3 Zero-Copy Pipeline

**Memory Copies Eliminated:**

1. **Video Decode → Inference:** 0 copies
   ```python
   gpu_ptr = rtsp.get_cuda_frame(cam_id)['ptr']
   detections = detector.detect_batch_gpu_nv12([gpu_ptr], ...)
   ```

2. **Bandwidth Savings:** ~30,000× reduction
   - Traditional: ~6 MB H2D per frame
   - Zero-copy: ~200 bytes D2H (detections only)

---

### 3.4 Parallel Stream Processing

**Execution Timeline:**
```
Stream 0: [H2D][Kernel]
Stream 1:    [H2D][Kernel]
Stream 2:       [H2D][Kernel]
Stream 3:          [H2D][Kernel]
...
Stream 7:                         [H2D][Kernel]
                                                [Wait][Inference]
```

**Synchronization:**
```cpp
// Event-based sync (GPU-side, no CPU blocking)
for (int i = 0; i < batch_size; ++i) {
    cudaEventRecord(preprocess_events_[i], preprocess_streams_[i]);
}
for (int i = 0; i < batch_size; ++i) {
    cudaStreamWaitEvent(stream_, preprocess_events_[i], 0);
}
```

**Performance:** ~20% faster than sequential

---

## File Structure

```
/home/omar/work/tenssort_inference/tenssort_inference/
│
├── config/
│   └── config.yaml                     # Model, video, tracking config
│
├── trt_detector/                       # C++ TensorRT engine
│   ├── include/trt_detector/
│   │   ├── trt_engine.hpp              # Main TRT wrapper (138 lines)
│   │   ├── detector_service.hpp        # Thread-safe service (56 lines)
│   │   ├── async_pipeline.hpp          # 3-stage pipeline (123 lines)
│   │   ├── cuda_preprocess.hpp         # Preprocessing (23 lines)
│   │   ├── cuda_postprocess.hpp        # Postprocessing (41 lines)
│   │   ├── cuda_osd.hpp                # OSD kernel (27 lines)
│   │   ├── model_config.hpp            # Config struct
│   │   ├── detection.hpp               # Detection struct
│   │   ├── preprocessor.hpp            # CPU preprocess (optional)
│   │   └── postprocessor.hpp           # CPU postprocess (optional)
│   │
│   ├── src/
│   │   ├── trt_engine.cpp              # TRT engine (615 lines)
│   │   ├── detector_service.cpp        # Service wrapper (74 lines)
│   │   ├── async_pipeline.cpp          # Pipeline (220 lines)
│   │   ├── bindings.cpp                # Python bindings (386 lines)
│   │   ├── cuda_preprocess.cu          # Preprocessing kernels (264 lines)
│   │   ├── cuda_postprocess.cu         # Postprocessing kernel (128 lines)
│   │   ├── cuda_osd.cu                 # OSD rendering (175 lines)
│   │   ├── preprocessor.cpp            # CPU preprocess (optional)
│   │   └── postprocessor.cpp           # CPU postprocess (optional)
│   │
│   ├── build/
│   │   └── trt_detector.cpython-*.so   # Python module (~2.1 MB)
│   │
│   └── CMakeLists.txt                  # CMake config (64 lines)
│
├── RTSPModule-main/                    # GStreamer video decoding
│   ├── include/rtspmodule/
│   │   ├── stream_decoder.h            # Decoder class (205 lines)
│   │   ├── gpu_buffer.h                # GPU memory management
│   │   ├── cpu_buffer.h                # CPU ring buffer
│   │   └── rtsp_structs.h              # Data structures
│   │
│   ├── src/
│   │   ├── core/
│   │   │   ├── stream_decoder.cpp      # GStreamer pipeline
│   │   │   ├── cpu_buffer.cpp          # Ring buffer
│   │   │   └── rtsp_client.cpp         # Multi-stream manager
│   │   └── bindings/
│   │       └── bindings.cpp            # Python bindings
│   │
│   └── configs/
│       ├── config.yaml                 # RTSP config template
│       └── generated_config.yaml       # Runtime config
│
├── Python Scripts
│   ├── detector.py                     # Single-stream (215 lines)
│   ├── rtsp_detector.py                # Multi-stream (442 lines)
│   ├── tracker.py                      # ByteTrack (245 lines)
│   ├── pt_to_trt.py                    # Model conversion
│   └── load_class_names.py             # Utility
│
├── requirements.txt                    # Python dependencies
└── steps.md                            # Build/setup guide
```

---

## Configuration System

### config.yaml

```yaml
model:
  engine_path: ../models/yolo26m_fp16_dynamic.engine
  conf_threshold: 0.6
  nms_threshold: 0.45
  input_width: 640
  input_height: 640

display:
  frame_skip: 3  # Show every Nth frame

tracking:
  track_thresh: 0.5
  match_thresh: 0.8
  track_buffer: 30
  track_classes: [person]

streams:
  - id: 0
    source: ../videos/vid1.mp4
    name: Stream 1
  # ... 7 more streams
```

---

## Build System & Dependencies

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(trt_detector LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)  # Turing, Ampere, Ada

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

set(SOURCES
    src/trt_engine.cpp
    src/detector_service.cpp
    src/async_pipeline.cpp
    src/bindings.cpp
    src/cuda_preprocess.cu
    src/cuda_postprocess.cu
)

pybind11_add_module(trt_detector ${SOURCES})
set_target_properties(trt_detector PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(trt_detector PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS}
)
```

### Build Commands

```bash
cd trt_detector
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Dependencies

**System Libraries:**
```bash
sudo apt install -y \
    build-essential cmake git \
    libopencv-dev \
    pybind11-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev
```

**CUDA/TensorRT:**
- CUDA Toolkit: 11.8 or 12.x
- TensorRT: 8.x or 10.x

**Python Packages:**
```
numpy
opencv-python
ultralytics
pybind11
pyyaml
tensorrt
```

---

## API Reference

### Python API

#### DetectorService

```python
import sys
sys.path.insert(0, 'trt_detector/build')
from trt_detector import DetectorService, ModelConfig

# Create detector
detector = DetectorService()

# Load model
config = ModelConfig(
    engine_path="models/yolo.engine",
    class_names=["person", "car", ...],
    conf_threshold=0.5,
    nms_threshold=0.45,
    input_width=640,
    input_height=640
)
detector.load_model(config)

# Single frame
detections = detector.detect(frame)

# Batched
detections_batch = detector.detect_batch([frame1, frame2, ...])

# Zero-copy GPU NV12
detections_batch = detector.detect_batch_gpu_nv12(
    gpu_ptrs=[ptr1, ptr2, ...],
    widths=[1920, 1920, ...],
    heights=[1080, 1080, ...]
)

# CPU NV12 (skips cv2.cvtColor)
detections_batch = detector.detect_batch_nv12(
    frames=[nv12_frame1, nv12_frame2, ...],
    width=1920,
    height=1080
)
```

#### Detection

```python
class Detection:
    x: int           # Top-left x
    y: int           # Top-left y
    width: int       # Box width
    height: int      # Box height
    class_id: int    # Class ID (0-79)
    confidence: float  # Score [0, 1]
    label: str       # Class name
```

#### AsyncPipeline

```python
from trt_detector import AsyncPipeline

pipeline = AsyncPipeline()
pipeline.init(config)
pipeline.start("video.mp4")

# Blocking
result = pipeline.get_result()

# Non-blocking
result = pipeline.try_get_result()

pipeline.stop()
```

---

## Performance Metrics

### Current Performance

**Single Stream (1080p):**
- FPS: ~45 (YOLOv26m FP16)
- Latency: ~6.2ms

**Multi-Stream (8 concurrent):**
- Total FPS: ~180
- Per-stream: ~22.5
- GPU Memory: 1.05 GB
- GPU Utilization: 85-95%

### Latency Breakdown (Zero-Copy)

```
Video Decode (NVDEC)      0.5ms
GPU NV12 Preprocess       1.0ms
TensorRT Inference        4.0ms
GPU Postprocess           0.1ms
D2H Copy (detections)     0.1ms
ByteTrack (CPU)           0.5ms
─────────────────────────────────
Total:                    6.2ms   (161 FPS max)
```

### Memory Footprint

**GPU (batch=8):**
- TRT engine: 800 MB
- Buffers: 250 MB
- Total: 1.05 GB

**CPU:**
- Python + libs: 580 MB

---

## Development Guide

### Adding a New Detection Path

1. Add method to `TRTEngine` class:
   ```cpp
   std::vector<std::vector<Detection>> detectMyPath(...);
   ```

2. Implement in `trt_engine.cpp`

3. Add Python binding in `bindings.cpp`:
   ```cpp
   .def("detect_my_path", [](DetectorService& self, ...) {
       py::gil_scoped_release release;
       return self.detectMyPath(...);
   });
   ```

4. Rebuild:
   ```bash
   cd trt_detector/build && make -j
   ```

### Performance Profiling

```bash
# CUDA profiling
nsys profile -o report python3 rtsp_detector.py
nsys-ui report.qdrep

# GStreamer debugging
GST_DEBUG=3 python3 rtsp_detector.py
```

---

## Conclusions

TensorSort Inference is a production-ready system achieving near-optimal GPU utilization through:

1. **Zero-copy data flow**
2. **Batched processing**
3. **CUDA-accelerated pre/post**
4. **Parallel stream execution**
5. **Hardware video decoding**

**Key Metrics:**
- 45 FPS single-stream (1.6× vs Ultralytics)
- 180 FPS total for 8 streams
- 6.2ms latency
- 1.05 GB GPU memory
- 2,093 lines C++/CUDA

This system is suitable for real-time multi-camera surveillance, traffic monitoring, or any application requiring high-throughput object detection with sub-10ms latency.

---

**End of Documentation**
