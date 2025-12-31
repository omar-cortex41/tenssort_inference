# TRT Detector - C++ Architecture

## Overview

High-performance TensorRT inference service with multi-model support, exposed as a Python module via pybind11.

## Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     Python Application                        │
│                  (14k line AI engine)                        │
└──────────────────────────┬───────────────────────────────────┘
                           │ import trt_detector
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  trt_detector.so (C++ Module)                │
├──────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐  │
│  │                  DetectorService                        │  │
│  │  - Model registry (name → TRTEngine)                   │  │
│  │  - load_model() / unload_model()                       │  │
│  │  - detect() / detect_multi()                           │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                   │
│  ┌────────────────────────▼───────────────────────────────┐  │
│  │                    TRTEngine                            │  │
│  │  - TensorRT runtime, engine, context                   │  │
│  │  - Pre-allocated GPU/CPU buffers                       │  │
│  │  - CUDA stream per engine                              │  │
│  │  - infer() method                                      │  │
│  └───────────┬─────────────────────────────┬──────────────┘  │
│              │                             │                  │
│  ┌───────────▼───────────┐   ┌─────────────▼──────────────┐  │
│  │    Preprocessor       │   │     Postprocessor          │  │
│  │  - letterbox()        │   │  - NMS                     │  │
│  │  - normalize()        │   │  - Coordinate transform    │  │
│  │  - HWC→CHW            │   │  - Threshold filtering     │  │
│  └───────────────────────┘   └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                        GPU (CUDA)                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │ Engine1 │ │ Engine2 │ │ EngineN │  (Multiple models)     │
│  │ Stream1 │ │ Stream2 │ │ StreamN │  (Parallel execution)  │
│  └─────────┘ └─────────┘ └─────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

## File Structure

```
trt_detector/
├── CMakeLists.txt          # Build configuration
├── setup.py                # Python package setup
├── include/
│   └── trt_detector/
│       ├── detection.hpp       # Detection struct
│       ├── model_config.hpp    # Model configuration
│       ├── preprocessor.hpp    # Image preprocessing
│       ├── postprocessor.hpp   # NMS and output processing
│       ├── trt_engine.hpp      # TensorRT engine wrapper
│       └── detector_service.hpp # Multi-model service
└── src/
    ├── preprocessor.cpp
    ├── postprocessor.cpp
    ├── trt_engine.cpp
    ├── detector_service.cpp
    └── bindings.cpp            # pybind11 bindings
```

## Data Flow

```
Frame (numpy) ──► Preprocessor ──► TRTEngine ──► Postprocessor ──► Detections
     │                │               │               │               │
     │           letterbox       GPU infer         NMS            list[Detection]
     │           normalize                     transform
     │           transpose
     ▼
  cv::Mat ────────► float* ────► float* ─────► vector<Detection>
```

## Key Design Decisions

1. **RAII Memory Management**: All GPU/CPU buffers allocated in constructor, freed in destructor
2. **One Stream Per Engine**: Enables parallel inference across models
3. **Zero-Copy Where Possible**: NumPy arrays passed directly to C++ without copying
4. **GIL Released During Inference**: Python threads not blocked during GPU work
5. **Pre-allocated Buffers**: No runtime memory allocation during inference

## Thread Safety

- Each `TRTEngine` is independent (own context, stream, buffers)
- `DetectorService` uses mutex for model registry modifications
- Inference calls are thread-safe across different models
- Same model concurrent calls require context pooling (future enhancement)
