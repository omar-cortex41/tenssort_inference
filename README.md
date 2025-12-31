# TensorRT YOLO Inference

High-performance YOLO object detection using TensorRT C++ with Python bindings.

## Performance

| Method | FPS (with display) | Speedup |
|--------|-------------------|---------|
| TRT C++ FP16 | ~45 FPS | 1.6x |
| TRT C++ FP32 | ~45 FPS | 1.6x |
| Ultralytics Python | ~28 FPS | 1.0x |

---

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Build the C++ TensorRT module
cd trt_detector
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../..

# 3. Run inference
python trt_inference.py
```

---

## Full Installation Guide

### System Requirements

- **OS:** Ubuntu 20.04/22.04 (or similar Linux)
- **GPU:** NVIDIA GPU with compute capability 7.5+ (RTX 20 series or newer)
- **CUDA:** 12.x
- **cuDNN:** 9.x
- **TensorRT:** 10.x
- **Python:** 3.10+

### Step 1: Install System Dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake git
sudo apt install -y libopencv-dev python3-dev python3-pip
sudo apt install -y pybind11-dev
```

### Step 2: Install CUDA Toolkit

Download and install from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

Verify installation:
```bash
nvcc --version
nvidia-smi
```

### Step 3: Install cuDNN

Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) and follow installation instructions.

### Step 4: Install TensorRT

Download from [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)

After installation, add `trtexec` to your PATH:
```bash
# Find trtexec location
sudo find / -type f -name trtexec 2>/dev/null
# Usually at: /usr/src/tensorrt/bin/trtexec

# Add to PATH
echo 'export PATH=$PATH:/usr/src/tensorrt/bin' >> ~/.bashrc
source ~/.bashrc

# Verify
trtexec --version
```

### Step 5: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Build the C++ TensorRT Module

```bash
cd trt_detector
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../..
```

This creates `trt_detector/build/trt_detector.cpython-*.so` which Python imports.

### Step 7: Prepare Your Model

#### Convert PyTorch model to ONNX:
```bash
python pt_to_onnx.py models/your_model.pt
```

Or manually:
```python
from ultralytics import YOLO
model = YOLO("models/your_model.pt")
model.export(format="onnx")
```

#### Convert ONNX to TensorRT Engine:

**FP16 (recommended - 2x faster, minimal accuracy loss):**
```bash
trtexec --onnx=models/your_model.onnx --saveEngine=models/your_model_fp16.engine --fp16
```

**FP32 (maximum accuracy):**
```bash
trtexec --onnx=models/your_model.onnx --saveEngine=models/your_model_fp32.engine
```

> ⚠️ **Important:** TensorRT engines are GPU-specific. You must rebuild the engine on each target machine.

---

## Usage

### TensorRT C++ Inference (Fast)

Edit `trt_inference.py` to set your paths:
```python
VIDEO_PATH = "videos/your_video.mp4"
ENGINE_PATH = "models/your_model.engine"
CLASS_NAMES = ['class1', 'class2', ...]  # Your model's class names
```

Run:
```bash
python trt_inference.py
```

### Ultralytics YOLO Inference (Reference)

Edit `yolo.py` to set your paths:
```python
VIDEO_PATH = "videos/your_video.mp4"
MODEL_PATH = "models/your_model.pt"
```

Run:
```bash
python yolo.py
```

### Controls

- Press `q` to quit the window

---

## Project Structure

```
.
├── trt_detector/           # C++ TensorRT detector module
│   ├── include/            # Header files
│   ├── src/                # Source files (.cpp, .cu)
│   ├── build/              # Build output (created after cmake/make)
│   └── CMakeLists.txt      # CMake build configuration
├── models/                 # Model files
│   ├── *.pt                # PyTorch weights
│   ├── *.onnx              # ONNX models
│   └── *.engine            # TensorRT engines
├── videos/                 # Input videos
├── trt_inference.py        # TensorRT C++ inference script
├── yolo.py                 # Ultralytics YOLO inference script
├── pt_to_onnx.py           # PyTorch to ONNX converter
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Troubleshooting

### "No module named 'trt_detector'"

The C++ module hasn't been built. Run:
```bash
cd trt_detector
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### "Failed to load model" or "Engine deserialization failed"

TensorRT engines are GPU-specific. Rebuild on your machine:
```bash
trtexec --onnx=models/your_model.onnx --saveEngine=models/your_model.engine --fp16
```

### CMake can't find TensorRT

Set the TensorRT path manually:
```bash
export TENSORRT_ROOT=/usr/local/tensorrt  # or your TensorRT location
cd trt_detector/build
cmake ..
make -j$(nproc)
```

### CMake can't find pybind11

Install pybind11:
```bash
sudo apt install pybind11-dev
# or
pip install pybind11
```

### Low FPS

- Use FP16 engine instead of FP32
- Check GPU usage with `nvidia-smi`
- Ensure you're using the discrete GPU (not integrated graphics)

### CUDA out of memory

- Close other GPU applications
- Use a smaller model
- Reduce input video resolution
