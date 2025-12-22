# YOLO TensorRT Detector

Real-time object detection using YOLO11 with TensorRT acceleration on NVIDIA GPUs.

## What is this?

This project runs YOLO object detection **directly on your GPU** using NVIDIA TensorRT, bypassing PyTorch entirely for maximum speed.

| Method | Speed | Dependencies |
|--------|-------|--------------|
| `yolo.py` (PyTorch) | ~30 FPS | Heavy (PyTorch, Ultralytics) |
| `main.py` (TensorRT) | ~100+ FPS | Light (TensorRT, PyCUDA) |

## Requirements

- NVIDIA GPU (tested on RTX 4060)
- CUDA 12.x
- cuDNN 9.x
- TensorRT 10.x
- Python 3.10+

## Installation

### 1. Check your GPU
```bash
nvidia-smi
```

### 2. Install CUDA Toolkit
Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

### 3. Install cuDNN
Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

### 4. Install TensorRT
Download the version compatible with your CUDA from [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)

Find and add `trtexec` to PATH:
```bash
sudo find / -type f -name trtexec 2>/dev/null
# Output: /usr/src/tensorrt/bin/trtexec

echo 'export PATH=$PATH:/usr/src/tensorrt/bin' >> ~/.bashrc
source ~/.bashrc

# Verify installation
trtexec --version
```

### 5. Install Python dependencies
```bash
pip install pycuda numpy opencv-python
```

## Usage

### Quick Start (TensorRT)
```bash
python main.py
```

### Quick Start (PyTorch/Ultralytics)
```bash
pip install ultralytics
python yolo.py
```

## Building TensorRT Engines

### Convert PyTorch → ONNX
```bash
python pt_to_onnx.py
```
Or manually:
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.export(format="onnx")
```

### Convert ONNX → TensorRT Engine

**FP16 (recommended - fast, minimal accuracy loss):**
```bash
trtexec \
  --onnx=yolo11n.onnx \
  --saveEngine=yolo11n.engine \
  --fp16 \
  --memPoolSize=workspace:4096
```

**FP32 (maximum accuracy):**
```bash
trtexec \
  --onnx=yolo11n.onnx \
  --saveEngine=yolo11n.engine \
  --memPoolSize=workspace:4096
```

## Project Structure

```
├── main.py          # TensorRT inference (fast, no PyTorch)
├── pt_to_onnx.py    # Convert .pt to .onnx
├── yolo11n.pt       # YOLO11 nano PyTorch weights
├── yolo11n.onnx     # YOLO11 nano ONNX model
├── yolo11n.engine   # TensorRT engine (generated)
└── people.mp4       # Test video
```

## Configuration

Edit `main.py`:
```python
VIDEO_PATH = "people.mp4"  # or 0 for webcam
ENGINE_PATH = "yolo11n.engine"
CONF_THRESHOLD = 0.25      # detection confidence
NMS_THRESHOLD = 0.45       # duplicate removal threshold
```

## Controls

- Press `q` to quit

## Troubleshooting

### CUDA Error 35 (Stub Library)
You have conflicting TensorRT versions. Remove all and reinstall:
```bash
pip uninstall tensorrt tensorrt_cu12 tensorrt_cu13 -y
pip install tensorrt_cu12
```

### Engine deserialization failed
Rebuild the engine without `--versionCompatible`:
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine
```

### Custom classes
Edit `coco.names` with your class names (one per line), then rebuild your engine from a custom-trained model.
