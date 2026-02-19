# TensorRT YOLO Inference

High-performance YOLO object detection using TensorRT C++ with Python bindings.

## Performance

| Method | FPS (with display) | Speedup |
|--------|-------------------|---------|
| TRT C++ FP16 | ~45 FPS | 1.6x |
| TRT C++ FP32 | ~45 FPS | 1.6x |
| Ultralytics Python | ~28 FPS | 1.0x |

---

## Quick Start (Docker) - Recommended

The easiest way to run this project. Requires [Docker](https://docs.docker.com/get-docker/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
# 1. Place your model and video files
cp your_model.pt models/
cp your_video.mp4 videos/

# 2. Edit config
nano config/config.yaml

# 3. Build and run
docker compose build
docker compose run detector ./scripts/build_model.sh models/your_model.pt fp16
docker compose run detector
```

That's it! The Docker container handles all dependencies, C++ compilation, and model conversion.

---

## Quick Start (Manual)

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install libopencv-dev pybind11-dev

# 2. Install Python dependencies (including TensorRT)
pip install -r requirements.txt

# 3. Download TensorRT C++ headers (if not a sudo user)
# Download TensorRT tar.gz from https://developer.nvidia.com/tensorrt
# Extract headers to project directory:
tar -xzf TensorRT-*.tar.gz
mkdir -p trt_detector/external/tensorrt/include
cp -r TensorRT-*/include/* trt_detector/external/tensorrt/include/

# 4. Create symlinks for TensorRT libraries
cd ~/.local/lib/python3.10/site-packages/tensorrt_libs  # or your venv path
ln -sf libnvinfer.so.10 libnvinfer.so
ln -sf libnvinfer_plugin.so.10 libnvinfer_plugin.so
ln -sf libnvonnxparser.so.10 libnvonnxparser.so
cd -

# 5. Build the C++ TensorRT module
cd trt_detector
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../..

# 6. Convert PyTorch model to TensorRT engine
python pt_to_trt.py --fp16  # or --fp32

# 7. Run inference
python main.py
```

---

## Full Installation Guide

### System Requirements

- **OS:** Ubuntu 20.04/22.04 (or similar Linux)
- **GPU:** NVIDIA GPU with compute capability 7.5+ (RTX 20 series or newer)
  - RTX 3070: compute capability 8.6 ✓
  - RTX 4090: compute capability 8.9 ✓
- **Python:** 3.10+

### Tested Configuration (Recommended)

> ⚠️ **IMPORTANT:** All CUDA components must be compatible with each other. Mismatched versions cause runtime errors like `cuTensor permutate failed` or `illegal memory access`.

| Component | Version | Notes |
|-----------|---------|-------|
| **NVIDIA Driver** | 570.211.01 | Supports up to CUDA 12.8 |
| **CUDA Toolkit** | 12.8.61 | Must match driver capability |
| **cuDNN** | 8.9.7.29+cuda12.2 | Works with CUDA 12.x |
| **TensorRT** | 10.9.0.34+cuda12.8 | Must match CUDA version |

**Verify your versions:**
```bash
# Driver version
nvidia-smi | grep "Driver Version"

# CUDA version
nvcc --version

# cuDNN version
dpkg -l | grep cudnn

# TensorRT version
dpkg -l | grep nvinfer
```

**Common version mismatch errors:**
- `cuTensor permutate failed` → TensorRT/cuDNN built for wrong CUDA version
- `illegal memory access` → Library version mismatch or buffer overflow
- `Engine deserialization failed` → Engine built with different TensorRT version

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

#### Option A: Via pip (Recommended for non-sudo users)

```bash
# Install TensorRT Python bindings and libraries
pip install tensorrt

# Verify installation
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
```

**For C++ development (required for this project):**

1. Download TensorRT tar.gz from [NVIDIA TensorRT Downloads](https://developer.nvidia.com/tensorrt)
   - Choose the version matching your CUDA version (e.g., TensorRT 10.7 for CUDA 11.x)
   - Download the tar.gz package (not the .deb)

2. Extract headers to project directory:
```bash
# Extract the downloaded tar.gz
tar -xzf TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-11.8.tar.gz

# Copy headers to project
mkdir -p trt_detector/external/tensorrt/include
cp -r TensorRT-10.7.0.23/include/* trt_detector/external/tensorrt/include/
```

3. Create symlinks for libraries (pip installs versioned .so files):
```bash
# Navigate to tensorrt_libs directory (adjust path for your environment)
cd ~/.local/lib/python3.10/site-packages/tensorrt_libs
# or for venv: cd /path/to/venv/lib/python3.10/site-packages/tensorrt_libs

# Create symlinks
ln -sf libnvinfer.so.10 libnvinfer.so
ln -sf libnvinfer_plugin.so.10 libnvinfer_plugin.so
ln -sf libnvonnxparser.so.10 libnvonnxparser.so

# Return to project directory
cd -
```

#### Option B: System-wide installation (Requires sudo)

Download from [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)

After downloading the local repository package (e.g., `nv-tensorrt-local-repo-ubuntu2204-*.deb`), install it:

```bash
# Install the local repository package
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-*.deb

# Copy the GPG key
sudo cp /var/nv-tensorrt-local-repo-*/nv-tensorrt-local-*-keyring.gpg /usr/share/keyrings/

# Update package list
sudo apt-get update

# Install TensorRT
sudo apt-get install tensorrt

# Verify installation
dpkg -l | grep tensorrt
```

After installation, add `trtexec` to your PATH:
```bash
# Find trtexec location
sudo find /usr -name trtexec 2>/dev/null
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

**Verify the build:**
```bash
# Check the compiled module exists
ls -lh trt_detector/build/*.so

# Test Python import
python -c "import sys; sys.path.insert(0, 'trt_detector/build'); import trt_detector; print('✓ Module imported successfully!')"
```

### Step 7: Prepare Your Model

#### Quick Method: Use the conversion script

```bash
# Convert PyTorch model to TensorRT engine (FP16 - recommended)
python pt_to_trt.py --fp16

# Or for FP32 (slower but more accurate)
python pt_to_trt.py --fp32
```

This script automatically:
1. Converts `.pt` → `.onnx`
2. Converts `.onnx` → `.engine` using TensorRT

#### Manual Method: Step-by-step conversion

**Step 1: Convert PyTorch model to ONNX:**
```bash
python pt_to_onnx.py models/your_model.pt
```

Or manually:
```python
from ultralytics import YOLO
model = YOLO("models/your_model.pt")
model.export(format="onnx")
```

**Step 2: Convert ONNX to TensorRT Engine:**

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

### TensorRT C++ Inference with Web Interface (Recommended for SSH)

The detector now runs as a web service with real-time video streaming, perfect for remote access via SSH.

**1. Configure your model and video:**

Edit `config/config.yaml`:
```yaml
model:
  engine_path: "models/sgm32.engine"
  conf_threshold: 0.5
  nms_threshold: 0.45

video:
  path: "videos/your_video.mp4"

class_names:
  - "class1"
  - "class2"
  # ... your class names
```

**2. Start the web server:**
```bash
python detector.py
```

**3. Access the interface:**
- **Local:** Open browser to `http://localhost:8000`
- **SSH/Remote:**
  ```bash
  # On your local machine, create SSH tunnel:
  ssh -L 8000:localhost:8000 user@remote-server

  # Then open browser to http://localhost:8000
  ```

The web interface shows:
- 📹 Real-time video stream with detections
- 📊 Live statistics (FPS, frame count, processing time)
- 🎨 Clean dark theme UI

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

**Note:** `yolo.py` uses OpenCV display windows which won't work over SSH. Use `detector.py` for remote access.

---

## Project Structure

```
.
├── trt_detector/           # C++ TensorRT detector module
│   ├── include/            # Header files
│   ├── src/                # Source files (.cpp, .cu)
│   ├── build/              # Build output (after cmake/make)
│   └── CMakeLists.txt      # CMake build configuration
├── config/
│   └── config.yaml         # Model/video/class configuration
├── models/                 # Model files
│   ├── *.pt                # PyTorch weights
│   ├── *.onnx              # ONNX models
│   └── *.engine            # TensorRT engines
├── videos/                 # Input videos
├── scripts/
│   └── build_model.sh      # Model conversion script
├── main.py                 # TensorRT inference script
├── yolo.py                 # Ultralytics YOLO inference script
├── pt_to_onnx.py           # PyTorch to ONNX converter
├── Dockerfile              # Docker build file
├── docker-compose.yml      # Docker compose config
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
python pt_to_trt.py --fp16
# or manually:
# trtexec --onnx=models/your_model.onnx --saveEngine=models/your_model.engine --fp16
```

### CMake can't find TensorRT headers (NvInfer.h)

**If you installed via pip (non-sudo):**

1. Download TensorRT tar.gz from NVIDIA
2. Extract and copy headers:
```bash
tar -xzf TensorRT-*.tar.gz
mkdir -p trt_detector/external/tensorrt/include
cp -r TensorRT-*/include/* trt_detector/external/tensorrt/include/
```

**If you have sudo access:**
```bash
# Copy headers to system location
sudo mkdir -p /usr/local/include/tensorrt
sudo cp -r /path/to/TensorRT-*/include/* /usr/local/include/tensorrt/
```

### CMake can't find TensorRT libraries (libnvinfer.so)

The pip-installed TensorRT libraries have version suffixes. Create symlinks:

```bash
# Find your tensorrt_libs directory
python -c "import tensorrt_libs; import os; print(os.path.dirname(tensorrt_libs.__file__))"

# Navigate there and create symlinks
cd /path/to/tensorrt_libs  # use the path from above
ln -sf libnvinfer.so.10 libnvinfer.so
ln -sf libnvinfer_plugin.so.10 libnvinfer_plugin.so
ln -sf libnvonnxparser.so.10 libnvonnxparser.so
```

### CMake can't find OpenCV

Install OpenCV development libraries:
```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

Verify installation:
```bash
pkg-config --modversion opencv4
```

### CMake can't find pybind11

Install pybind11:
```bash
sudo apt install pybind11-dev
# or
pip install pybind11
```

### CUDA architecture mismatch

If you see errors about unsupported compute capability, edit `trt_detector/CMakeLists.txt`:

```cmake
# Find this line (around line 9):
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# Remove architectures not supported by your CUDA version
# For CUDA 11.5, remove 89:
set(CMAKE_CUDA_ARCHITECTURES 75 86)
```

Common compute capabilities:
- RTX 20 series (Turing): 75
- RTX 30 series (Ampere): 86
- RTX 40 series (Ada Lovelace): 89 (requires CUDA 11.8+)

### Low FPS

- Use FP16 engine instead of FP32
- Check GPU usage with `nvidia-smi`
- Ensure you're using the discrete GPU (not integrated graphics)

### CUDA out of memory

- Close other GPU applications
- Use a smaller model
- Reduce input video resolution

---

## Docker Details

### Prerequisites

1. **Docker**: https://docs.docker.com/get-docker/
2. **NVIDIA Container Toolkit**:
```bash
# Add NVIDIA repo
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Docker Commands

```bash
# Build the Docker image
docker compose build

# Convert model to TensorRT engine
docker compose run detector ./scripts/build_model.sh models/your_model.pt fp16

# Run inference
docker compose run detector

# Run with custom command
docker compose run detector python3 yolo.py

# Interactive shell
docker compose run detector bash
```

### Display Issues (GUI)

If the video window doesn't appear:
```bash
# Allow Docker to access display
xhost +local:docker

# Then run
docker compose run detector
```
