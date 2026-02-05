# System Dependencies

## Ubuntu 22.04 / 24.04 LTS

To build from source, install the following development packages:

```bash
sudo apt update && sudo apt install -y \
    cmake \
    git \
    python3-dev \
    python3-pip \
    libopencv-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    libyaml-cpp-dev
```

> [!IMPORTANT]
> **NVIDIA CUDA Toolkit (12.x)** is required for GPU acceleration.

## Python Dependencies

Install the required Python packages. We recommend using a virtual environment.

```bash
# Core dependencies
pip install numpy opencv-python pyyaml

# GPU Acceleration (select matching CUDA version)
pip install cupy-cuda12x  # For CUDA 12.x
# pip install cupy-cuda11x  # For CUDA 11.x
```
