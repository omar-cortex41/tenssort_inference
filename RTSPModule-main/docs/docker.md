# RTSPModule Docker Guide

This guide details how to build and run the RTSPModule using various Docker strategies. Choose the one that best fits your deployment needs.

## Build Options

### Option 1: Standalone CUDA (Recommended)
**Best for:** Most users requiring a clean, high-performance environment with the latest CUDA (12.8) and GStreamer (1.24), without the overhead of PyTorch.

This image compiles GStreamer from source and installs the module in a lean runtime environment.

```bash
docker build -f docker/Dockerfile -t rtsp_module_cuda .
```

### Option 2: PyTorch & Runtime Artifacts
**Best for:** Deep Learning integration where PyTorch is required.

#### Standard Build
Builds the module on top of the generic PyTorch 2.6 devel/runtime images.

```bash
docker build -f docker/Dockerfile.pytorch -t rtsp_module_pytorch .
```

#### Optimized Runtime Build (Export Flow)
For production deployment, you can extract the compiled GStreamer artifacts and wheel to create a minimal runtime image without build dependencies.

1.  **Export Artifacts**:
    ```bash
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.pytorch --target export --output type=local,dest=./dist_artifacts .
    ```

2.  **Build Runtime Image**:
    ```bash
    cd dist_artifacts && docker build -f ../docker/Dockerfile.runtime -t rtsp_module_runtime .
    ```

### Option 3: NVIDIA DeepStream
**Best for:** Integration with existing DeepStream 7.1 pipelines.

This uses the official `nvcr.io/nvidia/deepstream:7.1-triton-multiarch` base image.

```bash
docker build -f docker/Dockerfile.deepstream -t rtsp_module_deepstream .
```

> **Note:** This build process automatically runs the `user_additional_install.sh` script provided by DeepStream to install necessary software decoders.

### Option 4: System GStreamer (APT)
**Best for:** Quick builds using system-provided packages (Ubuntu 22.04 default GStreamer).

**Warning:** This may use an older GStreamer version (1.20) compared to the source-built versions (1.24+), which might lack some features or fixes.

```bash
docker build -f docker/Dockerfile.apt -t rtsp_module_apt .
```

### Option 5: PyTorch & DeepStream
**Best for:** Deep learning pipelines requiring both PyTorch and standard DeepStream 7.1 features.

Combines the PyTorch 2.6 runtime with strict DeepStream 7.1 installation.

```bash
docker build -f docker/Dockerfile.pytorch_deepstream -t rtsp_module_pytorch_deepstream .
```

#### Export Build
Extracts the compiled wheels for **Python 3.9, 3.10, 3.11, and 3.12** for local usage.

```bash
DOCKER_BUILDKIT=1 docker build --target export --output type=local,dest=./wheels/deepstream . -f docker/Dockerfile.pytorch_deepstream
```

---

## Run the Container

Regardless of which image you built, the run command is similar. The following command enables GPU support, host networking (critical for low-latency RTSP), and GUI display.

Replace `IMAGE_NAME` with your built tag (e.g., `rtsp_module_cuda`, `rtsp_module_pytorch`, etc.).

```bash
docker run --rm -it \
  --net=host \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/configs/config.yaml:/workspace/config.yaml \
  -v $(pwd)/output:/workspace/output \
  IMAGE_NAME /bin/bash
```

### Explanation of Flags

-   `--net=host`: Uses the host's network stack. This is **required** for low-latency RTSP streaming and proper functionality of some network protocols, including WebRTC signaling.
-   `--gpus all`: Exposes all NVIDIA GPUs to the container.
-   `-e DISPLAY=$DISPLAY` & `-v /tmp/.X11-unix...`: Enables GUI applications (like OpenCV `imshow` or GStreamer `autovideosink`) to render on the host display.
-   `-v .../config.yaml`: Mounts your local configuration file into the workspace.
-   `-v .../output`: Mounts the output directory for recordings or logs.

---

### WebRTC Requirements

If you have enabled WebRTC streaming in your configuration (`webrtc_enabled: true`), using `--net=host` is the easiest way to ensure the WebRTC signaling server (default port `9000`) and the dynamically allocated ICE UDP ports are reachable from the host.

If you cannot use `--net=host`, you must explicitly map the signaling port and the required UDP port range for ICE negotiation. However, due to the dynamic nature of WebRTC UDP ports, `--net=host` is strongly recommended for Docker deployments.
