# RTSP Module

A high-performance RTSP streaming and processing module leveraging GPU acceleration for real-time video analytics.

## Documentation

### Core Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System design, components, data flow, and design rationale |
| [API Reference](api_reference.md) | Complete Python API documentation with usage patterns |
| [Docker](docker.md) | Docker build options and deployment strategies |
| [Dependencies](dependencies.md) | System and Python dependencies |

## Project Structure

```text
RTSPModule/
├── src/                    # C++ source files
│   ├── bindings/           # Python bindings (pybind11)
│   ├── core/               # Core C++ implementation
│   └── rtspmodule/         # Compiled module output
├── examples/               # Example Python scripts / C++
├── docs/                   # Detailed documentation
├── include/                # C++ headers
├── tests/                  # Integration and Unit tests
├── configs/                # Configuration files
├── docker/                 # Docker build files
└── CMakeLists.txt          # Build configuration
```

## Prerequisites

### Hardware
- **NVIDIA GPU** 

### Operating System
- **Linux** (Ubuntu 22.04 recommended) or **WSL2** on Windows

### Drivers
- **NVIDIA Drivers** supporting CUDA 12.X

### Software
- **Docker**
- **NVIDIA Container Toolkit**

## Getting Started

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/OCTeamAI/RTSPModule.git
cd RTSPModule
```

## Building with Docker

### Build the Image

```bash
docker build -t rtsp-module .
```

### Configuration

Ensure `configs/config.yaml` is present.

```yaml
settings:
  buffer_size: 3                   # GPU Frame buffer queue depth (default: 5 frames)
  retry_max_attempts: 0            # Max reconnection attempts per stream (0 = unlimited)
  backoff_multiplier: 1.5          # Delay multiplier on failed reconnects (1.0 = no backoff)
  gpu_id: 0                        # GPU device ID for hardware decoding
  log_base_path: ./logs            # Log base path
  
  # CPU Buffer settings
  cpu_buffer_enabled: true         # Enable CPU RAM ring buffer (replaces GPU queue when true)
  cpu_buffer_duration_sec: 2.0     # Seconds of video to buffer (default: 2.0)
  
  # Output settings
  output_format: NV12              # NV12 (default), RGB, BGR, RGBA, I420

  # Decoder preference: auto (default), nvv4l2, nvdec, cpu
  decoder_preference: auto

streams:
  # RTSP streams
  - name: "Camera 1"
    url: rtsp://user:pass@192.168.1.10:554/stream
  - name: "Camera 2"
    url: rtsp://user:pass@192.168.1.11:554/stream
    
  # MP4 files with FPS-capped decoding
  - name: "Demo Video"
    file: /path/to/demo.mp4
    loop: true
    fps: 25
```

| Setting | Default | Description |
|---------|---------|-------------|
| `buffer_size` | 3 | GPU Frame buffer queue depth (prevent overwrites) |
| `retry_max_attempts` | 0 | Max reconnection attempts (0 = unlimited) |
| `backoff_multiplier` | 1.5 | Reconnect delay multiplier |
| `gpu_id` | 0 | GPU device ID |
| `cpu_buffer_enabled` | true | Enable CPU Ring Buffer (replaces GPU queue) |
| `cpu_buffer_duration_sec`| 2.0 | Seconds of history to keep in CPU RAM |
| `output_format` | NV12 | Output format (NV12, RGB, BGR, etc.) |
| `decoder_preference` | auto | Decoder priority (nvv4l2 > nvdec > cpu) |

### Run the Container

```bash

docker run --rm -it \
  --net=host \
  -p 8080:8080 \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/configs/config.yaml:/app/config.yaml \
  -v $(pwd)/output:/app/output \
  rtsp-module /bin/bash
```
### Export the current directory to PYTHONPATH

```bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Usage Scripts

### 1. Record Streams (`recorder.py`)

Records all streams to an MP4 file in the `./output` directory.

```bash
python3 test/recorder.py
```

### 2. Stability Benchmark (`client.py`)

Runs the pipeline without visualization to test memory stability and calculate raw FPS throughput.

```bash
python3 test/client.py
```

### 3. Frame Sync Test (`test_frame_sync.py`)

Tests the frame synchronization mechanism across multiple streams.

```bash
python3 test/test_frame_sync.py
```

### 4. High-Performance Web Viewer (`05_web_viewer.py`)

A highly optimized web viewer that uses per-stream acquisition threads and per-client queues to broadcast streams at native FPS.

**Features:**
- Native FPS broadcasting
- Low-latency WebSocket streaming
- GPU/CPU decoder status indicators
- Parallel thread architecture

```bash
python3 examples/05_web_viewer.py
```
Open `http://localhost:8080` to view the dashboard.

## Architecture Overview

The module uses a **zero-copy GPU pipeline** where decoded frames remain in GPU memory throughout:

```
rtspsrc → rtph26Xdepay → h26Xparse → nvh26Xdec → cudaconvert → appsink → Python (CuPy)
```

**Key Features:**
- Shared CUDA context across all streams (saves ~250 MB per stream)
- Automatic H.264/H.265 codec detection
- Configurable frame queue to prevent overwrites
- **CPU Ring Buffer** for thread-safe, efficient frame access
- **Native Batch Support** for high-performance multi-stream inference
- Automatic reconnection for failed streams
- GIL-released bindings for true Python concurrency

**Frame Synchronization:**
- Implements a "Soft Barrier" that waits up to 40ms for all cameras
- Returns `None` for lagging streams (prevents pipeline freezing)

📖 **For detailed architecture diagrams and component documentation, see [architecture.md](architecture.md)**

