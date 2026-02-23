
# RTSPModule Examples

This directory contains example scripts demonstrating how to use the `RTSPModule` Python bindings.

## Prerequisites

- Python 3.10+
- The compiled `RTSPModule.so` library in the `../lib` directory.
- `numpy`
- `opencv-python` (for visualization in some examples)
- `cupy` (optional, for GPU examples)

### Web Viewer Dependencies (05_web_viewer.py)

The high-performance web viewer requires an async web server and a fast JPEG encoder.

```bash
pip3 install aiohttp simplejpeg
```

**Note:** `simplejpeg` provides significant performance benefits over OpenCV's encoder but may require system-level `libjpeg-turbo` development headers (e.g., `sudo apt install libjpeg-turbo8-dev` on Ubuntu) for successful compilation if a wheel is not available.

### WebRTC Viewer Dependencies (06_webrtc_viewer.py)

The WebRTC viewer requires an async web server and the GStreamer Rust WebRTC plugin.

```bash
pip3 install aiohttp
```

**Note:** You must have the `webrtcrs` GStreamer plugin installed and optionally set `GST_PLUGIN_PATH` (e.g., `export GST_PLUGIN_PATH=/path/to/gst-webrtcrs/target/release`).

## Usage

Run the scripts from this directory. 

```bash
cd examples
python3 01_basics.py
```

## Available Examples

| File | Description | Functions Covered |
|------|-------------|-------------------|
| `01_basics.py` | Basic initialization, starting/stopping streams, and monitoring stats. | `start`, `stop`, `is_running`, `stream_count`, `set_log_path`, `get_stats` |
| `02_cpu_capture.py` | Retrieving frames from the CPU ring buffer. | `get_cpu_frame`, `get_cpu_buffer_info`, `is_cpu_buffer_enabled` |
| `03_gpu_capture.py` | Zero-copy frame retrieval on GPU (requires CUDA). | `get_cuda_frame`, `is_gpu_available` |
| `04_batch_capture.py` | High-performance batch frame retrieval for inference. | `get_batch` |
| `05_web_viewer.py` | Ultra-High Performance WebSocket Web Viewer. | `start`, `stop`, `stream_count`, `get_cpu_frame`, `is_gpu_available`, `get_stats` |
| `06_webrtc_viewer.py` | WebRTC Viewer serving a dashboard that streams all cameras via native browser WebRTC. | `start_streaming`, `stop_streaming`, `start_streaming_all`, `stop_streaming_all`, `is_webrtc_streaming` |

## Configuration

The examples use `../configs/config.yaml` by default. Ensure this file exists and contains valid RTSP stream URLs.
