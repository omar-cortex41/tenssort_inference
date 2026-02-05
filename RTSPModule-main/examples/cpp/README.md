# RTSPCore C++ Example

This directory contains a standalone C++ example demonstrating how to use the `libRTSPCore.a` static library directly without Python bindings.

## Prerequisites

1. **Build the main RTSPModule project first** to generate `libRTSPCore.a`:

   ```bash
   # From the repository root
   cmake -B build -S .
   cmake --build build
   ```

2. **System dependencies** (same as main project):
   - GStreamer 1.0 with video, app, and sdp plugins
   - CUDA Toolkit
   - yaml-cpp

## Building the Example

```bash
cd examples/cpp

# Create build directory
cmake -B build -S .

# Build
cmake --build build
```

### Custom Library Path

If your `libRTSPCore.a` is in a different location:

```bash
cmake -B build -S . \
    -DRTSPCORE_LIB_PATH=/path/to/libRTSPCore.a \
    -DRTSPCORE_INCLUDE_DIR=/path/to/include
```

## Running

```bash
# Use the default config
./build/rtsp_example ../../configs/config.yaml

# Or provide your own config
./build/rtsp_example /path/to/your/config.yaml
```

## Example Output

```
=================================================
       RTSPCore C++ Example Application
=================================================

[INFO] Loading configuration: ../../configs/config.yaml
[INFO] Configured 4 stream(s)
[INFO] CPU buffer mode: ENABLED
[INFO] GPU available: YES

[INFO] Starting streams...
[INFO] Waiting for streams to stabilize...

=== Example 1: Single Frame Retrieval ===
  [Camera 0] Frame #42 | 1920x1080 | Format: NV12 | Size: 3110400 bytes
  [Camera 1] Frame #38 | 1920x1080 | Format: NV12 | Size: 3110400 bytes
  ...

=== Example 2: Batch Frame Retrieval ===
  Batch size: 4 | Valid: 4 | Resolution: 1920x1080 | Format: NV12
    [0] Camera 0 Frame #43 -> OK
    [1] Camera 1 Frame #39 -> OK
    ...

=== Example 3: Monitoring Loop (5s) ===
  [Camera 0] FPS: 30.0 | Frames: 150 | Drops: 0 | Queue: 3/5
  ...
```

## API Reference

### RtspClient

| Method | Description |
|--------|-------------|
| `loadConfig(path)` | Load YAML configuration file |
| `start()` | Start all streams |
| `stop()` | Stop all streams |
| `getStreamCount()` | Number of configured streams |
| `isRunning()` | Check if client is active |
| `isCpuBufferEnabled()` | Check if CPU buffer mode is active |
| `isGpuAvailable()` | Check if GPU acceleration is available |
| `getGpuFrame(cam, timeout)` | Get GPU frame (returns device pointer) |
| `getCpuFrame(cam, timeout)` | Get single frame from camera |
| `getBatchedFrames(config)` | Get frames from multiple cameras |
| `getStats(cam)` | Get stream statistics |
| `getCpuBufferInfo(cam)` | Get CPU buffer status |

### Configuration File

See `../../configs/config.yaml` for all available options including:
- `buffer_size`: Frame queue depth
- `cpu_buffer_enabled`: Force CPU buffer mode
- `output_format`: NV12, RGB, BGR, etc.
- `decoder_preference`: auto, nvdec, cpu
