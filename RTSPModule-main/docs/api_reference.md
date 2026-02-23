# RTSPModule Python API Reference

Complete API reference for the `rtspmodule` Python package.

---

## Table of Contents

- [RTSPModule Class](#rtspmodule-class)
  - [Initialization](#initialization)
  - [Stream Management](#stream-management)
  - [Frame Retrieval](#frame-retrieval)
  - [Statistics & Monitoring](#statistics--monitoring)
  - [Configuration](#configuration)
- [Data Structures](#data-structures)
  - [Frame Dictionary (GPU)](#frame-dictionary-gpu)
  - [Frame Dictionary (CPU)](#frame-dictionary-cpu)
  - [Batch Dictionary](#batch-dictionary)
  - [Statistics Dictionary](#statistics-dictionary)
  - [Buffer Info Dictionary](#buffer-info-dictionary)
- [Configuration File Schema](#configuration-file-schema)
- [Error Handling](#error-handling)
- [Usage Patterns](#usage-patterns)

---

## RTSPModule Class

Main class for managing multiple RTSP streams with hardware-accelerated decoding.

```python
from rtspmodule import RTSPModule

provider = RTSPModule()
```

### Initialization

#### `__init__()`

Create a new RTSPModule instance.

```python
provider = RTSPModule()
```

**Behavior:**
- Initializes GStreamer if not already initialized
- Sets up internal state (no streams loaded yet)
- Safe to create multiple instances (each manages independent stream sets)

---

### Stream Management

#### `start(config_file: str) -> None`

Start all RTSP streams defined in the configuration file.

**Parameters:**
- `config_file` (**str**): Absolute or relative path to YAML configuration file

**Raises:**
- `RuntimeError`: If configuration file cannot be loaded or parsed
- `RuntimeError`: If stream initialization fails (e.g., CUDA context creation)

**Example:**
```python
provider.start("config.yaml")
# Streams are now connecting in background threads
```

**Behavior:**
- Parses YAML configuration file
- Initializes CUDA context (if GPU mode enabled)
- Creates GStreamer pipeline for each stream
- Starts background threads for each stream
- Returns immediately (streams connect asynchronously)

---

#### `stop() -> None`

Stop all streams and release resources.

```python
provider.stop()
```

**Behavior:**
- Gracefully stops all GStreamer pipelines
- Joins all background threads
- Releases CUDA context
- Frees all GPU/CPU buffers
- Safe to call multiple times

**Best Practice:**
```python
try:
    provider.start("config.yaml")
    # ... use streams ...
finally:
    provider.stop()  # Always cleanup
```

---

#### `is_running() -> bool`

Check if streams are currently running.

```python
if provider.is_running():
    frame = provider.get_cuda_frame(0)
```

**Returns:**
- **bool**: `True` if streams are active, `False` otherwise

---

#### `stream_count() -> int`

Get the number of configured streams.

```python
count = provider.stream_count()
for cam_id in range(count):
    stats = provider.get_stats(cam_id)
```

**Returns:**
- **int**: Number of streams loaded from configuration file

---

### Frame Retrieval

#### `get_cuda_frame(camera_id: int, timeout_ms: int = 0) -> dict`

Get frame as CUDA device pointer (zero-copy GPU access).

**Requirements:**
- `cpu_buffer_enabled` must be `false` in config
- GPU hardware must be available

**Parameters:**
- `camera_id` (**int**): Camera index (0 to `stream_count() - 1`)
- `timeout_ms` (**int**, optional): Maximum wait time in milliseconds
  - `0` = non-blocking (default)
  - `> 0` = block until frame available or timeout

**Returns:**
- **dict**: Frame information (see [Frame Dictionary (GPU)](#frame-dictionary-gpu))

**Raises:**
- `RuntimeError`: If CPU buffer mode is enabled

**Example:**
```python
import cupy as cp

# Non-blocking: get latest frame immediately
frame = provider.get_cuda_frame(camera_id=0)

if frame['valid']:
    # Zero-copy: wrap CUDA pointer as CuPy array
    gpu_img = cp.ndarray(
        shape=frame['shape'],
        dtype=cp.uint8,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(frame['ptr'], frame['size'], None),
            0
        )
    )
    print(f"Frame {frame['frame_id']}: {frame['width']}x{frame['height']}")
```

**Performance:**
- No CPU-GPU transfers (frame stays on GPU)
- Minimal latency (~0.1ms overhead)
- Ideal for PyTorch/TensorRT inference

---

#### `get_cpu_frame(camera_id: int, timeout_ms: int = 0) -> dict`

Get next frame from CPU ring buffer.

**Requirements:**
- `cpu_buffer_enabled` must be `true` in config (or GPU unavailable)

**Parameters:**
- `camera_id` (**int**): Camera index
- `timeout_ms` (**int**, optional): Maximum wait time in milliseconds
  - `0` = non-blocking (default)
  - `> 0` = block until frame available or timeout

**Returns:**
- **dict**: Frame data (see [Frame Dictionary (CPU)](#frame-dictionary-cpu))

**Raises:**
- `RuntimeError`: If CPU buffer mode is disabled

**Example:**
```python
import cv2

# Blocking: wait up to 100ms for next frame
frame = provider.get_cpu_frame(camera_id=0, timeout_ms=100)

if frame['valid']:
    # Data is already a NumPy array
    img = frame['data']  # Shape: (H, W, 3) for BGR
    cv2.imshow(f"Camera {camera_id}", img)
```

---

#### `get_batch(camera_ids: List[int], timeout_ms: int = 10) -> dict`

Get frames from multiple cameras in a single batched call.

**Requirements:**
- `cpu_buffer_enabled` must be `true` in config

**Parameters:**
- `camera_ids` (**List[int]**): List of camera indices to retrieve
- `timeout_ms` (**int**, optional): Max wait time per frame (default: 10ms)

**Returns:**
- **dict**: Batch data (see [Batch Dictionary](#batch-dictionary))

**Behavior:**
- Pre-allocates output buffer (zero-copy from C++)
- Retrieves all frames in parallel using thread pool
- Offline/unavailable cameras return black (zeroed) frames
- Returns fixed-size batch regardless of stream availability

**Example:**
```python
# Get frames from cameras 0-3 in single call
batch = provider.get_batch([0, 1, 2, 3], timeout_ms=5)

frames = batch['data']       # Shape: (4, H, W, 3)
valid = batch['valid_mask']  # [True, True, False, True]

# Process only valid frames
for i in range(batch['count']):
    if valid[i]:
        img = frames[i]
        metadata = batch['metadata'][i]
        print(f"Cam {i}: Frame {metadata['frame_id']}")
```

**Performance:**
- **~2-3ms** for 4 streams @ 1080p (includes memcpy)
- Parallel copy using dedicated thread pool
- Ideal for batched AI inference

---

#### `get_multi_frames(camera_id: int, num_frames: int, timeout_ms: int = 10) -> dict`

Get multiple consecutive frames from a single camera's CPU ring buffer.

**Requirements:**
- `cpu_buffer_enabled` must be `true` in config

**Parameters:**
- `camera_id` (**int**): Camera index
- `num_frames` (**int**): Maximum number of frames to retrieve
- `timeout_ms` (**int**, optional): Max wait time for *first* frame (default: 10ms)

**Returns:**
- **dict**: Batch data covering multiple time steps (see [Batch Dictionary](#batch-dictionary))

**Behavior:**
- Pops up to `num_frames` from the ring buffer in FIFO order (oldest first)
- Frames are **consumed** (removed from buffer)
- Returns contiguous memory block `(N, H, W, C)`
- If buffer has fewer frames than requested, returns what is available

**Example:**
```python
# Get 8 frames from Camera 0
result = provider.get_multi_frames(camera_id=0, num_frames=8)

if result['count'] > 0:
    # Shape: (8, H, W, 3) for BGR
    frames = result['data']
    
    # Process temporal batch
    print(f"Retrieved {result['count']} frames")
```

**Performance:**
- Zero-copy from ring buffer to batch array (single memcpy per frame)
- Ideal for sliding window inference (e.g., action recognition)

---

### Statistics & Monitoring

#### `get_stats(camera_id: int) -> dict`

Get stream statistics for a camera.

**Parameters:**
- `camera_id` (**int**): Camera index

**Returns:**
- **dict**: Statistics dictionary (see [Statistics Dictionary](#statistics-dictionary))

**Example:**
```python
stats = provider.get_stats(0)

print(f"FPS: {stats['current_fps']:.1f}")
print(f"Resolution: {stats['source_width']}x{stats['source_height']}")
print(f"Frames consumed: {stats['frames_consumed']}")
print(f"Reconnections: {stats['reconnect_count']}")

if stats['frames_dropped_queue'] > 0:
    print("Warning: Consumer too slow!")
```

---

#### `get_cpu_buffer_info(camera_id: int) -> dict`

Get CPU ring buffer statistics.

**Requirements:**
- `cpu_buffer_enabled` must be `true` in config

**Parameters:**
- `camera_id` (**int**): Camera index

**Returns:**
- **dict**: Buffer statistics (see [Buffer Info Dictionary](#buffer-info-dictionary))

**Raises:**
- `RuntimeError`: If CPU buffer mode is disabled

**Example:**
```python
info = provider.get_cpu_buffer_info(0)

print(f"Buffer: {info['buffer_count']}/{info['buffer_capacity']} frames")
print(f"Duration: {info['buffer_duration_sec']:.1f}s")
print(f"Memory: {info['memory_usage_mb']:.1f} MB")
```

---

### WebRTC Streaming

#### `start_streaming(camera_id: int) -> bool`

Start WebRTC streaming for a specific camera. Safe to call at any time after `start()`. If the GStreamer pipeline is not yet ready, streaming will auto-start when the pipeline comes up.

**Parameters:**
- `camera_id` (**int**): Camera index (0 to stream_count-1).

**Returns:**
- **bool**: `True` on success or already-queued, `False` on error.

**Example:**
```python
success = provider.start_streaming(0)
```

---

#### `stop_streaming(camera_id: int) -> None`

Stop WebRTC streaming for a specific camera. Safely detaches the WebRTC branch from the live pipeline. The main decode pipeline continues uninterrupted.

**Parameters:**
- `camera_id` (**int**): Camera index (0 to stream_count-1).

---

#### `start_streaming_all() -> None`

Start WebRTC streaming for ALL cameras simultaneously.

---

#### `stop_streaming_all() -> None`

Stop WebRTC streaming for ALL cameras simultaneously.

---

#### `is_webrtc_streaming(camera_id: int) -> bool`

Check if WebRTC streaming is currently active for a camera.

**Parameters:**
- `camera_id` (**int**): Camera index (0 to stream_count-1).

**Returns:**
- **bool**: `True` if the WebRTC branch is live.

---

### WebRTC Signaling API (HTTP)

When WebRTC is enabled, a single Rust-powered HTTP signaling server is started (default port `9000`). All `webrtcrs_sink` elements share this single server. The architecture uses `stream_id` to route SDP exchange to the correct GStreamer pipeline.

The `stream_id` corresponds to the numeric camera index from the Python API (e.g., `"0"`, `"1"`), unless overridden.

#### `GET /health`
Check if the signaling server is running.
- **Returns:** HTTP 200 OK if successful.

#### `GET /streams`
List all currently active WebRTC streams registered with the server.
- **Returns JSON:**
  ```json
  [
    { "stream_id": "0" },
    { "stream_id": "1" }
  ]
  ```

#### `POST /webrtc/offer?stream_id={id}`
Send a WebRTC SDP offer from a browser/client to the `webrtcrs_sink`.
- **Query Parameters:** 
  - `stream_id` (string): The stream identifier (e.g., `"0"` for camera 0)
- **Request JSON:**
  ```json
  {
    "sdp": "v=0\r\no=- 46... (WebRTC offer SDP)",
    "type": "offer"
  }
  ```
- **Returns JSON:**
  ```json
  {
    "sdp": "v=0\r\no=- 52... (WebRTC answer SDP)",
    "type": "answer",
    "peer_id": "a1b2c3d4..."
  }
  ```
*Note: Save the `peer_id` if you wish to gracefully disconnect later without tearing down the browser connection.*

#### `POST /webrtc/disconnect`
Gracefully close a specific peer's connection.
- **Request JSON:**
  ```json
  {
    "peer_id": "a1b2c3d4...",
    "stream_id": "0"
  }
  ```
- **Returns:** HTTP 200 OK.

---

### Configuration

#### `set_log_path(base_path: str) -> None`

Set the base directory for camera logs.

**Parameters:**
- `base_path` (**str**): Directory path for log files

**Behavior:**
- Creates date-based subdirectories: `{base_path}/YYYY-MM-DD/`
- Each camera creates a log file: `camera_{name}.log`
- Logs include connection events, errors, FPS changes

**Example:**
```python
provider.set_log_path("./logs")
provider.start("config.yaml")
# Logs written to: ./logs/2026-02-05/camera_Front.log
```

---

#### `is_cpu_buffer_enabled() -> bool`

Check if CPU buffer mode is enabled.

```python
if provider.is_cpu_buffer_enabled():
    frame = provider.get_cpu_frame(0)
else:
    frame = provider.get_cuda_frame(0)
```

**Returns:**
- **bool**: `True` if CPU buffer mode active (config or GPU unavailable)

---

#### `is_gpu_available() -> bool`

Check if GPU hardware (NVDEC/cudaconvert) is available.

```python
if not provider.is_gpu_available():
    print("Warning: GPU decoding unavailable, using CPU fallback")
```

**Returns:**
- **bool**: `True` if GPU hardware initialized successfully

**Notes:**
- If `False`, CPU buffer mode is automatically enabled regardless of config

---

## Data Structures

### Frame Dictionary (GPU)

Returned by [`get_cuda_frame()`](#get_cuda_framecamera_id-int-timeout_ms-int--0---dict)

| Key | Type | Description |
|-----|------|-------------|
| `valid` | **bool** | `True` if frame available, `False` if timeout/error |
| `ptr` | **int** | CUDA device pointer (use with CuPy/PyTorch) |
| `width` | **int** | Frame width in pixels |
| `height` | **int** | Frame height in pixels |
| `stride` | **int** | Row stride in bytes (may differ from `width`) |
| `shape` | **tuple** | NumPy-compatible shape based on format |
| `size` | **int** | Total buffer size in bytes |
| `frame_id` | **int** | Sequential frame counter (increments per frame) |
| `format` | **str** | Pixel format: `"NV12"`, `"RGB"`, `"BGR"`, `"RGBA"`, `"I420"` |
| `dtype` | **str** | Data type (always `"uint8"`) |

**Shape Examples:**
- **NV12**: `(H*1.5, W)` - Y plane + UV plane
- **RGB/BGR**: `(H, W, 3)`
- **RGBA/BGRA**: `(H, W, 4)`

---

### Frame Dictionary (CPU)

Returned by [`get_cpu_frame()`](#get_cpu_framecamera_id-int-timeout_ms-int--0---dict)

| Key | Type | Description |
|-----|------|-------------|
| `valid` | **bool** | `True` if frame available |
| `data` | **numpy.ndarray** | Pixel data with format-appropriate shape |
| `width` | **int** | Frame width in pixels |
| `height` | **int** | Frame height in pixels |
| `format` | **str** | Pixel format: `"NV12"`, `"RGB"`, `"BGR"`, etc. |
| `frame_id` | **int** | Sequential frame counter |
| `timestamp_ns` | **int** | Presentation timestamp in nanoseconds |
| `data_size` | **int** | Actual data size in bytes (for debugging) |

---

### Batch Dictionary

Returned by [`get_batch()`](#get_batchcamera_ids-listint-timeout_ms-int--10---dict)

| Key | Type | Description |
|-----|------|-------------|
| `data` | **numpy.ndarray** | Contiguous array shape `(N, H, W, C)` or `(N, H*1.5, W)` |
| `valid_mask` | **numpy.ndarray[bool]** | Boolean mask indicating valid frames |
| `metadata` | **List[dict]** | Per-frame metadata (see below) |
| `count` | **int** | Total batch size (matches `len(camera_ids)`) |
| `valid_count` | **int** | Number of valid frames in batch |
| `width` | **int** | Common frame width |
| `height` | **int** | Common frame height |
| `format` | **str** | Pixel format |

**Metadata Item:**
```python
{
    'camera_id': int,      # Camera index
    'frame_id': int,       # Frame counter
    'timestamp_ns': int,   # PTS in nanoseconds
    'width': int,          # Frame width
    'height': int,         # Frame height
    'valid': bool          # True if frame available
}
```

---

### Statistics Dictionary

Returned by [`get_stats()`](#get_statscamera_id-int---dict)

#### Frame Counters

| Key | Type | Description |
|-----|------|-------------|
| `frames_received` | **int** | Frames received by appsink (after GStreamer drops) |
| `frames_decoded` | **int** | Frames successfully decoded to GPU/CPU |
| `frames_consumed` | **int** | Frames fetched by application |
| `frames_dropped_decode` | **int** | Frames dropped during decode/copy |
| `frames_dropped_queue` | **int** | Frames dropped because queue was full |
| `frames_overwritten` | **int** | Frames overwritten before consumption |
| `frames_duplicate` | **int** | Duplicate frames (same PTS) skipped |
| `decode_errors` | **int** | Decode/mapping errors encountered |
| `reconnect_count` | **int** | Number of stream reconnections |

#### FPS Metrics

| Key | Type | Description |
|-----|------|-------------|
| `current_fps` | **float** | Sliding window FPS (1-second window) |
| `instant_fps` | **float** | Instantaneous FPS (1 / last frame interval) |
| `source_fps` | **float** | FPS parsed from stream headers |

#### Stream Info

| Key | Type | Description |
|-----|------|-------------|
| `source_width` | **int** | Video width from stream headers |
| `source_height` | **int** | Video height from stream headers |

#### Queue Status

| Key | Type | Description |
|-----|------|-------------|
| `queue_depth` | **int** | Current queue depth |
| `queue_max_depth` | **int** | Maximum queue depth seen |

#### Derived Metrics (%)

| Key | Type | Description |
|-----|------|-------------|
| `decode_success_rate` | **float** | Percentage of successful decodes |
| `consumption_rate` | **float** | Percentage of decoded frames consumed |
| `overwrite_rate` | **float** | Percentage of frames overwritten |
| `queue_drop_rate` | **float** | Percentage of frames dropped due to full queue |

---

### Buffer Info Dictionary

Returned by [`get_cpu_buffer_info()`](#get_cpu_buffer_infocamera_id-int---dict)

| Key | Type | Description |
|-----|------|-------------|
| `buffer_count` | **int** | Current frames in buffer |
| `buffer_capacity` | **int** | Maximum buffer capacity (in frames) |
| `buffer_duration_sec` | **float** | Time span of buffered frames |
| `memory_usage_bytes` | **int** | RAM used by buffer |
| `memory_usage_mb` | **float** | RAM used in megabytes |
| `format` | **str** | Pixel format of buffered frames |

---

## Configuration File Schema

YAML configuration file structure:

```yaml
settings:
  # Frame buffer configuration
  buffer_size: 3                   # GPU queue depth (frames)
  
  # Reconnection strategy
  retry_max_attempts: 0            # Max reconnect attempts (0 = unlimited)
  backoff_multiplier: 1.5          # Delay multiplier on failures
  
  # Hardware
  gpu_id: 0                        # CUDA device ID
  
  # CPU Buffer (temporal frame access)
  cpu_buffer_enabled: true         # Enable CPU ring buffer
  cpu_buffer_duration_sec: 2.0     # Seconds of video to buffer
  
  # Output format
  output_format: NV12              # NV12, RGB, BGR, RGBA, I420
  
  # Decoder selection
  decoder_preference: auto         # auto, nvv4l2, nvdec, cpu
  
  # WebRTC streaming
  webrtc_enabled: true             # Auto-start WebRTC streaming
  webrtc_base_port: 9000           # Base port for signaling server
  
  # Logging
  log_base_path: ./logs            # Base directory for logs

streams:
  # RTSP streams
  - name: "Front Camera"
    url: rtsp://user:pass@192.168.1.10:554/stream
  - name: "Rear Camera"
    url: rtsp://user:pass@192.168.1.11:554/stream
  
  # MP4 files with FPS-capped decoding
  - name: "Demo Video"
    file: /path/to/demo.mp4
    loop: true
    fps: 25
  - name: "Test Clip"
    file: /path/to/test.mp4
```

### Settings Reference

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `buffer_size` | int | 3 | GPU frame queue depth (prevent overwrites) |
| `retry_max_attempts` | int | 0 | Max reconnection attempts (0 = unlimited) |
| `backoff_multiplier` | float | 1.5 | Reconnect delay multiplier |
| `gpu_id` | int | 0 | CUDA device ID for hardware decoding |
| `cpu_buffer_enabled` | bool | false | Enable CPU RAM ring buffer |
| `cpu_buffer_duration_sec` | float | 2.0 | Seconds of video history to keep |
| `output_format` | str | "NV12" | Output pixel format |
| `decoder_preference` | str | "auto" | Decoder priority mode |
| `webrtc_enabled` | bool | false | Auto-start WebRTC streaming |
| `webrtc_base_port` | int | 9000 | Base port for WebRTC signaling server |
| `log_base_path` | str | "./logs" | Directory for log files |

### Stream Configuration Reference

Each stream entry in the `streams` array can contain the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | str | Yes | Human-readable stream identifier (used for logging) |
| `url` | str | Yes¹ | RTSP URL for live streams |
| `file` | str | Yes¹ | Path to MP4 file for file-based streams |
| `loop` | bool | No | Loop the file when it reaches the end (MP4 only, default: false) |
| `fps` | int | No | Target FPS for frame rate capping (MP4 only, overrides file FPS) |

**Notes:**
- **Either** `url` **or** `file` must be specified, but not both
- `loop` and `fps` parameters are only applicable to MP4 file sources

### Decoder Preference Modes

- **`auto`**: Try nvv4l2 (DeepStream) → nvdec → CPU (default)
- **`nvv4l2`**: Force DeepStream decoder (fails if unavailable)
- **`nvdec`**: Force standard GStreamer NVDEC
- **`cpu`**: Force CPU decoding (for testing/fallback)

### Output Formats

| Format | Description | Bytes/Pixel | Shape (GPU) | Use Case |
|--------|-------------|-------------|-------------|----------|
| **NV12** | YUV 4:2:0 planar | 1.5 | `(H*1.5, W)` | Best for JPEG encoding, minimal bandwidth |
| **I420** | YUV 4:2:0 planar | 1.5 | `(H*1.5, W)` | Similar to NV12, different layout |
| **RGB** | 8-bit RGB | 3.0 | `(H, W, 3)` | PyTorch models |
| **BGR** | 8-bit BGR | 3.0 | `(H, W, 3)` | OpenCV compatibility |
| **RGBA** | 8-bit RGBA | 4.0 | `(H, W, 4)` | Alpha channel needed |

---

## Error Handling

### Exceptions

All methods may raise:
- **`RuntimeError`**: Configuration errors, initialization failures, invalid buffer mode

### Pattern: Safe Startup

```python
from rtspmodule import RTSPModule

provider = RTSPModule()

try:
    provider.start("config.yaml")
except RuntimeError as e:
    print(f"Failed to start streams: {e}")
    exit(1)

# Check GPU availability
if not provider.is_gpu_available():
    print("Warning: GPU unavailable, using CPU decoding")

try:
    # Main loop
    while True:
        for cam_id in range(provider.stream_count()):
            if provider.is_cpu_buffer_enabled():
                frame = provider.get_cpu_frame(cam_id)
            else:
                frame = provider.get_cuda_frame(cam_id)
            
            if frame['valid']:
                process_frame(frame)
finally:
    provider.stop()  # Always cleanup
```

### Pattern: Graceful Degradation

```python
# Try GPU mode first, fallback to CPU if unavailable
if provider.is_gpu_available() and not provider.is_cpu_buffer_enabled():
    try:
        frame = provider.get_cuda_frame(camera_id)
    except RuntimeError:
        # GPU mode failed, restart with CPU buffer
        provider.stop()
        # Update config to enable CPU buffer...
        provider.start("config_cpu.yaml")
```

---

## Usage Patterns

### Pattern 1: Real-Time Inference (GPU)

```python
import cupy as cp
import torch

provider = RTSPModule()
provider.start("config.yaml")  # cpu_buffer_enabled: false

# GPU inference loop
while True:
    frame = provider.get_cuda_frame(camera_id=0, timeout_ms=33)
    
    if not frame['valid']:
        continue
    
    # Zero-copy: wrap CUDA pointer as CuPy array
    gpu_frame = cp.ndarray(
        shape=frame['shape'],
        dtype=cp.uint8,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(frame['ptr'], frame['size'], None), 0
        )
    )
    
    # Convert to PyTorch tensor (still on GPU)
    tensor = torch.as_tensor(gpu_frame, device='cuda')
    
    # Run inference (e.g., YOLOv8)
    results = model(tensor)
```

**Advantages:**
- No CPU-GPU transfers
- ~0.1ms overhead per frame
- Ideal for real-time inference

---

### Pattern 2: Multi-Stream Batch Inference

```python
provider = RTSPModule()
provider.start("config.yaml")  # cpu_buffer_enabled: true

# Batched inference loop
camera_ids = [0, 1, 2, 3]  # 4 cameras

while True:
    batch = provider.get_batch(camera_ids, timeout_ms=10)
    
    if batch['valid_count'] == 0:
        continue  # No frames available
    
    # Batch shape: (4, H, W, 3) for BGR
    frames = batch['data']
    valid_mask = batch['valid_mask']
    
    # Run inference on all 4 frames at once
    results = model.predict(frames[valid_mask])
    
    # Map results back to cameras
    for i, result in enumerate(results):
        cam_idx = np.where(valid_mask)[0][i]
        cam_id = camera_ids[cam_idx]
        print(f"Camera {cam_id}: {result}")
```

**Performance:**
- **~2-3ms** for 4x 1080p frames
- Ideal for batched AI models
- Handles offline cameras gracefully

---

### Pattern 3: Stream Recording

```python
import cv2

provider = RTSPModule()
provider.start("config.yaml")  # cpu_buffer_enabled: true

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920, 1080))

try:
    while True:
        frame = provider.get_cpu_frame(camera_id=0, timeout_ms=100)
        
        if frame['valid']:
            # Convert NV12 to BGR if needed
            if frame['format'] == 'NV12':
                yuv = frame['data']
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            else:
                bgr = frame['data']
            
            out.write(bgr)
finally:
    out.release()
    provider.stop()
```

---

### Pattern 4: FPS Monitoring

```python
import time

provider = RTSPModule()
provider.start("config.yaml")

last_report = time.time()
frame_counts = [0] * provider.stream_count()

while True:
    for cam_id in range(provider.stream_count()):
        frame = provider.get_cpu_frame(cam_id)
        if frame['valid']:
            frame_counts[cam_id] += 1
    
    # Report every 5 seconds
    if time.time() - last_report > 5.0:
        for cam_id in range(provider.stream_count()):
            stats = provider.get_stats(cam_id)
            measured_fps = frame_counts[cam_id] / 5.0
            
            print(f"Camera {cam_id}:")
            print(f"  Stream FPS: {stats['current_fps']:.1f}")
            print(f"  Consume FPS: {measured_fps:.1f}")
            print(f"  Dropped: {stats['frames_dropped_queue']}")
            
            frame_counts[cam_id] = 0
        
        last_report = time.time()
```

---

### Pattern 5: Web Streaming (JPEG over WebSocket)

```python
import simplejpeg
from aiohttp import web

provider = RTSPModule()
provider.start("config.yaml")  # output_format: NV12

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    while True:
        frame = provider.get_cpu_frame(camera_id=0, timeout_ms=33)
        
        if frame['valid']:
            # Encode NV12 to JPEG (very fast with simplejpeg)
            jpeg_bytes = simplejpeg.encode_jpeg(
                frame['data'],
                quality=85,
                colorspace='NV12',
                fastdct=True
            )
            
            await ws.send_bytes(jpeg_bytes)
    
    return ws

app = web.Application()
app.router.add_route('GET', '/stream', websocket_handler)
web.run_app(app, port=8080)
```

**Performance:**
- NV12 encoding is **2-3x faster** than BGR
- Can stream 16+ cameras at 30 FPS

---

### Pattern 6: Automatic Fallback Detection

```python
provider = RTSPModule()
provider.start("config.yaml")

# Wait for streams to connect
time.sleep(2)

# Auto-detect buffer mode
if provider.is_cpu_buffer_enabled():
    print("Using CPU buffer mode")
    get_frame = lambda cam_id: provider.get_cpu_frame(cam_id)
else:
    print("Using GPU zero-copy mode")
    get_frame = lambda cam_id: provider.get_cuda_frame(cam_id)

# Unified frame retrieval
while True:
    for cam_id in range(provider.stream_count()):
        frame = get_frame(cam_id)
        if frame['valid']:
            process(frame)
```

---

## Performance Considerations

### GPU Mode (`cpu_buffer_enabled: false`)

**Pros:**
- **Zero-copy**: No CPU-GPU transfers
- **Low latency**: ~0.1ms overhead
- **High throughput**: Native decode speed

**Cons:**
- Requires CuPy/PyTorch for frame access
- No temporal buffering (latest frame only)
- GPU must be available

**Best for:** Real-time inference, low latency requirements

---

### CPU Buffer Mode (`cpu_buffer_enabled: true`)

**Pros:**
- **Temporal access**: 2-second ring buffer
- **NumPy arrays**: Direct CPU access
- **Automatic fallback**: Works without GPU
- **Batch API**: Optimized multi-stream retrieval

**Cons:**
- GPU→CPU transfer latency (~0.5ms per frame)
- Higher memory usage (buffer in RAM)

**Best for:** Recording, batch inference, web streaming

---

## Integration Examples

### PyTorch

```python
import torch
import cupy as cp

# GPU zero-copy
frame = provider.get_cuda_frame(0)
gpu_array = cp.ndarray(frame['shape'], cp.uint8, 
    cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(frame['ptr'], frame['size'], None), 0))
tensor = torch.as_tensor(gpu_array, device='cuda')
```

### TensorRT

```python
import pycuda.driver as cuda

frame = provider.get_cuda_frame(0)
# Use frame['ptr'] directly as input binding
context.execute_v2([frame['ptr'], output_ptr])
```

### OpenCV

```python
import cv2

frame = provider.get_cpu_frame(0)
img = frame['data']

# Convert if needed
if frame['format'] == 'NV12':
    bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)
else:
    bgr = img

cv2.imshow("Stream", bgr)
```

---

## Troubleshooting

### "get_cuda_frame() unavailable when cpu_buffer_enabled=true"

**Cause:** Config has `cpu_buffer_enabled: true` but code calls `get_cuda_frame()`

**Solution:** Use `get_cpu_frame()` instead or set `cpu_buffer_enabled: false`

---

### "get_cpu_frame() unavailable when cpu_buffer_enabled=false"

**Cause:** Config has `cpu_buffer_enabled: false` but code calls `get_cpu_frame()`

**Solution:** Use `get_cuda_frame()` instead or set `cpu_buffer_enabled: true`

---

### Frames Always Invalid (`valid: False`)

**Possible Causes:**
1. Stream not connected yet (wait 1-2 seconds after `start()`)
2. Invalid RTSP URL or authentication
3. Network issues
4. Codec not supported

**Debug:**
```python
import time
provider.start("config.yaml")
time.sleep(3)  # Wait for connection

stats = provider.get_stats(0)
if stats['reconnect_count'] > 0:
    print("Stream connection issues - check URL and network")
```

---

### High `frames_dropped_queue`

**Cause:** Consumer is too slow, decoder is overwriting frames

**Solutions:**
1. Increase `buffer_size` in config (e.g., `buffer_size: 10`)
2. Speed up frame processing
3. Use `get_batch()` for parallel processing
4. Skip frames if processing can't keep up

---

### Memory Usage Growing

**Cause:** Not releasing frame references (NumPy arrays from `get_cpu_frame()`)

**Solution:**
```python
# Good: Frame is dropped when out of scope
while True:
    frame = provider.get_cpu_frame(0)
    if frame['valid']:
        process(frame['data'])
    # 'frame' is released here

# Bad: Accumulating frames
frames = []
while True:
    frame = provider.get_cpu_frame(0)
    frames.append(frame)  # Memory leak!
```

---




