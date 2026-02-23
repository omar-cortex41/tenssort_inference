# MP4 File Support Implementation

This document describes the new MP4 file support that has been added to the RTSPModule, allowing to use both RTSP streams and MP4 files in the same configuration with **FPS-capped decoding** for precise frame rate control.

## Configuration Format

The configuration now supports two types of sources:

### RTSP Stream (existing)
```yaml
streams:
  - name: "Camera 1"
    url: rtsp://192.168.1.61:8554/cam1
```

### MP4 File (new)
```yaml
streams:
  - name: "Video File 1"
    file: /path/to/video.mp4
    loop: true  # Optional: loop the file when it reaches the end
    fps: 25     # Optional: specify playback FPS (overrides file's native FPS)
```

## Key Features

### 1. Mixed Stream Support
- You can mix RTSP streams and MP4 files in the same configuration
- Each stream entry must have either a `url` key (for RTSP) or a `file` key (for MP4), but not both

### 2. MP4 Loop Support
- Add `loop: true` to make an MP4 file loop continuously when it reaches the end
- If `loop` is not specified or set to `false`, the stream will stop when the file ends

### 3. FPS-Capped Decoding
- Add `fps: 25` to cap the decoding frame rate to the specified FPS
- This overrides the file's native FPS and ensures consistent frame timing
- Perfect for testing scenarios where you need controlled frame rates
- Works with both looping and non-looping MP4 files

### 4. Automatic Pipeline Selection
- **RTSP streams**: Use `rtspsrc → rtph264depay → h264parse → decoder → converter → appsink`
- **MP4 files**: Use `filesrc → qtdemux → decodebin → decoder → converter → appsink`

### 5. Hardware Acceleration
- MP4 files support the same 3-tier decoder fallback as RTSP streams:
  - Tier 1: DeepStream nvv4l2decoder (NVMM memory)
  - Tier 2: Standard GStreamer nvh264dec/nvh265dec (CUDA memory)  
  - Tier 3: Software avdec_h264/avdec_h265 (CPU memory)

## Example Configuration

```yaml
settings:
  buffer_size: 3
  retry_max_attempts: 0
  decoder_preference: auto
  output_format: NV12
  cpu_buffer_enabled: true
  cpu_buffer_duration_sec: 2.0

streams:
  # RTSP stream
  - name: "Living Room Camera"
    url: rtsp://192.168.1.100:554/stream1
  
   # Looping MP4 file with FPS-capped decoding (useful for testing/demo)
  - name: "Demo Video"
    file: /path/to/demo.mp4
    loop: true
    fps: 25
  
  # Non-looping MP4 file (plays once then stops)
  - name: "Recorded Clip"
    file: /path/to/clip.mp4
  
  # Another RTSP stream
  - name: "Front Door Camera"
    url: rtsp://192.168.1.101:554/stream2
```

## Implementation Details

### New Constructor Parameters
The `StreamDecoder` class now accepts additional parameters:
- `is_file_source`: Boolean flag to distinguish RTSP from file sources
- `loop_file`: Boolean flag to enable file looping
- `target_fps`: Optional FPS value for frame rate capping



### Error Handling
- Invalid configurations (missing both url/file, or having both) are detected and reported
- File not found errors are properly handled
- Loop functionality gracefully handles seek failures
- FPS capping gracefully handles invalid FPS values (defaults to file's native FPS)


## Compatibility

- **Backward Compatible**: Existing RTSP-only configurations continue to work unchanged
- **API Compatible**: Python bindings (`RTSPModule` class) work with the new functionality
- **Performance**: Same GPU acceleration and CPU buffer optimization for both source types


## Usage Notes

- MP4 files don't participate in the reconnection logic (no network issues)
- File sources don't need RTSP-specific settings like latency or protocol configuration
- Looping is handled at the GStreamer level for maximum efficiency
- FPS capping is implemented using `videorate` element in the pipeline for precise frame rate control
