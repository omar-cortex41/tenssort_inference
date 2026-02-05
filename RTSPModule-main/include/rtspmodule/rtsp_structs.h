#ifndef RTSP_STRUCTS_H
#define RTSP_STRUCTS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Frame {
  const uint8_t *data = nullptr;
  size_t data_size = 0;
  int width = 0;
  int height = 0;
  int format = 0;
  uint64_t frame_id = 0;
  uint64_t timestamp_ns = 0;
  bool valid = false;
};

struct GpuFrameInfo {
  uint64_t ptr;
  int width;
  int height;
  int stride;
  size_t size;
  uint64_t frame_id;
  std::string format;  // "NV12", "RGB", "BGR", etc.
  bool valid;
};

// Statistics for diagnosing frame drops
struct FrameStats {
  uint64_t frames_received =
      0; // Frames received by appsink (after GStreamer drops)
  uint64_t frames_decoded = 0;        // Frames successfully decoded to GPU
  uint64_t frames_dropped_decode = 0; // Frames dropped during decode/copy
  uint64_t frames_consumed = 0;       // Frames fetched by Python consumer
  uint64_t frames_overwritten =
      0; // Frames overwritten before consumption (producer too fast)
  uint64_t frames_dropped_queue = 0; // Frames dropped because queue was full
  uint64_t decode_errors = 0;        // Decode/mapping errors
  uint64_t reconnect_count = 0;      // Number of stream reconnections
  uint64_t queue_depth = 0;          // Current queue depth
  uint64_t queue_max_depth = 0;      // Max queue depth seen
  uint64_t frames_duplicate = 0;     // Duplicate frames (same PTS) skipped
  
  // Real-time FPS tracking (calculated in C++)
  double current_fps = 0.0;          // Sliding window FPS (1 second window)
  double instant_fps = 0.0;          // Instantaneous FPS (1 / last frame interval)
  double source_fps = 0.0;           // FPS parsed from stream headers
  int source_width = 0;              // Video width from stream headers
  int source_height = 0;             // Video height from stream headers
};

// CPU-side frame for ring buffer (owns its memory)
struct CpuFrame {
  std::vector<uint8_t> data;    // Frame pixels (format depends on config)
  int width = 0;
  int height = 0;
  size_t data_size = 0;
  uint64_t frame_id = 0;
  uint64_t timestamp_ns = 0;    // PTS from stream
  int64_t capture_time_ns = 0;  // Wall-clock capture time
  std::string format;           // "NV12", "RGB", "BGR", etc.
  bool valid = false;
};

// CPU buffer statistics
struct CpuBufferInfo {
  size_t buffer_count = 0;      // Frames currently in buffer
  size_t buffer_capacity = 0;   // Max frames buffer can hold
  double buffer_duration_sec = 0.0;
  size_t memory_usage_bytes = 0;
  std::string format;           // Current output format
};

#endif // RTSP_STRUCTS_H

