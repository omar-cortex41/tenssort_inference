#ifndef STREAM_DECODER_H
#define STREAM_DECODER_H

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "gpu_buffer.h"
#include "logger.h"
#include "rtsp_structs.h"
#include "cpu_buffer.h"

// Forward declarations
namespace rtsp {
  class StreamPipelineBuilder;
}

// Decoder type enumeration for tracking active decoder
enum class DecoderType {
  NVDEC_CUDA,    // Standard GStreamer nvh264dec/nvh265dec with cudaconvert
  NVV4L2_NVMM,   // DeepStream nvv4l2decoder with nvvideoconvert (NVMM memory)
  AVDEC_CPU      // Software decoder (avdec_h264/avdec_h265) with videoconvert
};

class StreamDecoder {
public:
StreamDecoder(int id, const std::string &name,
                const std::string &url, size_t max_queue_size = 5,
                const std::string& output_format = "NV12",
                const std::string& decoder_preference = "auto",
                bool is_file_source = false,
                bool loop_file = false,
                double target_fps = 0.0);
  ~StreamDecoder();

  // Logger configuration
  void setLogPath(const std::string &base_path);
  void logReconnected(int attempt_count);  // Log successful reconnection

  bool start();
  void stop();

  bool create();
  bool recreate();
  void destroy();

  GpuFrameInfo getGpuFrame(int timeout_ms = 0);
  FrameStats getStats() const;

  int getId() const { return id_; }
  const std::string &getName() const { return name_; }

  void markError() { has_error_ = true; }
  void clearError() { has_error_ = false; }
  bool hasError() const { return has_error_; }
  bool isPendingReconnect() const { return pending_reconnect_; }

  void updateFrameTime() {
    last_frame_time_ =
        std::chrono::steady_clock::now().time_since_epoch().count();
  }

  bool isStale(int timeout_sec) const {
    // Use single atomic load to prevent TOCTOU race
    int64_t last = last_frame_time_.load(std::memory_order_acquire);
    if (last == 0)
      return false;
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return ((now - last) / 1000000000) > timeout_sec;
  }

  bool isFileSource() const { return is_file_source_; }

  bool ownsElement(const std::string &element_name) const;

  void setSharedContext(GstContext *context);
  void setOnContextFoundCallback(std::function<void(GstContext *)> cb) {
    on_context_found_ = cb;
  }

  // CPU Buffer configuration and access
  void setCpuBufferConfig(bool enabled, double duration_sec, double fps = 25.0);
  void resizeCpuBuffer(double detected_fps);  // Resize buffer based on detected FPS
  CpuFrame getCpuFrame(int timeout_ms = 0);
  std::vector<CpuFrame> getCpuFrames(int count, int timeout_ms = 0);
  CpuBufferInfo getCpuBufferInfo() const;
  const CpuFrame* peekLatestFrame(int timeout_ms = 0) const;  // Zero-copy peek for batch
  std::string getOutputFormat() const { return output_format_; }
  bool isCpuBufferEnabled() const { return cpu_buffer_enabled_; }
  
  // GPU pipeline status - true if using hardware acceleration (NVDEC or nvv4l2)
  bool isUsingGpuPipeline() const { return (use_cuda_memory_ || use_nvmm_memory_) && !hardware_accel_failed_; }
  bool isUsingNvmmMemory() const { return use_nvmm_memory_; }  // True if using DeepStream NVMM path
  bool hasHardwareAccelFailed() const { return hardware_accel_failed_; }
  DecoderType getActiveDecoderType() const { return active_decoder_type_; }
  void enableCpuBufferFallback();  // Switch to CPU mode at runtime

  // -------------------------------------------------------------------------
  // WebRTC streaming control — fully independent per stream
  // Safe to call from any thread at any time after start()
  // -------------------------------------------------------------------------
  void setWebRtcConfig(int signaling_port, const std::string& stream_id);
  bool start_streaming();  // Returns false if already active or pipeline not ready
  void stop_streaming();
  bool isWebRtcStreamingEnabled() const { return webrtc_streaming_active_.load(); }
  
  static double getBytesPerPixel(const std::string& format);

private:
  int id_;
  std::string name_;
  std::string url_;
  std::string decoder_preference_;
  bool is_file_source_;
  bool loop_file_ = false;  // For MP4 file looping
  double target_fps_ = 0.0; // Target FPS for MP4 files (0 = native)

  GstElement *pipeline_ = nullptr;
  GstElement *source_ = nullptr;
  GstElement *demuxer_ = nullptr;  // For MP4 files (qtdemux)
  GstElement *decodebin_ = nullptr; // For MP4 files (decodebin)
  GstElement *depay_ = nullptr;
  GstElement *parse_ = nullptr;
  GstElement *decoder_ = nullptr;
  GstElement *convert_ = nullptr;
  GstElement *appsink_ = nullptr;
  std::atomic<bool> decoder_linked_{false};  // Accessed from GStreamer pad-added callbacks

  // Provide builder access to pipeline members
  friend class rtsp::StreamPipelineBuilder;

  std::atomic<bool> has_error_;
  std::atomic<int64_t> last_frame_time_;
  std::atomic<int> reconnect_count_{0};  // Accessed from reconnect thread and pipeline naming
  std::atomic<bool> pending_reconnect_{false};    // True after recreate(), cleared on first frame
  std::atomic<bool> pending_first_frame_{false};  // True after initial start(), cleared on first frame

  std::atomic<bool> running_;
  std::thread bus_thread_;
  void busLoop();

  mutable std::mutex frame_mutex_;  // mutable to allow locking in const methods
  GpuBuffer gpu_buffer_;
  uint64_t frame_counter_;
  bool use_cuda_memory_ = false; // True if using CUDA zero-copy path (nvh264dec + cudaconvert)
  bool use_nvmm_memory_ = false;  // True if using DeepStream NVMM path (nvv4l2decoder + nvvideoconvert)
  DecoderType active_decoder_type_ = DecoderType::AVDEC_CPU;  // Track which decoder is active
  std::atomic<bool> hardware_accel_failed_{false}; // True if GPU decode/convert failed at runtime
  uint64_t cuda_device_ptr_ = 0; // Direct CUDA pointer (zero-copy)

  // Frame drop statistics
  mutable std::mutex stats_mutex_;
  FrameStats stats_;

  GstContext *shared_context_ = nullptr;
  std::function<void(GstContext *)> on_context_found_;

  // State logging
  std::unique_ptr<rtsp::DateLogger> logger_;

  int current_stride_ = 0;

  // Frame queue for zero-overwrite buffering
  size_t max_queue_depth_ = 5; // Configurable via constructor
  struct QueuedFrame {
    GstSample *sample;
    uint64_t frame_id;
    int width;
    int height;
    int stride;
    size_t data_size;
    uint64_t cuda_ptr;
    uint64_t timestamp_ns;
  };
  std::queue<QueuedFrame> frame_queue_;
  std::condition_variable queue_cv_;
  
  // Watchdog for stalled streams
  std::atomic<int64_t> last_frame_rx_time_ms_{0};

  // -------------------------------------------------------------------------
  // WebRTC streaming control
  // -------------------------------------------------------------------------
  GstElement* webrtc_tee_ = nullptr;  // tee element inserted into main path
  GstElement* webrtc_bin_ = nullptr;  // encapsulated WebRTC branch

  std::atomic<bool> webrtc_enabled_{false};          // Auto-start on pipeline creation if true
  std::atomic<bool> webrtc_streaming_active_{false}; // Branch is currently live
  int         webrtc_signaling_port_ = 9000;
  std::string webrtc_stream_id_;
  bool        webrtc_is_h265_ = false;  // Set in onPadAdded when codec is detected
  
public:
  // Global flag for GPU failure detected via logs
  static std::atomic<bool> global_gpu_failure_;
  
  // Check if stream is healthy (receiving frames). Returns false if stalled.
  bool checkHealth();

  // FPS tracking with sliding window (measured at decoder)
  static constexpr size_t FPS_WINDOW_SIZE = 120;  // Max samples in window
  static constexpr double FPS_WINDOW_SEC = 1.0;   // 1 second sliding window
  std::deque<int64_t> fps_timestamps_ns_;         // Nanosecond timestamps
  int64_t last_frame_time_ns_ = 0;                // For instant FPS
  uint64_t last_pts_ = 0;                         // Last frame PTS for duplicate detection
  
  void updateFps();  // Calculate FPS from timestamps

  static GstBusSyncReply busSyncHandler(GstBus *bus, GstMessage *msg,
                                        gpointer user_data);
  static GstFlowReturn onNewSample(GstElement *sink, gpointer data);
  static GstPadProbeReturn onParserCaps(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);

  // Output format configuration
  std::string output_format_ = "NV12";
  
  // CPU Ring Buffer
  std::atomic<bool> cpu_buffer_enabled_{false};
  double cpu_buffer_duration_sec_ = 2.0;
  std::unique_ptr<CpuBuffer> cpu_buffer_;
};

#endif // STREAM_DECODER_H
