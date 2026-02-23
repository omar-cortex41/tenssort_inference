#ifndef RTSP_CLIENT_H
#define RTSP_CLIENT_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "rtsp_structs.h"
#include "stream_decoder.h"
#include "batch_types.h"

class RtspClient {
public:
  // Default configuration values
  static constexpr int DEFAULT_RECONNECT_DELAY_SEC = 5;
  static constexpr size_t DEFAULT_BUFFER_SIZE = 5;
  static constexpr int DEFAULT_RETRY_MAX_ATTEMPTS = 0;  // 0 = unlimited retries
  static constexpr float DEFAULT_BACKOFF_MULTIPLIER = 1.0f;
  static constexpr int DEFAULT_GPU_ID = 0;
  static constexpr bool DEFAULT_CPU_BUFFER_ENABLED = false;
  static constexpr double DEFAULT_CPU_BUFFER_DURATION = 2.0;

  RtspClient()
      : running_(false),
        buffer_size_(DEFAULT_BUFFER_SIZE),
        retry_max_attempts_(DEFAULT_RETRY_MAX_ATTEMPTS),
        backoff_multiplier_(DEFAULT_BACKOFF_MULTIPLIER),
        gpu_id_(DEFAULT_GPU_ID) {}
  ~RtspClient() { stop(); }

  bool loadConfig(const std::string &config_file);
  bool start();
  void stop();

  GpuFrameInfo getGpuFrame(int camera_id, int timeout_ms = 0);
  FrameStats getStats(int camera_id) const;

  int getStreamCount() const { return (int)decoders_.size(); }
  bool isRunning() const { return running_; }

  // Configure log path for all camera streams
  void setLogPath(const std::string &base_path);

  // CPU Buffer access (timeout_ms: 0=non-blocking, >0=wait for frame)
  CpuFrame getCpuFrame(int camera_id, int timeout_ms = 0);
  std::vector<CpuFrame> getCpuFrames(int camera_id, int count, int timeout_ms = 0);
  CpuBufferInfo getCpuBufferInfo(int camera_id) const;
  bool isCpuBufferEnabled() const { 
      return cpu_buffer_enabled_ || StreamDecoder::global_gpu_failure_.load(); 
  }
  bool isGpuAvailable() const { return gpu_available_; }

  // Batch Frame Retrieval - get frames from multiple cameras in single call
  // Returns fixed-size batch with valid_mask indicating which frames succeeded
  FrameBatch getBatchedFrames(const BatchConfig& config);

  // -------------------------------------------------------------------------
  // WebRTC streaming control — fully independent per stream, hot-switchable
  // -------------------------------------------------------------------------
  bool start_streaming(int camera_id);   // Returns false if invalid id or already streaming
  void stop_streaming(int camera_id);
  void start_streaming_all();
  void stop_streaming_all();
  bool isWebRtcStreamingEnabled(int camera_id) const;
  void setWebRtcBasePort(int base_port) { webrtc_base_port_ = base_port; }

private:
  std::vector<std::unique_ptr<StreamDecoder>> decoders_;
  std::atomic<bool> running_;
  std::thread reconnect_thread_;

  GstContext *cuda_context_ = nullptr;
  std::mutex context_mutex_;
  std::string log_base_path_;

  // Configurable settings
  size_t buffer_size_ = DEFAULT_BUFFER_SIZE;
  int retry_max_attempts_ = DEFAULT_RETRY_MAX_ATTEMPTS;
  float backoff_multiplier_ = DEFAULT_BACKOFF_MULTIPLIER;
  int gpu_id_ = DEFAULT_GPU_ID;

  bool initCudaContext();
  void reconnectLoop();

  // CPU Buffer settings
  bool cpu_buffer_enabled_ = DEFAULT_CPU_BUFFER_ENABLED;
  double cpu_buffer_duration_sec_ = DEFAULT_CPU_BUFFER_DURATION;
  std::string output_format_ = "NV12";
  bool gpu_available_ = true;  // Set to false if CUDA context init fails
  std::string decoder_preference_ = "auto";
  
  // Pre-allocated batch buffer (reused across getBatchedFrames calls)
  FrameBatch batch_buffer_;
  size_t batch_buffer_capacity_ = 0;  // Current allocated frame count
  size_t batch_buffer_stride_ = 0;    // Current bytes per frame
  
  // WebRTC settings
  bool webrtc_autostart_ = false;  // Auto-start all streams on launch
  int  webrtc_base_port_ = 9000;   // single signaling port for all streams
  
  // Thread pool for parallel batch copy (eliminates per-call thread spawn overhead)
  static constexpr size_t COPY_POOL_SIZE = 4;  // 4 workers is optimal for memcpy
  std::vector<std::thread> copy_workers_;
  std::queue<std::function<void()>> copy_tasks_;
  std::mutex copy_mutex_;
  std::condition_variable copy_cv_;
  std::condition_variable copy_done_cv_;
  std::atomic<bool> copy_pool_running_{false};
  std::atomic<size_t> pending_tasks_{0};
  
  void initCopyPool();
  void shutdownCopyPool();
  void parallelCopy(const std::vector<std::pair<uint8_t*, std::pair<const uint8_t*, size_t>>>& tasks);
};

#endif
