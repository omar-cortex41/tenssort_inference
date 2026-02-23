#include <rtspmodule/rtsp_client.h>
#include <algorithm>
#include <future>
#include <iomanip>
#include <yaml-cpp/yaml.h>

// RtspClient

bool RtspClient::loadConfig(const std::string &config_file) {
  try {
    YAML::Node config = YAML::LoadFile(config_file);

    // Read global settings
    if (config["settings"]) {
      auto settings = config["settings"];
      
      // Buffer size (frame queue depth) - supports both names for compatibility
      if (settings["buffer_size"]) {
        buffer_size_ = settings["buffer_size"].as<size_t>();
        std::cout << "Config: buffer_size = " << buffer_size_ << std::endl;
      } else if (settings["max_queue_size"]) {
        // Backward compatibility
        buffer_size_ = settings["max_queue_size"].as<size_t>();
        std::cout << "Config: buffer_size = " << buffer_size_ 
                  << " (via max_queue_size, consider renaming)" << std::endl;
      }
      
      // Retry configuration
      if (settings["retry_max_attempts"]) {
        retry_max_attempts_ = settings["retry_max_attempts"].as<int>();
        std::cout << "Config: retry_max_attempts = " << retry_max_attempts_
                  << (retry_max_attempts_ == 0 ? " (unlimited)" : "") << std::endl;
      }
      
      if (settings["backoff_multiplier"]) {
        backoff_multiplier_ = settings["backoff_multiplier"].as<float>();
        std::cout << "Config: backoff_multiplier = " << backoff_multiplier_ << std::endl;
      }
      
      // GPU selection
      if (settings["gpu_id"]) {
        gpu_id_ = settings["gpu_id"].as<int>();
        std::cout << "Config: gpu_id = " << gpu_id_ << std::endl;
      }
      
      // Logging path
      if (settings["log_base_path"]) {
        log_base_path_ = settings["log_base_path"].as<std::string>();
        std::cout << "Config: log_base_path = " << log_base_path_ << std::endl;
      }
      
      // CPU Buffer settings
      if (settings["cpu_buffer_enabled"]) {
        cpu_buffer_enabled_ = settings["cpu_buffer_enabled"].as<bool>();
        std::cout << "Config: cpu_buffer_enabled = " << (cpu_buffer_enabled_ ? "true" : "false") << std::endl;
      }
      
      if (settings["cpu_buffer_duration_sec"]) {
        cpu_buffer_duration_sec_ = settings["cpu_buffer_duration_sec"].as<double>();
        std::cout << "Config: cpu_buffer_duration_sec = " << cpu_buffer_duration_sec_ << std::endl;
      }
      
      // Output format
      if (settings["output_format"]) {
        output_format_ = settings["output_format"].as<std::string>();
        std::cout << "Config: output_format = " << output_format_ << std::endl;
      }
      
      // Decoder preference
      if (settings["decoder_preference"]) {
        decoder_preference_ = settings["decoder_preference"].as<std::string>();
        std::cout << "Config: decoder_preference = " << decoder_preference_ << std::endl;
      }

      // WebRTC settings
      if (settings["webrtc_enabled"]) {
        webrtc_autostart_ = settings["webrtc_enabled"].as<bool>();
        std::cout << "Config: webrtc_enabled = " << (webrtc_autostart_ ? "true" : "false") << std::endl;
      }
      if (settings["webrtc_base_port"]) {
        webrtc_base_port_ = settings["webrtc_base_port"].as<int>();
        std::cout << "Config: webrtc_base_port = " << webrtc_base_port_ << std::endl;
      }
    }

    if (!config["streams"])
      return false;

    int id = 0;
    for (const auto &stream : config["streams"]) {
      if (!stream.IsMap())
        continue;

      // Validate that either url or file is present, but not both
      bool has_url = stream["url"].IsDefined();
      bool has_file = stream["file"].IsDefined();
      
      if (!has_url && !has_file) {
        std::cerr << "Stream entry missing both 'url' and 'file' keys, skipping" << std::endl;
        continue;
      }
      
      if (has_url && has_file) {
        std::cerr << "Stream entry has both 'url' and 'file' keys, only one allowed, skipping" << std::endl;
        continue;
      }

      std::string url;
      std::string name;
      bool is_file_source = false;
      bool loop_file = false;
      double target_fps = 0.0; // Default: native FPS
      
      if (has_url) {
        url = stream["url"].as<std::string>();
        name = stream["name"] ? stream["name"].as<std::string>()
                              : "Camera " + std::to_string(id + 1);
      } else {
        url = stream["file"].as<std::string>();
        is_file_source = true;
        name = stream["name"] ? stream["name"].as<std::string>()
                              : "File " + std::to_string(id + 1);
        
        // Check for loop parameter (only applies to file sources)
        if (stream["loop"]) {
          loop_file = stream["loop"].as<bool>();
        }
        
        // Check for fps parameter (only applies to file sources)
        if (stream["fps"]) {
          target_fps = stream["fps"].as<double>();
        }
      }
      
      // Allow per-stream overrides of decoder preference
      std::string stream_pref = decoder_preference_;
      if (stream["decoder_preference"]) {
          stream_pref = stream["decoder_preference"].as<std::string>();
      }

      decoders_.push_back(
          std::make_unique<StreamDecoder>(id++, name, url, buffer_size_, output_format_, stream_pref, is_file_source, loop_file, target_fps));
    }
  } catch (...) {
    return false;
  }
  return !decoders_.empty();
}

void RtspClient::setLogPath(const std::string &base_path) {
  log_base_path_ = base_path;
  // Configure logging for existing decoders
  for (auto &d : decoders_) {
    d->setLogPath(base_path);
  }
}

bool RtspClient::initCudaContext() {
  // Create a minimal pipeline to obtain CUDA context from GStreamer
  // Use cudaupload as it handles System->CUDA transition from videotestsrc
  GstElement *probe = gst_element_factory_make("cudaupload", "context_probe");
  if (!probe) {
    std::cerr
        << "cudaupload not available - CUDA context pre-allocation disabled"
        << std::endl;
    return false;
  }

  GstElement *pipe = gst_pipeline_new("context_probe_pipe");
  GstElement *src = gst_element_factory_make("videotestsrc", "probe_src");
  GstElement *sink = gst_element_factory_make("fakesink", "probe_sink");

  if (!pipe || !src || !sink) {
    if (probe)
      gst_object_unref(probe);
    if (pipe)
      gst_object_unref(pipe);
    if (src)
      gst_object_unref(src);
    if (sink)
      gst_object_unref(sink);
    return false;
  }

  g_object_set(src, "num-buffers", 1, nullptr);
  gst_bin_add_many(GST_BIN(pipe), src, probe, sink, nullptr);
  gst_element_link_many(src, probe, sink, nullptr);

  // Context sync handler to capture created context
  GstContext *captured_ctx = nullptr;
  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipe));
  gst_bus_set_sync_handler(
      bus,
      [](GstBus *, GstMessage *msg, gpointer data) -> GstBusSyncReply {
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_HAVE_CONTEXT) {
          GstContext *ctx = nullptr;
          gst_message_parse_have_context(msg, &ctx);
          if (ctx && g_strcmp0(gst_context_get_context_type(ctx),
                               "gst.cuda.context") == 0) {
            auto *out = static_cast<GstContext **>(data);
            if (!*out)
              *out = gst_context_ref(ctx);
            gst_context_unref(ctx);
          }
        }
        return GST_BUS_PASS;
      },
      &captured_ctx, nullptr);

  // Run probe pipeline briefly
  gst_element_set_state(pipe, GST_STATE_PLAYING);

  for (int i = 0; i < 20 && !captured_ctx; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  gst_element_set_state(pipe, GST_STATE_NULL);
  gst_object_unref(bus);
  gst_object_unref(pipe);

  if (captured_ctx) {
    cuda_context_ = captured_ctx;
    std::cout << "Pre-allocated CUDA context successfully" << std::endl;
    return true;
  }

  std::cerr << "Failed to pre-allocate CUDA context" << std::endl;
  return false;
}

bool RtspClient::start() {
  if (running_)
    return true;

  gst_init(nullptr, nullptr);
  running_ = true;

  // Pre-allocate CUDA context before starting any streams
  gpu_available_ = initCudaContext();
  if (!gpu_available_) {
    std::cerr << "Warning: CUDA context pre-allocation failed - GPU unavailable" << std::endl;
    std::cout << "GPU hardware not available - forcing CPU buffer mode for all streams" << std::endl;
    // Force CPU buffer mode when GPU is not available
    cpu_buffer_enabled_ = true;
  }

  // Start ALL streams in parallel with shared context
  for (auto &d : decoders_) {
    if (cuda_context_) {
      d->setSharedContext(cuda_context_);
    }
    // Configure logging if path is set
    if (!log_base_path_.empty()) {
      d->setLogPath(log_base_path_);
    }
    // Configure CPU buffer if enabled (either by config or forced by GPU unavailability)
    if (cpu_buffer_enabled_) {
      d->setCpuBufferConfig(true, cpu_buffer_duration_sec_, 25.0);  // Default 25fps, will adjust with detected FPS
    }
    // Configure WebRTC — all streams share one signaling port, identified by stream_id
    std::string stream_id = d->getName();
    std::replace(stream_id.begin(), stream_id.end(), ' ', '-');
    d->setWebRtcConfig(webrtc_base_port_, stream_id);
    d->start();
    
    // After starting, check if decoder is actually using GPU pipeline
    // If cudaconvert is unavailable but nvdec is present, decoder won't have full GPU path
    if (!cpu_buffer_enabled_ && !d->isUsingGpuPipeline()) {
      std::cout << "[" << d->getId() << "] GPU pipeline incomplete - enabling CPU buffer fallback" << std::endl;
      d->enableCpuBufferFallback();
    }
    
    // If webrtc_autostart_, defer start_streaming until onPadAdded via webrtc_enabled_ flag
    // (The pipeline may not be ready yet, start_streaming() handles this automatically)
    if (webrtc_autostart_) {
      d->start_streaming();  // Sets webrtc_enabled_=true if tee not ready yet
    }
  }

  std::cout << "Started " << decoders_.size() << " streams" 
            << (gpu_available_ ? " (GPU mode)" : " (CPU fallback mode)") << std::endl;

  // Initialize thread pool for batch copy operations
  initCopyPool();

  reconnect_thread_ = std::thread(&RtspClient::reconnectLoop, this);
  return true;
}

void RtspClient::stop() {
  if (!running_)
    return;
  running_ = false;

  // Shutdown thread pool first
  shutdownCopyPool();

  if (reconnect_thread_.joinable())
    reconnect_thread_.join();
  for (auto &d : decoders_)
    d->stop();

  if (cuda_context_) {
    gst_context_unref(cuda_context_);
    cuda_context_ = nullptr;
  }
}

GpuFrameInfo RtspClient::getGpuFrame(int id, int timeout_ms) {
  if (id < 0 || id >= (int)decoders_.size())
    return GpuFrameInfo{};
  return decoders_[id]->getGpuFrame(timeout_ms);
}

FrameStats RtspClient::getStats(int id) const {
  if (id < 0 || id >= (int)decoders_.size())
    return FrameStats{};
  return decoders_[id]->getStats();
}

// ---------------------------------------------------------------------------
// WebRTC streaming API — per-stream, hot-switchable
// ---------------------------------------------------------------------------

bool RtspClient::start_streaming(int camera_id) {
  if (camera_id < 0 || camera_id >= (int)decoders_.size())
    return false;
  return decoders_[camera_id]->start_streaming();
}

void RtspClient::stop_streaming(int camera_id) {
  if (camera_id < 0 || camera_id >= (int)decoders_.size())
    return;
  decoders_[camera_id]->stop_streaming();
}

void RtspClient::start_streaming_all() {
  for (auto& d : decoders_)
    d->start_streaming();
}

void RtspClient::stop_streaming_all() {
  for (auto& d : decoders_)
    d->stop_streaming();
}

bool RtspClient::isWebRtcStreamingEnabled(int camera_id) const {
  if (camera_id < 0 || camera_id >= (int)decoders_.size())
    return false;
  return decoders_[camera_id]->isWebRtcStreamingEnabled();
}

void RtspClient::reconnectLoop() {
  int current_delay_sec = DEFAULT_RECONNECT_DELAY_SEC;
  std::vector<int> retry_counts(decoders_.size(), 0);
  std::vector<bool> pending_reconnections(decoders_.size(), false);
  
  while (running_) {
    // Check if any pending reconnections have actually succeeded (received frames)
    bool any_actually_recovered = false;
    for (size_t idx = 0; idx < decoders_.size(); ++idx) {
      if (pending_reconnections[idx]) {
        // Check if first frame was received (pending_reconnect_ was cleared)
        if (!decoders_[idx]->isPendingReconnect()) {
          any_actually_recovered = true;
          pending_reconnections[idx] = false;
        }
      }
    }
    
    // Apply backoff or reset based on actual recovery
    if (any_actually_recovered) {
      current_delay_sec = DEFAULT_RECONNECT_DELAY_SEC;  // Reset on actual success
    }
    
    // Wait with current delay (in 100ms increments for responsiveness)
    for (int i = 0; i < current_delay_sec * 10 && running_; i++)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!running_)
      break;

    bool any_attempted = false;
    for (size_t idx = 0; idx < decoders_.size(); ++idx) {
      auto &d = decoders_[idx];
      if (d->hasError() || (!d->isFileSource() && d->isStale(10))) {
        // Check retry limit (0 = unlimited)
        if (retry_max_attempts_ > 0 && retry_counts[idx] >= retry_max_attempts_) {
          continue;  // Skip this stream, max retries reached
        }
        
        // Check if GPU hardware failed - enable CPU buffer fallback before reconnecting
        if (d->hasHardwareAccelFailed() && !d->isCpuBufferEnabled()) {
          std::cout << "[" << d->getId() << "] GPU hardware failed - switching to CPU buffer for reconnect" << std::endl;
          d->enableCpuBufferFallback();
        }
        
        {
          std::lock_guard<std::mutex> lock(context_mutex_);
          if (cuda_context_)
            d->setSharedContext(cuda_context_);
        }
        
        if (d->recreate()) {
          // Pipeline started - mark as pending, will check for actual frame later
          pending_reconnections[idx] = true;
          retry_counts[idx] = 0;
        } else {
          retry_counts[idx]++;
        }
        any_attempted = true;
      }
    }
    
    // Apply backoff if we attempted reconnections but none succeeded yet
    if (any_attempted && !any_actually_recovered && backoff_multiplier_ > 1.0f) {
      current_delay_sec = static_cast<int>(current_delay_sec * backoff_multiplier_);
    }
  }
}

// CPU Buffer pass-through methods

CpuFrame RtspClient::getCpuFrame(int id, int timeout_ms) {
  if (id < 0 || id >= (int)decoders_.size())
    return CpuFrame{};
  return decoders_[id]->getCpuFrame(timeout_ms);
}

CpuBufferInfo RtspClient::getCpuBufferInfo(int id) const {
  if (id < 0 || id >= (int)decoders_.size())
    return CpuBufferInfo{};
  return decoders_[id]->getCpuBufferInfo();
}

std::vector<CpuFrame> RtspClient::getCpuFrames(int id, int count, int timeout_ms) {
  if (id < 0 || id >= (int)decoders_.size())
    return {};
  return decoders_[id]->getCpuFrames(count, timeout_ms);
}

// Batch Frame Retrieval Implementation (Pre-allocated Buffer + Parallel Copy)
FrameBatch RtspClient::getBatchedFrames(const BatchConfig& config) {

  
  const size_t batch_size = config.camera_ids.size();
  
  // Early exit for empty batch
  if (batch_size == 0) {
    return FrameBatch{};
  }
  double bytes_per_pixel = 3.0;
  if (output_format_ == "NV12" || output_format_ == "I420") {
    bytes_per_pixel = 1.5;
  } else if (output_format_ == "RGBA" || output_format_ == "BGRA") {
    bytes_per_pixel = 4.0;
  }
  
  
  std::vector<const CpuFrame*> frame_ptrs(batch_size, nullptr);
  int common_w = config.target_width;
  int common_h = config.target_height;
  size_t valid_count = 0;
  
  for (size_t i = 0; i < batch_size; ++i) {
    int cam_id = config.camera_ids[i];
    
    // Bounds check
    if (cam_id < 0 || cam_id >= (int)decoders_.size()) {
      continue;
    }
    
    // Fast pointer lookup (no data copy)
    frame_ptrs[i] = decoders_[cam_id]->peekLatestFrame(config.timeout_ms);
    
    if (frame_ptrs[i] && frame_ptrs[i]->valid) {
      valid_count++;
      
      // Use first valid frame to determine common resolution
      if (common_w == 0 || common_h == 0) {
        common_w = frame_ptrs[i]->width;
        common_h = frame_ptrs[i]->height;
      }
    }
  }
  

  
  // Calculate required stride based on detected resolution
  size_t required_stride = (common_w > 0 && common_h > 0) 
      ? static_cast<size_t>(common_w * common_h * bytes_per_pixel) 
      : 0;
  
  // Only reallocate if:
  //   1. Batch size increased beyond current capacity
  //   2. Frame stride increased (resolution change)
  
  // Determine output destination: use external buffer if provided, else internal
  uint8_t* output_base = config.output_ptr;
  size_t output_stride = batch_buffer_.frame_stride;
  
  // If internal stride is invalid/small (first run), calculate needed stride
  if (output_stride < required_stride) {
      output_stride = required_stride;
  }
  
  bool using_external_buffer = (output_base != nullptr && config.output_size >= batch_size * output_stride);
  
  if (batch_buffer_capacity_ < batch_size || batch_buffer_stride_ < required_stride) {
    size_t new_capacity = std::max(batch_size, batch_buffer_capacity_ * 3 / 2);
    size_t new_stride = std::max(required_stride, batch_buffer_stride_ * 3 / 2);
    
    if (new_stride == 0) {
      new_stride = static_cast<size_t>(1920 * 1080 * bytes_per_pixel);
    }
    
    batch_buffer_.valid_mask.resize(new_capacity);
    batch_buffer_.metadata.resize(new_capacity);
    
    if (!using_external_buffer) {
        batch_buffer_.data.resize(new_capacity * new_stride);
    }
    
    batch_buffer_capacity_ = new_capacity;
    batch_buffer_stride_ = new_stride;
    
    // Update stride if we just grew it
    if (!using_external_buffer) {
        output_stride = new_stride;
    }
  }
  
  if (!using_external_buffer) {
    output_base = batch_buffer_.data.data();
  }
  
  std::fill(batch_buffer_.valid_mask.begin(), 
            batch_buffer_.valid_mask.begin() + batch_size, false);
  
  batch_buffer_.batch_size = batch_size;
  batch_buffer_.format = output_format_;
  batch_buffer_.valid_count = 0;
  batch_buffer_.width = common_w;
  batch_buffer_.height = common_h;
  
  // Early exit if no valid frames
  if (valid_count == 0 || required_stride == 0) {
    batch_buffer_.width = 0;
    batch_buffer_.height = 0;
    batch_buffer_.frame_stride = 0; // This line was moved/modified
    return batch_buffer_;
  }
  
  struct CopyTask {
    uint8_t* dst;
    const uint8_t* src;
    size_t size;
  };
  std::vector<CopyTask> tasks;
  tasks.reserve(valid_count);
  
  // Determine output destination: use external buffer if provided, else internal
  // This block was moved up
  
  for (size_t i = 0; i < batch_size; ++i) {
    int cam_id = config.camera_ids[i];
    batch_buffer_.metadata[i].camera_id = cam_id;
    batch_buffer_.metadata[i].valid = false;
    
    const CpuFrame* frame = frame_ptrs[i];
    if (!frame || !frame->valid) {
      continue;
    }
    
    // Resolution mismatch check
    if (frame->width != common_w || frame->height != common_h) {
      continue;
    }
    
    // Mark as valid
    batch_buffer_.valid_mask[i] = true;
    batch_buffer_.metadata[i].valid = true;
    batch_buffer_.metadata[i].frame_id = frame->frame_id;
    batch_buffer_.metadata[i].timestamp_ns = frame->timestamp_ns;
    batch_buffer_.metadata[i].width = frame->width;
    batch_buffer_.metadata[i].height = frame->height;
    batch_buffer_.valid_count++;
    
    // Add copy task - destination is either external NumPy buffer or internal buffer
    CopyTask task;
    task.dst = output_base + (i * output_stride);
    task.src = frame->data.data();
    task.size = std::min(frame->data.size(), output_stride); // Safe clip
    tasks.push_back(task);
  }
  

  if (tasks.size() >= 4 && copy_pool_running_) {
    // Convert to thread pool task format
    std::vector<std::pair<uint8_t*, std::pair<const uint8_t*, size_t>>> pool_tasks;
    pool_tasks.reserve(tasks.size());
    for (const auto& task : tasks) {
      pool_tasks.emplace_back(task.dst, std::make_pair(task.src, task.size));
    }
    parallelCopy(pool_tasks);
  } else {
    // Sequential copy (few tasks or pool not running)
    for (const auto& task : tasks) {
      std::memcpy(task.dst, task.src, task.size);
    }
  }
  
  // Use zero-copy return if external buffer was used
  if (using_external_buffer) {
      FrameBatch result;
      result.metadata = batch_buffer_.metadata;
      result.valid_mask = batch_buffer_.valid_mask;
      result.batch_size = batch_buffer_.batch_size;
      result.frame_stride = output_stride;
      result.width = batch_buffer_.width;
      result.height = batch_buffer_.height;
      result.format = batch_buffer_.format;
      result.valid_count = batch_buffer_.valid_count;
      // result.data remains empty! No massive copy!
      return result;
  }
  
  return batch_buffer_;
}

void RtspClient::initCopyPool() {
  if (copy_pool_running_.exchange(true)) {
    return;  // Already running
  }
  
  copy_workers_.reserve(COPY_POOL_SIZE);
  for (size_t i = 0; i < COPY_POOL_SIZE; ++i) {
    copy_workers_.emplace_back([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(copy_mutex_);
          copy_cv_.wait(lock, [this] {
            return !copy_tasks_.empty() || !copy_pool_running_;
          });
          
          if (!copy_pool_running_ && copy_tasks_.empty()) {
            return;  // Shutdown requested
          }
          
          if (!copy_tasks_.empty()) {
            task = std::move(copy_tasks_.front());
            copy_tasks_.pop();
          }
        }
        
        if (task) {
          task();
          if (pending_tasks_.fetch_sub(1) == 1) {
            copy_done_cv_.notify_all();
          }
        }
      }
    });
  }
  
  std::cout << "[Batch] Thread pool initialized with " << COPY_POOL_SIZE << " workers" << std::endl;
}

void RtspClient::shutdownCopyPool() {
  if (!copy_pool_running_.exchange(false)) {
    return;  // Already stopped
  }
  
  // Wake up all workers
  copy_cv_.notify_all();
  
  // Join all workers
  for (auto& worker : copy_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  copy_workers_.clear();
  
  // Clear any remaining tasks
  std::queue<std::function<void()>> empty;
  std::swap(copy_tasks_, empty);
  pending_tasks_ = 0;
}

void RtspClient::parallelCopy(
    const std::vector<std::pair<uint8_t*, std::pair<const uint8_t*, size_t>>>& tasks) {
  if (tasks.empty()) return;
  
  const size_t num_tasks = tasks.size();
  const size_t num_workers = std::min((size_t)COPY_POOL_SIZE, num_tasks);
  const size_t chunk_size = (num_tasks + num_workers - 1) / num_workers;
  
  // Use futures for true parallel execution without lock contention
  // Each worker handles a contiguous chunk of tasks
  std::vector<std::future<void>> futures;
  futures.reserve(num_workers);
  
  for (size_t w = 0; w < num_workers; ++w) {
    size_t start = w * chunk_size;
    size_t end = std::min(start + chunk_size, num_tasks);
    
    if (start >= end) break;  // No more work
    
    // Launch async task - each worker copies its chunk sequentially
    futures.push_back(std::async(std::launch::async, [&tasks, start, end] {
      for (size_t i = start; i < end; ++i) {
        std::memcpy(tasks[i].first, tasks[i].second.first, tasks[i].second.second);
      }
    }));
  }
  
  // Wait for all workers to complete
  for (auto& f : futures) {
    f.get();
  }
}
