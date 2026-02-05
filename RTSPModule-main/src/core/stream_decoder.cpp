#include <rtspmodule/stream_decoder.h>
#define GST_USE_UNSTABLE_API
#include <cuda_runtime.h>
#include <gst/rtsp/gstrtsp.h>
#include <gst/sdp/gstsdpmessage.h>
#include <iostream>

// Conditional GStreamer CUDA memory support (for standard NVDEC path)
#ifdef HAVE_GST_CUDA
#include <gst/cuda/gstcuda.h>
#endif

// Conditional DeepStream NVMM support - requires DMA-BUF for buffer access
#ifdef HAVE_DEEPSTREAM
#include <nvbufsurface.h>
#include <gst/allocators/gstdmabuf.h>
#endif

// Static initialization
std::atomic<bool> StreamDecoder::global_gpu_failure_{false};

// Custom GStreamer log handler to sniff for critical GPU errors
static void gst_gpu_error_sniffer(GstDebugCategory *category,
                                  GstDebugLevel level,
                                  const gchar *file,
                                  const gchar *function,
                                  gint line,
                                  GObject *object,
                                  GstDebugMessage *message,
                                  gpointer user_data) {
    if (level <= GST_LEVEL_ERROR) {
        const gchar *cat_name = gst_debug_category_get_name(category);
        std::string category_name = cat_name ? cat_name : "";
        
        const gchar *msg_str = gst_debug_message_get(message);
        std::string err_msg = msg_str ? msg_str : "";
        
        bool is_gpu_category = (category_name.find("nvdec") != std::string::npos ||
                                category_name.find("nvcodec") != std::string::npos ||
                                category_name.find("cuda") != std::string::npos);
                                
        bool is_gpu_msg = (err_msg.find("nvdec") != std::string::npos || 
                           err_msg.find("cuda") != std::string::npos ||
                           err_msg.find("CUDA") != std::string::npos);
        
        bool is_fatal_msg = (err_msg.find("Could not create decoder") != std::string::npos || 
                             err_msg.find("Couldn't create decoder") != std::string::npos || 
                             err_msg.find("failed to init") != std::string::npos ||
                             err_msg.find("cuInit") != std::string::npos);

        if ((is_gpu_category || is_gpu_msg) && is_fatal_msg) {
             // Flag global failure
             StreamDecoder::global_gpu_failure_ = true;
             std::cerr << "\n[CRITICAL] GPU FAILURE DETECTED VIA LOGS (" << category_name << "): " << err_msg 
                       << " -> Triggering global fallback\n" << std::endl;
        }
    }
    // Forward to default handler
    gst_debug_log_default(category, level, file, function, line, object, message, user_data);
}

StreamDecoder::StreamDecoder(int id, const std::string &name,
                             const std::string &url, size_t max_queue_size,
                             const std::string& output_format)
    : id_(id), name_(name), url_(url), frame_counter_(0), has_error_(false),
      pipeline_(nullptr), last_frame_time_(0), reconnect_count_(0),
      running_(false), max_queue_depth_(max_queue_size),
      gpu_buffer_(), output_format_(output_format) {
  std::cout << "[" << id_ << "] Queue depth: " << max_queue_depth_ 
            << ", Output format: " << output_format_ << std::endl;

  static std::once_flag log_handler_flag;
  std::call_once(log_handler_flag, [](){
      gst_debug_remove_log_function(gst_debug_log_default);
      gst_debug_add_log_function(gst_gpu_error_sniffer, nullptr, nullptr);
  });
}

void StreamDecoder::setLogPath(const std::string &base_path) {
  logger_ = std::make_unique<rtsp::DateLogger>(base_path, name_);
  logger_->logInfo("Logger initialized for camera: " + name_);
}

void StreamDecoder::logReconnected(int attempt_count) {
  if (logger_) {
    logger_->logStateChange(rtsp::CameraState::Reconnected,
                            "Successfully recovered after " + std::to_string(attempt_count) + " attempt(s)");
  }
}

StreamDecoder::~StreamDecoder() {
  destroy();
  if (shared_context_) {
    gst_context_unref(shared_context_);
    shared_context_ = nullptr;
  }
}

void StreamDecoder::setSharedContext(GstContext *ctx) {
  // Protect shared_context_ access - also accessed from busSyncHandler on GStreamer thread
  std::lock_guard<std::mutex> lock(frame_mutex_);
  if (shared_context_)
    gst_context_unref(shared_context_);
  shared_context_ = ctx;
  if (shared_context_)
    gst_context_ref(shared_context_);
}

GstBusSyncReply StreamDecoder::busSyncHandler(GstBus *bus, GstMessage *msg,
                                              gpointer user_data) {
  auto self = static_cast<StreamDecoder *>(user_data);

  if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_NEED_CONTEXT) {
    const gchar *type;
    gst_message_parse_context_type(msg, &type);
    if (g_strcmp0(type, "gst.cuda.context") == 0) {
      // Protect shared_context_ access - also modified by setSharedContext()
      std::lock_guard<std::mutex> lock(self->frame_mutex_);
      if (self->shared_context_) {
        gst_element_set_context(GST_ELEMENT(GST_MESSAGE_SRC(msg)),
                                self->shared_context_);
        return GST_BUS_DROP;
      }
    }
  }
  return GST_BUS_PASS;
}

bool StreamDecoder::create() {
  std::string id_str =
      std::to_string(id_) + "_" + std::to_string(reconnect_count_);

  // Sync with global failure flag from log sniffer
  if (global_gpu_failure_.load()) {
      hardware_accel_failed_ = true;
  }

  // Force CPU buffer mode if hardware acceleration failed
  // This ensures frames are pushed to cpu_buffer_ instead of trying dead GPU copy
  if (hardware_accel_failed_) {
      cpu_buffer_enabled_ = true;
      std::cout << "[" << id_ << "] Hardware failure detected/persisted - forcing CPU buffer mode" << std::endl;
  }

  // Determine source type: RTSP, file://, or direct file path
  is_file_source_ = false;
  std::string file_path;

  if (url_.find("file://") == 0) {
      is_file_source_ = true;
      file_path = url_.substr(7);  // Remove "file://" prefix
  } else if (url_.find("rtsp://") != 0 && url_.find("rtsps://") != 0) {
      // Check if it's a direct file path (starts with / or contains common video extensions)
      if (url_[0] == '/' || url_.find(".mp4") != std::string::npos ||
          url_.find(".mkv") != std::string::npos || url_.find(".avi") != std::string::npos ||
          url_.find(".mov") != std::string::npos || url_.find(".ts") != std::string::npos) {
          is_file_source_ = true;
          file_path = url_;
      } else {
          std::string err_msg = "Invalid URL scheme (must start with rtsp://, rtsps://, file://, or be a file path): " + url_;
          std::cerr << "[" << id_ << "] [ERROR] " << err_msg << std::endl;
          if (logger_) {
              logger_->logError(rtsp::ErrorCategory::InvalidConfig, err_msg);
          }
          return false;
      }
  }

  if (is_file_source_) {
      std::cout << "[" << id_ << "] Opening file: " << file_path << std::endl;
  } else {
      std::cout << "[" << id_ << "] Connecting to RTSP URL: " << url_ << std::endl;
  }

  pipeline_ = gst_pipeline_new(("pipeline-" + id_str).c_str());
  if (!pipeline_)
    return false;

 
  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
  gst_bus_set_sync_handler(bus, busSyncHandler, this, nullptr);
  gst_object_unref(bus);

  // Create source element based on URL type
  if (is_file_source_) {
    source_ = gst_element_factory_make("filesrc", ("src-" + id_str).c_str());
    demux_ = gst_element_factory_make("qtdemux", ("demux-" + id_str).c_str());
  } else {
    source_ = gst_element_factory_make("rtspsrc", ("src-" + id_str).c_str());
    demux_ = nullptr;
  }

  // 3-tier converter selection: nvvideoconvert (DeepStream) → cudaconvert (NVDEC) → videoconvert (CPU)
  GstElement* convert = nullptr;
  use_nvmm_memory_ = false;
  use_cuda_memory_ = false;

  if (!hardware_accel_failed_) {
    // First, verify GPU is actually accessible via CUDA
    int cuda_device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&cuda_device_count);
    bool cuda_available = (cuda_err == cudaSuccess && cuda_device_count > 0);
    
    if (!cuda_available) {
      hardware_accel_failed_ = true;
      std::cout << "[" << id_ << "] CUDA not available (devices=" << cuda_device_count 
                << ", err=" << cudaGetErrorString(cuda_err) << ") - forcing CPU decoder" << std::endl;
      if (logger_) {
        logger_->logError(rtsp::ErrorCategory::HardwareAccelFailed,
                          "CUDA not available, forcing software decoder");
      }
    }
  }

  if (!hardware_accel_failed_) {
    // Tier 1: Check if DeepStream's nvv4l2decoder is available
    // nvv4l2decoder requires NVIDIA driver to be loaded (/dev/nvidia0)
    GstElementFactory* nvv4l2_factory = gst_element_factory_find("nvv4l2decoder");
    if (nvv4l2_factory) {
      gst_object_unref(nvv4l2_factory);
      
      // Check for NVIDIA driver - /dev/nvidia0 confirms driver is loaded
      // Note: /dev/video* are V4L2 capture devices (webcams), NOT related to NVDEC
      bool nvidia_driver_available = false;
      FILE* fp = fopen("/dev/nvidia0", "r");
      if (fp) {
        fclose(fp);
        nvidia_driver_available = true;
      }
      
      if (!nvidia_driver_available) {
        std::cout << "[" << id_ << "] NVIDIA driver not available (/dev/nvidia0 missing) - skipping nvv4l2decoder" << std::endl;
        if (logger_) {
          logger_->logInfo("NVIDIA driver not found, falling back to software decoder");
        }
        hardware_accel_failed_ = true;  // Force CPU fallback
      } else {
        convert = gst_element_factory_make("nvvideoconvert", ("convert-" + id_str).c_str());
        if (convert) {
          use_nvmm_memory_ = true;
          std::cout << "[" << id_ << "] Using DeepStream NVMM path (nvvideoconvert)" << std::endl;
          if (logger_) {
            logger_->logInfo("Converter selected: nvvideoconvert (DeepStream NVMM)");
          }
        }
      }
    }
  }

  // Tier 2: NVDEC decode outputs NV12 directly - no color conversion needed
  // The nvh264dec element outputs NV12 to CPU memory by default
  // We just need a passthrough element (or skip converter entirely)
  // For now, fall through to CPU videoconvert which handles NV12 passthrough

  // Tier 3: CPU fallback
  if (!convert) {
    convert = gst_element_factory_make("videoconvert", ("convert-" + id_str).c_str());
    std::cout << "[" << id_ << "] Using CPU videoconvert (hardware acceleration unavailable)" << std::endl;
    if (logger_) {
      logger_->logWarning("Hardware acceleration unavailable, using CPU videoconvert");
      if (hardware_accel_failed_) {
        logger_->logError(rtsp::ErrorCategory::HardwareAccelFailed,
                          "GPU decode/convert failed, fell back to CPU");
      }
    }
  }

  appsink_ = gst_element_factory_make("appsink", ("sink-" + id_str).c_str());

  // Validate required elements
  if (!source_ || !convert || !appsink_) {
    destroy();
    return false;
  }
  if (is_file_source_ && !demux_) {
    destroy();
    return false;
  }

  convert_ = convert;

  // Configure source element based on type
  if (is_file_source_) {
    g_object_set(source_, "location", file_path.c_str(), nullptr);
  } else {
    g_object_set(source_, "location", url_.c_str(), "latency", 500,
                 "drop-on-latency", TRUE, "udp-buffer-size", 524288,
                 "protocols", 7,
                 nullptr);
  }

  // Request configured output format with appropriate memory type
  GstCaps *caps;
  std::string caps_str;
  if (use_nvmm_memory_) {
    // DeepStream NVMM memory path
    caps_str = "video/x-raw(memory:NVMM), format=" + output_format_;
  } else if (use_cuda_memory_) {
    // Standard GStreamer CUDA memory path
    caps_str = "video/x-raw(memory:CUDAMemory), format=" + output_format_;
  } else {
    // CPU memory path
    caps_str = "video/x-raw, format=" + output_format_;
  }
  caps = gst_caps_from_string(caps_str.c_str());

  g_object_set(appsink_, "emit-signals", TRUE, "drop", TRUE, "max-buffers", 2,
               "caps", caps, "sync", FALSE, nullptr);
  gst_caps_unref(caps);

  g_signal_connect(appsink_, "new-sample", G_CALLBACK(onNewSample), this);

  // Build pipeline based on source type
  if (is_file_source_) {
    // File pipeline: filesrc -> qtdemux -> (dynamic pad) -> parse -> decode -> convert -> appsink
    gst_bin_add_many(GST_BIN(pipeline_), source_, demux_, convert, appsink_, nullptr);

    if (!gst_element_link(source_, demux_)) {
      std::cerr << "[" << id_ << "] Failed to link filesrc to demux" << std::endl;
      destroy();
      return false;
    }

    if (!gst_element_link(convert, appsink_)) {
      destroy();
      return false;
    }

    // Connect to demux pad-added signal for dynamic linking
    g_signal_connect(demux_, "pad-added", G_CALLBACK(onDemuxPadAdded), this);
  } else {
    // RTSP pipeline: rtspsrc -> (dynamic pad) -> depay -> parse -> decode -> convert -> appsink
    gst_bin_add_many(GST_BIN(pipeline_), source_, convert, appsink_, nullptr);

    if (!gst_element_link(convert, appsink_)) {
      destroy();
      return false;
    }

    g_signal_connect(source_, "pad-added", G_CALLBACK(onPadAdded), this);
  }

  std::cout << "[" << id_ << "] Created pipeline: " << name_ << std::endl;
  if (logger_) {
    logger_->logStateChange(rtsp::CameraState::Connecting,
                            "Pipeline created for " + url_);
  }
  return true;
}

bool StreamDecoder::start() {
  if (running_)
    return true;
  if (!create())
    return false;

  if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) ==
      GST_STATE_CHANGE_FAILURE) {
    if (logger_) {
      logger_->logError(rtsp::ErrorCategory::DecoderInitFailed,
                        "Failed to set pipeline to PLAYING state");
    }
    destroy();
    return false;
  }

  running_ = true;
  pending_first_frame_ = true;  // Will log CONNECTED when first frame arrives
  bus_thread_ = std::thread(&StreamDecoder::busLoop, this);
  
  // Recreate CPU buffer if it was reset during destroy() but cpu_buffer_enabled_ is still true
  // This ensures the buffer exists when frames start arriving after reconnect
  // The buffer will be resized when actual FPS is detected via onParserCaps
  if (cpu_buffer_enabled_ && !cpu_buffer_) {
    size_t initial_capacity = static_cast<size_t>(cpu_buffer_duration_sec_ * 25.0) + 1;
    cpu_buffer_ = std::make_unique<CpuBuffer>(initial_capacity);
    std::cout << "[" << id_ << "] CPU buffer recreated on reconnect. Initial capacity=" << initial_capacity << std::endl;
  }
  
  return true;
}

void StreamDecoder::stop() {
  if (logger_) {
    logger_->logStateChange(rtsp::CameraState::Disconnected,
                            "Stopping stream");
  }
  running_ = false;
  // Unblock any threads waiting for frames
  queue_cv_.notify_all();
  
  if (bus_thread_.joinable())
    bus_thread_.join();
  destroy();
}

void StreamDecoder::destroy() {
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }

  source_ = nullptr;
  demux_ = nullptr;
  depay_ = nullptr;
  parse_ = nullptr;
  decoder_ = nullptr;
  convert_ = nullptr;
  appsink_ = nullptr;
  decoder_linked_ = false;
  is_file_source_ = false;
  last_frame_time_ = 0;

  // Clean up frame queue
  while (!frame_queue_.empty()) {
    gst_sample_unref(frame_queue_.front().sample);
    frame_queue_.pop();
  }
  
  // Deallocate GPU buffer to prevent memory leak on reconnect
  // This releases: CUDA device memory, pinned host memory, stream, and event
  gpu_buffer_.deallocate();
  cuda_device_ptr_ = 0;
  cpu_buffer_.reset();
  
  // Reset FPS tracking - protected by stats_mutex_ since fps_timestamps_ns_ is shared
  {
    std::lock_guard<std::mutex> slock(stats_mutex_);
    fps_timestamps_ns_.clear();
    last_frame_time_ns_ = 0;
    last_pts_ = 0;
    stats_.current_fps = 0.0;
    stats_.instant_fps = 0.0;
  }
}

bool StreamDecoder::recreate() {
  std::cout << "[" << id_ << "] Reconnecting..." << std::endl;
  if (logger_) {
    logger_->logStateChange(rtsp::CameraState::Retrying, "Attempting reconnection #" + std::to_string(reconnect_count_ + 1));
  }
  stop();
  clearError();
  reconnect_count_++;
  pending_reconnect_ = true;  // Will be cleared when first frame received
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.reconnect_count++;
  }
  return start();
}

void StreamDecoder::busLoop() {
  if (!pipeline_)
    return;

  GstBus *bus = gst_element_get_bus(pipeline_);

  while (running_) {
    GstMessage *msg = gst_bus_timed_pop_filtered(
        bus, 100 * GST_MSECOND,
        (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS |
                         GST_MESSAGE_HAVE_CONTEXT));

    if (msg) {
      switch (GST_MESSAGE_TYPE(msg)) {
      case GST_MESSAGE_ERROR: {
        if (!hasError()) {
          GError *err = nullptr;
          gchar *debug = nullptr;
          gst_message_parse_error(msg, &err, &debug);
          std::cerr << "[" << id_ << "] Error: " << err->message << std::endl;
          
          if (logger_) {
            std::string error_msg = err->message;
            // Detect specific error types based on error message patterns
            rtsp::ErrorCategory category = rtsp::ErrorCategory::Unknown;
            
            if (error_msg.find("slice header") != std::string::npos ||
                error_msg.find("bitstream") != std::string::npos ||
                error_msg.find("NAL") != std::string::npos) {
              category = rtsp::ErrorCategory::BitstreamError;
            } else if (error_msg.find("Internal data stream") != std::string::npos ||
                       error_msg.find("Could not read") != std::string::npos ||
                       error_msg.find("Could not open resource") != std::string::npos ||
                       error_msg.find("connection") != std::string::npos ||
                       error_msg.find("socket") != std::string::npos ||
                       error_msg.find("network") != std::string::npos) {
              category = rtsp::ErrorCategory::NetworkError;
            } else if (error_msg.find("timeout") != std::string::npos ||
                       error_msg.find("Timeout") != std::string::npos ||
                       error_msg.find("timed out") != std::string::npos) {
              category = rtsp::ErrorCategory::Timeout;
            } else if (error_msg.find("corrupt") != std::string::npos ||
                       error_msg.find("malformed") != std::string::npos ||
                       error_msg.find("invalid") != std::string::npos) {
              category = rtsp::ErrorCategory::FrameCorruption;
            } else if (error_msg.find("decoder") != std::string::npos ||
                       error_msg.find("Decoder") != std::string::npos ||
                       error_msg.find("nvdec") != std::string::npos ||
                       error_msg.find("nvdecoder") != std::string::npos ||
                       error_msg.find("cuda") != std::string::npos ||
                       error_msg.find("CUDA") != std::string::npos ||
                       error_msg.find("cudacontext") != std::string::npos ||
                       error_msg.find("hardware") != std::string::npos ||
                       error_msg.find("create decoder") != std::string::npos) {
              category = rtsp::ErrorCategory::HardwareAccelFailed;
              // Mark GPU as failed for this stream - will use CPU buffer on reconnect
              hardware_accel_failed_ = true;
              std::cout << "[" << id_ << "] GPU hardware failure detected - will use CPU mode on reconnect" << std::endl;
            }
            
            logger_->logError(category, error_msg, err->code);
            logger_->logStateChange(rtsp::CameraState::StreamLost,
                                    "Pipeline error detected");
          }
          
          markError();
          g_error_free(err);
          g_free(debug);
        }
        break;
      }
      case GST_MESSAGE_EOS:
        std::cout << "[" << id_ << "] EOS" << std::endl;
        if (logger_) {
          logger_->logStateChange(rtsp::CameraState::StreamLost,
                                  "End of stream received");
        }
        markError();
        break;

      case GST_MESSAGE_WARNING: {
        GError *err = nullptr;
        gchar *debug = nullptr;
        gst_message_parse_warning(msg, &err, &debug);
        
        std::string warn_msg = (err && err->message) ? err->message : "Unknown warning";
        if (debug) {
            warn_msg += " (" + std::string(debug) + ")";
        }
        
        // Treat GPU/Decoder warnings as fatal errors to trigger fallback
        if (warn_msg.find("decoder") != std::string::npos ||
            warn_msg.find("nvdec") != std::string::npos ||
            warn_msg.find("nvdecoder") != std::string::npos ||
            warn_msg.find("cuda") != std::string::npos ||
            warn_msg.find("CUDA") != std::string::npos ||
            warn_msg.find("cudacontext") != std::string::npos ||
            warn_msg.find("resource") != std::string::npos) {
            
            std::cout << "[" << id_ << "] Critical GPU warning: " << warn_msg << std::endl;
            
            hardware_accel_failed_ = true;
            markError();
        } else {
            std::cout << "[" << id_ << "] Warning: " << warn_msg << std::endl;
        }

        g_error_free(err);
        g_free(debug);
        break;
      }

      case GST_MESSAGE_HAVE_CONTEXT: {
        GstContext *ctx = nullptr;
        gst_message_parse_have_context(msg, &ctx);
        if (ctx) {
          const gchar *type = gst_context_get_context_type(ctx);
          if (g_strcmp0(type, "gst.cuda.context") == 0 && on_context_found_) {
            on_context_found_(ctx);
          }
          gst_context_unref(ctx);
        }
        break;
      }
      default:
        break;
      }
      gst_message_unref(msg);
    }
  }
  gst_object_unref(bus);
}

GpuFrameInfo StreamDecoder::getGpuFrame(int timeout_ms) {
  std::unique_lock<std::mutex> lock(frame_mutex_);
  
  // Wait if queue is empty and timeout is requested
  if (frame_queue_.empty() && timeout_ms > 0 && running_) {
    queue_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                       [this] { return !frame_queue_.empty() || !running_; });
  }

  GpuFrameInfo info;
  info.valid = false;

  // Pop from queue instead of reading single buffer
  if (!frame_queue_.empty()) {
    QueuedFrame &qf = frame_queue_.front();

    // Map the sample to get GPU pointer
    GstBuffer *buffer = gst_sample_get_buffer(qf.sample);
    if (buffer) {
      // NVMM memory path (DeepStream) - extract pointer from NvBufSurface
      if (use_nvmm_memory_) {
#ifdef HAVE_DEEPSTREAM
        // DeepStream NVMM: Access NvBufSurface via DMA-BUF file descriptor
        // This is the correct approach for DeepStream 7.x+
        GstMemory *mem = gst_buffer_peek_memory(buffer, 0);
        if (mem && gst_is_dmabuf_memory(mem)) {
          int dmabuf_fd = gst_dmabuf_memory_get_fd(mem);
          NvBufSurface *surf = nullptr;
          
          if (NvBufSurfaceFromFd(dmabuf_fd, (void **)&surf) == 0 && surf) {
            if (surf->numFilled > 0 && surf->surfaceList[0].dataPtr) {
              info.ptr = (uint64_t)surf->surfaceList[0].dataPtr;
              info.width = surf->surfaceList[0].width;
              info.height = surf->surfaceList[0].height;
              info.stride = surf->surfaceList[0].pitch;
              info.size = surf->surfaceList[0].dataSize;
              info.frame_id = qf.frame_id;
              info.format = output_format_;
              info.valid = true;
              
              cuda_device_ptr_ = info.ptr;
              current_stride_ = info.stride;
            }
          }
        }
#else
        // Non-DeepStream build: NVMM path selected but no NvBufSurface support
        // This is a configuration error - NVMM requires DeepStream SDK
        static std::once_flag warn_flag;
        std::call_once(warn_flag, [this]() {
          std::cerr << "[" << id_ << "] WARNING: NVMM path selected but HAVE_DEEPSTREAM not defined" << std::endl;
        });
#endif
      }
      // CUDA memory path (requires HAVE_GST_CUDA headers)
#ifdef HAVE_GST_CUDA
      else if (use_cuda_memory_) {
        GstMemory *mem = gst_buffer_peek_memory(buffer, 0);
        if (mem && gst_is_cuda_memory(mem)) {
          GstMapInfo map;
          if (gst_memory_map(mem, &map,
                             (GstMapFlags)(GST_MAP_READ | GST_MAP_CUDA))) {
            info.ptr = reinterpret_cast<uint64_t>(map.data);
            info.width = qf.width;
            info.height = qf.height;
            info.stride = qf.stride;
            info.size = qf.data_size;
            info.frame_id = qf.frame_id;
            info.format = output_format_;
            info.valid = true;

            // Store stride for internal use
            cuda_device_ptr_ = info.ptr;
            current_stride_ = qf.stride;

            gst_memory_unmap(mem, &map);
          }
        }
      }
#endif
      if (!info.valid && gpu_buffer_.isReady()) {
        // Fallback path (CPU copy to GPU)
        info.ptr = gpu_buffer_.devicePtrAsInt();
        info.width = gpu_buffer_.width();
        info.height = gpu_buffer_.height();
        info.stride = gpu_buffer_.width();
        info.size = gpu_buffer_.dataSize();
        info.frame_id = qf.frame_id;
        info.format = output_format_;
        info.valid = true;
      }
    }

    if (info.valid) {
      // Release the sample and remove from queue
      gst_sample_unref(qf.sample);
      frame_queue_.pop();

      std::lock_guard<std::mutex> slock(stats_mutex_);
      stats_.frames_consumed++;
      stats_.queue_depth = frame_queue_.size();
    }
  }

  return info;
}

FrameStats StreamDecoder::getStats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return stats_;
}

GstFlowReturn StreamDecoder::onNewSample(GstElement *sink, gpointer data) {
  auto self = static_cast<StreamDecoder *>(data);

  GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
  if (!sample)
    return GST_FLOW_OK;

  // Track that we received a frame from GStreamer
  {
    std::lock_guard<std::mutex> slock(self->stats_mutex_);
    self->stats_.frames_received++;
  }
  
  // Update watchdog timestamp
  self->last_frame_rx_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();

  // Check if this is first frame after initial connection or reconnection
  if (self->pending_first_frame_) {
    self->pending_first_frame_ = false;
    std::cout << "[" << self->id_ << "] Connected (first frame received)" << std::endl;
    if (self->logger_) {
      self->logger_->logStateChange(rtsp::CameraState::Connected,
                                    "First frame received - connection confirmed");
    }
  } else if (self->pending_reconnect_) {
    self->pending_reconnect_ = false;
    std::cout << "[" << self->id_ << "] Recovered (first frame received)" << std::endl;
    if (self->logger_) {
      self->logger_->logStateChange(rtsp::CameraState::Reconnected,
                                    "Successfully recovered - first frame received");
    }
  }

  GstBuffer *buffer = gst_sample_get_buffer(sample);
  GstCaps *caps = gst_sample_get_caps(sample);
  GstVideoInfo info;
  
  // Guard: caps can be NULL during pipeline negotiation
  if (!caps) {
    gst_sample_unref(sample);
    return GST_FLOW_OK;
  }

  // Check for duplicate frame using PTS (presentation timestamp)
  uint64_t pts = GST_BUFFER_PTS(buffer);
  if (pts != GST_CLOCK_TIME_NONE && pts == self->last_pts_ && self->last_pts_ != 0) {
    // Duplicate frame detected - skip it
    std::lock_guard<std::mutex> slock(self->stats_mutex_);
    self->stats_.frames_duplicate++;
    gst_sample_unref(sample);
    return GST_FLOW_OK;
  }
  self->last_pts_ = pts;

  if (gst_video_info_from_caps(&info, caps)) {
    int w = GST_VIDEO_INFO_WIDTH(&info);
    int h = GST_VIDEO_INFO_HEIGHT(&info);
    size_t frame_size = (size_t)(w * h * 1.5); // NV12

    std::lock_guard<std::mutex> lock(self->frame_mutex_);
    bool success = false;

    // NVMM memory path (DeepStream) - extract pointer from NvBufSurface via DMA-BUF
    if (self->use_nvmm_memory_) {
#ifdef HAVE_DEEPSTREAM
      // DeepStream NVMM: Access NvBufSurface via DMA-BUF file descriptor
      // This is the correct approach for DeepStream 7.x+
      GstMemory *mem = gst_buffer_peek_memory(buffer, 0);
      if (mem && gst_is_dmabuf_memory(mem)) {
        int dmabuf_fd = gst_dmabuf_memory_get_fd(mem);
        NvBufSurface *surf = nullptr;
        
        if (NvBufSurfaceFromFd(dmabuf_fd, (void **)&surf) == 0 && surf) {
          if (surf->numFilled > 0 && surf->surfaceList[0].dataPtr) {
            self->cuda_device_ptr_ = (uint64_t)surf->surfaceList[0].dataPtr;
            self->current_stride_ = surf->surfaceList[0].pitch;
            success = true;
          }
        }
      }
#else
      // Non-DeepStream build: NVMM path selected but no NvBufSurface support
      // This is a configuration error - log warning once and fall through to CPU
      static std::once_flag warn_flag;
      std::call_once(warn_flag, [self]() {
        std::cerr << "[" << self->id_ << "] WARNING: NVMM path selected but HAVE_DEEPSTREAM not defined" << std::endl;
      });
#endif
    }

    // CUDA memory path (requires HAVE_GST_CUDA headers)
#ifdef HAVE_GST_CUDA
    if (!success && self->use_cuda_memory_) {
      GstMemory *mem = gst_buffer_peek_memory(buffer, 0);
      if (mem && gst_is_cuda_memory(mem)) {
        GstCudaMemory *cuda_mem = GST_CUDA_MEMORY_CAST(mem);
        GstMapInfo map;
        if (gst_memory_map(mem, &map,
                           (GstMapFlags)(GST_MAP_READ | GST_MAP_CUDA))) {
          self->cuda_device_ptr_ = reinterpret_cast<uint64_t>(map.data);
          int inferred_stride = (int)(map.size / (h * 1.5));

          if (inferred_stride > info.stride[0] &&
              (map.size % (int)(h * 1.5) == 0)) {
            self->current_stride_ = inferred_stride;
          } else {
            self->current_stride_ = info.stride[0];
          }

          success = true;
          gst_memory_unmap(mem, &map);
        }
      }
    }
#endif

    // CPU buffer mode: Direct path without GPU buffer (for when GPU is unavailable)
    if (!success && self->cpu_buffer_enabled_ && self->cpu_buffer_) {
      // CPU buffer mode doesn't need gpu_buffer_, just mark success
      // The actual data copy happens in pushToCpuBuffer()
      self->current_stride_ = w;  // Packed stride
      success = true;
    }

    // Fallback: CPU memory path with copy to GPU (for GPU queue mode only)
    if (!success && !self->cpu_buffer_enabled_) {
      GstMapInfo map;
      if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        size_t required_size = (size_t)(w * h * 1.5); // NV12

        // Dynamically allocate or reallocate GPU buffer as needed
        if (!self->gpu_buffer_.isAllocated() ||
            self->gpu_buffer_.size() < required_size) {
          self->gpu_buffer_.allocate(required_size);
        }

        if (self->gpu_buffer_.isAllocated() &&
            self->gpu_buffer_.copyToDevice(map.data, map.size, w, h)) {
          self->cuda_device_ptr_ = 0; // Use gpu_buffer_ path
          self->current_stride_ = w;  // Fallback is packed
          success = true;
        }
        gst_buffer_unmap(buffer, &map);
      }
    }

    if (!success) {
        static std::atomic<int> log_counter{0};
        int count = log_counter.fetch_add(1);
        if (count < 50) {
             std::cout << "[" << self->id_ << "] Processing FAIL. CUDA=" << self->use_cuda_memory_ 
                       << ", CPU_En=" << self->cpu_buffer_enabled_ 
                       << ", Buf=" << (self->cpu_buffer_ ? "YES" : "NULL") << std::endl;
        }
    }

    if (success) {
      // CPU buffer mode OR GPU queue mode (mutually exclusive)
      if (self->cpu_buffer_enabled_ && self->cpu_buffer_) {
        // ...

        // CPU buffer mode: copy frame to ring buffer
        {
          std::lock_guard<std::mutex> slock(self->stats_mutex_);
          self->stats_.frames_decoded++;
        }
        self->pushToCpuBuffer(buffer, w, h, ++self->frame_counter_);
        self->updateFps();
        self->updateFrameTime();
        gst_sample_unref(sample);
        return GST_FLOW_OK;
      } else {
        // GPU queue mode: queue-based buffering for zero overwrites
        // Original pattern: stats_mutex_ protects both stats AND frame_queue_
        {
          std::lock_guard<std::mutex> slock(self->stats_mutex_);
          self->stats_.frames_decoded++;

          // If queue is full, drop oldest frame to make room
          if (self->frame_queue_.size() >= self->max_queue_depth_) {
            QueuedFrame &oldest = self->frame_queue_.front();
            gst_sample_unref(oldest.sample);
            self->frame_queue_.pop();
            self->stats_.frames_dropped_queue++;
          }

          // Create queued frame entry
          QueuedFrame qf;
          qf.sample = sample; // Transfer ownership to queue
          qf.frame_id = ++self->frame_counter_;
          qf.width = w;
          qf.height = h;
          qf.stride = self->current_stride_;
          qf.data_size = frame_size;
          qf.cuda_ptr = self->cuda_device_ptr_;
          qf.timestamp_ns = GST_BUFFER_PTS(buffer);

          self->frame_queue_.push(qf);
          self->queue_cv_.notify_one(); 

          // Update stats
          size_t depth = self->frame_queue_.size();
          self->stats_.queue_depth = depth;
          if (depth > self->stats_.queue_max_depth) {
            self->stats_.queue_max_depth = depth;
          }
        }

        self->updateFps();
        self->updateFrameTime();
        return GST_FLOW_OK;
      }
    } else {
      // Track decode/mapping failure
      std::lock_guard<std::mutex> slock(self->stats_mutex_);
      self->stats_.frames_dropped_decode++;
      self->stats_.decode_errors++;
    }
  }

  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

void StreamDecoder::updateFps() {
  auto now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
  
  // Detect large gaps (>2 seconds) indicating disconnect/reconnect
  // Reset FPS tracking on gap to avoid incorrect calculations
  static constexpr int64_t GAP_THRESHOLD_NS = 2000000000LL;  // 2 seconds
  
  // Protect fps_timestamps_ns_ with stats_mutex_ to prevent data races
  std::lock_guard<std::mutex> lock(stats_mutex_);
  
  if (last_frame_time_ns_ > 0) {
    int64_t delta_ns = now_ns - last_frame_time_ns_;
    
    if (delta_ns > GAP_THRESHOLD_NS) {
      // Large gap detected - likely a reconnection
      // Reset sliding window to start fresh
      fps_timestamps_ns_.clear();
      // Don't calculate instant FPS from stale data
    } else if (delta_ns > 0) {
      // Normal case - calculate instant FPS
      stats_.instant_fps = 1e9 / static_cast<double>(delta_ns);
    }
  }
  last_frame_time_ns_ = now_ns;
  
  // Sliding window FPS calculation
  fps_timestamps_ns_.push_back(now_ns);
  
  // Remove timestamps older than the window (1 second)
  int64_t cutoff_ns = now_ns - static_cast<int64_t>(FPS_WINDOW_SEC * 1e9);
  while (!fps_timestamps_ns_.empty() && fps_timestamps_ns_.front() < cutoff_ns) {
    fps_timestamps_ns_.pop_front();
  }
  
  // Limit buffer size to prevent unbounded growth
  while (fps_timestamps_ns_.size() > FPS_WINDOW_SIZE) {
    fps_timestamps_ns_.pop_front();
  }
  
  // Calculate FPS using rate formula: (count - 1) / duration
  // This correctly counts intervals, not endpoints
  if (fps_timestamps_ns_.size() >= 2) {
    int64_t duration_ns = fps_timestamps_ns_.back() - fps_timestamps_ns_.front();
    double duration_sec = static_cast<double>(duration_ns) / 1e9;
    
    // Only use rate calculation if window has at least 0.5s of data
    // This prevents wild fluctuations during startup
    if (duration_sec >= 0.5) {
      stats_.current_fps = static_cast<double>(fps_timestamps_ns_.size() - 1) / duration_sec;
    }
  }
}

GstPadProbeReturn StreamDecoder::onParserCaps(GstPad *pad,
                                              GstPadProbeInfo *info,
                                              gpointer user_data) {
  auto self = static_cast<StreamDecoder *>(user_data);
  GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);

  if (GST_EVENT_TYPE(event) == GST_EVENT_CAPS) {
    GstCaps *caps;
    gst_event_parse_caps(event, &caps);
    if (caps) {
      GstStructure *s = gst_caps_get_structure(caps, 0);
      int fps_n = 0, fps_d = 1;
      int width = 0, height = 0;
      double new_fps = 0.0;
      
      // Parse metadata
      gst_structure_get_int(s, "width", &width);
      gst_structure_get_int(s, "height", &height);
      
      // Skip invalid 0x0 caps (can occur during pipeline negotiation)
      if (width <= 0 || height <= 0) {
        return GST_PAD_PROBE_OK;
      }
      
      if (gst_structure_get_fraction(s, "framerate", &fps_n, &fps_d)) {
        if (fps_d > 0) {
          new_fps = (double)fps_n / (double)fps_d;
        }
      }

      bool perform_update = false;
      bool is_startup = false;
      std::string log_message;
      
      {
          std::lock_guard<std::mutex> lock(self->stats_mutex_);
          // Capture old values for comparison
          int old_w = self->stats_.source_width;
          int old_h = self->stats_.source_height;
          double old_fps = self->stats_.source_fps;
          
          // Check if this is the first detection (Startup)
          if (old_w == 0 && old_h == 0) {
              is_startup = true;
              perform_update = true;
              
              std::ostringstream oss;
              oss << "Source detected: " << width << "x" << height << " @ " << new_fps << " fps";
              log_message = oss.str();
          } 
          // Check for Runtime Changes
          else if (width != old_w || height != old_h || std::abs(new_fps - old_fps) > 0.1) {
              perform_update = true;
              is_startup = false;
              
              std::ostringstream oss;
              oss << "Stream format changed: ";
              bool added = false;
              
              if (width != old_w || height != old_h) {
                  oss << "Resolution " << old_w << "x" << old_h << " -> " << width << "x" << height;
                  added = true;
              }
              
              if (std::abs(new_fps - old_fps) > 0.1) {
                  if (added) oss << ", ";
                  oss << "FPS " << old_fps << " -> " << new_fps;
              }
              log_message = oss.str();
          }

          // Apply updates if needed (inside lock)
          if (perform_update) {
            self->stats_.source_fps = new_fps;
            self->stats_.source_width = width;
            self->stats_.source_height = height;
          }
      } 

      // Log and Side Effects (outside lock)
      if (perform_update) {
          std::cout << "[" << self->id_ << "] " << log_message << std::endl;

          if (self->logger_) {
             if (is_startup) {
                 self->logger_->logInfo(log_message);
             } else {
                 // Runtime changes are logged as State Changes
                 self->logger_->logStateChange(rtsp::CameraState::Connected, log_message);
             }
          }
          
          // Reset FPS window on any format change to allow instant stats update
          {
            std::lock_guard<std::mutex> slock(self->stats_mutex_);
            self->fps_timestamps_ns_.clear();
          }
          
          // Resize buffer if FPS changed (or startup), with default fallback
          double fps_to_configure = (new_fps > 0) ? new_fps : 25.0;
          self->resizeCpuBuffer(fps_to_configure);
          
          // Update appsink caps if resolution changed
          if (width > 0 && height > 0 && self->appsink_) {
             std::string caps_str;
             if (self->use_cuda_memory_) {
               caps_str = "video/x-raw(memory:CUDAMemory), format=" + self->output_format_ +
                          ", width=" + std::to_string(width) +
                          ", height=" + std::to_string(height);
             } else {
               caps_str = "video/x-raw, format=" + self->output_format_ +
                          ", width=" + std::to_string(width) +
                          ", height=" + std::to_string(height);
             }
             
             GstCaps* new_caps = gst_caps_from_string(caps_str.c_str());
             if (new_caps) {
               g_object_set(self->appsink_, "caps", new_caps, nullptr);
               gst_caps_unref(new_caps);
             }
          }
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

void StreamDecoder::onPadAdded(GstElement *element, GstPad *pad,
                               gpointer data) {
  auto self = static_cast<StreamDecoder *>(data);
  
  // Sync with global failure flag
  if (global_gpu_failure_.load()) {
      self->hardware_accel_failed_ = true;
  }

  if (self->decoder_linked_)
    return;

  GstCaps *caps = gst_pad_get_current_caps(pad);
  if (!caps)
    caps = gst_pad_query_caps(pad, nullptr);

  GstStructure *str = gst_caps_get_structure(caps, 0);
  const gchar *name = gst_structure_get_name(str);

  if (g_str_has_prefix(name, "application/x-rtp")) {
    const gchar *media = gst_structure_get_string(str, "media");
    if (media && g_strcmp0(media, "video") == 0) {

      const gchar *encoding = gst_structure_get_string(str, "encoding-name");
      std::string id_str = std::to_string(self->id_) + "_" +
                           std::to_string(self->reconnect_count_);
      bool is_h265 = false;

      if (encoding) {
        std::string enc(encoding);
        is_h265 = (enc == "H265" || enc == "HEVC");
        std::cout << "[" << self->id_ << "] Codec: " << encoding << " ("
                  << (is_h265 ? "H265" : "H264") << ")" << std::endl;
      }

      if (is_h265) {
        self->depay_ = gst_element_factory_make("rtph265depay",
                                                ("depay-" + id_str).c_str());
        self->parse_ =
            gst_element_factory_make("h265parse", ("parse-" + id_str).c_str());
        
        // Add probe to parser src pad to get FPS
        if (self->parse_) {
            GstPad *src_pad = gst_element_get_static_pad(self->parse_, "src");
            if (src_pad) {
                gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                                  onParserCaps, self, nullptr);
                gst_object_unref(src_pad);
            }
        }

        // 3-tier decoder selection: nvv4l2decoder → nvh265dec → avdec_h265
        self->decoder_ = nullptr;
        
        // Tier 1: DeepStream nvv4l2decoder (if NVMM path active)
        if (self->use_nvmm_memory_ && !self->hardware_accel_failed_) {
          self->decoder_ = gst_element_factory_make("nvv4l2decoder", ("decode-" + id_str).c_str());
          if (self->decoder_) {
            self->active_decoder_type_ = DecoderType::NVV4L2_NVMM;
            std::cout << "[" << self->id_ << "] Using nvv4l2decoder (DeepStream NVMM) for H265" << std::endl;
            if (self->logger_) {
              self->logger_->logInfo("Decoder selected: NVV4L2_NVMM (DeepStream) for H265");
            }
          }
        }
        
        // Tier 2: Standard NVDEC
        if (!self->decoder_ && !self->hardware_accel_failed_) {
          self->decoder_ = gst_element_factory_make("nvh265dec", ("decode-" + id_str).c_str());
          if (self->decoder_) {
            g_object_set(self->decoder_, "num-output-surfaces", 1, nullptr);
            self->active_decoder_type_ = DecoderType::NVDEC_CUDA;
            std::cout << "[" << self->id_ << "] Using nvh265dec (NVDEC CUDA) for H265" << std::endl;
            if (self->logger_) {
              self->logger_->logInfo("Decoder selected: NVDEC_CUDA for H265");
            }
          }
        }
        
        // Tier 3: CPU fallback
        if (!self->decoder_) {
          self->decoder_ = gst_element_factory_make("avdec_h265", ("decode-" + id_str).c_str());
          self->active_decoder_type_ = DecoderType::AVDEC_CPU;
          std::cout << "[" << self->id_ << "] Using avdec_h265 (CPU) for H265" << std::endl;
          if (self->logger_) {
            self->logger_->logWarning("Hardware acceleration unavailable, falling back to AVDEC_CPU for H265");
            if (self->hardware_accel_failed_) {
              self->logger_->logError(rtsp::ErrorCategory::HardwareAccelFailed,
                                      "GPU decoder failed, using CPU decoder for H265");
            }
          }
        }
      } else {
        self->depay_ = gst_element_factory_make("rtph264depay",
                                                ("depay-" + id_str).c_str());
        self->parse_ =
            gst_element_factory_make("h264parse", ("parse-" + id_str).c_str());
        
        // Add probe to parser src pad to get FPS
        if (self->parse_) {
            GstPad *src_pad = gst_element_get_static_pad(self->parse_, "src");
            if (src_pad) {
                gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                                  onParserCaps, self, nullptr);
                gst_object_unref(src_pad);
            }
        }

        // 3-tier decoder selection: nvv4l2decoder → nvh264dec → avdec_h264
        self->decoder_ = nullptr;
        
        // Tier 1: DeepStream nvv4l2decoder (if NVMM path active)
        if (self->use_nvmm_memory_ && !self->hardware_accel_failed_) {
          self->decoder_ = gst_element_factory_make("nvv4l2decoder", ("decode-" + id_str).c_str());
          if (self->decoder_) {
            self->active_decoder_type_ = DecoderType::NVV4L2_NVMM;
            std::cout << "[" << self->id_ << "] Using nvv4l2decoder (DeepStream NVMM) for H264" << std::endl;
            if (self->logger_) {
              self->logger_->logInfo("Decoder selected: NVV4L2_NVMM (DeepStream) for H264");
            }
          }
        }
        
        // Tier 2: Standard NVDEC
        if (!self->decoder_ && !self->hardware_accel_failed_) {
          self->decoder_ = gst_element_factory_make("nvh264dec", ("decode-" + id_str).c_str());
          if (self->decoder_) {
            g_object_set(self->decoder_, "num-output-surfaces", 1, nullptr);
            self->active_decoder_type_ = DecoderType::NVDEC_CUDA;
            std::cout << "[" << self->id_ << "] Using nvh264dec (NVDEC CUDA) for H264" << std::endl;
            if (self->logger_) {
              self->logger_->logInfo("Decoder selected: NVDEC_CUDA for H264");
            }
          }
        }
        
        // Tier 3: CPU fallback
        if (!self->decoder_) {
          self->decoder_ = gst_element_factory_make("avdec_h264", ("decode-" + id_str).c_str());
          self->active_decoder_type_ = DecoderType::AVDEC_CPU;
          std::cout << "[" << self->id_ << "] Using avdec_h264 (CPU) for H264" << std::endl;
          if (self->logger_) {
            self->logger_->logWarning("Hardware acceleration unavailable, falling back to AVDEC_CPU for H264");
            if (self->hardware_accel_failed_) {
              self->logger_->logError(rtsp::ErrorCategory::HardwareAccelFailed,
                                      "GPU decoder failed, using CPU decoder for H264");
            }
          }
        }
      }

      if (self->depay_ && self->parse_ && self->decoder_) {
        g_object_set(self->parse_, "config-interval", -1, nullptr);
        gst_bin_add_many(GST_BIN(self->pipeline_), self->depay_, self->parse_,
                         self->decoder_, nullptr);

        if (gst_element_link_many(self->depay_, self->parse_, self->decoder_,
                                  self->convert_, nullptr)) {
          gst_element_sync_state_with_parent(self->depay_);
          gst_element_sync_state_with_parent(self->parse_);
          gst_element_sync_state_with_parent(self->decoder_);

          GstPad *sink = gst_element_get_static_pad(self->depay_, "sink");
          if (sink) {
            gst_pad_link(pad, sink);
            gst_object_unref(sink);
          }
          self->decoder_linked_ = true;
        } else {
          // Linking failed - remove elements from bin and unref to prevent leak
          gst_bin_remove_many(GST_BIN(self->pipeline_), self->depay_,
                              self->parse_, self->decoder_, nullptr);
          // MUST unref after removing from bin (bin doesn't own the ref after remove)
          gst_object_unref(self->depay_);
          gst_object_unref(self->parse_);
          gst_object_unref(self->decoder_);
          self->depay_ = nullptr;
          self->parse_ = nullptr;
          self->decoder_ = nullptr;
        }
      } else {
        // Partial creation - clean up any successfully created elements
        if (self->depay_) {
          gst_object_unref(self->depay_);
          self->depay_ = nullptr;
        }
        if (self->parse_) {
          gst_object_unref(self->parse_);
          self->parse_ = nullptr;
        }
        if (self->decoder_) {
          gst_object_unref(self->decoder_);
          self->decoder_ = nullptr;
        }
      }
    }
  }
  gst_caps_unref(caps);
}

// Callback for file source demuxer (qtdemux) pad-added
void StreamDecoder::onDemuxPadAdded(GstElement *element, GstPad *pad,
                                    gpointer data) {
  auto self = static_cast<StreamDecoder *>(data);

  // Sync with global failure flag
  if (global_gpu_failure_.load()) {
      self->hardware_accel_failed_ = true;
  }

  if (self->decoder_linked_)
    return;

  GstCaps *caps = gst_pad_get_current_caps(pad);
  if (!caps)
    caps = gst_pad_query_caps(pad, nullptr);

  GstStructure *str = gst_caps_get_structure(caps, 0);
  const gchar *name = gst_structure_get_name(str);

  // Check for video stream (video/x-h264, video/x-h265, video/mpeg, etc.)
  if (g_str_has_prefix(name, "video/")) {
    std::string id_str = std::to_string(self->id_) + "_" +
                         std::to_string(self->reconnect_count_);
    bool is_h265 = (g_strcmp0(name, "video/x-h265") == 0);
    bool is_h264 = (g_strcmp0(name, "video/x-h264") == 0);

    std::cout << "[" << self->id_ << "] File codec: " << name << std::endl;

    if (is_h265) {
      self->parse_ = gst_element_factory_make("h265parse", ("parse-" + id_str).c_str());

      // Add probe to parser src pad to get FPS
      if (self->parse_) {
          GstPad *src_pad = gst_element_get_static_pad(self->parse_, "src");
          if (src_pad) {
              gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                                onParserCaps, self, nullptr);
              gst_object_unref(src_pad);
          }
      }

      // 3-tier decoder selection for H265
      self->decoder_ = nullptr;

      if (self->use_nvmm_memory_ && !self->hardware_accel_failed_) {
        self->decoder_ = gst_element_factory_make("nvv4l2decoder", ("decode-" + id_str).c_str());
        if (self->decoder_) {
          self->active_decoder_type_ = DecoderType::NVV4L2_NVMM;
          std::cout << "[" << self->id_ << "] Using nvv4l2decoder (DeepStream NVMM) for H265" << std::endl;
        }
      }

      if (!self->decoder_ && !self->hardware_accel_failed_) {
        self->decoder_ = gst_element_factory_make("nvh265dec", ("decode-" + id_str).c_str());
        if (self->decoder_) {
          g_object_set(self->decoder_, "num-output-surfaces", 1, nullptr);
          self->active_decoder_type_ = DecoderType::NVDEC_CUDA;
          std::cout << "[" << self->id_ << "] Using nvh265dec (NVDEC CUDA) for H265" << std::endl;
        }
      }

      if (!self->decoder_) {
        self->decoder_ = gst_element_factory_make("avdec_h265", ("decode-" + id_str).c_str());
        self->active_decoder_type_ = DecoderType::AVDEC_CPU;
        std::cout << "[" << self->id_ << "] Using avdec_h265 (CPU) for H265" << std::endl;
      }
    } else if (is_h264) {
      self->parse_ = gst_element_factory_make("h264parse", ("parse-" + id_str).c_str());

      // Add probe to parser src pad to get FPS
      if (self->parse_) {
          GstPad *src_pad = gst_element_get_static_pad(self->parse_, "src");
          if (src_pad) {
              gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                                onParserCaps, self, nullptr);
              gst_object_unref(src_pad);
          }
      }

      // 3-tier decoder selection for H264
      self->decoder_ = nullptr;

      if (self->use_nvmm_memory_ && !self->hardware_accel_failed_) {
        self->decoder_ = gst_element_factory_make("nvv4l2decoder", ("decode-" + id_str).c_str());
        if (self->decoder_) {
          self->active_decoder_type_ = DecoderType::NVV4L2_NVMM;
          std::cout << "[" << self->id_ << "] Using nvv4l2decoder (DeepStream NVMM) for H264" << std::endl;
        }
      }

      if (!self->decoder_ && !self->hardware_accel_failed_) {
        self->decoder_ = gst_element_factory_make("nvh264dec", ("decode-" + id_str).c_str());
        if (self->decoder_) {
          g_object_set(self->decoder_, "num-output-surfaces", 1, nullptr);
          self->active_decoder_type_ = DecoderType::NVDEC_CUDA;
          std::cout << "[" << self->id_ << "] Using nvh264dec (NVDEC CUDA) for H264" << std::endl;
        }
      }

      if (!self->decoder_) {
        self->decoder_ = gst_element_factory_make("avdec_h264", ("decode-" + id_str).c_str());
        self->active_decoder_type_ = DecoderType::AVDEC_CPU;
        std::cout << "[" << self->id_ << "] Using avdec_h264 (CPU) for H264" << std::endl;
      }
    } else {
      // Unsupported codec - try decodebin as fallback
      std::cout << "[" << self->id_ << "] Unsupported codec: " << name << ", skipping" << std::endl;
      gst_caps_unref(caps);
      return;
    }

    // Link: demux -> parse -> decoder -> [cudaupload ->] convert
    if (self->parse_ && self->decoder_) {
      g_object_set(self->parse_, "config-interval", -1, nullptr);
      gst_bin_add_many(GST_BIN(self->pipeline_), self->parse_, self->decoder_, nullptr);

      if (gst_element_link_many(self->parse_, self->decoder_, self->convert_, nullptr)) {
        gst_element_sync_state_with_parent(self->parse_);
        gst_element_sync_state_with_parent(self->decoder_);

        GstPad *sink = gst_element_get_static_pad(self->parse_, "sink");
        if (sink) {
          GstPadLinkReturn ret = gst_pad_link(pad, sink);
          if (ret != GST_PAD_LINK_OK) {
            std::cerr << "[" << self->id_ << "] Failed to link demux to parse: " << ret << std::endl;
          }
          gst_object_unref(sink);
        }
        self->decoder_linked_ = true;
        std::cout << "[" << self->id_ << "] File decoder pipeline linked successfully" << std::endl;
      } else {
        std::cerr << "[" << self->id_ << "] Failed to link parse -> decoder -> convert" << std::endl;
        gst_bin_remove_many(GST_BIN(self->pipeline_), self->parse_, self->decoder_, nullptr);
        gst_object_unref(self->parse_);
        gst_object_unref(self->decoder_);
        self->parse_ = nullptr;
        self->decoder_ = nullptr;
      }
    } else {
      if (self->parse_) {
        gst_object_unref(self->parse_);
        self->parse_ = nullptr;
      }
      if (self->decoder_) {
        gst_object_unref(self->decoder_);
        self->decoder_ = nullptr;
      }
    }
  }
  gst_caps_unref(caps);
}

// CPU Buffer Methods

void StreamDecoder::setCpuBufferConfig(bool enabled, double duration_sec, double fps) {

  cpu_buffer_enabled_ = enabled;
  cpu_buffer_duration_sec_ = duration_sec;
  
  std::lock_guard<std::mutex> lock(frame_mutex_);
  if (enabled) {
    if (!cpu_buffer_) {
        // Start with minimal capacity
        size_t initial_capacity = static_cast<size_t>(fps);
        cpu_buffer_ = std::make_unique<CpuBuffer>(initial_capacity);

    }
  } else {

    cpu_buffer_.reset();
  }
}

void StreamDecoder::resizeCpuBuffer(double detected_fps) {

  
  if (!cpu_buffer_enabled_ || detected_fps <= 0) return;
  
  size_t new_capacity = static_cast<size_t>(cpu_buffer_duration_sec_ * detected_fps) + 1;

  {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      
      size_t current_capacity = cpu_buffer_ ? cpu_buffer_->capacity() : 0;
      
      // Only resize if capacity changed significantly (more than 10% difference)
      if (current_capacity > 0) {
        double diff_ratio = std::abs((double)new_capacity - (double)current_capacity) / (double)current_capacity;
        if (diff_ratio < 0.1) {
          return;  // Capacity is close enough, no need to resize
        }
      }
      
      // Resize existing buffer to preserve frames, or create new if needed
      if (cpu_buffer_) {
        cpu_buffer_->resize(new_capacity);
      } else {
        cpu_buffer_ = std::make_unique<CpuBuffer>(new_capacity);
      }
  }


  
  if (logger_) {
    std::ostringstream oss;
    oss << "CPU buffer resized to " << new_capacity << " frames for " << detected_fps << " fps";
    logger_->logInfo(oss.str());
  }
}

void StreamDecoder::enableCpuBufferFallback() {
  if (cpu_buffer_enabled_) {
    return;  // Already enabled
  }
  
  std::cout << "[" << id_ << "] Enabling CPU buffer fallback due to GPU unavailability" << std::endl;
  if (logger_) {
    logger_->logInfo("GPU unavailable - switching to CPU buffer mode");
  }
  
  // Enable CPU buffer with default settings (will be resized when FPS is detected)
  setCpuBufferConfig(true, cpu_buffer_duration_sec_, 25.0);
}

void StreamDecoder::pushToCpuBuffer(GstBuffer* buffer, int width, int height, uint64_t frame_id) {
  // PRE: Caller (onNewSample) must hold frame_mutex_
  // DO NOT acquire frame_mutex_ here - it would cause deadlock (std::mutex is non-recursive)
  if (!cpu_buffer_) return;
  
  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    return;
  }
  
  CpuFrame frame;
  frame.width = width;
  frame.height = height;
  frame.frame_id = frame_id;
  frame.timestamp_ns = GST_BUFFER_PTS(buffer);
  frame.capture_time_ns = std::chrono::steady_clock::now().time_since_epoch().count();
  frame.format = output_format_;
  frame.valid = true;
  
  // Calculate expected packed size and actual stride
  size_t expected_size = 0;
  int bytes_per_pixel = 0;
  
  if (output_format_ == "NV12" || output_format_ == "I420") {
    expected_size = static_cast<size_t>(width * height * 1.5);
    bytes_per_pixel = 1;  // For Y plane
  } else if (output_format_ == "RGB" || output_format_ == "BGR") {
    expected_size = static_cast<size_t>(width * height * 3);
    bytes_per_pixel = 3;
  } else if (output_format_ == "RGBA" || output_format_ == "BGRA") {
    expected_size = static_cast<size_t>(width * height * 4);
    bytes_per_pixel = 4;
  } else {
    // Unknown format - copy as-is
    expected_size = map.size;
    bytes_per_pixel = 0;
  }
  
  // Check if we have stride padding
  bool has_stride_padding = (map.size > expected_size);
  
  if (!has_stride_padding || bytes_per_pixel == 0) {
    // No stride padding or unknown format - direct copy
    frame.data.assign(map.data, map.data + std::min(map.size, expected_size));
  } else if (output_format_ == "NV12") {
    // NV12: Y plane (height rows) + UV plane (height/2 rows)
    // Stride is likely current_stride_ member
    int stride = current_stride_;
    if (stride == 0) stride = width;  // Fallback
    
    // Allocate packed buffer
    frame.data.resize(expected_size);
    uint8_t* dst = frame.data.data();
    const uint8_t* src = map.data;
    
    // Copy Y plane row by row
    for (int y = 0; y < height; y++) {
      std::memcpy(dst, src + y * stride, width);
      dst += width;
    }
    
    // Copy UV plane row by row (interleaved UV, height/2 rows)
    const uint8_t* uv_src = map.data + stride * height;
    for (int y = 0; y < height / 2; y++) {
      std::memcpy(dst, uv_src + y * stride, width);
      dst += width;
    }
  } else {
    // RGB/BGR/RGBA/BGRA: copy row by row
    int stride = static_cast<int>(map.size / height);  // Estimate stride
    int row_bytes = width * bytes_per_pixel;
    
    frame.data.resize(expected_size);
    uint8_t* dst = frame.data.data();
    const uint8_t* src = map.data;
    
    for (int y = 0; y < height; y++) {
      std::memcpy(dst, src + y * stride, row_bytes);
      dst += row_bytes;
    }
  }
  
  frame.data_size = frame.data.size();
  
  gst_buffer_unmap(buffer, &map);
  
  cpu_buffer_->push(std::move(frame));
  
  // Update queue stats for CPU buffer
  {
    std::lock_guard<std::mutex> slock(stats_mutex_);
    stats_.queue_depth = cpu_buffer_->size();
    if (stats_.queue_depth > stats_.queue_max_depth) {
      stats_.queue_max_depth = stats_.queue_depth;
    }
  }
}

CpuFrame StreamDecoder::getCpuFrame(int timeout_ms) {
  // Acquire frame_mutex_ to prevent race with setCpuBufferConfig() resetting cpu_buffer_
  std::unique_lock<std::mutex> lock(frame_mutex_);
  if (!cpu_buffer_) {
    return CpuFrame{};
  }
  // Release lock before blocking get() to avoid blocking producers
  CpuBuffer* buf = cpu_buffer_.get();
  lock.unlock();
  
  CpuFrame frame = buf->get(timeout_ms);
  if (frame.valid) {
    std::lock_guard<std::mutex> slock(stats_mutex_);
    stats_.frames_consumed++;
  }
  return frame;
}

CpuBufferInfo StreamDecoder::getCpuBufferInfo() const {
  CpuBufferInfo info;
  // Acquire frame_mutex_ to prevent race with setCpuBufferConfig() resetting cpu_buffer_
  std::lock_guard<std::mutex> lock(frame_mutex_);
  if (cpu_buffer_) {
    info.buffer_count = cpu_buffer_->size();
    info.buffer_capacity = cpu_buffer_->capacity();
    info.buffer_duration_sec = cpu_buffer_duration_sec_;
    info.memory_usage_bytes = cpu_buffer_->memoryUsage();
    info.format = output_format_;
  }
  return info;
}

const CpuFrame* StreamDecoder::peekLatestFrame(int timeout_ms) const {
  // Acquire frame_mutex_ to prevent race with setCpuBufferConfig() resetting cpu_buffer_
  std::lock_guard<std::mutex> lock(frame_mutex_);
  if (!cpu_buffer_) {
    return nullptr;
  }
  return cpu_buffer_->peekLatest(timeout_ms);
}

double StreamDecoder::getBytesPerPixel(const std::string& format) {
  if (format == "NV12" || format == "I420") return 1.5;
  if (format == "RGB" || format == "BGR") return 3.0;
  if (format == "RGBA" || format == "BGRA") return 4.0;
  return 1.5;  // Default to NV12
}

bool StreamDecoder::checkHealth() {
  // Check global GPU failure flag
  if (global_gpu_failure_.load() && !cpu_buffer_enabled_) {
     std::cout << "[" << id_ << "] Global GPU failure detected - triggering fallback" << std::endl;
     hardware_accel_failed_ = true;
     markError();
     return false;
  }

  if (running_ && pipeline_ && !pending_first_frame_) {
    int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    int64_t last_frame = last_frame_rx_time_ms_.load();
    
    // If no frames for 5 seconds when we are supposed to be connected
    if (now - last_frame > 5000) {
      // Only trigger if we have received at least one frame (or initialized properly)
      if (last_frame > 0) { 
         std::cout << "[" << id_ << "] Stream stalled (no frames for 5s)" << std::endl;
         
         // If we are in GPU mode, assume it might be a decoder failure
         if (!cpu_buffer_enabled_) {
             std::cout << "[" << id_ << "] Suspected GPU failure (stall) - triggering fallback" << std::endl;
             hardware_accel_failed_ = true;
         }
         
         markError();
         return false;
      }
    }
  }
  return true;
}

