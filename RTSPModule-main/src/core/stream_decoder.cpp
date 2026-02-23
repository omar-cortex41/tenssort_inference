#include <rtspmodule/stream_decoder.h>
#include <rtspmodule/webrtc_sink_bin.hpp>
#include <rtspmodule/stream_pipeline_builder.hpp>
#include <rtspmodule/buffer_mapper.hpp>

#define GST_USE_UNSTABLE_API
#include <cuda_runtime.h>
#include <gst/rtsp/gstrtsp.h>
#include <gst/sdp/gstsdpmessage.h>
#include <iostream>

using namespace rtsp;

// Conditional GStreamer CUDA memory support
#ifdef HAVE_GST_CUDA
#include <gst/cuda/gstcuda.h>
#endif

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
                             const std::string& output_format,
                             const std::string& decoder_preference,
                             bool is_file_source,
                             bool loop_file,
                             double target_fps)
    : id_(id), name_(name), url_(url), frame_counter_(0), has_error_(false),
      pipeline_(nullptr), last_frame_time_(0), reconnect_count_(0),
      running_(false), max_queue_depth_(max_queue_size),
      gpu_buffer_(), output_format_(output_format),
      decoder_preference_(decoder_preference), is_file_source_(is_file_source),
      loop_file_(loop_file), target_fps_(target_fps) {
  std::cout << "[" << id_ << "] Queue depth: " << max_queue_depth_ 
            << ", Output format: " << output_format_
            << ", Decoder preference: " << decoder_preference_ << std::endl;

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
  StreamPipelineBuilder builder(this);
  PipelineElements el;
  
  if (!builder.build(el)) {
    return false;
  }

  pipeline_ = el.pipeline;
  source_ = el.source;
  demuxer_ = el.demuxer;
  decodebin_ = el.decodebin;
  depay_ = el.depay;
  parse_ = el.parse;
  decoder_ = el.decoder;
  convert_ = el.convert;
  appsink_ = el.appsink;
  webrtc_tee_ = el.webrtc_tee;

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

  // Cleanly detach the WebRTC branch before tearing down the pipeline.
  // This sets elements to NULL, unlinks pads, and removes them from the bin
  // — preventing GStreamer-CRITICAL "Trying to dispose element in PLAYING".
  stop_streaming();

  destroy();
}

void StreamDecoder::destroy() {
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }

  source_ = nullptr;
  demuxer_ = nullptr;
  decodebin_ = nullptr;
  depay_ = nullptr;
  parse_ = nullptr;
  decoder_ = nullptr;
  convert_ = nullptr;
  appsink_ = nullptr;
  decoder_linked_ = false;
  last_frame_time_ = 0;

  // Clean up frame queue
  while (!frame_queue_.empty()) {
    gst_sample_unref(frame_queue_.front().sample);
    frame_queue_.pop();
  }
  
  gpu_buffer_.deallocate();
  cuda_device_ptr_ = 0;
  cpu_buffer_.reset();
  
  {
    std::lock_guard<std::mutex> slock(stats_mutex_);
    fps_timestamps_ns_.clear();
    last_frame_time_ns_ = 0;
    last_pts_ = 0;
    stats_.current_fps = 0.0;
    stats_.instant_fps = 0.0;
  }
  
  webrtc_tee_ = nullptr;
  webrtc_bin_ = nullptr;
  webrtc_streaming_active_ = false;
  webrtc_is_h265_ = false;
}

void StreamDecoder::setWebRtcConfig(int signaling_port, const std::string& stream_id) {
  webrtc_signaling_port_ = signaling_port;
  webrtc_stream_id_      = stream_id;
}

// WebRTC and Pipeline building logic moved to WebrtcSinkBin and StreamPipelineBuilder

bool StreamDecoder::start_streaming() {
  if (webrtc_streaming_active_.load()) {
    return false;
  }
  if (!pipeline_ || !webrtc_tee_) {
    webrtc_enabled_ = true;
    return true;
  }

  std::string id_str = std::to_string(id_) + "_" + std::to_string(reconnect_count_.load());

  webrtc_bin_ = create_webrtc_sink_bin(id_str, webrtc_signaling_port_, webrtc_stream_id_, 
                                       use_cuda_memory_, use_nvmm_memory_, 
                                       webrtc_is_h265_, webrtc_is_h265_);

  if (!webrtc_bin_) {
    std::cerr << "[" << id_ << "] WebRTC: failed to create sink bin" << std::endl;
    return false;
  }

  gst_bin_add(GST_BIN(pipeline_), webrtc_bin_);
  gst_element_sync_state_with_parent(webrtc_bin_);

  GstPad* tee_src = gst_element_request_pad_simple(webrtc_tee_, "src_%u");
  GstPad* bin_sink = gst_element_get_static_pad(webrtc_bin_, "sink");
  GstPadLinkReturn link_ret = gst_pad_link(tee_src, bin_sink);
  gst_object_unref(tee_src);
  gst_object_unref(bin_sink);

  if (link_ret != GST_PAD_LINK_OK) {
    std::cerr << "[" << id_ << "] WebRTC: failed to link tee to bin" << std::endl;
    gst_bin_remove(GST_BIN(pipeline_), webrtc_bin_);
    webrtc_bin_ = nullptr;
    return false;
  }

  webrtc_streaming_active_ = true;
  std::cout << "[" << id_ << "] WebRTC streaming STARTED on port " << webrtc_signaling_port_
            << " (stream-id: " << webrtc_stream_id_ << ")" << std::endl;
  return true;
}

void StreamDecoder::stop_streaming() {
  if (!webrtc_streaming_active_.load()) {
    return;
  }
  if (!webrtc_tee_ || !webrtc_bin_) {
    webrtc_streaming_active_ = false;
    return;
  }

  gst_element_set_state(webrtc_bin_, GST_STATE_NULL);

  GstPad* bin_sink = gst_element_get_static_pad(webrtc_bin_, "sink");
  if (bin_sink) {
    GstPad* tee_src = gst_pad_get_peer(bin_sink);
    if (tee_src) {
      gst_pad_unlink(tee_src, bin_sink);
      gst_element_release_request_pad(webrtc_tee_, tee_src);
      gst_object_unref(tee_src);
    }
    gst_object_unref(bin_sink);
  }

  gst_bin_remove(GST_BIN(pipeline_), webrtc_bin_);
  webrtc_bin_ = nullptr;
  webrtc_streaming_active_ = false;
  std::cout << "[" << id_ << "] WebRTC streaming STOPPED" << std::endl;
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
        if (is_file_source_ && loop_file_) {
          std::cout << "[" << id_ << "] EOS received, seeking to beginning for loop" << std::endl;
          // Seek to beginning to loop the file
          gst_element_seek_simple(pipeline_, GST_FORMAT_TIME, 
                                 static_cast<GstSeekFlags>(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT), 0);
        } else {
          std::cout << "[" << id_ << "] EOS received" << std::endl;
          if (logger_) {
            logger_->logStateChange(rtsp::CameraState::StreamLost,
                                    "End of stream received");
          }
          markError();
        }
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
    QueuedFrame q_frame = {nullptr, 0, 0, 0, 0, 0, 0, 0};
    {
        std::unique_lock<std::mutex> lock(frame_mutex_);
        if (frame_queue_.empty()) {
            if (timeout_ms <= 0) return {0, 0, 0, 0, 0, 0, "", false};
            if (queue_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms)) == std::cv_status::timeout) {
                return {0, 0, 0, 0, 0, 0, "", false};
            }
        }
        if (frame_queue_.empty()) return {0, 0, 0, 0, 0, 0, "", false};
        q_frame = frame_queue_.front();
        frame_queue_.pop();
    }

    GpuFrameInfo info;
    info.width = q_frame.width;
    info.height = q_frame.height;
    info.stride = q_frame.stride;
    info.size = q_frame.data_size;
    info.frame_id = q_frame.frame_id;
    info.valid = true;
    info.format = output_format_;

    GstBuffer* buffer = gst_sample_get_buffer(q_frame.sample);
    uint64_t ptr = 0;
    int stride = 0;

    if (BufferMapper::mapGpuBuffer(buffer, use_nvmm_memory_, use_cuda_memory_, 
                                  info.width, info.height, &q_frame.stride, 
                                  ptr, stride)) {
        info.ptr = ptr;
        info.stride = stride;
    } else {
        info.ptr = 0;
    }

    gst_sample_unref(q_frame.sample);
    
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        stats_.frames_consumed++;
        stats_.queue_depth = frame_queue_.size();
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
    if (!sample) return GST_FLOW_OK; // Changed from GST_FLOW_ERROR to GST_FLOW_OK to match original behavior on no sample

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
    GstVideoInfo v_info;
    
    // Guard: caps can be NULL during pipeline negotiation
    if (!caps || !gst_video_info_from_caps(&v_info, caps)) {
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

    uint64_t frame_id = ++self->frame_counter_;
    self->updateFrameTime();
    self->updateFps();

    bool success = false;
    uint64_t cuda_ptr = 0;
    int current_stride = 0;

    if (self->cpu_buffer_enabled_ && self->cpu_buffer_) {
        // CPU buffer mode: copy frame to ring buffer (preferred when enabled)
        // This must be checked FIRST because Python reads from CPU buffer via get_batch().
        // With NVMM, mapGpuBuffer() would succeed and put frames in GPU queue,
        // but Python's get_batch() reads from CPU buffer — causing starvation.
        BufferMapper::pushToCpuBuffer(self->cpu_buffer_.get(), buffer,
                                      self->output_format_, v_info.width, v_info.height,
                                      v_info.stride[0], frame_id);
        success = true;
    } else if (BufferMapper::mapGpuBuffer(buffer, self->use_nvmm_memory_, self->use_cuda_memory_,
                                   v_info.width, v_info.height, &v_info.stride[0],
                                   cuda_ptr, current_stride)) {
        success = true;
    } else {
        // Fallback: CPU memory path with copy to GPU (for GPU queue mode only)
        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            size_t required_size = (size_t)(v_info.width * v_info.height * 1.5); // NV12

            // Dynamically allocate or reallocate GPU buffer as needed
            if (!self->gpu_buffer_.isAllocated() ||
                self->gpu_buffer_.size() < required_size) {
                self->gpu_buffer_.allocate(required_size);
            }

            if (self->gpu_buffer_.isAllocated() &&
                self->gpu_buffer_.copyToDevice(map.data, map.size, v_info.width, v_info.height)) {
                cuda_ptr = 0; // Use gpu_buffer_ path
                current_stride = v_info.width;  // Fallback is packed
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
        std::lock_guard<std::mutex> slock(self->stats_mutex_);
        self->stats_.frames_dropped_decode++;
        self->stats_.decode_errors++;
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

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
        qf.frame_id = frame_id;
        qf.width = v_info.width;
        qf.height = v_info.height;
        qf.stride = current_stride;
        qf.data_size = gst_buffer_get_size(buffer); // Use actual buffer size
        qf.cuda_ptr = cuda_ptr;
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

// onPadAdded logic moved to StreamPipelineBuilder

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

std::vector<CpuFrame> StreamDecoder::getCpuFrames(int count, int timeout_ms) {
  std::unique_lock<std::mutex> lock(frame_mutex_);
  if (!cpu_buffer_) {
    return {};
  }
  CpuBuffer* buf = cpu_buffer_.get();
  lock.unlock();
  
  auto frames = buf->getMulti(count, timeout_ms);
  if (!frames.empty()) {
    std::lock_guard<std::mutex> slock(stats_mutex_);
    stats_.frames_consumed += frames.size();
  }
  return frames;
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

// getBytesPerPixel logic moved to BufferMapper
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

