#include <rtspmodule/stream_pipeline_builder.hpp>
#include <iostream>

namespace rtsp {

StreamPipelineBuilder::StreamPipelineBuilder(StreamDecoder* decoder) : decoder_(decoder) {}

bool StreamPipelineBuilder::build(PipelineElements& elements) {
    if (!decoder_) return false;
    
    std::string id_str = std::to_string(decoder_->getId()) + "_" + std::to_string(decoder_->reconnect_count_.load());

    // Sync with global failure flag from log sniffer
    if (StreamDecoder::global_gpu_failure_.load()) {
        decoder_->hardware_accel_failed_ = true;
    }

    if (decoder_->hardware_accel_failed_) {
        decoder_->cpu_buffer_enabled_ = true;
        std::cout << "[" << decoder_->getId() << "] Hardware failure detected/persisted - forcing CPU buffer mode" << std::endl;
    }

    // Validate source based on type
    if (decoder_->is_file_source_) {
        if (decoder_->url_.empty()) {
            std::cerr << "[" << decoder_->getId() << "] [ERROR] File path cannot be empty for file source" << std::endl;
            if (decoder_->logger_) {
                decoder_->logger_->logError(rtsp::ErrorCategory::InvalidConfig, "File path cannot be empty for file source");
            }
            return false;
        }
        std::cout << "[" << decoder_->getId() << "] Opening MP4 file: " << decoder_->url_;
        if (decoder_->target_fps_ > 0.0) std::cout << " (target FPS: " << decoder_->target_fps_ << ")";
        std::cout << std::endl;
    } else {
        if (decoder_->url_.find("rtsp://") != 0 && decoder_->url_.find("rtsps://") != 0) {
            std::cerr << "[" << decoder_->getId() << "] [ERROR] Invalid RTSP URL scheme: " << decoder_->url_ << std::endl;
            if (decoder_->logger_) {
                decoder_->logger_->logError(rtsp::ErrorCategory::InvalidConfig, "Invalid RTSP URL scheme: " + decoder_->url_);
            }
            return false;
        }
        std::cout << "[" << decoder_->getId() << "] Connecting to RTSP URL: " << decoder_->url_ << std::endl;
    }

    elements.pipeline = gst_pipeline_new(("pipeline-" + id_str).c_str());
    if (!elements.pipeline) return false;

    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(elements.pipeline));
    gst_bus_set_sync_handler(bus, StreamDecoder::busSyncHandler, decoder_, nullptr);
    gst_object_unref(bus);

    if (decoder_->is_file_source_) {
        elements.source = gst_element_factory_make("filesrc", ("src-" + id_str).c_str());
    } else {
        elements.source = gst_element_factory_make("rtspsrc", ("src-" + id_str).c_str());
    }

    // 3-tier converter selection
    elements.convert = nullptr;
    decoder_->use_nvmm_memory_ = false;
    decoder_->use_cuda_memory_ = false;

    if (!decoder_->hardware_accel_failed_) {
        bool try_nvv4l2 = (decoder_->decoder_preference_ == "auto" || decoder_->decoder_preference_ == "nvv4l2");
        bool try_nvdec = (decoder_->decoder_preference_ == "auto" || decoder_->decoder_preference_ == "nvdec");
        bool force_cpu = (decoder_->decoder_preference_ == "cpu");

        if (force_cpu) {
            decoder_->hardware_accel_failed_ = true;
            std::cout << "[" << decoder_->getId() << "] CPU decoder forced by preference 'cpu'" << std::endl;
        } else {
            // First, verify GPU is actually accessible via CUDA
            int cuda_device_count = 0;
            cudaError_t cuda_err = cudaGetDeviceCount(&cuda_device_count);
            bool cuda_available = (cuda_err == cudaSuccess && cuda_device_count > 0);
            
            if (!cuda_available) {
                decoder_->hardware_accel_failed_ = true;
                std::cout << "[" << decoder_->getId() << "] CUDA not available (devices=" << cuda_device_count 
                          << ", err=" << cudaGetErrorString(cuda_err) << ") - forcing CPU decoder" << std::endl;
                if (decoder_->logger_) {
                    decoder_->logger_->logError(rtsp::ErrorCategory::HardwareAccelFailed, "CUDA not available, forcing software decoder");
                }
            }
        }

        if (!decoder_->hardware_accel_failed_ && try_nvv4l2) {
            GstElementFactory* nvv4l2_factory = gst_element_factory_find("nvv4l2decoder");
            if (nvv4l2_factory) {
                gst_object_unref(nvv4l2_factory);
                bool driver_found = false;
                FILE* fp = fopen("/dev/nvidia0", "r");
                if (fp) {
                    driver_found = true;
                    fclose(fp);
                } else {
                    fp = fopen("/dev/dxg", "r");
                    if (fp) {
                        driver_found = true;
                        fclose(fp);
                    }
                }

                if (!driver_found) {
                    std::cout << "[" << decoder_->getId() << "] NVIDIA driver not available (/dev/nvidia0 and /dev/dxg missing) - skipping nvv4l2decoder" << std::endl;
                    if (decoder_->logger_) decoder_->logger_->logInfo("NVIDIA driver not found (or WSL GPU not detected), falling back to software decoder");
                    if (decoder_->decoder_preference_ == "nvv4l2") decoder_->hardware_accel_failed_ = true;
                } else {
                    elements.convert = gst_element_factory_make("nvvideoconvert", ("convert-" + id_str).c_str());
                    if (elements.convert) {
                        decoder_->use_nvmm_memory_ = true;
                        std::cout << "[" << decoder_->getId() << "] Using DeepStream NVMM path (nvvideoconvert)" << std::endl;
                        if (decoder_->logger_) decoder_->logger_->logInfo("Converter selected: nvvideoconvert (DeepStream NVMM)");
                    }
                }
            } else if (decoder_->decoder_preference_ == "nvv4l2") {
                std::cout << "[" << decoder_->getId() << "] nvv4l2decoder not found in GStreamer registry" << std::endl;
                decoder_->hardware_accel_failed_ = true;
            }
        }

#ifdef HAVE_GST_CUDA
        if (!elements.convert && !decoder_->hardware_accel_failed_ && try_nvdec) {
            elements.convert = gst_element_factory_make("cudaconvert", ("convert-" + id_str).c_str());
            if (elements.convert) {
                decoder_->use_cuda_memory_ = true;
                std::cout << "[" << decoder_->getId() << "] Using CUDA zero-copy path (cudaconvert)" << std::endl;
                if (decoder_->logger_) decoder_->logger_->logInfo("Converter selected: cudaconvert (CUDA zero-copy)");
            } else if (decoder_->decoder_preference_ == "nvdec") {
                std::cout << "[" << decoder_->getId() << "] cudaconvert creation failed" << std::endl; 
                decoder_->hardware_accel_failed_ = true;
            }
        }
#endif
    }

    if (!elements.convert) {
        elements.convert = gst_element_factory_make("videoconvert", ("convert-" + id_str).c_str());
        std::cout << "[" << decoder_->getId() << "] Using CPU videoconvert (hardware acceleration unavailable)" << std::endl;
        if (decoder_->logger_) {
            decoder_->logger_->logWarning("Hardware acceleration unavailable, using CPU videoconvert");
            if (decoder_->hardware_accel_failed_) {
                decoder_->logger_->logError(rtsp::ErrorCategory::HardwareAccelFailed, "GPU decode/convert failed, fell back to CPU");
            }
        }
    }

    elements.appsink = gst_element_factory_make("appsink", ("sink-" + id_str).c_str());

    if (!elements.source || !elements.convert || !elements.appsink) return false;

    // Apply specific configuration based on source type
    if (decoder_->is_file_source_) {
        if (!configureFileSource(elements, id_str)) return false;
    } else {
        if (!configureRtspSource(elements)) return false;
    }

    // AppSink setup
    GstCaps *caps;
    std::string caps_str;
    if (decoder_->cpu_buffer_enabled_) {
        // CPU buffer mode: appsink needs system memory for gst_buffer_map()
        // nvvideoconvert will handle NVMM → system memory conversion
        caps_str = "video/x-raw, format=" + decoder_->output_format_;
    } else if (decoder_->use_nvmm_memory_) {
        caps_str = "video/x-raw(memory:NVMM), format=" + decoder_->output_format_;
    } else if (decoder_->use_cuda_memory_) {
        caps_str = "video/x-raw(memory:CUDAMemory), format=" + decoder_->output_format_;
    } else {
        caps_str = "video/x-raw, format=" + decoder_->output_format_;
    }
    caps = gst_caps_from_string(caps_str.c_str());

    g_object_set(elements.appsink, "emit-signals", TRUE, "drop", TRUE, "max-buffers", 2,
                 "caps", caps, "sync", (decoder_->target_fps_ > 0.0) ? TRUE : FALSE, nullptr);
    gst_caps_unref(caps);

    g_signal_connect(elements.appsink, "new-sample", G_CALLBACK(StreamDecoder::onNewSample), decoder_);

    std::cout << "[" << decoder_->getId() << "] Created pipeline: " << decoder_->name_ << std::endl;
    if (decoder_->logger_) {
        decoder_->logger_->logStateChange(rtsp::CameraState::Connecting, "Pipeline created for " + decoder_->url_);
    }
    return true;
}

bool StreamPipelineBuilder::configureFileSource(PipelineElements& elements, const std::string& id_str) {
    g_object_set(elements.source, "location", decoder_->url_.c_str(), nullptr);
    
    elements.demuxer = gst_element_factory_make("qtdemux", ("demux-" + id_str).c_str());
    if (!elements.demuxer) {
        std::cerr << "[" << decoder_->getId() << "] Failed to create qtdemux" << std::endl;
        return false;
    }

    // For NVMM path: skip decodebin entirely, create nvv4l2decoder manually in onPadAdded
    // For CPU/CUDA path: use decodebin for auto decoder selection
    if (!decoder_->use_nvmm_memory_) {
        elements.decodebin = gst_element_factory_make("decodebin", ("decodebin-" + id_str).c_str());
        if (!elements.decodebin) {
            std::cerr << "[" << decoder_->getId() << "] Failed to create decodebin" << std::endl;
            return false;
        }
        g_signal_connect(elements.decodebin, "pad-added", G_CALLBACK(StreamPipelineBuilder::onPadAdded), decoder_);
        std::cout << "[" << decoder_->getId() << "] Using decodebin path (non-NVMM)" << std::endl;
    } else {
        elements.decodebin = nullptr;
        std::cout << "[" << decoder_->getId() << "] Using direct nvv4l2decoder path (NVMM, no decodebin)" << std::endl;
    }

    g_signal_connect(elements.demuxer, "pad-added", G_CALLBACK(StreamPipelineBuilder::onPadAdded), decoder_);
    
    GstElement *rate_control = nullptr;
    if (decoder_->target_fps_ > 0.0) {
        rate_control = gst_element_factory_make("videorate", ("rate-" + id_str).c_str());
        if (rate_control) {
            g_object_set(rate_control, "drop-only", TRUE, "skip-to-first", TRUE, nullptr);
            std::cout << "[" << decoder_->getId() << "] Added FPS control for target: " << decoder_->target_fps_ << " fps" << std::endl;
        } else {
            std::cerr << "[" << decoder_->getId() << "] Failed to create videorate element for FPS control" << std::endl;
        }
    }

    // Add elements to pipeline bin
    if (elements.decodebin) {
        if (rate_control) {
            gst_bin_add_many(GST_BIN(elements.pipeline), elements.source, elements.demuxer, elements.decodebin, rate_control, elements.convert, elements.appsink, nullptr);
        } else {
            gst_bin_add_many(GST_BIN(elements.pipeline), elements.source, elements.demuxer, elements.decodebin, elements.convert, elements.appsink, nullptr);
        }
    } else {
        // NVMM path: no decodebin; parse + decoder added dynamically in onPadAdded
        if (rate_control) {
            gst_bin_add_many(GST_BIN(elements.pipeline), elements.source, elements.demuxer, rate_control, elements.convert, elements.appsink, nullptr);
        } else {
            gst_bin_add_many(GST_BIN(elements.pipeline), elements.source, elements.demuxer, elements.convert, elements.appsink, nullptr);
        }
    }
    
    if (!gst_element_link(elements.source, elements.demuxer)) {
        std::cerr << "[" << decoder_->getId() << "] Failed to link filesrc to qtdemux" << std::endl;
        return false;
    }

    // Link converter → [rate] → appsink (static part, downstream of dynamic link)
    // IMPORTANT: videorate is a CPU-only element - CANNOT be used with NVMM memory
    if (decoder_->use_nvmm_memory_) {
        // NVMM path: skip videorate, link converter → appsink directly
        // Frame pacing handled by appsink sync=TRUE (set in build()) + nvv4l2decoder timing
        if (rate_control) {
            // Remove unused rate_control from pipeline
            gst_bin_remove(GST_BIN(elements.pipeline), rate_control);
            rate_control = nullptr;
            std::cout << "[" << decoder_->getId() << "] NVMM mode: skipping videorate (CPU-only element)" << std::endl;
        }
        if (!gst_element_link(elements.convert, elements.appsink)) {
            std::cerr << "[" << decoder_->getId() << "] Failed to link NVMM converter to appsink" << std::endl;
            return false;
        }
    } else if (rate_control) {
        // Non-NVMM path: use videorate for FPS control with plain video/x-raw caps
        if (!gst_element_link(elements.convert, rate_control)) {
            std::cerr << "[" << decoder_->getId() << "] Failed to link converter to rate control" << std::endl;
            return false;
        }
        std::string rate_caps_str = "video/x-raw, format=" + decoder_->output_format_ + ",framerate=" + std::to_string(static_cast<int>(decoder_->target_fps_)) + "/1";
        GstCaps *rate_caps = gst_caps_from_string(rate_caps_str.c_str());
        if (!gst_element_link_filtered(rate_control, elements.appsink, rate_caps)) {
            std::cerr << "[" << decoder_->getId() << "] Failed to link rate control to appsink" << std::endl;
            gst_caps_unref(rate_caps);
            return false;
        }
        gst_caps_unref(rate_caps);
        std::cout << "[" << decoder_->getId() << "] Successfully linked pipeline with FPS control at " << decoder_->target_fps_ << " fps" << std::endl;
    } else {
        if (!gst_element_link(elements.convert, elements.appsink)) {
            return false;
        }
    }
    return true;
}

bool StreamPipelineBuilder::configureRtspSource(PipelineElements& elements) {
    g_object_set(elements.source, "location", decoder_->url_.c_str(), "latency", 200, "drop-on-latency", FALSE, "protocols", 4, nullptr);
    gst_bin_add_many(GST_BIN(elements.pipeline), elements.source, elements.convert, elements.appsink, nullptr);
    g_signal_connect(elements.source, "pad-added", G_CALLBACK(StreamPipelineBuilder::onPadAdded), decoder_);
    if (!gst_element_link(elements.convert, elements.appsink)) {
        return false;
    }
    return true;
}

void StreamPipelineBuilder::onPadAdded(GstElement *element, GstPad *pad, gpointer data) {
    auto self = static_cast<StreamDecoder *>(data);
    
    if (StreamDecoder::global_gpu_failure_.load()) {
        self->hardware_accel_failed_ = true;
    }

    if (self->decoder_linked_) return;

    GstCaps *caps = gst_pad_get_current_caps(pad);
    if (!caps) caps = gst_pad_query_caps(pad, nullptr);

    if (!caps || gst_caps_is_empty(caps)) {
        std::cerr << "[" << self->id_ << "] onPadAdded: NULL or empty caps from pad" << std::endl;
        if (caps) gst_caps_unref(caps);
        return;
    }

    GstStructure *str = gst_caps_get_structure(caps, 0);
    if (!str) {
        std::cerr << "[" << self->id_ << "] onPadAdded: No structure in caps" << std::endl;
        gst_caps_unref(caps);
        return;
    }

    const gchar *name = gst_structure_get_name(str);
    if (!name) {
        std::cerr << "[" << self->id_ << "] onPadAdded: No name in structure" << std::endl;
        gst_caps_unref(caps);
        return;
    }

    if (self->is_file_source_) {
        if (element == self->demuxer_) {
            if (g_str_has_prefix(name, "video/x-")) {
                std::cout << "[" << self->id_ << "] Stream from demuxer: " << name << std::endl;

                if (self->use_nvmm_memory_ && !self->hardware_accel_failed_) {
                    // NVMM path: bypass decodebin, create h264parse + nvv4l2decoder manually
                    // This ensures nvv4l2decoder outputs NVMM memory that nvvideoconvert can consume
                    bool is_h265 = (g_strcmp0(name, "video/x-h265") == 0);
                    std::string id_str = std::to_string(self->id_) + "_" + std::to_string(self->reconnect_count_.load());

                    self->parse_ = gst_element_factory_make(is_h265 ? "h265parse" : "h264parse", ("parse-" + id_str).c_str());
                    self->decoder_ = gst_element_factory_make("nvv4l2decoder", ("decode-" + id_str).c_str());

                    if (!self->parse_ || !self->decoder_) {
                        std::cerr << "[" << self->id_ << "] Failed to create parse/nvv4l2decoder for NVMM file path" << std::endl;
                        if (self->parse_) { gst_object_unref(self->parse_); self->parse_ = nullptr; }
                        if (self->decoder_) { gst_object_unref(self->decoder_); self->decoder_ = nullptr; }
                        // Fall through to decodebin path below
                    } else {
                        self->active_decoder_type_ = DecoderType::NVV4L2_NVMM;
                        g_object_set(self->parse_, "config-interval", -1, nullptr);

                        gst_bin_add_many(GST_BIN(self->pipeline_), self->parse_, self->decoder_, nullptr);

                        // Link: demuxer_pad → parse → nvv4l2decoder → nvvideoconvert (convert_)
                        bool linked = false;
                        if (gst_element_link(self->parse_, self->decoder_) &&
                            gst_element_link(self->decoder_, self->convert_)) {

                            gst_element_sync_state_with_parent(self->parse_);
                            gst_element_sync_state_with_parent(self->decoder_);

                            GstPad *parse_sink = gst_element_get_static_pad(self->parse_, "sink");
                            if (parse_sink) {
                                GstPadLinkReturn ret = gst_pad_link(pad, parse_sink);
                                if (ret == GST_PAD_LINK_OK) {
                                    linked = true;
                                    self->decoder_linked_ = true;
                                    std::cout << "[" << self->id_ << "] NVMM file pipeline linked: demux → h264parse → nvv4l2decoder → nvvideoconvert" << std::endl;
                                    if (self->logger_) {
                                        self->logger_->logStateChange(rtsp::CameraState::Connected, "NVMM file stream pipeline linked");
                                    }
                                } else {
                                    std::cerr << "[" << self->id_ << "] Failed to link demuxer pad to parse sink: " << ret << std::endl;
                                }
                                gst_object_unref(parse_sink);
                            }
                        } else {
                            std::cerr << "[" << self->id_ << "] Failed to link parse → decoder → converter chain" << std::endl;
                        }

                        if (!linked) {
                            // Clean up failed NVMM elements
                            gst_element_set_state(self->parse_, GST_STATE_NULL);
                            gst_element_set_state(self->decoder_, GST_STATE_NULL);
                            gst_bin_remove(GST_BIN(self->pipeline_), self->parse_);
                            gst_bin_remove(GST_BIN(self->pipeline_), self->decoder_);
                            self->parse_ = nullptr;
                            self->decoder_ = nullptr;
                            std::cerr << "[" << self->id_ << "] NVMM file path failed, stream will error" << std::endl;
                        }

                        gst_caps_unref(caps);
                        return;
                    }
                }

                // Non-NVMM path (or NVMM creation failed): link demuxer → decodebin
                if (self->decodebin_) {
                    if (!gst_element_link_pads(element, gst_pad_get_name(pad), self->decodebin_, nullptr)) {
                        std::cerr << "[" << self->id_ << "] Failed to link demuxer to decodebin" << std::endl;
                    } else {
                        std::cout << "[" << self->id_ << "] Successfully linked demuxer to decodebin" << std::endl;
                    }
                } else {
                    std::cerr << "[" << self->id_ << "] No decodebin available for non-NVMM fallback" << std::endl;
                }
            }
        } else if (element == self->decodebin_) {
            // decodebin path (non-NVMM): link decoded video to converter
            if (g_str_has_prefix(name, "video/x-")) {
                std::cout << "[" << self->id_ << "] Decoded video stream: " << name << std::endl;

                GstPad *convert_sink_pad = gst_element_get_static_pad(self->convert_, "sink");
                if (convert_sink_pad) {
                    GstPadLinkReturn ret = gst_pad_link(pad, convert_sink_pad);

                    if (ret == GST_PAD_LINK_OK) {
                        std::cout << "[" << self->id_ << "] Successfully linked decodebin to converter" << std::endl;
                        self->decoder_linked_ = true;
                        if (self->logger_) {
                            self->logger_->logStateChange(rtsp::CameraState::Connected, "File stream pipeline linked");
                        }
                    } else {
                        std::cerr << "[" << self->id_ << "] Failed to link decodebin to converter: " << ret << std::endl;

                        // Debug: Print caps
                        GstCaps *pad_caps = gst_pad_query_caps(pad, NULL);
                        GstCaps *sink_caps = gst_pad_query_caps(convert_sink_pad, NULL);
                        gchar *pad_caps_str = pad_caps ? gst_caps_to_string(pad_caps) : nullptr;
                        gchar *sink_caps_str = sink_caps ? gst_caps_to_string(sink_caps) : nullptr;
                        std::cerr << "[" << self->id_ << "] Pad caps: " << (pad_caps_str ? pad_caps_str : "NULL") << std::endl;
                        std::cerr << "[" << self->id_ << "] Sink caps: " << (sink_caps_str ? sink_caps_str : "NULL") << std::endl;
                        if (pad_caps_str) g_free(pad_caps_str);
                        if (sink_caps_str) g_free(sink_caps_str);
                        if (pad_caps) gst_caps_unref(pad_caps);
                        if (sink_caps) gst_caps_unref(sink_caps);
                    }

                    gst_object_unref(convert_sink_pad);
                }
            }
        }
        gst_caps_unref(caps);
        return;
    }

    if (g_str_has_prefix(name, "application/x-rtp")) {
        const gchar *media = gst_structure_get_string(str, "media");
        if (media && g_strcmp0(media, "video") == 0) {
            const gchar *encoding = gst_structure_get_string(str, "encoding-name");
            std::string id_str = std::to_string(self->id_) + "_" + std::to_string(self->reconnect_count_);
            bool is_h265 = false;

            if (encoding) {
                std::string enc(encoding);
                is_h265 = (enc == "H265" || enc == "HEVC");
                std::cout << "[" << self->id_ << "] Codec: " << encoding << " (" << (is_h265 ? "H265" : "H264") << ")" << std::endl;
            }

            self->depay_ = gst_element_factory_make(is_h265 ? "rtph265depay" : "rtph264depay", ("depay-" + id_str).c_str());
            self->parse_ = gst_element_factory_make(is_h265 ? "h265parse" : "h264parse", ("parse-" + id_str).c_str());

            if (self->parse_) {
                GstPad *src_pad = gst_element_get_static_pad(self->parse_, "src");
                if (src_pad) {
                    gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, StreamDecoder::onParserCaps, self, nullptr);
                    gst_object_unref(src_pad);
                }
            }

            self->decoder_ = nullptr;
            
            if (self->use_nvmm_memory_ && !self->hardware_accel_failed_) {
                self->decoder_ = gst_element_factory_make(is_h265 ? "nvv4l2decoder" : "nvv4l2decoder", ("decode-" + id_str).c_str());
                if (self->decoder_) {
                    self->active_decoder_type_ = DecoderType::NVV4L2_NVMM;
                    std::cout << "[" << self->id_ << "] Using nvv4l2decoder (DeepStream NVMM)" << std::endl;
                }
            }
            
            if (!self->decoder_ && !self->hardware_accel_failed_) {
                self->decoder_ = gst_element_factory_make(is_h265 ? "nvh265dec" : "nvh264dec", ("decode-" + id_str).c_str());
                if (self->decoder_) {
                    g_object_set(self->decoder_, "num-output-surfaces", 1, nullptr);
                    self->active_decoder_type_ = DecoderType::NVDEC_CUDA;
                    std::cout << "[" << self->id_ << "] Using " << (is_h265 ? "nvh265dec" : "nvh264dec") << " (NVDEC CUDA)" << std::endl;
                }
            }
            
            if (!self->decoder_) {
                self->decoder_ = gst_element_factory_make(is_h265 ? "avdec_h265" : "avdec_h264", ("decode-" + id_str).c_str());
                self->active_decoder_type_ = DecoderType::AVDEC_CPU;
                std::cout << "[" << self->id_ << "] Using " << (is_h265 ? "avdec_h265" : "avdec_h264") << " (CPU)" << std::endl;
            }

            self->webrtc_is_h265_ = is_h265;

            if (self->depay_ && self->parse_ && self->decoder_) {
                g_object_set(self->parse_, "config-interval", -1, nullptr);

                self->webrtc_tee_ = gst_element_factory_make("tee", ("webrtc-tee-" + id_str).c_str());
                if (!self->webrtc_tee_) {
                    std::cerr << "[" << self->id_ << "] Failed to create tee element" << std::endl;
                }

                bool linked = false;
                GstElement* dec_caps = nullptr;

                if (self->webrtc_tee_) {
                    g_object_set(self->webrtc_tee_, "allow-not-linked", TRUE, nullptr);

                    if (!is_h265) {
                        gst_bin_add_many(GST_BIN(self->pipeline_), self->depay_, self->parse_, self->webrtc_tee_, self->decoder_, nullptr);
                        if (gst_element_link_many(self->depay_, self->parse_, self->webrtc_tee_, nullptr)) {
                            GstPad* tee_main = gst_element_request_pad_simple(self->webrtc_tee_, "src_%u");
                            GstPad* dec_sink = gst_element_get_static_pad(self->decoder_, "sink");
                            if (tee_main && dec_sink && gst_pad_link(tee_main, dec_sink) == GST_PAD_LINK_OK) {
                                if (gst_element_link(self->decoder_, self->convert_)) linked = true;
                            }
                            if (tee_main) gst_object_unref(tee_main);
                            if (dec_sink) gst_object_unref(dec_sink);
                        }
                    } else {
                        if (self->use_cuda_memory_) {
                            dec_caps = gst_element_factory_make("capsfilter", ("dec-caps-" + id_str).c_str());
                            GstCaps* caps = gst_caps_from_string("video/x-raw(memory:CUDAMemory), format=NV12");
                            if (dec_caps && caps) g_object_set(dec_caps, "caps", caps, nullptr);
                            if (caps) gst_caps_unref(caps);
                        } else if (self->use_nvmm_memory_) {
                            dec_caps = gst_element_factory_make("capsfilter", ("dec-caps-" + id_str).c_str());
                            GstCaps* caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
                            if (dec_caps && caps) g_object_set(dec_caps, "caps", caps, nullptr);
                            if (caps) gst_caps_unref(caps);
                        }

                        if (dec_caps) {
                            gst_bin_add_many(GST_BIN(self->pipeline_), self->depay_, self->parse_, self->decoder_, dec_caps, self->webrtc_tee_, nullptr);
                            if (gst_element_link_many(self->depay_, self->parse_, self->decoder_, dec_caps, self->webrtc_tee_, nullptr)) {
                                GstPad* tee_main = gst_element_request_pad_simple(self->webrtc_tee_, "src_%u");
                                GstPad* cvt_sink = gst_element_get_static_pad(self->convert_, "sink");
                                if (tee_main && cvt_sink && gst_pad_link(tee_main, cvt_sink) == GST_PAD_LINK_OK) linked = true;
                                if (tee_main) gst_object_unref(tee_main);
                                if (cvt_sink) gst_object_unref(cvt_sink);
                            }
                        } else {
                            gst_bin_add_many(GST_BIN(self->pipeline_), self->depay_, self->parse_, self->decoder_, self->webrtc_tee_, nullptr);
                            if (gst_element_link_many(self->depay_, self->parse_, self->decoder_, self->webrtc_tee_, nullptr)) {
                                GstPad* tee_main = gst_element_request_pad_simple(self->webrtc_tee_, "src_%u");
                                GstPad* cvt_sink = gst_element_get_static_pad(self->convert_, "sink");
                                if (tee_main && cvt_sink && gst_pad_link(tee_main, cvt_sink) == GST_PAD_LINK_OK) linked = true;
                                if (tee_main) gst_object_unref(tee_main);
                                if (cvt_sink) gst_object_unref(cvt_sink);
                            }
                        }
                    }
                } else {
                    gst_bin_add_many(GST_BIN(self->pipeline_), self->depay_, self->parse_, self->decoder_, nullptr);
                    linked = gst_element_link_many(self->depay_, self->parse_, self->decoder_, self->convert_, nullptr);
                }

                if (linked) {
                    gst_element_sync_state_with_parent(self->depay_);
                    gst_element_sync_state_with_parent(self->parse_);
                    gst_element_sync_state_with_parent(self->decoder_);
                    if (dec_caps) gst_element_sync_state_with_parent(dec_caps);
                    if (self->webrtc_tee_) gst_element_sync_state_with_parent(self->webrtc_tee_);

                    GstPad *sink = gst_element_get_static_pad(self->depay_, "sink");
                    if (sink) {
                        gst_pad_link(pad, sink);
                        gst_object_unref(sink);
                    }
                    self->decoder_linked_ = true;

                    if (self->webrtc_enabled_.load() && self->webrtc_tee_) {
                        std::cout << "[" << self->id_ << "] WebRTC tee ready, auto-starting streaming" << std::endl;
                        self->start_streaming(); 
                    }
                } else {
                    std::cerr << "[" << self->id_ << "] onPadAdded: failed to link pipeline" << std::endl;
                    std::vector<GstElement*> to_remove = {self->depay_, self->parse_, self->decoder_};
                    if (self->webrtc_tee_) to_remove.push_back(self->webrtc_tee_);
                    if (dec_caps) to_remove.push_back(dec_caps);
                    for (auto* el : to_remove) {
                        if (el) {
                            gst_bin_remove(GST_BIN(self->pipeline_), el);
                            gst_object_unref(el);
                        }
                    }
                    self->depay_ = nullptr;
                    self->parse_ = nullptr;
                    self->decoder_ = nullptr;
                    self->webrtc_tee_ = nullptr;
                }
            } else {
                if (self->depay_) { gst_object_unref(self->depay_); self->depay_ = nullptr; }
                if (self->parse_) { gst_object_unref(self->parse_); self->parse_ = nullptr; }
                if (self->decoder_) { gst_object_unref(self->decoder_); self->decoder_ = nullptr; }
            }
        }
    }
    gst_caps_unref(caps);
}

} // namespace rtsp
