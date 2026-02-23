#include <rtspmodule/webrtc_sink_bin.hpp>
#include <iostream>
#include <vector>

namespace rtsp {

static GstElement* createH264Encoder(const std::string& name_suffix) {
    GstElement* enc = nullptr;

    // Tier 1: DeepStream NVMM encoder
    enc = gst_element_factory_make("nvv4l2h264enc", ("webrtc-enc-" + name_suffix).c_str());
    if (enc) {
        g_object_set(enc, "iframeinterval", 30, nullptr);  // IDR every 30 frames
        std::cout << "[WebrtcSinkBin] Encoder: nvv4l2h264enc (DeepStream NVMM)" << std::endl;
        return enc;
    }

    // Tier 2: Standard NVENC (GStreamer nvcodec)
    enc = gst_element_factory_make("nvh264enc", ("webrtc-enc-" + name_suffix).c_str());
    if (enc) {
        g_object_set(enc, "preset", 4, "bitrate", 4000, "gop-size", 30, nullptr);
        std::cout << "[WebrtcSinkBin] Encoder: nvh264enc (NVENC)" << std::endl;
        return enc;
    }

    // Tier 3: CPU fallback
    enc = gst_element_factory_make("x264enc", ("webrtc-enc-" + name_suffix).c_str());
    if (enc) {
        g_object_set(enc, "tune", 4, "speed-preset", 1, "bitrate", 4000, "key-int-max", 30, nullptr);
        std::cout << "[WebrtcSinkBin] Encoder: x264enc (CPU)" << std::endl;
    }
    return enc;
}

GstElement* create_webrtc_sink_bin(const std::string& name_prefix,
                                   int signaling_port,
                                   const std::string& stream_id,
                                   bool use_cuda_memory,
                                   bool use_nvmm_memory,
                                   bool is_h265,
                                   bool is_raw_video) {

    GstElement* bin = gst_bin_new(("webrtc_bin_" + name_prefix).c_str());
    if (!bin) {
        std::cerr << "[WebrtcSinkBin] Failed to create GstBin." << std::endl;
        return nullptr;
    }

    GstElement* webrtc_queue = gst_element_factory_make("queue", ("webrtc-q-" + name_prefix).c_str());
    GstElement* webrtc_sink = gst_element_factory_make("webrtcrs_sink", ("webrtc-sink-" + name_prefix).c_str());

    if (!webrtc_queue || !webrtc_sink) {
        std::cerr << "[WebrtcSinkBin] Failed to create queue or webrtcrs_sink." << std::endl;
        if (webrtc_queue) gst_object_unref(webrtc_queue);
        if (webrtc_sink) gst_object_unref(webrtc_sink);
        gst_object_unref(bin);
        return nullptr;
    }

    g_object_set(webrtc_sink,
                 "stream-id", stream_id.c_str(),
                 "signaling-port", signaling_port,
                 nullptr);

    // If the input is already encoded H.264, we don't need a transcoder, just RTP payloading.
    if (!is_h265 && !is_raw_video) {
        GstElement* webrtc_pay = gst_element_factory_make("rtph264pay", ("webrtc-pay-" + name_prefix).c_str());
        if (!webrtc_pay) {
            std::cerr << "[WebrtcSinkBin] Failed to create rtph264pay for H.264 passthrough." << std::endl;
            gst_object_unref(webrtc_queue);
            gst_object_unref(webrtc_sink);
            gst_object_unref(bin);
            return nullptr;
        }
        g_object_set(webrtc_pay, "config-interval", -1, nullptr);

        gst_bin_add_many(GST_BIN(bin), webrtc_queue, webrtc_pay, webrtc_sink, nullptr);

        if (!gst_element_link_many(webrtc_queue, webrtc_pay, webrtc_sink, nullptr)) {
            std::cerr << "[WebrtcSinkBin] Failed to link H.264 passthrough elements." << std::endl;
            gst_object_unref(bin);
            return nullptr;
        }
    } 
    // Otherwise, we need to transcode from raw video (or decode-then-encode if it was parsed H.265).
    else {
        GstElement* webrtc_pay = gst_element_factory_make("rtph264pay", ("webrtc-pay-" + name_prefix).c_str());
        GstElement* webrtc_parse = gst_element_factory_make("h264parse", ("webrtc-parse-" + name_prefix).c_str());
        GstElement* webrtc_caps = gst_element_factory_make("capsfilter", ("webrtc-caps-" + name_prefix).c_str());

        if (!webrtc_pay || !webrtc_parse || !webrtc_caps) {
            std::cerr << "[WebrtcSinkBin] Failed to create rtph264pay, h264parse, or capsfilter." << std::endl;
            if (webrtc_pay) gst_object_unref(webrtc_pay);
            if (webrtc_parse) gst_object_unref(webrtc_parse);
            if (webrtc_caps) gst_object_unref(webrtc_caps);
            gst_object_unref(webrtc_queue);
            gst_object_unref(webrtc_sink);
            gst_object_unref(bin);
            return nullptr;
        }

        g_object_set(webrtc_pay, "config-interval", -1, "pt", 96, nullptr);
        g_object_set(webrtc_parse, "config-interval", -1, nullptr);
        
        GstCaps* baseline_caps = gst_caps_from_string("video/x-h264, profile=baseline");
        g_object_set(webrtc_caps, "caps", baseline_caps, nullptr);
        gst_caps_unref(baseline_caps);

        bool branch_ok = false;
        GstElement* webrtc_convert = nullptr;
        GstElement* webrtc_encoder = nullptr;
        GstElement* webrtc_download = nullptr;

        // Try CUDA zero-copy
        if (use_cuda_memory) {
            webrtc_convert = gst_element_factory_make("cudaconvert", ("webrtc-cvt-" + name_prefix).c_str());
            webrtc_encoder = gst_element_factory_make("nvh264enc", ("webrtc-enc-" + name_prefix).c_str());

            if (webrtc_convert && webrtc_encoder) {
                g_object_set(webrtc_encoder, "preset", 4, "bitrate", 2000, "gop-size", 30, nullptr);
                gst_bin_add_many(GST_BIN(bin), webrtc_queue, webrtc_convert,
                                 webrtc_encoder, webrtc_caps, webrtc_parse, webrtc_pay, webrtc_sink, nullptr);
                branch_ok = gst_element_link_many(webrtc_queue, webrtc_convert,
                                                  webrtc_encoder, webrtc_caps, webrtc_parse, webrtc_pay, webrtc_sink, nullptr);
                if (branch_ok) {
                    std::cout << "[WebrtcSinkBin] branch: cudaconvert -> nvh264enc (zero-copy GPU)" << std::endl;
                }
            }
        }

        // Try NVMM zero-copy
        if (!branch_ok && use_nvmm_memory) {
            if (webrtc_convert) { gst_object_unref(webrtc_convert); webrtc_convert = nullptr; }
            if (webrtc_encoder) { gst_object_unref(webrtc_encoder); webrtc_encoder = nullptr; }

            webrtc_convert = gst_element_factory_make("nvvideoconvert", ("webrtc-cvt-" + name_prefix).c_str());
            webrtc_encoder = gst_element_factory_make("nvv4l2h264enc", ("webrtc-enc-" + name_prefix).c_str());

            if (webrtc_convert && webrtc_encoder) {
                g_object_set(webrtc_encoder, "iframeinterval", 30, nullptr);
                gst_bin_add_many(GST_BIN(bin), webrtc_queue, webrtc_convert,
                                 webrtc_encoder, webrtc_caps, webrtc_parse, webrtc_pay, webrtc_sink, nullptr);
                branch_ok = gst_element_link_many(webrtc_queue, webrtc_convert,
                                                  webrtc_encoder, webrtc_caps, webrtc_parse, webrtc_pay, webrtc_sink, nullptr);
                if (branch_ok) {
                    std::cout << "[WebrtcSinkBin] branch: nvvideoconvert -> nvv4l2h264enc (zero-copy NVMM)" << std::endl;
                }
            }
        }

        // Try CPU fallback
        if (!branch_ok) {
            if (webrtc_convert) { gst_object_unref(webrtc_convert); webrtc_convert = nullptr; }
            if (webrtc_encoder) { gst_object_unref(webrtc_encoder); webrtc_encoder = nullptr; }

            if (use_cuda_memory || use_nvmm_memory) {
                webrtc_download = gst_element_factory_make("cudadownload", ("webrtc-dl-" + name_prefix).c_str());
            }
            webrtc_convert = gst_element_factory_make("videoconvert", ("webrtc-cvt-" + name_prefix).c_str());
            webrtc_encoder = gst_element_factory_make("x264enc", ("webrtc-enc-" + name_prefix).c_str());

            if (webrtc_convert && webrtc_encoder) {
                g_object_set(webrtc_encoder, "tune", 4, "speed-preset", 1, "bitrate", 4000, "key-int-max", 30, nullptr);

                if (webrtc_download) {
                    gst_bin_add_many(GST_BIN(bin), webrtc_queue, webrtc_download,
                                     webrtc_convert, webrtc_encoder, webrtc_caps, webrtc_parse, webrtc_pay, webrtc_sink, nullptr);
                    branch_ok = gst_element_link_many(webrtc_queue, webrtc_download, webrtc_convert,
                                                      webrtc_encoder, webrtc_caps, webrtc_parse, webrtc_pay, webrtc_sink, nullptr);
                } else {
                    gst_bin_add_many(GST_BIN(bin), webrtc_queue, webrtc_convert,
                                     webrtc_encoder, webrtc_caps, webrtc_parse, webrtc_pay, webrtc_sink, nullptr);
                    branch_ok = gst_element_link_many(webrtc_queue, webrtc_convert,
                                                      webrtc_encoder, webrtc_caps, webrtc_parse, webrtc_pay, webrtc_sink, nullptr);
                }
                if (branch_ok) {
                    std::cout << "[WebrtcSinkBin] branch: " << (webrtc_download ? "cudadownload -> " : "")
                              << "videoconvert -> x264enc (CPU fallback)" << std::endl;
                }
            }
        }

        if (!branch_ok) {
            std::cerr << "[WebrtcSinkBin] Failed to create/link transcoding branch." << std::endl;
            gst_object_unref(bin); // This unrefs and destroys all added children
            return nullptr;
        }
    }

    // Attach ghost pad to the external bin
    GstPad* pad = gst_element_get_static_pad(webrtc_queue, "sink");
    if (!pad) {
        std::cerr << "[WebrtcSinkBin] Failed to get sink pad from queue." << std::endl;
        gst_object_unref(bin);
        return nullptr;
    }

    GstPad* ghost_pad = gst_ghost_pad_new("sink", pad);
    gst_pad_set_active(ghost_pad, TRUE);
    gst_element_add_pad(bin, ghost_pad);
    gst_object_unref(pad);

    return bin;
}

} // namespace rtsp
