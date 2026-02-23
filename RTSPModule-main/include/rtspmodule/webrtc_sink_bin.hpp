#ifndef WEBRTC_SINK_BIN_HPP
#define WEBRTC_SINK_BIN_HPP

#include <gst/gst.h>
#include <string>

namespace rtsp {

/**
 * @brief Creates a standalone GStreamer Bin that handles H.264/H.265 transcoding and WebRTC signaling.
 * 
 * This Bin connects to a pipeline producing raw video (CPU or GPU memory) or parsed H.264 video.
 * It contains the necessary format converters, hardware/software encoders, parsers, payloaders,
 * and the `webrtcrs_sink` element. The entire sub-pipeline is encapsulated and exposes a
 * single sink pad ("sink").
 *
 * @param name_prefix Unique prefix for naming the internal GStreamer elements.
 * @param signaling_port The port the webrtc-rs server will listen on.
 * @param stream_id The logical ID of the WebRTC stream (used by the browser to connect).
 * @param use_cuda_memory True if the input video is in standard GStreamer CUDAMemory (NVDEC).
 * @param use_nvmm_memory True if the input video is in DeepStream NVMM memory.
 * @param is_h265 True if the incoming raw video originates from an H.265 source (affects encoder selection).
 * @param is_raw_video True if the input is raw video requiring encoding. False if it's already H.264.
 * @return GstElement* The constructed GstBin, or nullptr on failure. Ownership is transferred to the caller.
 */
GstElement* create_webrtc_sink_bin(const std::string& name_prefix,
                                   int signaling_port,
                                   const std::string& stream_id,
                                   bool use_cuda_memory,
                                   bool use_nvmm_memory,
                                   bool is_h265,
                                   bool is_raw_video);

} // namespace rtsp

#endif // WEBRTC_SINK_BIN_HPP
