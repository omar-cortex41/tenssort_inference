#ifndef STREAM_PIPELINE_BUILDER_HPP
#define STREAM_PIPELINE_BUILDER_HPP

#include <gst/gst.h>
#include <string>
#include <memory>
#include "rtspmodule/stream_decoder.h"

namespace rtsp {

struct PipelineElements {
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* demuxer = nullptr;
    GstElement* decodebin = nullptr;
    GstElement* depay = nullptr;
    GstElement* parse = nullptr;
    GstElement* decoder = nullptr;
    GstElement* convert = nullptr;
    GstElement* appsink = nullptr;
    GstElement* webrtc_tee = nullptr;
};

class StreamPipelineBuilder {
public:
    StreamPipelineBuilder(StreamDecoder* decoder);
    
    // Builds the initial static parts of the pipeline based on configuration.
    // Handles difference between RTSP and MP4 file, and creates the appropriate elements.
    bool build(PipelineElements& out_elements);
    
    // Static pad-added callback handler (used for RTSP dynamic payloads and MP4 Demuxer).
    static void onPadAdded(GstElement* element, GstPad* pad, gpointer data);

private:
    StreamDecoder* decoder_;
    
    // Helper to configure file sources
    bool configureFileSource(PipelineElements& elements, const std::string& id_str);
    
    // Helper to configure RTSP sources
    bool configureRtspSource(PipelineElements& elements);
};

} // namespace rtsp

#endif // STREAM_PIPELINE_BUILDER_HPP
