#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>

namespace bytetrack {

// Track states
enum class TrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
};

// Input detection structure
struct Detection {
    float x;        // top-left x
    float y;        // top-left y
    float width;
    float height;
    float confidence;
    int class_id;
    std::string label;
    
    Detection() : x(0), y(0), width(0), height(0), confidence(0), class_id(0) {}
    
    Detection(float x_, float y_, float w_, float h_, float conf_, int cls_id_ = 0, const std::string& lbl_ = "")
        : x(x_), y(y_), width(w_), height(h_), confidence(conf_), class_id(cls_id_), label(lbl_) {}
    
    // Convert to tlbr format [x1, y1, x2, y2]
    Eigen::Vector4f tlbr() const {
        return Eigen::Vector4f(x, y, x + width, y + height);
    }
    
    // Convert to tlwh format [x, y, w, h]
    Eigen::Vector4f tlwh() const {
        return Eigen::Vector4f(x, y, width, height);
    }
};

// Tracker configuration
struct TrackerConfig {
    float track_thresh = 0.5f;      // Detection confidence threshold
    float high_thresh = 0.6f;       // High confidence threshold (track_thresh + 0.1)
    float match_thresh = 0.8f;      // IoU matching threshold
    int track_buffer = 30;          // Frames to keep lost tracks
    int frame_rate = 30;            // Video frame rate
    
    TrackerConfig() : high_thresh(track_thresh + 0.1f) {}
    
    TrackerConfig(float track_th, float match_th, int buffer, int fps = 30)
        : track_thresh(track_th)
        , high_thresh(track_th + 0.1f)
        , match_thresh(match_th)
        , track_buffer(buffer)
        , frame_rate(fps) {}
};

// Output track info (for Python bindings)
struct TrackInfo {
    int track_id;
    float x, y, width, height;  // tlwh format
    float confidence;
    int class_id;
    TrackState state;
    
    TrackInfo() : track_id(0), x(0), y(0), width(0), height(0), 
                  confidence(0), class_id(0), state(TrackState::New) {}
};

} // namespace bytetrack

