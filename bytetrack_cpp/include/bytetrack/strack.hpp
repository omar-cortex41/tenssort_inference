#pragma once

#include "data_types.hpp"
#include "kalman_filter.hpp"
#include <memory>

namespace bytetrack {

/**
 * Single object track.
 */
class STrack {
public:
    STrack(const Eigen::Vector4f& tlwh, float score, int class_id = 0);
    
    // Activate a new track
    void activate(const KalmanFilter& kf, int frame_id);
    
    // Re-activate a lost track
    void reActivate(const STrack& new_track, int frame_id, bool new_id = false);
    
    // Update track with new detection
    void update(const STrack& new_track, int frame_id);
    
    // Predict next state using Kalman filter
    void predict();
    
    // Mark track as lost
    void markLost() { state_ = TrackState::Lost; }
    
    // Mark track as removed
    void markRemoved() { state_ = TrackState::Removed; }
    
    // Static prediction for multiple tracks
    static void multiPredict(std::vector<std::shared_ptr<STrack>>& tracks, 
                             const KalmanFilter& kf);
    
    // Coordinate conversions
    Eigen::Vector4f tlwh() const;           // [x, y, w, h]
    Eigen::Vector4f tlbr() const;           // [x1, y1, x2, y2]
    Eigen::Vector4f xyah() const;           // [cx, cy, aspect, h]
    
    static Eigen::Vector4f tlwhToXyah(const Eigen::Vector4f& tlwh);
    static Eigen::Vector4f tlbrToTlwh(const Eigen::Vector4f& tlbr);
    static Eigen::Vector4f tlwhToTlbr(const Eigen::Vector4f& tlwh);
    
    // Getters
    int trackId() const { return track_id_; }
    int classId() const { return class_id_; }
    float score() const { return score_; }
    TrackState state() const { return state_; }
    bool isActivated() const { return is_activated_; }
    int frameId() const { return frame_id_; }
    int startFrame() const { return start_frame_; }
    int endFrame() const { return frame_id_; }
    int trackletLen() const { return tracklet_len_; }
    
    const KalmanFilter::StateVector& mean() const { return mean_; }
    const KalmanFilter::StateCov& covariance() const { return covariance_; }
    
    // Get next track ID
    static int nextId();
    static void resetId();

private:
    // Track ID management
    static int id_count_;
    
    int track_id_ = 0;
    int class_id_ = 0;
    float score_ = 0;
    TrackState state_ = TrackState::New;
    bool is_activated_ = false;
    
    int frame_id_ = 0;
    int start_frame_ = 0;
    int tracklet_len_ = 0;
    
    // Kalman filter state
    Eigen::Vector4f tlwh_;                  // Original detection
    KalmanFilter::StateVector mean_;
    KalmanFilter::StateCov covariance_;
    const KalmanFilter* kalman_filter_ = nullptr;
};

} // namespace bytetrack

