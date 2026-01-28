#pragma once

#include "data_types.hpp"
#include "strack.hpp"
#include "kalman_filter.hpp"
#include <vector>
#include <memory>

namespace bytetrack {

/**
 * ByteTrack multi-object tracker.
 * 
 * Two-stage association:
 * 1. Match high-confidence detections to existing tracks
 * 2. Match low-confidence detections to remaining unmatched tracks
 */
class BYTETracker {
public:
    using STrackPtr = std::shared_ptr<STrack>;
    
    explicit BYTETracker(const TrackerConfig& config = TrackerConfig());
    
    /**
     * Update tracker with new detections.
     * 
     * @param detections Vector of detections for current frame
     * @return Vector of active tracks
     */
    std::vector<STrackPtr> update(const std::vector<Detection>& detections);
    
    /**
     * Get current track info (for Python bindings).
     */
    std::vector<TrackInfo> getTrackInfo() const;
    
    /**
     * Reset tracker state.
     */
    void reset();
    
    // Getters
    int frameId() const { return frame_id_; }
    const std::vector<STrackPtr>& trackedTracks() const { return tracked_stracks_; }
    const std::vector<STrackPtr>& lostTracks() const { return lost_stracks_; }

private:
    // Association helpers
    std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr>& a,
                                         const std::vector<STrackPtr>& b) const;
    
    std::vector<STrackPtr> subStracks(const std::vector<STrackPtr>& a,
                                       const std::vector<STrackPtr>& b) const;
    
    std::pair<std::vector<STrackPtr>, std::vector<STrackPtr>>
    removeDuplicateStracks(const std::vector<STrackPtr>& a,
                           const std::vector<STrackPtr>& b) const;

    TrackerConfig config_;
    KalmanFilter kalman_filter_;
    
    int frame_id_ = 0;
    int max_time_lost_;
    
    std::vector<STrackPtr> tracked_stracks_;
    std::vector<STrackPtr> lost_stracks_;
    std::vector<STrackPtr> removed_stracks_;
};

} // namespace bytetrack

