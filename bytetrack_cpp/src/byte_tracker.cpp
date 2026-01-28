#include "bytetrack/byte_tracker.hpp"
#include "bytetrack/lapjv.hpp"
#include <algorithm>
#include <unordered_set>

namespace bytetrack {

BYTETracker::BYTETracker(const TrackerConfig& config)
    : config_(config)
    , max_time_lost_(static_cast<int>(config.frame_rate / 30.0f * config.track_buffer)) {
}

void BYTETracker::reset() {
    frame_id_ = 0;
    tracked_stracks_.clear();
    lost_stracks_.clear();
    removed_stracks_.clear();
    STrack::resetId();
}

std::vector<BYTETracker::STrackPtr> BYTETracker::update(const std::vector<Detection>& detections) {
    frame_id_++;
    
    // Split detections by confidence
    std::vector<STrackPtr> det_high, det_low;
    
    for (const auto& det : detections) {
        Eigen::Vector4f tlwh(det.x, det.y, det.width, det.height);
        auto strack = std::make_shared<STrack>(tlwh, det.confidence, det.class_id);
        
        if (det.confidence >= config_.high_thresh) {
            det_high.push_back(strack);
        } else if (det.confidence >= config_.track_thresh) {
            det_low.push_back(strack);
        }
    }
    
    // Separate tracked and unconfirmed tracks
    std::vector<STrackPtr> unconfirmed, tracked_stracks;
    for (auto& track : tracked_stracks_) {
        if (!track->isActivated()) {
            unconfirmed.push_back(track);
        } else {
            tracked_stracks.push_back(track);
        }
    }
    
    // Combine tracked and lost for matching
    std::vector<STrackPtr> strack_pool = jointStracks(tracked_stracks, lost_stracks_);
    
    // Predict current locations
    STrack::multiPredict(strack_pool, kalman_filter_);
    
    // === First association: high confidence detections ===
    std::vector<Eigen::Vector4f> track_tlbrs, det_tlbrs;
    for (const auto& t : strack_pool) track_tlbrs.push_back(t->tlbr());
    for (const auto& d : det_high) det_tlbrs.push_back(d->tlbr());
    
    Eigen::MatrixXf dists = iouDistance(track_tlbrs, det_tlbrs);
    
    auto [matches, u_track, u_det] = LAPJV::solve(dists, config_.match_thresh);
    
    std::vector<STrackPtr> activated, refind;
    std::vector<STrackPtr> u_tracks_first, u_dets_first;
    
    for (auto& [it, id] : matches) {
        auto& track = strack_pool[it];
        auto& det = det_high[id];
        
        if (track->state() == TrackState::Tracked) {
            track->update(*det, frame_id_);
            activated.push_back(track);
        } else {
            track->reActivate(*det, frame_id_, false);
            refind.push_back(track);
        }
    }
    
    for (int i : u_track) u_tracks_first.push_back(strack_pool[i]);
    for (int i : u_det) u_dets_first.push_back(det_high[i]);
    
    // === Second association: low confidence detections to remaining tracks ===
    std::vector<STrackPtr> r_tracked;
    for (auto& t : u_tracks_first) {
        if (t->state() == TrackState::Tracked) {
            r_tracked.push_back(t);
        }
    }
    
    track_tlbrs.clear();
    det_tlbrs.clear();
    for (const auto& t : r_tracked) track_tlbrs.push_back(t->tlbr());
    for (const auto& d : det_low) det_tlbrs.push_back(d->tlbr());
    
    dists = iouDistance(track_tlbrs, det_tlbrs);
    
    auto [matches2, u_track2, u_det2] = LAPJV::solve(dists, 0.5f);
    
    for (auto& [it, id] : matches2) {
        auto& track = r_tracked[it];
        auto& det = det_low[id];
        
        if (track->state() == TrackState::Tracked) {
            track->update(*det, frame_id_);
            activated.push_back(track);
        } else {
            track->reActivate(*det, frame_id_, false);
            refind.push_back(track);
        }
    }
    
    // Mark unmatched tracks as lost
    std::vector<STrackPtr> lost_tracks;
    for (int i : u_track2) {
        auto& track = r_tracked[i];
        if (track->state() != TrackState::Lost) {
            track->markLost();
            lost_tracks.push_back(track);
        }
    }
    
    // === Third association: unconfirmed tracks ===
    track_tlbrs.clear();
    det_tlbrs.clear();
    for (const auto& t : unconfirmed) track_tlbrs.push_back(t->tlbr());
    for (const auto& d : u_dets_first) det_tlbrs.push_back(d->tlbr());
    
    dists = iouDistance(track_tlbrs, det_tlbrs);
    auto [matches3, u_track3, u_det3] = LAPJV::solve(dists, 0.7f);
    
    for (auto& [it, id] : matches3) {
        unconfirmed[it]->update(*u_dets_first[id], frame_id_);
        activated.push_back(unconfirmed[it]);
    }
    
    // Remove unmatched unconfirmed tracks
    std::vector<STrackPtr> removed;
    for (int i : u_track3) {
        unconfirmed[i]->markRemoved();
        removed.push_back(unconfirmed[i]);
    }
    
    // Initialize new tracks from remaining high-conf detections
    for (int i : u_det3) {
        auto& det = u_dets_first[i];
        if (det->score() >= config_.high_thresh) {
            det->activate(kalman_filter_, frame_id_);
            activated.push_back(det);
        }
    }

    // Update lost tracks
    for (auto& track : lost_stracks_) {
        if (frame_id_ - track->endFrame() > max_time_lost_) {
            track->markRemoved();
            removed.push_back(track);
        }
    }

    // Update track lists
    tracked_stracks_ = jointStracks(jointStracks(tracked_stracks, activated), refind);
    tracked_stracks_ = subStracks(tracked_stracks_, lost_tracks);
    lost_stracks_ = jointStracks(subStracks(lost_stracks_, tracked_stracks_), lost_tracks);
    lost_stracks_ = subStracks(lost_stracks_, removed);

    auto [res_tracked, res_lost] = removeDuplicateStracks(tracked_stracks_, lost_stracks_);
    tracked_stracks_ = res_tracked;
    lost_stracks_ = res_lost;

    // Return active tracks
    std::vector<STrackPtr> output;
    for (auto& track : tracked_stracks_) {
        if (track->isActivated()) {
            output.push_back(track);
        }
    }

    return output;
}

std::vector<TrackInfo> BYTETracker::getTrackInfo() const {
    std::vector<TrackInfo> info;
    for (const auto& track : tracked_stracks_) {
        if (track->isActivated()) {
            TrackInfo ti;
            ti.track_id = track->trackId();
            auto box = track->tlwh();
            ti.x = box(0);
            ti.y = box(1);
            ti.width = box(2);
            ti.height = box(3);
            ti.confidence = track->score();
            ti.class_id = track->classId();
            ti.state = track->state();
            info.push_back(ti);
        }
    }
    return info;
}

std::vector<BYTETracker::STrackPtr> BYTETracker::jointStracks(
    const std::vector<STrackPtr>& a, const std::vector<STrackPtr>& b) const {

    std::unordered_set<int> seen;
    std::vector<STrackPtr> result;

    for (const auto& t : a) {
        seen.insert(t->trackId());
        result.push_back(t);
    }

    for (const auto& t : b) {
        if (seen.find(t->trackId()) == seen.end()) {
            result.push_back(t);
        }
    }

    return result;
}

std::vector<BYTETracker::STrackPtr> BYTETracker::subStracks(
    const std::vector<STrackPtr>& a, const std::vector<STrackPtr>& b) const {

    std::unordered_set<int> b_ids;
    for (const auto& t : b) {
        b_ids.insert(t->trackId());
    }

    std::vector<STrackPtr> result;
    for (const auto& t : a) {
        if (b_ids.find(t->trackId()) == b_ids.end()) {
            result.push_back(t);
        }
    }

    return result;
}

std::pair<std::vector<BYTETracker::STrackPtr>, std::vector<BYTETracker::STrackPtr>>
BYTETracker::removeDuplicateStracks(const std::vector<STrackPtr>& a,
                                     const std::vector<STrackPtr>& b) const {
    std::vector<Eigen::Vector4f> atlbrs, btlbrs;
    for (const auto& t : a) atlbrs.push_back(t->tlbr());
    for (const auto& t : b) btlbrs.push_back(t->tlbr());

    Eigen::MatrixXf dists = iouDistance(atlbrs, btlbrs);

    std::vector<bool> keep_a(a.size(), true);
    std::vector<bool> keep_b(b.size(), true);

    for (int i = 0; i < dists.rows(); ++i) {
        for (int j = 0; j < dists.cols(); ++j) {
            if (dists(i, j) < 0.15f) {  // IoU > 0.85
                int timep_a = a[i]->frameId() - a[i]->startFrame();
                int timep_b = b[j]->frameId() - b[j]->startFrame();
                if (timep_a > timep_b) {
                    keep_b[j] = false;
                } else {
                    keep_a[i] = false;
                }
            }
        }
    }

    std::vector<STrackPtr> res_a, res_b;
    for (size_t i = 0; i < a.size(); ++i) {
        if (keep_a[i]) res_a.push_back(a[i]);
    }
    for (size_t i = 0; i < b.size(); ++i) {
        if (keep_b[i]) res_b.push_back(b[i]);
    }

    return {res_a, res_b};
}

} // namespace bytetrack

