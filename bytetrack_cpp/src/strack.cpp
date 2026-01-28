#include "bytetrack/strack.hpp"

namespace bytetrack {

int STrack::id_count_ = 0;

int STrack::nextId() {
    return ++id_count_;
}

void STrack::resetId() {
    id_count_ = 0;
}

STrack::STrack(const Eigen::Vector4f& tlwh, float score, int class_id)
    : class_id_(class_id)
    , score_(score)
    , tlwh_(tlwh) {
    mean_ = KalmanFilter::StateVector::Zero();
    covariance_ = KalmanFilter::StateCov::Zero();
}

void STrack::activate(const KalmanFilter& kf, int frame_id) {
    kalman_filter_ = &kf;
    track_id_ = nextId();
    
    auto [mean, cov] = kf.initiate(tlwhToXyah(tlwh_));
    mean_ = mean;
    covariance_ = cov;
    
    tracklet_len_ = 0;
    state_ = TrackState::Tracked;
    
    if (frame_id == 1) {
        is_activated_ = true;
    }
    
    frame_id_ = frame_id;
    start_frame_ = frame_id;
}

void STrack::reActivate(const STrack& new_track, int frame_id, bool new_id) {
    auto [mean, cov] = kalman_filter_->update(mean_, covariance_, 
                                               tlwhToXyah(new_track.tlwh()));
    mean_ = mean;
    covariance_ = cov;
    
    tracklet_len_ = 0;
    state_ = TrackState::Tracked;
    is_activated_ = true;
    frame_id_ = frame_id;
    
    if (new_id) {
        track_id_ = nextId();
    }
    
    score_ = new_track.score();
}

void STrack::update(const STrack& new_track, int frame_id) {
    frame_id_ = frame_id;
    tracklet_len_++;
    
    auto [mean, cov] = kalman_filter_->update(mean_, covariance_,
                                               tlwhToXyah(new_track.tlwh()));
    mean_ = mean;
    covariance_ = cov;
    
    state_ = TrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.score();
}

void STrack::predict() {
    if (state_ != TrackState::Tracked) {
        mean_(7) = 0;  // Reset height velocity if not tracked
    }
    
    auto [mean, cov] = kalman_filter_->predict(mean_, covariance_);
    mean_ = mean;
    covariance_ = cov;
}

void STrack::multiPredict(std::vector<std::shared_ptr<STrack>>& tracks,
                          const KalmanFilter& kf) {
    for (auto& track : tracks) {
        if (track->state_ != TrackState::Tracked) {
            track->mean_(7) = 0;
        }
        auto [mean, cov] = kf.predict(track->mean_, track->covariance_);
        track->mean_ = mean;
        track->covariance_ = cov;
    }
}

Eigen::Vector4f STrack::tlwh() const {
    if (kalman_filter_ == nullptr) {
        return tlwh_;
    }
    
    // Convert from xyah to tlwh
    Eigen::Vector4f ret;
    ret(0) = mean_(0) - mean_(2) * mean_(3) / 2;  // x = cx - w/2
    ret(1) = mean_(1) - mean_(3) / 2;              // y = cy - h/2
    ret(2) = mean_(2) * mean_(3);                  // w = a * h
    ret(3) = mean_(3);                             // h
    return ret;
}

Eigen::Vector4f STrack::tlbr() const {
    Eigen::Vector4f box = tlwh();
    box(2) += box(0);  // x2 = x + w
    box(3) += box(1);  // y2 = y + h
    return box;
}

Eigen::Vector4f STrack::xyah() const {
    return tlwhToXyah(tlwh());
}

Eigen::Vector4f STrack::tlwhToXyah(const Eigen::Vector4f& tlwh) {
    Eigen::Vector4f ret;
    ret(0) = tlwh(0) + tlwh(2) / 2;  // cx = x + w/2
    ret(1) = tlwh(1) + tlwh(3) / 2;  // cy = y + h/2
    ret(2) = tlwh(2) / tlwh(3);      // a = w / h
    ret(3) = tlwh(3);                // h
    return ret;
}

Eigen::Vector4f STrack::tlbrToTlwh(const Eigen::Vector4f& tlbr) {
    Eigen::Vector4f ret;
    ret(0) = tlbr(0);
    ret(1) = tlbr(1);
    ret(2) = tlbr(2) - tlbr(0);  // w = x2 - x1
    ret(3) = tlbr(3) - tlbr(1);  // h = y2 - y1
    return ret;
}

Eigen::Vector4f STrack::tlwhToTlbr(const Eigen::Vector4f& tlwh) {
    Eigen::Vector4f ret;
    ret(0) = tlwh(0);
    ret(1) = tlwh(1);
    ret(2) = tlwh(0) + tlwh(2);  // x2 = x + w
    ret(3) = tlwh(1) + tlwh(3);  // y2 = y + h
    return ret;
}

} // namespace bytetrack

