#include "bytetrack/kalman_filter.hpp"

namespace bytetrack {

KalmanFilter::KalmanFilter() 
    : std_weight_position_(1.0f / 20.0f)
    , std_weight_velocity_(1.0f / 160.0f) {
    
    // Initialize motion matrix (state transition) F
    // State: [x, y, a, h, vx, vy, va, vh]
    // x' = x + vx, y' = y + vy, etc.
    motion_mat_ = Eigen::Matrix<float, 8, 8>::Identity();
    for (int i = 0; i < 4; ++i) {
        motion_mat_(i, i + 4) = 1.0f;  // dt = 1
    }
    
    // Initialize measurement matrix H
    // We observe [x, y, a, h] from state [x, y, a, h, vx, vy, va, vh]
    update_mat_ = Eigen::Matrix<float, 4, 8>::Zero();
    for (int i = 0; i < 4; ++i) {
        update_mat_(i, i) = 1.0f;
    }
}

std::pair<KalmanFilter::StateVector, KalmanFilter::StateCov>
KalmanFilter::initiate(const MeasVector& measurement) const {
    // Initialize mean: [x, y, a, h, 0, 0, 0, 0]
    StateVector mean = StateVector::Zero();
    mean.head<4>() = measurement;
    
    // Initialize covariance
    float h = measurement(3);
    std::array<float, 8> std = {
        2.0f * std_weight_position_ * h,
        2.0f * std_weight_position_ * h,
        1e-2f,
        2.0f * std_weight_position_ * h,
        10.0f * std_weight_velocity_ * h,
        10.0f * std_weight_velocity_ * h,
        1e-5f,
        10.0f * std_weight_velocity_ * h
    };
    
    StateCov covariance = StateCov::Zero();
    for (int i = 0; i < 8; ++i) {
        covariance(i, i) = std[i] * std[i];
    }
    
    return {mean, covariance};
}

std::pair<KalmanFilter::StateVector, KalmanFilter::StateCov>
KalmanFilter::predict(const StateVector& mean, const StateCov& covariance) const {
    float h = mean(3);
    
    // Process noise
    std::array<float, 8> std = {
        std_weight_position_ * h,
        std_weight_position_ * h,
        1e-2f,
        std_weight_position_ * h,
        std_weight_velocity_ * h,
        std_weight_velocity_ * h,
        1e-5f,
        std_weight_velocity_ * h
    };
    
    StateCov motion_cov = StateCov::Zero();
    for (int i = 0; i < 8; ++i) {
        motion_cov(i, i) = std[i] * std[i];
    }
    
    // Predict: x' = F * x, P' = F * P * F^T + Q
    StateVector new_mean = motion_mat_ * mean;
    StateCov new_cov = motion_mat_ * covariance * motion_mat_.transpose() + motion_cov;
    
    return {new_mean, new_cov};
}

std::pair<KalmanFilter::MeasVector, KalmanFilter::MeasCov>
KalmanFilter::project(const StateVector& mean, const StateCov& covariance) const {
    float h = mean(3);
    
    // Measurement noise
    std::array<float, 4> std = {
        std_weight_position_ * h,
        std_weight_position_ * h,
        1e-1f,
        std_weight_position_ * h
    };
    
    MeasCov innovation_cov = MeasCov::Zero();
    for (int i = 0; i < 4; ++i) {
        innovation_cov(i, i) = std[i] * std[i];
    }
    
    // Project: z = H * x, S = H * P * H^T + R
    MeasVector proj_mean = update_mat_ * mean;
    MeasCov proj_cov = update_mat_ * covariance * update_mat_.transpose() + innovation_cov;
    
    return {proj_mean, proj_cov};
}

std::pair<KalmanFilter::StateVector, KalmanFilter::StateCov>
KalmanFilter::update(const StateVector& mean, const StateCov& covariance,
                     const MeasVector& measurement) const {
    auto [proj_mean, proj_cov] = project(mean, covariance);
    
    // Kalman gain: K = P * H^T * S^-1
    Eigen::Matrix<float, 8, 4> kalman_gain = 
        covariance * update_mat_.transpose() * proj_cov.inverse();
    
    // Innovation (residual)
    MeasVector innovation = measurement - proj_mean;
    
    // Update: x' = x + K * y, P' = P - K * S * K^T
    StateVector new_mean = mean + kalman_gain * innovation;
    StateCov new_cov = covariance - kalman_gain * proj_cov * kalman_gain.transpose();
    
    return {new_mean, new_cov};
}

} // namespace bytetrack

