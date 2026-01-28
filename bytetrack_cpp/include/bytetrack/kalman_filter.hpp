#pragma once

#include <Eigen/Dense>
#include <utility>

namespace bytetrack {

/**
 * Kalman filter for tracking bounding boxes in image space.
 * 
 * State space (8D): [x, y, a, h, vx, vy, va, vh]
 *   - (x, y): bounding box center
 *   - a: aspect ratio (width / height)
 *   - h: height
 *   - vx, vy, va, vh: respective velocities
 */
class KalmanFilter {
public:
    using StateVector = Eigen::Matrix<float, 8, 1>;
    using StateCov = Eigen::Matrix<float, 8, 8>;
    using MeasVector = Eigen::Matrix<float, 4, 1>;
    using MeasCov = Eigen::Matrix<float, 4, 4>;
    
    KalmanFilter();
    
    /**
     * Initialize a new track from measurement.
     * @param measurement [x, y, a, h] - center x, center y, aspect ratio, height
     * @return (mean, covariance) of new track
     */
    std::pair<StateVector, StateCov> initiate(const MeasVector& measurement) const;
    
    /**
     * Predict next state.
     * @param mean Current state mean
     * @param covariance Current state covariance
     * @return (predicted_mean, predicted_covariance)
     */
    std::pair<StateVector, StateCov> predict(const StateVector& mean, 
                                              const StateCov& covariance) const;
    
    /**
     * Update state with measurement.
     * @param mean Predicted state mean
     * @param covariance Predicted state covariance
     * @param measurement New measurement [x, y, a, h]
     * @return (updated_mean, updated_covariance)
     */
    std::pair<StateVector, StateCov> update(const StateVector& mean,
                                             const StateCov& covariance,
                                             const MeasVector& measurement) const;

private:
    // Project state to measurement space
    std::pair<MeasVector, MeasCov> project(const StateVector& mean,
                                            const StateCov& covariance) const;
    
    Eigen::Matrix<float, 8, 8> motion_mat_;      // State transition matrix F
    Eigen::Matrix<float, 4, 8> update_mat_;      // Measurement matrix H
    float std_weight_position_;
    float std_weight_velocity_;
};

} // namespace bytetrack

