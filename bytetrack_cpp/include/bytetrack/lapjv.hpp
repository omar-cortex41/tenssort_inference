#pragma once

#include <Eigen/Dense>
#include <vector>
#include <tuple>

namespace bytetrack {

/**
 * Linear Assignment Problem solver using Jonker-Volgenant algorithm.
 * This is a C++ implementation of the LAPJV algorithm used in the Python 'lap' library.
 */
class LAPJV {
public:
    /**
     * Solve the linear assignment problem.
     * 
     * @param cost_matrix NxM cost matrix
     * @param thresh Cost threshold (assignments above this are rejected)
     * @return tuple of (matches, unmatched_a, unmatched_b)
     *         - matches: Kx2 matrix of (row, col) pairs
     *         - unmatched_a: indices of unmatched rows
     *         - unmatched_b: indices of unmatched columns
     */
    static std::tuple<std::vector<std::pair<int, int>>, 
                      std::vector<int>, 
                      std::vector<int>>
    solve(const Eigen::MatrixXf& cost_matrix, float thresh);

private:
    static constexpr float LARGE = 1e9f;
    
    // Core LAPJV algorithm
    static void lapjv_internal(int n, const float* cost, 
                               int* x, int* y,
                               float* u, float* v);
};

/**
 * Compute IoU distance matrix between two sets of tracks.
 * 
 * @param atlbrs First set of bounding boxes in tlbr format
 * @param btlbrs Second set of bounding boxes in tlbr format
 * @return NxM distance matrix (1 - IoU)
 */
Eigen::MatrixXf iouDistance(const std::vector<Eigen::Vector4f>& atlbrs,
                            const std::vector<Eigen::Vector4f>& btlbrs);

/**
 * Compute IoU between two boxes.
 */
float computeIou(const Eigen::Vector4f& box1, const Eigen::Vector4f& box2);

/**
 * Fuse detection scores with IoU cost matrix.
 */
Eigen::MatrixXf fuseScore(const Eigen::MatrixXf& cost_matrix,
                          const std::vector<float>& scores);

} // namespace bytetrack

