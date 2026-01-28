#include "bytetrack/lapjv.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

namespace bytetrack {

float computeIou(const Eigen::Vector4f& box1, const Eigen::Vector4f& box2) {
    float x1 = std::max(box1(0), box2(0));
    float y1 = std::max(box1(1), box2(1));
    float x2 = std::min(box1(2), box2(2));
    float y2 = std::min(box1(3), box2(3));
    
    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;
    
    float area1 = (box1(2) - box1(0)) * (box1(3) - box1(1));
    float area2 = (box2(2) - box2(0)) * (box2(3) - box2(1));
    float union_area = area1 + area2 - inter_area;
    
    return union_area > 0 ? inter_area / union_area : 0.0f;
}

Eigen::MatrixXf iouDistance(const std::vector<Eigen::Vector4f>& atlbrs,
                            const std::vector<Eigen::Vector4f>& btlbrs) {
    size_t n = atlbrs.size();
    size_t m = btlbrs.size();
    
    Eigen::MatrixXf cost_matrix(n, m);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            cost_matrix(i, j) = 1.0f - computeIou(atlbrs[i], btlbrs[j]);
        }
    }
    
    return cost_matrix;
}

Eigen::MatrixXf fuseScore(const Eigen::MatrixXf& cost_matrix,
                          const std::vector<float>& scores) {
    if (cost_matrix.size() == 0) return cost_matrix;
    
    Eigen::MatrixXf iou_sim = Eigen::MatrixXf::Ones(cost_matrix.rows(), cost_matrix.cols()) 
                              - cost_matrix;
    
    Eigen::MatrixXf fused = cost_matrix;
    for (int i = 0; i < cost_matrix.rows(); ++i) {
        for (int j = 0; j < cost_matrix.cols(); ++j) {
            fused(i, j) = 1.0f - iou_sim(i, j) * scores[j];
        }
    }
    
    return fused;
}

// LAPJV implementation based on the Jonker-Volgenant algorithm
void LAPJV::lapjv_internal(int n, const float* cost, int* x, int* y, 
                           float* u, float* v) {
    const float INF = std::numeric_limits<float>::max();
    
    std::vector<int> free_rows(n);
    std::vector<float> d(n);
    std::vector<int> pred(n);
    std::vector<bool> in_v(n, false);
    
    // Initialize
    for (int i = 0; i < n; ++i) {
        x[i] = -1;
        y[i] = -1;
        u[i] = 0;
        v[i] = 0;
    }
    
    // Column reduction
    for (int j = n - 1; j >= 0; --j) {
        float min_val = cost[j];
        int i_min = 0;
        for (int i = 1; i < n; ++i) {
            if (cost[i * n + j] < min_val) {
                min_val = cost[i * n + j];
                i_min = i;
            }
        }
        v[j] = min_val;
        
        if (x[i_min] < 0) {
            x[i_min] = j;
            y[j] = i_min;
        }
    }
    
    // Reduction transfer and augmenting row reduction
    int num_free = 0;
    for (int i = 0; i < n; ++i) {
        if (x[i] < 0) {
            free_rows[num_free++] = i;
        } else {
            int j1 = x[i];
            float min_val = INF;
            for (int j = 0; j < n; ++j) {
                if (j != j1) {
                    float val = cost[i * n + j] - v[j];
                    if (val < min_val) min_val = val;
                }
            }
            u[i] = min_val;
        }
    }
    
    // Augmentation
    for (int f = 0; f < num_free; ++f) {
        int i0 = free_rows[f];
        
        for (int j = 0; j < n; ++j) {
            d[j] = cost[i0 * n + j] - u[i0] - v[j];
            pred[j] = i0;
            in_v[j] = false;
        }
        
        int j0 = -1, j1 = -1;
        float min_val = INF;
        
        while (true) {
            // Find minimum
            min_val = INF;
            for (int j = 0; j < n; ++j) {
                if (!in_v[j] && d[j] < min_val) {
                    min_val = d[j];
                    j0 = j;
                }
            }
            
            in_v[j0] = true;
            int i1 = y[j0];
            
            if (i1 < 0) break;  // Augmenting path found
            
            // Update distances
            for (int j = 0; j < n; ++j) {
                if (!in_v[j]) {
                    float new_d = cost[i1 * n + j] - u[i1] - v[j] + min_val;
                    if (new_d < d[j]) {
                        d[j] = new_d;
                        pred[j] = i1;
                    }
                }
            }
        }
        
        // Update dual variables
        for (int j = 0; j < n; ++j) {
            if (in_v[j]) {
                v[j] += d[j] - min_val;
            }
        }
        u[i0] += min_val;

        // Augment
        while (j0 >= 0) {
            int i1 = pred[j0];
            y[j0] = i1;
            std::swap(j0, x[i1]);
        }
    }
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
LAPJV::solve(const Eigen::MatrixXf& cost_matrix, float thresh) {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_a, unmatched_b;

    int rows = cost_matrix.rows();
    int cols = cost_matrix.cols();

    if (rows == 0 || cols == 0) {
        for (int i = 0; i < rows; ++i) unmatched_a.push_back(i);
        for (int j = 0; j < cols; ++j) unmatched_b.push_back(j);
        return {matches, unmatched_a, unmatched_b};
    }

    // Make square matrix for LAPJV (pad with LARGE values)
    int n = std::max(rows, cols);
    std::vector<float> cost(n * n, LARGE);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cost[i * n + j] = cost_matrix(i, j);
        }
    }

    std::vector<int> x(n), y(n);
    std::vector<float> u(n), v(n);

    lapjv_internal(n, cost.data(), x.data(), y.data(), u.data(), v.data());

    // Extract matches
    std::vector<bool> matched_b(cols, false);
    for (int i = 0; i < rows; ++i) {
        int j = x[i];
        if (j < cols && cost_matrix(i, j) < thresh) {
            matches.push_back({i, j});
            matched_b[j] = true;
        } else {
            unmatched_a.push_back(i);
        }
    }

    for (int j = 0; j < cols; ++j) {
        if (!matched_b[j]) {
            unmatched_b.push_back(j);
        }
    }

    return {matches, unmatched_a, unmatched_b};
}

} // namespace bytetrack

