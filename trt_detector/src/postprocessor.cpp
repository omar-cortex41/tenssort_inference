#include "trt_detector/postprocessor.hpp"
#include <algorithm>
#include <numeric>

namespace trt_detector {

float Postprocessor::iou(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    
    int inter_w = std::max(0, x2 - x1);
    int inter_h = std::max(0, y2 - y1);
    int inter_area = inter_w * inter_h;
    int union_area = a.width * a.height + b.width * b.height - inter_area;
    
    return union_area > 0 ? static_cast<float>(inter_area) / union_area : 0.0f;
}

std::vector<int> Postprocessor::nms(const std::vector<cv::Rect>& boxes,
                                    const std::vector<float>& scores,
                                    float nms_threshold) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(indices.size(), false);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx_j = indices[j];
            if (suppressed[idx_j]) continue;
            
            if (iou(boxes[idx], boxes[idx_j]) > nms_threshold) {
                suppressed[idx_j] = true;
            }
        }
    }
    
    return keep;
}

std::vector<Detection> Postprocessor::process(
    const float* raw_output, int num_detections, int num_classes,
    float conf_threshold, float nms_threshold,
    float scale, float pad_x, float pad_y,
    int frame_w, int frame_h,
    const std::vector<std::string>& class_names
) {
    std::vector<Detection> detections;

    // Post-NMS format: [num_detections, 6] where each row is [x1, y1, x2, y2, conf, class_id]
    // num_detections here is actually 6 (the stride), real count is from engine shape
    // The output is [300, 6] flattened, so we iterate over 300 detections
    const int stride = 6;  // x1, y1, x2, y2, conf, class_id
    const int max_dets = 300;  // Ultralytics default max detections

    for (int i = 0; i < max_dets; ++i) {
        const float* det = raw_output + i * stride;

        float x1_raw = det[0];
        float y1_raw = det[1];
        float x2_raw = det[2];
        float y2_raw = det[3];
        float conf = det[4];
        int class_id = static_cast<int>(det[5]);

        // Skip low confidence or invalid detections
        if (conf < conf_threshold || x2_raw <= x1_raw || y2_raw <= y1_raw) continue;

        // Transform from letterboxed 640x640 back to original frame coordinates
        int x1 = static_cast<int>((x1_raw - pad_x) / scale);
        int y1 = static_cast<int>((y1_raw - pad_y) / scale);
        int x2 = static_cast<int>((x2_raw - pad_x) / scale);
        int y2 = static_cast<int>((y2_raw - pad_y) / scale);

        // Clamp to frame bounds
        x1 = std::max(0, std::min(x1, frame_w - 1));
        y1 = std::max(0, std::min(y1, frame_h - 1));
        x2 = std::max(0, std::min(x2, frame_w));
        y2 = std::max(0, std::min(y2, frame_h));

        int w = x2 - x1;
        int h = y2 - y1;

        if (w <= 0 || h <= 0) continue;

        std::string label = (class_id < static_cast<int>(class_names.size()))
            ? class_names[class_id] : std::to_string(class_id);

        detections.emplace_back(x1, y1, w, h, class_id, conf, label);
    }

    return detections;
}

} // namespace trt_detector

