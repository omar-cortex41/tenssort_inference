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
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    
    for (int i = 0; i < num_detections; ++i) {
        float cx = raw_output[0 * num_detections + i];
        float cy = raw_output[1 * num_detections + i];
        float bw = raw_output[2 * num_detections + i];
        float bh = raw_output[3 * num_detections + i];
        
        float max_score = 0.0f;
        int max_class = 0;
        for (int c = 0; c < num_classes; ++c) {
            float score = raw_output[(4 + c) * num_detections + i];
            if (score > max_score) {
                max_score = score;
                max_class = c;
            }
        }
        
        if (max_score < conf_threshold) continue;
        
        int x1 = static_cast<int>((cx - bw / 2 - pad_x) / scale);
        int y1 = static_cast<int>((cy - bh / 2 - pad_y) / scale);
        int w = static_cast<int>(bw / scale);
        int h = static_cast<int>(bh / scale);
        
        x1 = std::max(0, std::min(x1, frame_w - 1));
        y1 = std::max(0, std::min(y1, frame_h - 1));
        w = std::min(w, frame_w - x1);
        h = std::min(h, frame_h - y1);
        
        boxes.emplace_back(x1, y1, w, h);
        confidences.push_back(max_score);
        class_ids.push_back(max_class);
    }
    
    std::vector<int> keep = nms(boxes, confidences, nms_threshold);
    
    std::vector<Detection> detections;
    detections.reserve(keep.size());
    
    for (int idx : keep) {
        std::string label = (class_ids[idx] < static_cast<int>(class_names.size())) 
            ? class_names[class_ids[idx]] : std::to_string(class_ids[idx]);
        detections.emplace_back(
            boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height,
            class_ids[idx], confidences[idx], label
        );
    }
    
    return detections;
}

} // namespace trt_detector

