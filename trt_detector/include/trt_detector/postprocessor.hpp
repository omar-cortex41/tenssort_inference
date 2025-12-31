#pragma once

#include "detection.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace trt_detector {

class Postprocessor {
public:
    static std::vector<Detection> process(
        const float* raw_output,
        int num_detections,
        int num_classes,
        float conf_threshold,
        float nms_threshold,
        float scale,
        float pad_x,
        float pad_y,
        int frame_w,
        int frame_h,
        const std::vector<std::string>& class_names
    );

private:
    static std::vector<int> nms(
        const std::vector<cv::Rect>& boxes,
        const std::vector<float>& scores,
        float nms_threshold
    );
    
    static float iou(const cv::Rect& a, const cv::Rect& b);
};

} // namespace trt_detector

