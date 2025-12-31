#pragma once

#include <string>

namespace trt_detector {

struct Detection {
    int x;
    int y;
    int width;
    int height;
    int class_id;
    float confidence;
    std::string label;
    
    Detection() = default;
    
    Detection(int x, int y, int w, int h, int cls_id, float conf, const std::string& lbl)
        : x(x), y(y), width(w), height(h), class_id(cls_id), confidence(conf), label(lbl) {}
};

} // namespace trt_detector

