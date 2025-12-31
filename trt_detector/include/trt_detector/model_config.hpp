#pragma once

#include <string>
#include <vector>

namespace trt_detector {

struct ModelConfig {
    std::string engine_path;
    int input_width = 640;
    int input_height = 640;
    float conf_threshold = 0.5f;
    float nms_threshold = 0.45f;
    std::vector<std::string> class_names;
    
    ModelConfig() = default;
    
    ModelConfig(const std::string& path,
                const std::vector<std::string>& classes,
                float conf = 0.5f,
                float nms = 0.45f,
                int width = 640,
                int height = 640)
        : engine_path(path)
        , input_width(width)
        , input_height(height)
        , conf_threshold(conf)
        , nms_threshold(nms)
        , class_names(classes) {}
};

} // namespace trt_detector

