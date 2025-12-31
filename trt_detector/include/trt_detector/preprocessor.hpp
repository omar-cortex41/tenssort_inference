#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace trt_detector {

struct LetterboxInfo {
    float scale;
    int pad_x;
    int pad_y;
};

class Preprocessor {
public:
    // Letterbox resize: preserves aspect ratio, pads with gray
    static LetterboxInfo letterbox(const cv::Mat& src, cv::Mat& dst, 
                                   int target_w, int target_h,
                                   const cv::Scalar& color = cv::Scalar(114, 114, 114));
    
    // Full preprocessing: letterbox + BGR2RGB + normalize + HWC2CHW
    static LetterboxInfo process(const cv::Mat& src, float* dst,
                                 int target_w, int target_h);
};

} // namespace trt_detector

