#include "trt_detector/preprocessor.hpp"
#include <cstring>

namespace trt_detector {

LetterboxInfo Preprocessor::letterbox(const cv::Mat& src, cv::Mat& dst,
                                       int target_w, int target_h,
                                       const cv::Scalar& color) {
    LetterboxInfo info;
    
    int src_h = src.rows;
    int src_w = src.cols;
    
    info.scale = std::min(static_cast<float>(target_h) / src_h,
                          static_cast<float>(target_w) / src_w);
    
    int new_h = static_cast<int>(src_h * info.scale);
    int new_w = static_cast<int>(src_w * info.scale);
    
    info.pad_x = (target_w - new_w) / 2;
    info.pad_y = (target_h - new_h) / 2;
    
    dst.create(target_h, target_w, CV_8UC3);
    dst.setTo(color);
    
    cv::Mat roi = dst(cv::Rect(info.pad_x, info.pad_y, new_w, new_h));
    cv::resize(src, roi, roi.size(), 0, 0, cv::INTER_LINEAR);
    
    return info;
}

LetterboxInfo Preprocessor::process(const cv::Mat& src, float* dst,
                                    int target_w, int target_h) {
    thread_local cv::Mat letterboxed;
    LetterboxInfo info = letterbox(src, letterboxed, target_w, target_h);
    
    thread_local cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    
    if (!rgb.isContinuous()) {
        rgb = rgb.clone();
    }
    
    // Optimized HWC to CHW conversion
    const int hw = target_h * target_w;
    const float scale = 1.0f / 255.0f;
    const uint8_t* __restrict src_ptr = rgb.ptr<uint8_t>();
    float* __restrict dst_r = dst;
    float* __restrict dst_g = dst + hw;
    float* __restrict dst_b = dst + hw * 2;
    
    int i = 0;
    const int hw_aligned = hw - (hw % 4);
    
    for (; i < hw_aligned; i += 4) {
        const uint8_t* p = src_ptr + i * 3;
        dst_r[i]     = p[0]  * scale;
        dst_g[i]     = p[1]  * scale;
        dst_b[i]     = p[2]  * scale;
        dst_r[i + 1] = p[3]  * scale;
        dst_g[i + 1] = p[4]  * scale;
        dst_b[i + 1] = p[5]  * scale;
        dst_r[i + 2] = p[6]  * scale;
        dst_g[i + 2] = p[7]  * scale;
        dst_b[i + 2] = p[8]  * scale;
        dst_r[i + 3] = p[9]  * scale;
        dst_g[i + 3] = p[10] * scale;
        dst_b[i + 3] = p[11] * scale;
    }
    
    for (; i < hw; ++i) {
        const uint8_t* p = src_ptr + i * 3;
        dst_r[i] = p[0] * scale;
        dst_g[i] = p[1] * scale;
        dst_b[i] = p[2] * scale;
    }
    
    return info;
}

} // namespace trt_detector

