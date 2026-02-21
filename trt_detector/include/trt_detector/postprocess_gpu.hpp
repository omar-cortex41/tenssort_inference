#pragma once

#include <cuda_runtime.h>

namespace trt_detector {

// GPU-accelerated postprocessing
void postprocess_detections_gpu(
    const float* d_raw_output,      // Device pointer to raw output [300, 6]
    int* d_valid_count,             // Device pointer to output count
    float* d_boxes,                 // Device pointer to output boxes [max_dets, 4]
    float* d_scores,                // Device pointer to output scores [max_dets]
    int* d_class_ids,               // Device pointer to output class IDs [max_dets]
    int max_dets,                   // Maximum detections (300)
    float conf_threshold,           // Confidence threshold
    float scale,                    // Letterbox scale factor
    float pad_x,                    // Letterbox padding X
    float pad_y,                    // Letterbox padding Y
    int frame_w,                    // Original frame width
    int frame_h,                    // Original frame height
    cudaStream_t stream = 0         // CUDA stream
);

} // namespace trt_detector
