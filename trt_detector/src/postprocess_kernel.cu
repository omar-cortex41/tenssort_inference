#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace trt_detector {

// CUDA kernel for GPU-accelerated postprocessing
// Processes detection output: filter by confidence, transform coordinates, clamp bounds
__global__ void postprocess_detections_kernel(
    const float* __restrict__ raw_output,  // [300, 6] flattened
    int* __restrict__ d_valid_count,       // Output: number of valid detections
    float* __restrict__ d_boxes,           // Output: [max_dets, 4] (x1, y1, w, h)
    float* __restrict__ d_scores,          // Output: [max_dets]
    int* __restrict__ d_class_ids,         // Output: [max_dets]
    int max_dets,
    float conf_threshold,
    float scale,
    float pad_x,
    float pad_y,
    int frame_w,
    int frame_h
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_dets) return;

    const int stride = 6;  // x1, y1, x2, y2, conf, class_id
    const float* det = raw_output + idx * stride;

    float x1_raw = det[0];
    float y1_raw = det[1];
    float x2_raw = det[2];
    float y2_raw = det[3];
    float conf = det[4];
    int class_id = static_cast<int>(det[5]);

    // Early exit for invalid detections
    if (conf < conf_threshold || x2_raw <= x1_raw || y2_raw <= y1_raw) {
        return;
    }

    // Transform from letterboxed 640x640 back to original frame coordinates
    int x1 = static_cast<int>((x1_raw - pad_x) / scale);
    int y1 = static_cast<int>((y1_raw - pad_y) / scale);
    int x2 = static_cast<int>((x2_raw - pad_x) / scale);
    int y2 = static_cast<int>((y2_raw - pad_y) / scale);

    // Clamp to frame bounds
    x1 = max(0, min(x1, frame_w - 1));
    y1 = max(0, min(y1, frame_h - 1));
    x2 = max(0, min(x2, frame_w));
    y2 = max(0, min(y2, frame_h));

    int w = x2 - x1;
    int h = y2 - y1;

    if (w <= 0 || h <= 0) return;

    // Atomic increment to get output index
    int out_idx = atomicAdd(d_valid_count, 1);

    if (out_idx < max_dets) {
        d_boxes[out_idx * 4 + 0] = static_cast<float>(x1);
        d_boxes[out_idx * 4 + 1] = static_cast<float>(y1);
        d_boxes[out_idx * 4 + 2] = static_cast<float>(w);
        d_boxes[out_idx * 4 + 3] = static_cast<float>(h);
        d_scores[out_idx] = conf;
        d_class_ids[out_idx] = class_id;
    }
}

// Host wrapper for GPU postprocessing
void postprocess_detections_gpu(
    const float* d_raw_output,
    int* d_valid_count,
    float* d_boxes,
    float* d_scores,
    int* d_class_ids,
    int max_dets,
    float conf_threshold,
    float scale,
    float pad_x,
    float pad_y,
    int frame_w,
    int frame_h,
    cudaStream_t stream
) {
    // Reset valid count to 0
    cudaMemsetAsync(d_valid_count, 0, sizeof(int), stream);

    int threads = 256;
    int blocks = (max_dets + threads - 1) / threads;

    postprocess_detections_kernel<<<blocks, threads, 0, stream>>>(
        d_raw_output,
        d_valid_count,
        d_boxes,
        d_scores,
        d_class_ids,
        max_dets,
        conf_threshold,
        scale,
        pad_x,
        pad_y,
        frame_w,
        frame_h
    );
}

} // namespace trt_detector
