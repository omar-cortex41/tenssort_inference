#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/**
 * GPU-accelerated postprocessing for YOLO TRT output.
 * Performs confidence filtering + coordinate transformation on GPU,
 * avoiding CPU-side iteration over 300 detections per batch element.
 *
 * Input:  Raw TRT output [batch, 300, 6] format: [x1, y1, x2, y2, conf, class_id]
 * Output: Compact detection buffer with valid detections only
 *
 * @param d_raw_output    Device pointer to raw TRT output (batch * 300 * 6 floats)
 * @param d_detections    Device output buffer (batch * max_dets * 7 floats: x,y,w,h,cls,conf,valid)
 * @param d_det_counts    Device output: number of valid detections per batch element (batch ints)
 * @param batch_size      Number of images in batch
 * @param max_dets        Max detections per image (300)
 * @param conf_threshold  Minimum confidence threshold
 * @param scales          Array of letterbox scale factors (one per batch, host memory)
 * @param pad_xs          Array of horizontal padding (one per batch, host memory)
 * @param pad_ys          Array of vertical padding (one per batch, host memory)
 * @param frame_ws        Array of original frame widths (one per batch, host memory)
 * @param frame_hs        Array of original frame heights (one per batch, host memory)
 * @param stream          CUDA stream for async execution
 */
extern "C" void cudaPostprocess(
    const float* d_raw_output,
    float* d_detections,
    int* d_det_counts,
    int batch_size,
    int max_dets,
    float conf_threshold,
    const float* d_scales,
    const int* d_pad_xs,
    const int* d_pad_ys,
    const int* d_frame_ws,
    const int* d_frame_hs,
    cudaStream_t stream
);
