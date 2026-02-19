#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/**
 * GPU On-Screen Display (OSD) kernel.
 * Renders bounding boxes and labels directly on the GPU frame buffer.
 * 
 * Capabilities:
 * - Filled rectangles for labels (simulated)
 * - Outline rectangles for bounding boxes
 * - Configurable line thickness and colors
 * - Supports BGR (HWC) format
 * 
 * TODO: Add support for NV12 rendering if needed (requires UV plane handling)
 * TODO: Add text rendering via bitmap texture
 */
extern "C" void cudaOSD(
    uint8_t* d_frame,          // RGB/BGR frame on device (HWC)
    int width, int height,
    const float* d_detections, // [max_dets, 7] array: [x,y,w,h,cls,conf,valid]
    int num_detections,
    int class_count,           // Total classes for color generation
    cudaStream_t stream
);
