/**
 * CUDA OSD Kernel implementation.
 * Renders bounding boxes and labels on GPU frames.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// Simple color generation (HSV to RGB logic or fixed palette)
__device__ void getClassColor(int class_id, uint8_t& r, uint8_t& g, uint8_t& b) {
    // Golden ratio conjugate based color generation
    const float golden_ratio_conjugate = 0.618033988749895;
    float h = (class_id * golden_ratio_conjugate);
    h = h - (int)h;
    h *= 6.0f;
    
    float s = 0.8f;
    float v = 0.95f;
    
    int i = (int)h;
    float f = h - i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);
    
    float r_f, g_f, b_f;
    switch (i % 6) {
        case 0: r_f = v; g_f = t; b_f = p; break;
        case 1: r_f = q; g_f = v; b_f = p; break;
        case 2: r_f = p; g_f = v; b_f = t; break;
        case 3: r_f = p; g_f = q; b_f = v; break;
        case 4: r_f = t; g_f = p; b_f = v; break;
        case 5: r_f = v; g_f = p; b_f = q; break;
    }
    
    r = (uint8_t)(r_f * 255);
    g = (uint8_t)(g_f * 255);
    b = (uint8_t)(b_f * 255);
}

// Draw a filled or outlined rectangle
// Each thread handles a portion of the box
__global__ void renderBBoxesKernel(
    uint8_t* __restrict__ frame,
    int width, int height,
    const float* __restrict__ detections,
    int num_detections,
    int stride_floats = 7 // [x,y,w,h,cls,conf,valid]
) {
    // One block per detection
    int det_idx = blockIdx.x;
    if (det_idx >= num_detections) return;
    
    const float* det = detections + det_idx * stride_floats;
    
    // Check valid flag
    if (det[6] < 0.5f) return;
    
    int x1 = (int)det[0];
    int y1 = (int)det[1];
    int w = (int)det[2];
    int h = (int)det[3];
    int cls = (int)det[4];
    
    int x2 = x1 + w;
    int y2 = y1 + h;
    
    // Clip to frame
    x1 = max(0, min(x1, width - 1));
    y1 = max(0, min(y1, height - 1));
    x2 = max(0, min(x2, width));
    y2 = max(0, min(y2, height));
    
    if (x2 <= x1 || y2 <= y1) return;
    
    uint8_t r, g, b;
    getClassColor(cls, r, g, b);
    
    // Thickness of border
    const int thickness = 2;
    
    // Parallelize drawing across threads in the block
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    // Total perimeter pixels approx 2*(w+h)
    // We can iterate over all pixels in the bounding box outer shell
    // Top and Bottom horizontal lines
    for (int x = x1 + tid; x < x2; x += blockSize) {
        for (int t = 0; t < thickness; ++t) {
            // Top
            int y_top = y1 + t;
            if (y_top < y2) {
                int idx = (y_top * width + x) * 3;
                frame[idx + 0] = b; // BGR
                frame[idx + 1] = g;
                frame[idx + 2] = r;
            }
            
            // Bottom
            int y_bot = y2 - 1 - t;
            if (y_bot >= y1) {
                int idx = (y_bot * width + x) * 3;
                frame[idx + 0] = b;
                frame[idx + 1] = g;
                frame[idx + 2] = r;
            }
        }
    }
    
    // Left and Right vertical lines
    for (int y = y1 + tid; y < y2; y += blockSize) {
        for (int t = 0; t < thickness; ++t) {
            // Left
            int x_left = x1 + t;
            if (x_left < x2) {
                int idx = (y * width + x_left) * 3;
                frame[idx + 0] = b;
                frame[idx + 1] = g;
                frame[idx + 2] = r;
            }
            
            // Right
            int x_right = x2 - 1 - t;
            if (x_right >= x1) {
                int idx = (y * width + x_right) * 3;
                frame[idx + 0] = b;
                frame[idx + 1] = g;
                frame[idx + 2] = r;
            }
        }
    }
    
    // Draw label background (filled rect at top-left)
    // Fixed size or proportional to text (simulated size here: 50x20)
    int lbl_w = 60;
    int lbl_h = 20;
    int lbl_x2 = min(x1 + lbl_w, width);
    int lbl_y1 = max(0, y1 - lbl_h);
    int lbl_y2 = y1;
    
    if (lbl_y1 < lbl_y2) {
        // Iterate over pixels in label box
        int area = (lbl_x2 - x1) * (lbl_y2 - lbl_y1);
        for (int i = tid; i < area; i += blockSize) {
            int local_y = i / (lbl_x2 - x1);
            int local_x = i % (lbl_x2 - x1);
            int py = lbl_y1 + local_y;
            int px = x1 + local_x;
            
            int idx = (py * width + px) * 3;
            frame[idx + 0] = b;
            frame[idx + 1] = g;
            frame[idx + 2] = r;
        }
    }
}

extern "C" void cudaOSD(
    uint8_t* d_frame,          
    int width, int height,
    const float* d_detections, 
    int num_detections,
    int class_count,
    cudaStream_t stream
) {
    if (num_detections == 0) return;
    
    int block_size = 128; // Enough for border drawing
    renderBBoxesKernel<<<num_detections, block_size, 0, stream>>>(
        d_frame, width, height, d_detections, num_detections
    );
}
