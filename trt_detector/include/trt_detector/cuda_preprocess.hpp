#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// BGR preprocessing - letterbox + BGR2RGB + normalize + HWC2CHW
extern "C" void cudaPreprocess(
    const uint8_t* d_src, float* d_dst,
    int src_w, int src_h, int dst_w, int dst_h,
    float* out_scale, int* out_pad_x, int* out_pad_y,
    cudaStream_t stream
);

// NV12 preprocessing - NV12 to RGB + letterbox + normalize + HWC2CHW
// Skips CPU color conversion entirely
extern "C" void cudaPreprocessNV12(
    const uint8_t* d_nv12, float* d_dst,
    int src_w, int src_h, int dst_w, int dst_h,
    float* out_scale, int* out_pad_x, int* out_pad_y,
    cudaStream_t stream
);

