/**
 * CUDA Preprocessing Kernels - Letterbox + BGR2RGB + Normalize + HWC2CHW
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

__global__ void preprocessKernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    int new_w, int new_h,
    int pad_x, int pad_y,
    float scale
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dx >= dst_w || dy >= dst_h) return;
    
    const int hw = dst_w * dst_h;
    const int dst_idx = dy * dst_w + dx;
    const float norm = 1.0f / 255.0f;
    
    int rx = dx - pad_x;
    int ry = dy - pad_y;
    
    float r, g, b;
    
    if (rx < 0 || rx >= new_w || ry < 0 || ry >= new_h) {
        r = g = b = 114.0f * norm;
    } else {
        float sx = rx / scale;
        float sy = ry / scale;
        
        int x0 = (int)sx;
        int y0 = (int)sy;
        int x1 = min(x0 + 1, src_w - 1);
        int y1 = min(y0 + 1, src_h - 1);
        
        float fx = sx - x0;
        float fy = sy - y0;
        
        x0 = min(max(x0, 0), src_w - 1);
        y0 = min(max(y0, 0), src_h - 1);
        
        const uint8_t* p00 = src + (y0 * src_w + x0) * 3;
        const uint8_t* p01 = src + (y0 * src_w + x1) * 3;
        const uint8_t* p10 = src + (y1 * src_w + x0) * 3;
        const uint8_t* p11 = src + (y1 * src_w + x1) * 3;
        
        float w00 = (1 - fx) * (1 - fy);
        float w01 = fx * (1 - fy);
        float w10 = (1 - fx) * fy;
        float w11 = fx * fy;
        
        // BGR to RGB
        r = (p00[2] * w00 + p01[2] * w01 + p10[2] * w10 + p11[2] * w11) * norm;
        g = (p00[1] * w00 + p01[1] * w01 + p10[1] * w10 + p11[1] * w11) * norm;
        b = (p00[0] * w00 + p01[0] * w01 + p10[0] * w10 + p11[0] * w11) * norm;
    }
    
    // CHW format
    dst[dst_idx]          = r;
    dst[hw + dst_idx]     = g;
    dst[2 * hw + dst_idx] = b;
}

extern "C" void cudaPreprocess(
    const uint8_t* d_src, float* d_dst,
    int src_w, int src_h, int dst_w, int dst_h,
    float* out_scale, int* out_pad_x, int* out_pad_y,
    cudaStream_t stream
) {
    float scale = min((float)dst_w / src_w, (float)dst_h / src_h);
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);
    int pad_x = (dst_w - new_w) / 2;
    int pad_y = (dst_h - new_h) / 2;
    
    *out_scale = scale;
    *out_pad_x = pad_x;
    *out_pad_y = pad_y;
    
    dim3 block(32, 32);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    
    preprocessKernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, src_w, src_h, dst_w, dst_h,
        new_w, new_h, pad_x, pad_y, scale
    );
}

