/**
 * CUDA Preprocessing Kernels - Letterbox + BGR2RGB + Normalize + HWC2CHW
 * Also includes NV12 to RGB conversion for zero-copy RTSP integration
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

/**
 * NV12 to RGB + Letterbox + Normalize + HWC2CHW kernel
 * Combines color conversion and preprocessing in a single pass
 *
 * NV12 format: Y plane (H x W) followed by interleaved UV plane (H/2 x W)
 */
__global__ void nv12ToRgbPreprocessKernel(
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ uv_plane,
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
        // Padding area - gray (114/255)
        r = g = b = 114.0f * norm;
    } else {
        // Map to source coordinates
        float sx = rx / scale;
        float sy = ry / scale;

        int x0 = (int)sx;
        int y0 = (int)sy;

        // Clamp to valid range
        x0 = min(max(x0, 0), src_w - 1);
        y0 = min(max(y0, 0), src_h - 1);

        // Get Y value
        float Y = (float)y_plane[y0 * src_w + x0];

        // Get UV values (subsampled 2x2)
        int uv_x = x0 / 2;
        int uv_y = y0 / 2;
        int uv_idx = uv_y * src_w + uv_x * 2;  // UV is interleaved

        float U = (float)uv_plane[uv_idx] - 128.0f;
        float V = (float)uv_plane[uv_idx + 1] - 128.0f;

        // YUV to RGB conversion (BT.601)
        float rf = Y + 1.402f * V;
        float gf = Y - 0.344136f * U - 0.714136f * V;
        float bf = Y + 1.772f * U;

        // Clamp and normalize
        r = fminf(fmaxf(rf, 0.0f), 255.0f) * norm;
        g = fminf(fmaxf(gf, 0.0f), 255.0f) * norm;
        b = fminf(fmaxf(bf, 0.0f), 255.0f) * norm;
    }

    // CHW format (RGB order for YOLO)
    dst[dst_idx]          = r;
    dst[hw + dst_idx]     = g;
    dst[2 * hw + dst_idx] = b;
}

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

/**
 * NV12 preprocessing - converts NV12 GPU buffer to preprocessed float tensor
 * Zero-copy path for RTSPModule integration
 *
 * @param d_nv12 Device pointer to NV12 frame (Y plane followed by UV plane)
 * @param d_dst  Device pointer to output float tensor (CHW format)
 * @param src_w  Source frame width
 * @param src_h  Source frame height
 * @param dst_w  Target width (e.g., 640)
 * @param dst_h  Target height (e.g., 640)
 * @param out_scale  Output: scale factor used
 * @param out_pad_x  Output: horizontal padding
 * @param out_pad_y  Output: vertical padding
 * @param stream CUDA stream for async execution
 */
extern "C" void cudaPreprocessNV12(
    const uint8_t* d_nv12, float* d_dst,
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

    // NV12 layout: Y plane is src_h * src_w, UV plane follows
    const uint8_t* y_plane = d_nv12;
    const uint8_t* uv_plane = d_nv12 + (src_h * src_w);

    dim3 block(32, 32);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    nv12ToRgbPreprocessKernel<<<grid, block, 0, stream>>>(
        y_plane, uv_plane, d_dst, src_w, src_h, dst_w, dst_h,
        new_w, new_h, pad_x, pad_y, scale
    );
}
