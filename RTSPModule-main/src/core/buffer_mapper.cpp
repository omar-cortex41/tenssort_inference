#include <rtspmodule/buffer_mapper.hpp>
#include <gst/allocators/gstdmabuf.h>
#include <cstring>
#include <iostream>
#include <mutex>

#ifdef HAVE_DEEPSTREAM
#include "nvbufsurface.h"
#endif

#ifdef HAVE_GST_CUDA
#include <gst/cuda/gstcudamemory.h>
#endif

namespace rtsp {

bool BufferMapper::mapGpuBuffer(GstBuffer* buffer, 
                                bool use_nvmm_memory, 
                                bool use_cuda_memory, 
                                int width, 
                                int height,
                                const int* info_stride,
                                uint64_t& out_ptr, 
                                int& out_stride) {
    if (use_nvmm_memory) {
#ifdef HAVE_DEEPSTREAM
        GstMemory *mem = gst_buffer_peek_memory(buffer, 0);
        if (mem && gst_is_dmabuf_memory(mem)) {
            int dmabuf_fd = gst_dmabuf_memory_get_fd(mem);
            NvBufSurface *surf = nullptr;
            if (NvBufSurfaceFromFd(dmabuf_fd, (void **)&surf) == 0 && surf) {
                if (surf->numFilled > 0 && surf->surfaceList[0].dataPtr) {
                    out_ptr = (uint64_t)surf->surfaceList[0].dataPtr;
                    out_stride = surf->surfaceList[0].pitch;
                    return true;
                }
            }
        }
#else
        static std::once_flag warn_flag;
        std::call_once(warn_flag, []() {
            std::cerr << "[BufferMapper] WARNING: NVMM path selected but HAVE_DEEPSTREAM not defined" << std::endl;
        });
#endif
    }

#ifdef HAVE_GST_CUDA
    if (use_cuda_memory) {
        GstMemory *mem = gst_buffer_peek_memory(buffer, 0);
        if (mem && gst_is_cuda_memory(mem)) {
            GstMapInfo map;
            if (gst_memory_map(mem, &map, (GstMapFlags)(GST_MAP_READ | GST_MAP_CUDA))) {
                out_ptr = reinterpret_cast<uint64_t>(map.data);
                int inferred_stride = (int)(map.size / (height * 1.5)); // NV12
                if (inferred_stride > info_stride[0] && (map.size % (int)(height * 1.5) == 0)) {
                    out_stride = inferred_stride;
                } else {
                    out_stride = info_stride[0];
                }
                gst_memory_unmap(mem, &map);
                return true;
            }
        }
    }
#endif

    return false;
}

void BufferMapper::getFormatRequirements(const std::string& format, 
                                         int width, int height, 
                                         size_t& out_expected_size, 
                                         int& out_bytes_per_pixel) {
    if (format == "NV12" || format == "I420") {
        out_expected_size = static_cast<size_t>(width * height * 1.5);
        out_bytes_per_pixel = 1;  // For Y plane
    } else if (format == "RGB" || format == "BGR") {
        out_expected_size = static_cast<size_t>(width * height * 3);
        out_bytes_per_pixel = 3;
    } else if (format == "RGBA" || format == "BGRA") {
        out_expected_size = static_cast<size_t>(width * height * 4);
        out_bytes_per_pixel = 4;
    } else {
        out_expected_size = 0;
        out_bytes_per_pixel = 0;
    }
}

void BufferMapper::pushToCpuBuffer(CpuBuffer* cpu_buffer, 
                                   GstBuffer* buffer, 
                                   const std::string& format,
                                   int width, 
                                   int height, 
                                   int current_stride,
                                   uint64_t frame_id) {
    if (!cpu_buffer) return;

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        return;
    }

    CpuFrame frame;
    frame.width = width;
    frame.height = height;
    frame.frame_id = frame_id;
    frame.timestamp_ns = GST_BUFFER_PTS(buffer);
    frame.capture_time_ns = std::chrono::steady_clock::now().time_since_epoch().count();
    frame.format = format;
    frame.valid = true;

    size_t expected_size = 0;
    int bytes_per_pixel = 0;
    getFormatRequirements(format, width, height, expected_size, bytes_per_pixel);

    if (expected_size == 0) {
        expected_size = map.size;
    }

    bool has_stride_padding = (map.size > expected_size);

    if (!has_stride_padding || bytes_per_pixel == 0) {
        frame.data.assign(map.data, map.data + std::min(map.size, expected_size));
    } else if (format == "NV12") {
        int stride = current_stride > 0 ? current_stride : width;
        frame.data.resize(expected_size);
        uint8_t* dst = frame.data.data();
        const uint8_t* src = map.data;

        // Copy Y plane row by row
        for (int y = 0; y < height; y++) {
            std::memcpy(dst, src + y * stride, width);
            dst += width;
        }

        // Copy UV plane row by row
        const uint8_t* uv_src = map.data + stride * height;
        for (int y = 0; y < height / 2; y++) {
            std::memcpy(dst, uv_src + y * stride, width);
            dst += width;
        }
    } else {
        int stride = static_cast<int>(map.size / height);
        int row_bytes = width * bytes_per_pixel;

        frame.data.resize(expected_size);
        uint8_t* dst = frame.data.data();
        const uint8_t* src = map.data;

        for (int y = 0; y < height; y++) {
            std::memcpy(dst, src + y * stride, row_bytes);
            dst += row_bytes;
        }
    }

    frame.data_size = frame.data.size();
    gst_buffer_unmap(buffer, &map);

    cpu_buffer->push(std::move(frame));
}

} // namespace rtsp
