#ifndef BUFFER_MAPPER_HPP
#define BUFFER_MAPPER_HPP

#include <gst/gst.h>
#include <string>
#include <memory>
#include "rtspmodule/stream_decoder.h"

namespace rtsp {

class BufferMapper {
public:
    // Extract GPU memory pointer from a GstBuffer (supporting NVMM DeepStream and CUDA)
    // Returns true if successfully mapped, updates ptr and stride
    static bool mapGpuBuffer(GstBuffer* buffer, 
                             bool use_nvmm_memory, 
                             bool use_cuda_memory, 
                             int width, 
                             int height,
                             const int* info_stride,
                             uint64_t& out_ptr, 
                             int& out_stride);

    // Compute expected size and bytes_per_pixel for a given format
    static void getFormatRequirements(const std::string& format, 
                                      int width, 
                                      int height, 
                                      size_t& out_expected_size, 
                                      int& out_bytes_per_pixel);

    // Push a GstBuffer to the CPU ring buffer, handling stride and planar formats (like NV12)
    static void pushToCpuBuffer(CpuBuffer* cpu_buffer, 
                                GstBuffer* buffer, 
                                const std::string& format,
                                int width, 
                                int height, 
                                int current_stride,
                                uint64_t frame_id);
};

} // namespace rtsp

#endif // BUFFER_MAPPER_HPP
