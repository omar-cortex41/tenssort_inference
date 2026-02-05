#include <rtspmodule/rtsp_client.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class FrameProvider {
public:
  FrameProvider() {
      // Ensure GStreamer is initialized before we try to create factories
      // This is critical for loading plugins from GST_PLUGIN_PATH
      if (!gst_is_initialized()) {
          gst_init(nullptr, nullptr);
      }
  }
  ~FrameProvider() { stop(); }

  void start(const std::string &config_file) {
    if (!client_.loadConfig(config_file)) {
      throw std::runtime_error("Failed to load config: " + config_file);
    }
    if (!client_.start()) {
      throw std::runtime_error("Failed to start client");
    }
  }

  void stop() { client_.stop(); }

  void set_log_path(const std::string &base_path) {
    client_.setLogPath(base_path);
  }

  bool is_running() const { return client_.isRunning(); }

  int stream_count() const { return client_.getStreamCount(); }


  py::dict get_cuda_frame(int camera_id, int timeout_ms = 0) {
    // Validate buffer mode
    if (client_.isCpuBufferEnabled()) {
      throw std::runtime_error(
          "get_cuda_frame() unavailable when cpu_buffer_enabled=true in config. "
          "Use get_cpu_frame() instead, or set cpu_buffer_enabled=false.");
    }
    
    GpuFrameInfo info;
    {
      py::gil_scoped_release release;
      info = client_.getGpuFrame(camera_id, timeout_ms);
    }
    py::dict result;

    result["valid"] = info.valid;
    if (!info.valid)
      return result;

    // Calculate shape based on format
    py::tuple shape;
    if (info.format == "NV12" || info.format == "I420") {
      int h_yuv = static_cast<int>(info.height * 1.5);
      shape = py::make_tuple(h_yuv, info.width);
    } else if (info.format == "RGB" || info.format == "BGR") {
      shape = py::make_tuple(info.height, info.width, 3);
    } else if (info.format == "RGBA" || info.format == "BGRA") {
      shape = py::make_tuple(info.height, info.width, 4);
    } else {
      // Default to flat buffer
      shape = py::make_tuple(info.size);
    }

    result["ptr"] = info.ptr;
    result["width"] = info.width;
    result["height"] = info.height;
    result["stride"] = info.stride;
    result["shape"] = shape;
    result["size"] = info.size;
    result["frame_id"] = info.frame_id;
    result["format"] = info.format;
    result["dtype"] = "uint8";

    return result;
  }

  py::dict get_stats(int camera_id) {
    FrameStats stats;
    {
      py::gil_scoped_release release;
      stats = client_.getStats(camera_id);
    }
    py::dict result;

    result["frames_received"] = stats.frames_received;
    result["frames_decoded"] = stats.frames_decoded;
    result["frames_dropped_decode"] = stats.frames_dropped_decode;
    result["frames_consumed"] = stats.frames_consumed;
    result["frames_overwritten"] = stats.frames_overwritten;
    result["frames_dropped_queue"] = stats.frames_dropped_queue;
    result["decode_errors"] = stats.decode_errors;
    result["reconnect_count"] = stats.reconnect_count;
    result["queue_depth"] = stats.queue_depth;
    result["queue_max_depth"] = stats.queue_max_depth;
    result["frames_duplicate"] = stats.frames_duplicate;
    
    // Real-time FPS from C++ (most accurate)
    result["current_fps"] = stats.current_fps;    // Sliding window average
    result["instant_fps"] = stats.instant_fps;    // 1 / last frame interval
    result["source_fps"] = stats.source_fps;      // Stream header FPS
    result["source_width"] = stats.source_width;  // Video width from headers
    result["source_height"] = stats.source_height; // Video height from headers

    // Derived stats
    uint64_t total_potential =
        stats.frames_decoded + stats.frames_dropped_decode;
    result["total_potential_frames"] = total_potential;
    result["decode_success_rate"] =
        total_potential > 0
            ? (double)stats.frames_decoded / total_potential * 100.0
            : 100.0;
    result["consumption_rate"] =
        stats.frames_decoded > 0
            ? (double)stats.frames_consumed / stats.frames_decoded * 100.0
            : 0.0;
    result["overwrite_rate"] =
        stats.frames_decoded > 0
            ? (double)stats.frames_overwritten / stats.frames_decoded * 100.0
            : 0.0;
    result["queue_drop_rate"] =
        stats.frames_decoded > 0
            ? (double)stats.frames_dropped_queue / stats.frames_decoded * 100.0
            : 0.0;

    return result;
  }

  // CPU Buffer methods
  py::dict get_cpu_frame(int camera_id, int timeout_ms = 0) {
    // Validate buffer mode
    if (!client_.isCpuBufferEnabled()) {
      throw std::runtime_error(
          "get_cpu_frame() unavailable when cpu_buffer_enabled=false in config. "
          "Use get_cuda_frame() instead, or set cpu_buffer_enabled=true.");
    }
    
    CpuFrame frame;
    {
      py::gil_scoped_release release;
      frame = client_.getCpuFrame(camera_id, timeout_ms);
    }
    
    py::dict result;
    result["valid"] = frame.valid;
    
    if (!frame.valid || frame.data.empty()) {
      return result;
    }
    
    // Calculate expected size based on format
    size_t expected_size = 0;
    std::vector<py::ssize_t> shape;
    
    if (frame.format == "NV12" || frame.format == "I420") {
      int h_yuv = static_cast<int>(frame.height * 1.5);
      expected_size = static_cast<size_t>(h_yuv * frame.width);
      shape = {h_yuv, frame.width};
    } else if (frame.format == "RGB" || frame.format == "BGR") {
      expected_size = static_cast<size_t>(frame.height * frame.width * 3);
      shape = {frame.height, frame.width, 3};
    } else if (frame.format == "RGBA" || frame.format == "BGRA") {
      expected_size = static_cast<size_t>(frame.height * frame.width * 4);
      shape = {frame.height, frame.width, 4};
    } else {
      // Default to flat buffer using actual data size
      expected_size = frame.data.size();
      shape = {static_cast<py::ssize_t>(frame.data.size())};
    }
    
    // Create NumPy array and copy data
    // Use minimum of expected size and actual data size to avoid overflow
    size_t copy_size = std::min(expected_size, frame.data.size());
    py::array_t<uint8_t> arr(shape);
    auto buf = arr.request();
    // Zero-initialize to avoid uninitialized memory if frame.data is smaller than expected
    if (copy_size < expected_size) {
      std::memset(buf.ptr, 0, expected_size);
    }
    std::memcpy(buf.ptr, frame.data.data(), copy_size);
    
    result["data"] = arr;
    result["width"] = frame.width;
    result["height"] = frame.height;
    result["format"] = frame.format;
    result["frame_id"] = frame.frame_id;
    result["timestamp_ns"] = frame.timestamp_ns;
    result["data_size"] = frame.data.size();  // Actual size for debugging
    
    return result;
  }


  py::dict get_cpu_buffer_info(int camera_id) {
    // Validate buffer mode
    if (!client_.isCpuBufferEnabled()) {
      throw std::runtime_error(
          "get_cpu_buffer_info() unavailable when cpu_buffer_enabled=false in config. "
          "Set cpu_buffer_enabled=true to use CPU buffer features.");
    }
    
    CpuBufferInfo info;
    {
      py::gil_scoped_release release;
      info = client_.getCpuBufferInfo(camera_id);
    }
    
    py::dict result;
    result["buffer_count"] = info.buffer_count;
    result["buffer_capacity"] = info.buffer_capacity;
    result["buffer_duration_sec"] = info.buffer_duration_sec;
    result["memory_usage_bytes"] = info.memory_usage_bytes;
    result["memory_usage_mb"] = info.memory_usage_bytes / (1024.0 * 1024.0);
    result["format"] = info.format;
    return result;
  }

  // Batch Frame Retrieval - get frames from multiple cameras in single call
  // ZERO-COPY: Allocates NumPy array first, passes pointer to C++ for direct writes
  py::dict get_batch(py::list camera_ids, int timeout_ms = 10) {

    
    // Validate buffer mode (batch currently requires CPU buffer)
    if (!client_.isCpuBufferEnabled()) {
      throw std::runtime_error(
          "get_batch() currently requires cpu_buffer_enabled=true in config. "
          "Set cpu_buffer_enabled=true to use batch frame retrieval.");
    }
    
    // Build config from Python list
    BatchConfig config;
    config.timeout_ms = timeout_ms;
    for (auto item : camera_ids) {
      config.camera_ids.push_back(item.cast<int>());
    }
    
    size_t batch_size = config.camera_ids.size();
    if (batch_size == 0) {
      py::dict result;
      result["data"] = py::none();
      result["valid_mask"] = py::array_t<bool>(0);
      result["metadata"] = py::list();
      result["count"] = 0;
      result["valid_count"] = 0;
      return result;
    }
    
    // Get expected dimensions from first valid frame (peek without copy)
    int width = 0, height = 0;
    std::string format = "BGR";
    double bytes_per_pixel = 3.0;
    
    for (size_t i = 0; i < batch_size && width == 0; ++i) {
      int cam_id = config.camera_ids[i];
      if (cam_id >= 0 && cam_id < client_.getStreamCount()) {
        auto info = client_.getCpuBufferInfo(cam_id);
        if (info.buffer_count > 0) {
          // Get dimensions from a quick peek
          CpuFrame frame = client_.getCpuFrame(cam_id, 0);
          if (frame.valid) {
            width = frame.width;
            height = frame.height;
            format = frame.format;
          }
        }
      }
    }
    
    // Fallback to 1080p if no frames available yet
    if (width == 0 || height == 0) {
      width = 1920;
      height = 1080;
    }
    
    // Calculate bytes per pixel based on format
    if (format == "NV12" || format == "I420") {
      bytes_per_pixel = 1.5;
    } else if (format == "RGBA" || format == "BGRA") {
      bytes_per_pixel = 4.0;
    }
    
    size_t frame_size = static_cast<size_t>(width * height * bytes_per_pixel);
    size_t total_size = batch_size * frame_size;
    
    // Pre-allocate NumPy array with proper shape BEFORE C++ call
    py::array_t<uint8_t> arr;
    if (format == "NV12" || format == "I420") {
      int h_yuv = static_cast<int>(height * 1.5);
      arr = py::array_t<uint8_t>({
        static_cast<py::ssize_t>(batch_size),
        static_cast<py::ssize_t>(h_yuv),
        static_cast<py::ssize_t>(width)
      });
    } else if (format == "RGBA" || format == "BGRA") {
      arr = py::array_t<uint8_t>({
        static_cast<py::ssize_t>(batch_size),
        static_cast<py::ssize_t>(height),
        static_cast<py::ssize_t>(width),
        static_cast<py::ssize_t>(4)
      });
    } else {
      arr = py::array_t<uint8_t>({
        static_cast<py::ssize_t>(batch_size),
        static_cast<py::ssize_t>(height),
        static_cast<py::ssize_t>(width),
        static_cast<py::ssize_t>(3)
      });
    }
    

    
    // Get NumPy buffer pointer for zero-copy writes
    auto buf = arr.request();
    config.output_ptr = static_cast<uint8_t*>(buf.ptr);
    config.output_size = total_size;
    config.target_width = width;
    config.target_height = height;
    
    // Zero the buffer for invalid frames (offline cameras show as black)
    std::memset(buf.ptr, 0, total_size);
    
    FrameBatch batch;
    {
      py::gil_scoped_release release;  // Single GIL release for entire batch
      batch = client_.getBatchedFrames(config);
    }
    

    
    py::dict result;
    result["count"] = batch.batch_size;
    result["valid_count"] = batch.valid_count;
    result["format"] = batch.format.empty() ? format : batch.format;
    result["width"] = batch.width > 0 ? batch.width : width;
    result["height"] = batch.height > 0 ? batch.height : height;
    result["data"] = arr;  // Already filled by C++ - no copy!
    
    // Create valid_mask as NumPy boolean array
    py::array_t<bool> valid_mask(batch.batch_size > 0 ? batch.batch_size : batch_size);
    auto mask_buf = valid_mask.request();
    bool* mask_ptr = static_cast<bool*>(mask_buf.ptr);
    for (size_t i = 0; i < batch_size; ++i) {
      mask_ptr[i] = (i < batch.valid_mask.size()) ? batch.valid_mask[i] : false;
    }
    result["valid_mask"] = valid_mask;
    
    // Build metadata list
    py::list metadata_list;
    for (size_t i = 0; i < batch_size; ++i) {
      py::dict meta;
      if (i < batch.metadata.size()) {
        meta["camera_id"] = batch.metadata[i].camera_id;
        meta["frame_id"] = batch.metadata[i].frame_id;
        meta["timestamp_ns"] = batch.metadata[i].timestamp_ns;
        meta["width"] = batch.metadata[i].width;
        meta["height"] = batch.metadata[i].height;
        meta["valid"] = batch.metadata[i].valid;
      } else {
        meta["camera_id"] = config.camera_ids[i];
        meta["frame_id"] = 0;
        meta["timestamp_ns"] = 0;
        meta["width"] = width;
        meta["height"] = height;
        meta["valid"] = false;
      }
      metadata_list.append(meta);
    }
    result["metadata"] = metadata_list;
    

    
    return result;
  }

  bool is_cpu_buffer_enabled() const { return client_.isCpuBufferEnabled(); }
  bool is_gpu_available() const { return client_.isGpuAvailable(); }



private:
  RtspClient client_;
};

PYBIND11_MODULE(_core, m) {
  m.doc() = "RTSP Client Module - Hardware-accelerated multi-stream RTSP decoder with GPU/CPU buffer support";

  py::class_<FrameProvider>(m, "RTSPModule",
      "Multi-stream RTSP client with NVDEC hardware decoding.\n\n"
      "Supports both GPU zero-copy frames (via CUDA) and CPU ring buffer for temporal access.")
      .def(py::init<>(), "Create a new RTSPModule instance.")
      
      .def("start", &FrameProvider::start, py::arg("config_file"),
           "Start all RTSP streams defined in the configuration file.\n\n"
           "Args:\n"
           "    config_file (str): Path to YAML configuration file.\n\n"
           "Raises:\n"
           "    RuntimeError: If config loading or stream initialization fails.")
      
      .def("stop", &FrameProvider::stop,
           "Stop all streams and release resources.")
      
      .def("set_log_path", &FrameProvider::set_log_path, py::arg("base_path"),
           "Set the base path for camera logs.\n\n"
           "Args:\n"
           "    base_path (str): Directory path for log files (creates date-based subdirectories).")
      
      .def("is_running", &FrameProvider::is_running,
           "Check if streams are currently running.\n\n"
           "Returns:\n"
           "    bool: True if streams are active, False otherwise.")
      
      .def("stream_count", &FrameProvider::stream_count,
           "Get the number of configured streams.\n\n"
           "Returns:\n"
           "    int: Number of streams loaded from config.")
      
      .def("is_cpu_buffer_enabled", &FrameProvider::is_cpu_buffer_enabled,
           "Check if CPU buffer mode is enabled.\n\n"
           "Returns:\n"
           "    bool: True if cpu_buffer_enabled=true in config or GPU unavailable.\n\n"
           "Use this to determine which frame retrieval method to use:\n"
           "    - If True: use get_cpu_frame()\n"
           "    - If False: use get_cuda_frame()")
      
      .def("is_gpu_available", &FrameProvider::is_gpu_available,
           "Check if GPU hardware (NVDEC/cudaconvert) is available.\n\n"
           "Returns:\n"
           "    bool: True if GPU hardware initialized successfully.\n\n"
           "Note: If False, CPU buffer mode is automatically enabled\n"
           "regardless of config setting.")

      .def("get_cuda_frame", &FrameProvider::get_cuda_frame,
           py::arg("camera_id"), py::arg("timeout_ms") = 0,
           "Get frame as CUDA device pointer (zero-copy GPU access).\n\n"
           "Args:\n"
           "    camera_id (int): Camera index (0 to stream_count-1).\n"
           "    timeout_ms (int): Max wait time in ms (0=non-blocking, >0=blocking wait).\n\n"
           "Returns:\n"
           "    dict: Frame info with keys:\n"
           "        - valid (bool): True if frame available\n"
           "        - ptr (int): CUDA device pointer (use with CuPy)\n"
           "        - width (int): Frame width in pixels\n"
           "        - height (int): Frame height in pixels\n"
           "        - stride (int): Row stride in bytes\n"
           "        - shape (tuple): NumPy-compatible shape based on format\n"
           "        - size (int): Total buffer size in bytes\n"
           "        - frame_id (int): Sequential frame counter\n"
           "        - format (str): Pixel format ('NV12', 'BGR', 'RGB', etc.)\n"
           "        - dtype (str): Data type ('uint8')")
      
      .def("get_stats", &FrameProvider::get_stats, py::arg("camera_id"),
           "Get stream statistics for a camera.\n\n"
           "Args:\n"
           "    camera_id (int): Camera index (0 to stream_count-1).\n\n"
           "Returns:\n"
           "    dict: Statistics with keys:\n"
           "        - frames_received, frames_decoded, frames_consumed (int)\n"
           "        - frames_duplicate, frames_dropped_queue (int)\n"
           "        - current_fps, instant_fps, source_fps (float)\n"
           "        - source_width, source_height (int)\n"
           "        - queue_depth, reconnect_count (int)\n"
           "        - decode_success_rate, consumption_rate (float %)")
      
      .def("get_cpu_frame", &FrameProvider::get_cpu_frame,
           py::arg("camera_id"), py::arg("timeout_ms") = 0,
           "Get next frame from CPU ring buffer.\n\n"
           "Args:\n"
           "    camera_id (int): Camera index (0 to stream_count-1).\n"
           "    timeout_ms (int): Max wait time in ms (0=non-blocking).\n\n"
           "Returns:\n"
           "    dict: Frame data with keys:\n"
           "        - valid (bool): True if frame available\n"
           "        - data (numpy.ndarray): Pixel data with format-appropriate shape\n"
           "        - width, height (int): Frame dimensions\n"
           "        - format (str): Pixel format ('NV12', 'BGR', etc.)\n"
           "        - frame_id (int): Sequential frame counter\n"
           "        - timestamp_ns (int): Presentation timestamp in nanoseconds")
      
      .def("get_cpu_buffer_info", &FrameProvider::get_cpu_buffer_info,
           py::arg("camera_id"),
           "Get CPU ring buffer statistics.\n\n"
           "Args:\n"
           "    camera_id (int): Camera index (0 to stream_count-1).\n\n"
           "Returns:\n"
           "    dict: Buffer info with keys:\n"
           "        - buffer_count (int): Current frames in buffer\n"
           "        - buffer_capacity (int): Maximum buffer capacity\n"
           "        - buffer_duration_sec (float): Time span of buffered frames\n"
           "        - memory_usage_bytes (int): RAM used by buffer\n"
           "        - memory_usage_mb (float): RAM used in megabytes\n"
           "        - format (str): Pixel format of buffered frames")
      
      .def("get_batch", &FrameProvider::get_batch,
           py::arg("camera_ids"), py::arg("timeout_ms") = 10,
           "Get frames from multiple cameras in a single batched call.\n\n"
           "This is optimized for AI inference pipelines that process multiple streams.\n"
           "Returns a fixed-size batch where offline/unavailable cameras have zeroed frames.\n\n"
           "Args:\n"
           "    camera_ids (list[int]): List of camera indices to fetch frames from.\n"
           "    timeout_ms (int): Max wait time per frame in ms (default: 10).\n\n"
           "Returns:\n"
           "    dict: Batch data with keys:\n"
           "        - data (numpy.ndarray): Contiguous array shape (N, H, W, C) or (N, H*1.5, W) for NV12\n"
           "        - valid_mask (numpy.ndarray[bool]): Boolean array indicating which frames are valid\n"
           "        - metadata (list[dict]): Per-frame metadata (camera_id, frame_id, timestamp_ns, etc.)\n"
           "        - count (int): Total batch size (matches len(camera_ids))\n"
           "        - valid_count (int): Number of valid frames in batch\n"
           "        - width, height (int): Common frame dimensions\n"
           "        - format (str): Pixel format\n\n"
           "Example:\n"
           "    result = provider.get_batch([0, 1, 2, 3], timeout_ms=5)\n"
           "    frames = result['data']  # shape: (4, H, W, 3) for BGR\n"
           "    valid = result['valid_mask']  # [True, True, False, True]\n"
           "    # Offline camera 2 has zeroed (black) frame");
}


