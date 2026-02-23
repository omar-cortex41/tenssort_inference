#ifndef BATCH_TYPES_H
#define BATCH_TYPES_H

#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief Configuration for batch frame retrieval
 *
 * Controls which cameras to fetch from and how to handle offline streams.
 */
struct BatchConfig {
    std::vector<int> camera_ids;    ///< List of camera IDs to fetch frames from
    int timeout_ms = 10;            ///< Max wait time per frame (ms)
    int target_width = 0;           ///< Target width for uniformity check (0 = use first valid frame)
    int target_height = 0;          ///< Target height for uniformity check (0 = use first valid frame)

    // Zero-copy output (optional): if set, writes directly to external buffer
    uint8_t* output_ptr = nullptr;  ///< External buffer to write frames to (null = use internal buffer)
    size_t output_size = 0;         ///< Size of external buffer in bytes
};

/**
 * @brief Configuration for adaptive batch frame retrieval
 *
 * Extends BatchConfig with adaptive batching logic - will automatically
 * attempt to collect min_batch_size frames, retrying if initial fetch
 * doesn't have enough valid frames.
 */
struct AdaptiveBatchConfig {
    std::vector<int> camera_ids;    ///< List of camera IDs to fetch frames from
    int min_batch_size = 1;         ///< Minimum frames required (will retry if below this)
    int max_batch_size = 8;         ///< Maximum frames to collect
    int timeout_ms = 10;            ///< Max wait time per frame on first attempt (ms)
    int retry_timeout_ms = 5;       ///< Max wait time per frame on retry attempt (ms)
    int target_width = 0;           ///< Target width for uniformity check
    int target_height = 0;          ///< Target height for uniformity check

    // Zero-copy output (optional)
    uint8_t* output_ptr = nullptr;  ///< External buffer to write frames to
    size_t output_size = 0;         ///< Size of external buffer in bytes
};

/**
 * @brief Metadata for a single frame within a batch
 * 
 * Provides per-frame information without the pixel data (which is packed separately).
 */
struct BatchedFrameMeta {
    int camera_id = -1;             ///< Camera index this frame came from
    uint64_t frame_id = 0;          ///< Sequential frame counter from decoder
    uint64_t timestamp_ns = 0;      ///< Presentation timestamp in nanoseconds
    int width = 0;                  ///< Frame width in pixels
    int height = 0;                 ///< Frame height in pixels
    bool valid = false;             ///< True if frame was successfully retrieved
};

/**
 * @brief Result of a batched frame retrieval operation
 * 
 * Contains contiguous pixel data for all frames plus per-frame metadata.
 * Invalid frames (offline cameras, timeouts) are zeroed in the data buffer.
 * 
 * Memory Layout (for NV12 format):
 *   data = [Frame0_Y][Frame0_UV][Frame1_Y][Frame1_UV]...[FrameN_Y][FrameN_UV]
 * 
 * Memory Layout (for BGR/RGB format):
 *   data = [Frame0_BGR][Frame1_BGR]...[FrameN_BGR]
 */
struct FrameBatch {
    std::vector<BatchedFrameMeta> metadata;  ///< Per-frame metadata (same order as data)
    std::vector<bool> valid_mask;            ///< Quick validity check per slot
    std::vector<uint8_t> data;               ///< Contiguous pixel data (all frames packed)
    
    size_t batch_size = 0;          ///< Number of frames in batch (matches requested count)
    size_t frame_stride = 0;        ///< Bytes per frame (for indexing into data buffer)
    int width = 0;                  ///< Common frame width (all frames same size)
    int height = 0;                 ///< Common frame height
    std::string format;             ///< Pixel format: "NV12", "BGR", "RGB", etc.
    
    size_t valid_count = 0;         ///< Number of frames with valid=true
    
    /**
     * @brief Get pointer to start of frame N in data buffer
     */
    const uint8_t* frame_ptr(size_t index) const {
        if (index >= batch_size || frame_stride == 0) return nullptr;
        return data.data() + (index * frame_stride);
    }
    
    /**
     * @brief Check if batch has uniform resolution
     */
    bool is_uniform() const {
        return width > 0 && height > 0;
    }
};

#endif // BATCH_TYPES_H
