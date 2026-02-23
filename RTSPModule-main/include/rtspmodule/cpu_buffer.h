#ifndef CPU_BUFFER_H
#define CPU_BUFFER_H

#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
#include "rtsp_structs.h"

/**
 * Ring Buffer implementation with:
 * - Lazy allocation: slots allocated on first use
 * - Explicit head (read) and tail (write) pointers
 * - FIFO semantics: get() returns oldest unread frame
 * - Memory-efficient: reuses allocated frame wrappers (object pooling)
 */
class CpuBuffer {
public:
  explicit CpuBuffer(size_t capacity = 60);
  ~CpuBuffer() = default;
  
  // Non-copyable
  CpuBuffer(const CpuBuffer&) = delete;
  CpuBuffer& operator=(const CpuBuffer&) = delete;
  
  // Push new frame at tail (overwrites oldest if full)
  void push(CpuFrame&& frame);
  
  // Get frame from head
  // timeout_ms: 0 = non-blocking (returns empty if no frames), >0 = wait up to timeout
  CpuFrame get(int timeout_ms = 0);
  
  // Get multiple frames from head (FIFO order, oldest first)
  // Returns up to 'count' frames. Waits up to timeout_ms for at least 1 frame.
  std::vector<CpuFrame> getMulti(int count, int timeout_ms = 0);
  
  // Clear buffer and release all memory
  void clear();
  
  // Resize capacity (clears existing data and releases memory)
  void resize(size_t new_capacity);
  
  // Stats
  size_t size() const;
  size_t capacity() const;
  size_t memoryUsage() const;
  bool isEmpty() const;
  bool isFull() const;
  
  // Peek at most recent frame without removing (for batch retrieval)
  // Returns pointer to internal data - caller MUST NOT hold this across push() calls
  // timeout_ms: 0 = non-blocking, >0 = wait for frame
  const CpuFrame* peekLatest(int timeout_ms = 0) const;

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<std::unique_ptr<CpuFrame>> buffer_;  // Lazy-allocated slots
  size_t capacity_;                // Maximum capacity
  size_t head_ = 0;                // Read pointer
  size_t tail_ = 0;                // Write pointer
  size_t count_ = 0;               // Current element count
};

#endif // CPU_BUFFER_H
