#include <rtspmodule/cpu_buffer.h>
#include <stdexcept>

CpuBuffer::CpuBuffer(size_t capacity) 
    : capacity_(capacity) {
  if (capacity_ == 0) {
    throw std::invalid_argument("CpuBuffer capacity must be > 0");
  }
  buffer_.resize(capacity_);  // Allocates only nullptr pointers (8 bytes each)
}

void CpuBuffer::push(CpuFrame&& frame) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Allocate and write at tail position
    // If slot is already allocated (from previous use), reuse it
    if (buffer_[tail_]) {
      *buffer_[tail_] = std::move(frame);
    } else {
      buffer_[tail_] = std::make_unique<CpuFrame>(std::move(frame));
    }
    
    // Advance tail with wrap-around
    tail_ = (tail_ + 1) % capacity_;
    
    if (count_ < capacity_) {
      ++count_;
    } else {
      // Buffer was full, oldest was overwritten, advance head
      head_ = (head_ + 1) % capacity_;
    }
  }
  cv_.notify_one();
}

CpuFrame CpuBuffer::get(int timeout_ms) {
  std::unique_lock<std::mutex> lock(mutex_);
  
  // Wait for data if empty and timeout requested
  if (count_ == 0 && timeout_ms > 0) {
    cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                 [this] { return count_ > 0; });
  }
  
  if (count_ == 0) {
    return CpuFrame{};  // Empty or timeout
  }
  
  // Read from head
  // Move data out, but keep the CpuFrame unique_ptr allocated for reuse
  CpuFrame result = std::move(*buffer_[head_]);
  result.valid = true;
  head_ = (head_ + 1) % capacity_;
  --count_;
  
  return result;
}

std::vector<CpuFrame> CpuBuffer::getMulti(int count, int timeout_ms) {
  std::vector<CpuFrame> results;
  if (count <= 0) return results;
  
  results.reserve(static_cast<size_t>(count));
  
  std::unique_lock<std::mutex> lock(mutex_);
  
  // Wait for at least 1 frame if empty and timeout requested
  if (count_ == 0 && timeout_ms > 0) {
    cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                 [this] { return count_ > 0; });
  }
  
  // Pop up to 'count' frames from head (FIFO: oldest first)
  const size_t to_pop = std::min(static_cast<size_t>(count), count_);
  for (size_t i = 0; i < to_pop; ++i) {
    if (buffer_[head_] && buffer_[head_]->valid) {
      results.push_back(std::move(*buffer_[head_]));
      results.back().valid = true;
    }
    head_ = (head_ + 1) % capacity_;
    --count_;
  }
  
  return results;
}

void CpuBuffer::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Release all allocated frames
  for (auto& frame : buffer_) {
    frame.reset();
  }
  
  head_ = 0;
  tail_ = 0;
  count_ = 0;
}

void CpuBuffer::resize(size_t new_capacity) {
  if (new_capacity == 0) {
    throw std::invalid_argument("CpuBuffer capacity must be > 0");
  }
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Create fresh empty buffer (only nullptr pointers)
  std::vector<std::unique_ptr<CpuFrame>> new_buffer(new_capacity);
  buffer_.swap(new_buffer);
  capacity_ = new_capacity;
  
  // Reset to empty state
  head_ = 0;
  tail_ = 0;
  count_ = 0;
}

size_t CpuBuffer::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return count_;
}

size_t CpuBuffer::capacity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return capacity_;
}

size_t CpuBuffer::memoryUsage() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t total = 0;
  for (const auto& frame : buffer_) {
    if (frame) {
      total += frame->data.size();
    }
  }
  return total;
}

bool CpuBuffer::isEmpty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return count_ == 0;
}

bool CpuBuffer::isFull() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return count_ == capacity_;
}

const CpuFrame* CpuBuffer::peekLatest(int timeout_ms) const {
  (void)timeout_ms;  // Unused - peek is always non-blocking
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (count_ == 0) {
    return nullptr;  // Empty buffer
  }
  
  // Return pointer to most recent frame (one before tail, with wrap-around)
  size_t latest_idx = (tail_ + capacity_ - 1) % capacity_;
  if (buffer_[latest_idx] && buffer_[latest_idx]->valid) {
    return buffer_[latest_idx].get();
  }
  return nullptr;
}
