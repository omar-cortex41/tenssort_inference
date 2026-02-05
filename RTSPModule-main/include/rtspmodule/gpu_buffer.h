#ifndef GPU_BUFFER_H
#define GPU_BUFFER_H

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <utility>
#include <iostream>

class GpuBuffer {
public:
    GpuBuffer() = default;
    ~GpuBuffer() { deallocate(); }
    
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    
    GpuBuffer(GpuBuffer&& other) noexcept {
        *this = std::move(other);
    }
    
    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            deallocate();
            d_ptr_ = other.d_ptr_;
            h_ptr_ = other.h_ptr_;
            size_ = other.size_;
            width_ = other.width_;
            height_ = other.height_;
            stream_ = other.stream_;
            event_ = other.event_;
            data_size_ = other.data_size_;
            host_valid_ = other.host_valid_;
            
            other.d_ptr_ = nullptr;
            other.h_ptr_ = nullptr;
            other.size_ = 0;
            other.data_size_ = 0;
            other.width_ = 0;
            other.height_ = 0;
            other.stream_ = nullptr;
            other.event_ = nullptr;
            other.host_valid_ = false;
        }
        return *this;
    }
    
    bool isAllocated() const { return d_ptr_ != nullptr; }
    
    bool allocate(size_t size) {
        if (size <= size_ && d_ptr_ && h_ptr_) return true;
        deallocate();
        
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) return false;
        if (cudaEventCreateWithFlags(&event_, cudaEventDisableTiming) != cudaSuccess) {
            cudaStreamDestroy(stream_);
            return false;
        }
        if (cudaHostAlloc(&h_ptr_, size, cudaHostAllocDefault) != cudaSuccess) {
            cudaEventDestroy(event_);
            cudaStreamDestroy(stream_);
            return false;
        }
        if (cudaMalloc(&d_ptr_, size) != cudaSuccess) {
            cudaFreeHost(h_ptr_);
            cudaEventDestroy(event_);
            cudaStreamDestroy(stream_);
            h_ptr_ = nullptr;
            return false;
        }
        size_ = size;
        return true;
    }
    
    void deallocate() {
        if (d_ptr_) { 
            cudaError_t err = cudaFree(d_ptr_); 
            if (err != cudaSuccess) {
                std::cerr << "[GpuBuffer] cudaFree failed: " << cudaGetErrorString(err) << std::endl;
            }
            d_ptr_ = nullptr; 
        }
        if (h_ptr_) { 
            cudaError_t err = cudaFreeHost(h_ptr_); 
            if (err != cudaSuccess) {
                std::cerr << "[GpuBuffer] cudaFreeHost failed: " << cudaGetErrorString(err) << std::endl;
            }
            h_ptr_ = nullptr; 
        }
        if (event_) { 
            cudaError_t err = cudaEventDestroy(event_); 
            if (err != cudaSuccess) {
                std::cerr << "[GpuBuffer] cudaEventDestroy failed: " << cudaGetErrorString(err) << std::endl;
            }
            event_ = nullptr; 
        }
        if (stream_) { 
            cudaError_t err = cudaStreamDestroy(stream_); 
            if (err != cudaSuccess) {
                std::cerr << "[GpuBuffer] cudaStreamDestroy failed: " << cudaGetErrorString(err) << std::endl;
            }
            stream_ = nullptr; 
        }
        size_ = 0;
    }
    
    // Direct GPU transfer with event recording for non-blocking completion check
    bool copyToDevice(const void* src, size_t src_size, int width, int height) {
        if (!d_ptr_ || src_size > size_) return false;
        
        if (cudaMemcpyAsync(d_ptr_, src, src_size, cudaMemcpyHostToDevice, stream_) != cudaSuccess) return false;
        // Record event after copy for non-blocking completion check
        cudaEventRecord(event_, stream_);
        
        width_ = width;
        height_ = height;
        data_size_ = src_size;
        host_valid_ = false;
        return true;
    }
    
    // Copy to both GPU and host (when CPU access is also needed)
    bool copyToDeviceAndHost(const void* src, size_t src_size, int width, int height) {
        if (!h_ptr_ || !d_ptr_ || src_size > size_) return false;
        
        std::memcpy(h_ptr_, src, src_size);
        if (cudaMemcpyAsync(d_ptr_, h_ptr_, src_size, cudaMemcpyHostToDevice, stream_) != cudaSuccess) return false;
        cudaEventRecord(event_, stream_);
        
        width_ = width;
        height_ = height;
        data_size_ = src_size;
        host_valid_ = true;
        return true;
    }
    
    // Non-blocking check if transfer is complete
    bool isReady() const {
        if (!event_) return true;
        return cudaEventQuery(event_) == cudaSuccess;
    }
    
    // Lazy copy from GPU to host only when needed
    bool ensureHostCopy() {
        if (host_valid_ || !d_ptr_ || !h_ptr_ || data_size_ == 0) return host_valid_;
        
        sync();
        if (cudaMemcpy(h_ptr_, d_ptr_, data_size_, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        host_valid_ = true;
        return true;
    }
    
    bool isHostValid() const { return host_valid_; }
    
    void sync() {
        if (event_) cudaEventSynchronize(event_);  // Lighter than stream sync
    }
    
    void* devicePtr() const { return d_ptr_; }
    void* hostPtr() const { return h_ptr_; }
    size_t size() const { return size_; }
    size_t dataSize() const { return data_size_; }
    int width() const { return width_; }
    int height() const { return height_; }
    cudaStream_t stream() const { return stream_; }
    uint64_t devicePtrAsInt() const { return reinterpret_cast<uint64_t>(d_ptr_); }

private:
    void* d_ptr_ = nullptr;
    void* h_ptr_ = nullptr;
    size_t size_ = 0;
    size_t data_size_ = 0;
    int width_ = 0;
    int height_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t event_ = nullptr;
    bool host_valid_ = false;
};

#endif
