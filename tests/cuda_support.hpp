#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
    }
}

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        if (count > 0) {
            cuda_check(cudaMalloc(&ptr_, count * sizeof(T)), "cudaMalloc");
        }
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const noexcept { return ptr_; }

private:
    T* ptr_ = nullptr;
};

class Stream {
public:
    Stream() { cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};
