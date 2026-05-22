#pragma once

#include <iostream>
#include <stdexcept>

#if defined(USE_CUDA)

#include <cuda_runtime.h>

#define GPU_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA API Call Failed"); \
        } \
    } while (0)

inline cudaStream_t default_cuda_stream() { return nullptr; }
using GpuStream_t = cudaStream_t;
using GpuEvent_t  = cudaEvent_t;

#elif defined(USE_ROCM)

#include <hip/hip_runtime.h>

#define GPU_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("HIP API Call Failed"); \
        } \
    } while (0)

inline hipStream_t default_hip_stream() { return nullptr; }
using GpuStream_t = hipStream_t;
using GpuEvent_t  = hipEvent_t;

#else
    #define GPU_CHECK(call)
#endif

class GpuStream {
public:
    GpuStream() {
#if defined(USE_CUDA) || defined(USE_ROCM)
        GPU_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
#endif
    }

    ~GpuStream() {
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (stream_) {
            (void)hipStreamSynchronize(stream_);
            (void)hipStreamDestroy(stream_);
        }
#endif
    }

    GpuStream(const GpuStream&) = delete;
    GpuStream& operator=(const GpuStream&) = delete;
    GpuStream(GpuStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    GpuStream& operator=(GpuStream&& other) noexcept {
        if (this != &other) {
            if (stream_) { (void)hipStreamDestroy(stream_); }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    GpuStream_t get() const noexcept { return stream_; }
    void synchronize() { if (stream_) GPU_CHECK(hipStreamSynchronize(stream_)); }

private:
    GpuStream_t stream_ = nullptr;
};

class GpuEvent {
public:
    GpuEvent() {
#if defined(USE_CUDA) || defined(USE_ROCM)
        GPU_CHECK(hipEventCreateWithFlags(&event_, hipEventDisableTiming));
#endif
    }

    ~GpuEvent() {
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (event_) (void)hipEventDestroy(event_);
#endif
    }

    GpuEvent(const GpuEvent&) = delete;
    GpuEvent& operator=(const GpuEvent&) = delete;

    void record(GpuStream_t stream = nullptr) {
#if defined(USE_CUDA) || defined(USE_ROCM)
        GPU_CHECK(hipEventRecord(event_, stream));
#endif
    }

    void wait(GpuStream_t stream = nullptr) const {
#if defined(USE_CUDA) || defined(USE_ROCM)
        GPU_CHECK(hipStreamWaitEvent(stream, event_, 0));
#endif
    }

    void synchronize() {
#if defined(USE_CUDA) || defined(USE_ROCM)
        GPU_CHECK(hipEventSynchronize(event_));
#endif
    }

    bool query() const {
#if defined(USE_CUDA) || defined(USE_ROCM)
        hipError_t err = hipEventQuery(event_);
        return err == hipSuccess;
#else
        return true;
#endif
    }

    GpuEvent_t get() const noexcept { return event_; }

private:
    GpuEvent_t event_ = nullptr;
};