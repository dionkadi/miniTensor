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

// Thread-local stream override for graph capture — set during capture,
// nullptr otherwise. Kernel launches use this if non-null.
inline cudaStream_t& active_stream() {
    thread_local cudaStream_t s = nullptr;
    return s;
}

// Capture-safe variant — cudaGetLastError returns cudaErrorStreamCaptureUnsupported
// during stream capture even if the kernel launched fine. We ignore it since
// warmup runs verify valid launch args before capture.
inline cudaError_t get_last_error_capture_safe() {
    cudaError_t err = cudaGetLastError();
    if (err == cudaErrorStreamCaptureUnsupported) return cudaSuccess;
    return err;
}

using GpuStream_t = cudaStream_t;
using GpuEvent_t  = cudaEvent_t;
using GpuGraph_t     = cudaGraph_t;
using GpuGraphExec_t = cudaGraphExec_t;

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

// Thread-local stream override for graph capture — set during capture,
// nullptr otherwise. Kernel launches use this if non-null.
inline hipStream_t& active_stream() {
    thread_local hipStream_t s = nullptr;
    return s;
}

// Capture-safe variant — hipGetLastError returns hipErrorStreamCaptureUnsupported
// during stream capture even if the kernel launched fine. We ignore it since
// warmup runs verify valid launch args before capture.
inline hipError_t get_last_error_capture_safe() {
    hipError_t err = hipGetLastError();
    if (err == hipErrorStreamCaptureUnsupported) return hipSuccess;
    return err;
}
using GpuStream_t = hipStream_t;
using GpuEvent_t  = hipEvent_t;
using GpuGraph_t     = hipGraph_t;
using GpuGraphExec_t = hipGraphExec_t;

#else
    #define GPU_CHECK(call)
    using GpuStream_t = void*;
    using GpuEvent_t  = void*;
    using GpuGraph_t     = void*;
    using GpuGraphExec_t = void*;
#endif

class GpuStream {
public:
    GpuStream() {
#if defined(USE_CUDA)
        GPU_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
#elif defined(USE_ROCM)
        GPU_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
#endif
    }

    ~GpuStream() {
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (stream_) {
#if defined(USE_CUDA)
            (void)cudaStreamSynchronize(stream_);
            (void)cudaStreamDestroy(stream_);
#elif defined(USE_ROCM)
            (void)hipStreamSynchronize(stream_);
            (void)hipStreamDestroy(stream_);
#endif
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
#if defined(USE_CUDA) || defined(USE_ROCM)
            if (stream_) {
#if defined(USE_CUDA)
                (void)cudaStreamDestroy(stream_);
#elif defined(USE_ROCM)
                (void)hipStreamDestroy(stream_);
#endif
            }
#endif
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    GpuStream_t get() const noexcept { return stream_; }
    void synchronize() {
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (stream_) {
#if defined(USE_CUDA)
            GPU_CHECK(cudaStreamSynchronize(stream_));
#elif defined(USE_ROCM)
            GPU_CHECK(hipStreamSynchronize(stream_));
#endif
        }
#endif
    }

private:
    GpuStream_t stream_ = nullptr;
};

class GpuEvent {
public:
    GpuEvent() {
#if defined(USE_CUDA)
        GPU_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
#elif defined(USE_ROCM)
        GPU_CHECK(hipEventCreateWithFlags(&event_, hipEventDisableTiming));
#endif
    }

    ~GpuEvent() {
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (event_) {
#if defined(USE_CUDA)
            (void)cudaEventDestroy(event_);
#elif defined(USE_ROCM)
            (void)hipEventDestroy(event_);
#endif
        }
#endif
    }

    GpuEvent(const GpuEvent&) = delete;
    GpuEvent& operator=(const GpuEvent&) = delete;

    void record(GpuStream_t stream = nullptr) {
#if defined(USE_CUDA)
        GPU_CHECK(cudaEventRecord(event_, stream));
#elif defined(USE_ROCM)
        GPU_CHECK(hipEventRecord(event_, stream));
#endif
    }

    void wait(GpuStream_t stream = nullptr) const {
#if defined(USE_CUDA)
        GPU_CHECK(cudaStreamWaitEvent(stream, event_, 0));
#elif defined(USE_ROCM)
        GPU_CHECK(hipStreamWaitEvent(stream, event_, 0));
#endif
    }

    void synchronize() {
#if defined(USE_CUDA)
        GPU_CHECK(cudaEventSynchronize(event_));
#elif defined(USE_ROCM)
        GPU_CHECK(hipEventSynchronize(event_));
#endif
    }

    bool query() const {
#if defined(USE_CUDA)
        cudaError_t err = cudaEventQuery(event_);
        return err == cudaSuccess;
#elif defined(USE_ROCM)
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