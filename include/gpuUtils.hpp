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

using GpuStream_t = cudaStream_t;

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

using GpuStream_t = hipStream_t;

#else
    // Define empty or stub structures if compiling CPU-only
    #define GPU_CHECK(call) 
#endif