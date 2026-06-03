#pragma once

#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstdlib>
#include <cstring>

#include "Defines.hpp"
#include "GpuUtils.hpp"

static size_t next_pow2(size_t n) {
    if (n <= 1) return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

class MemoryPool {
private:
    std::mutex mtx_;
    std::unordered_map<size_t, std::vector<void*>> cpu_pool_;
    std::unordered_map<size_t, std::vector<void*>> gpu_pool_;

    MemoryPool() = default;

public:
    static MemoryPool& get() {
        static MemoryPool instance;
        return instance;
    }

    ~MemoryPool() {
        empty_cache();
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    void* allocate(size_t bytes, Device device) {
        if (bytes == 0) return nullptr;

        if (device.type == DeviceType::CPU) {
            std::lock_guard<std::mutex> lock(mtx_);
            auto it = cpu_pool_.find(bytes);
            if (it != cpu_pool_.end() && !it->second.empty()) {
                void* ptr = it->second.back();
                it->second.pop_back();
                return ptr;
            }
            void* ptr = nullptr;
#if defined(USE_CUDA)
            GPU_CHECK(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
#elif defined(USE_ROCM)
            GPU_CHECK(hipHostMalloc(&ptr, bytes));
#else
            ptr = std::malloc(bytes);
            if (!ptr) throw std::bad_alloc();
#endif
            return ptr;
        }

        // GPU: round up to power-of-2 to reduce fragmentation
        size_t slot = next_pow2(bytes);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            auto it = gpu_pool_.find(slot);
            if (it != gpu_pool_.end() && !it->second.empty()) {
                void* ptr = it->second.back();
                it->second.pop_back();
                return ptr;
            }
        }
        void* ptr = nullptr;
#if defined(USE_CUDA)
        GPU_CHECK(cudaMalloc(&ptr, slot));
#elif defined(USE_ROCM)
        GPU_CHECK(hipMalloc(&ptr, slot));
#endif
        return ptr;
    }

    void free(void* ptr, size_t bytes, Device device) {
        if (!ptr || bytes == 0) return;

        if (device.type == DeviceType::CPU) {
            std::lock_guard<std::mutex> lock(mtx_);
            cpu_pool_[bytes].push_back(ptr);
        } else {
            size_t slot = next_pow2(bytes);
            std::lock_guard<std::mutex> lock(mtx_);
            gpu_pool_[slot].push_back(ptr);
        }
    }

    void empty_cache() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& pair : cpu_pool_) {
            for (void* ptr : pair.second) {
#if defined(USE_CUDA)
                (void)cudaFreeHost(ptr);
#elif defined(USE_ROCM)
                (void)hipHostFree(ptr);
#else
                std::free(ptr);
#endif
            }
        }
        cpu_pool_.clear();

        for (auto& pair : gpu_pool_) {
            for (void* ptr : pair.second) {
#if defined(USE_CUDA)
                (void)cudaFree(ptr);
#elif defined(USE_ROCM)
                (void)hipFree(ptr);
#endif
            }
        }
        gpu_pool_.clear();
    }
};