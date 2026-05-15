#pragma once

#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstdlib>
#include <cstring>

#include "Defines.hpp"
#include "GpuUtils.hpp"

class MemoryPool {
private:
    std::mutex mtx_;
    // Map: Allocation Size (in bytes) -> Vector of free pointers
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

        std::lock_guard<std::mutex> lock(mtx_);
        auto& pool = (device.type == DeviceType::CPU) ? cpu_pool_ : gpu_pool_;

        // Check if we have a free block of the EXACT size
        auto it = pool.find(bytes);
        if (it != pool.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }

        // No free block found, make an expensive OS/Driver call
        void* ptr = nullptr;
        if (device.type == DeviceType::CPU) {
#if defined(USE_CUDA) || defined(USE_ROCM)
            // Use pinned (page-locked) memory for faster CPU↔GPU transfers
            GPU_CHECK(hipHostMalloc(&ptr, bytes));
#else
            ptr = std::malloc(bytes);
            if (!ptr) throw std::bad_alloc();
#endif
        } else if (device.type == DeviceType::CUDA) {
#if defined(USE_CUDA)
            GPU_CHECK(cudaMalloc(&ptr, bytes));
#elif defined(USE_ROCM)
            GPU_CHECK(hipMalloc(&ptr, bytes));
#endif
        }
        return ptr;
    }

    void free(void* ptr, size_t bytes, Device device) {
        if (!ptr || bytes == 0) return;

        std::lock_guard<std::mutex> lock(mtx_);
        auto& pool = (device.type == DeviceType::CPU) ? cpu_pool_ : gpu_pool_;
        
        // Put the pointer back on the shelf for this specific size
        pool[bytes].push_back(ptr);
    }

    void empty_cache() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& pair : cpu_pool_) {
            for (void* ptr : pair.second) {
#if defined(USE_CUDA) || defined(USE_ROCM)
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