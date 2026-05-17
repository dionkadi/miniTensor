#pragma once

#include "Tensor.hpp"
#include "Module.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <cstring>

template<typename T>
void save_state_dict(const std::string& path, const std::vector<Tensor<T>>& params) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path + " for writing");

    uint32_t num_params = params.size();
    f.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

    for (const auto& p : params) {
        // Move to CPU and make contiguous for serialization
        Tensor<T> cpu_t = p.to({DeviceType::CPU}).contiguous();
        auto shape = cpu_t.shape();
        uint32_t ndim = shape.size();
        uint64_t total = cpu_t.total_elements();

        // Write number of dimensions
        f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        // Write shape
        for (auto d : shape) {
            uint64_t dim = d;
            f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        }
        // Write raw data
        f.write(reinterpret_cast<const char*>(cpu_t.data() + cpu_t.offset()),
                total * sizeof(T));
    }
}

template<typename T>
std::vector<Tensor<T>> load_state_dict(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path + " for reading");

    uint32_t num_params;
    f.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

    std::vector<Tensor<T>> params;
    for (uint32_t i = 0; i < num_params; ++i) {
        uint32_t ndim;
        f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

        std::vector<size_t> shape(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            uint64_t dim;
            f.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            shape[d] = static_cast<size_t>(dim);
        }

        Tensor<T> t(shape);
        uint64_t total = t.total_elements();
        f.read(reinterpret_cast<char*>(t.data()), total * sizeof(T));
        params.push_back(t);
    }

    return params;
}

template<typename T>
void load_state_dict_into(const std::string& path, std::vector<Tensor<T>>& target_params) {
    auto loaded = load_state_dict<T>(path);
    if (loaded.size() != target_params.size())
        throw std::runtime_error("Parameter count mismatch in state dict");

    for (size_t i = 0; i < loaded.size(); ++i) {
        if (loaded[i].total_elements() != target_params[i].total_elements())
            throw std::runtime_error("Parameter size mismatch in state dict");

        Tensor<T> cpu_loaded = loaded[i].to({DeviceType::CPU}).contiguous();
        size_t bytes = cpu_loaded.total_elements() * sizeof(T);

        if (target_params[i].device().type == DeviceType::CPU) {
            Tensor<T> cpu_target = target_params[i].to({DeviceType::CPU}).contiguous();
            std::memcpy(cpu_target.data() + cpu_target.offset(),
                        cpu_loaded.data() + cpu_loaded.offset(), bytes);
        } else {
            const T* src = cpu_loaded.data() + cpu_loaded.offset();
            T* dst = target_params[i].data();
#if defined(USE_CUDA)
            GPU_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
#elif defined(USE_ROCM)
            GPU_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
#endif
        }
    }
}
    }
}
