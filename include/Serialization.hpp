#pragma once

#include "Tensor.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <cstring>

template<typename T> class LRScheduler;

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

// ---------------------------------------------------------------------------
// Generic Checkpoint — model params + optimizer state + scheduler
// ---------------------------------------------------------------------------
// Binary format (all little-endian):
//   [uint32_t magic      = 0x4D544E53]   "MTNS"
//   [uint32_t num_params]
//   [num_params × tensor]                 model parameters
//   [uint32_t num_opt_buffers]
//   [num_opt_buffers × tensor]            optimizer state buffers (flattened;
//                                          e.g. 1 velocity for SGD+momentum,
//                                          2 m+v for Adam/AdamW)
//   [uint64_t step]
//   [uint32_t num_opt_scalars]
//   [num_opt_scalars × T]                 scalars (e.g. lr, wd, momentum
//                                          or lr, beta1, beta2, eps, wd)
//   [uint8_t  has_scheduler]
//   [if has_scheduler]                    scheduler save_state()
// ---------------------------------------------------------------------------

namespace {

template<typename T>
void write_tensor(std::ofstream& f, const Tensor<T>& t) {
    Tensor<T> cpu_t = t.to({DeviceType::CPU}).contiguous();
    auto shape = cpu_t.shape();
    uint32_t ndim = static_cast<uint32_t>(shape.size());
    uint64_t total = cpu_t.total_elements();

    f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (auto d : shape) {
        uint64_t dim = static_cast<uint64_t>(d);
        f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    f.write(reinterpret_cast<const char*>(cpu_t.data() + cpu_t.offset()),
            total * sizeof(T));
}

template<typename T>
Tensor<T> read_tensor(std::ifstream& f) {
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
    return t;
}

template<typename T>
void copy_into_maybe_gpu(const Tensor<T>& src, Tensor<T>& dst) {
    Tensor<T> cpu_src = src.to({DeviceType::CPU}).contiguous();
    size_t bytes = cpu_src.total_elements() * sizeof(T);
    const T* src_ptr = cpu_src.data() + cpu_src.offset();

    if (dst.device().type == DeviceType::CPU) {
        Tensor<T> cpu_dst = dst.to({DeviceType::CPU}).contiguous();
        std::memcpy(cpu_dst.data() + cpu_dst.offset(), src_ptr, bytes);
    } else {
#if defined(USE_CUDA)
        GPU_CHECK(cudaMemcpy(dst.data(), src_ptr, bytes, cudaMemcpyHostToDevice));
#elif defined(USE_ROCM)
        GPU_CHECK(hipMemcpy(dst.data(), src_ptr, bytes, hipMemcpyHostToDevice));
#endif
    }
}

} // anonymous namespace

template<typename T>
void save_checkpoint(const std::string& path,
                     const std::vector<Tensor<T>>& params,
                     const std::vector<Tensor<T>>& opt_buffers,
                     size_t step,
                     const std::vector<T>& opt_scalars,
                     const LRScheduler<T>* scheduler = nullptr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path + " for writing");

    uint32_t magic = 0x4D544E53;  // "MTNS"
    f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

    uint32_t num_params = static_cast<uint32_t>(params.size());
    f.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
    for (const auto& p : params) write_tensor(f, p);

    uint32_t num_opt_buffers = static_cast<uint32_t>(opt_buffers.size());
    f.write(reinterpret_cast<const char*>(&num_opt_buffers), sizeof(num_opt_buffers));
    for (const auto& buf : opt_buffers) write_tensor(f, buf);

    uint64_t step64 = static_cast<uint64_t>(step);
    f.write(reinterpret_cast<const char*>(&step64), sizeof(step64));

    uint32_t num_scalars = static_cast<uint32_t>(opt_scalars.size());
    f.write(reinterpret_cast<const char*>(&num_scalars), sizeof(num_scalars));
    for (auto s : opt_scalars)
        f.write(reinterpret_cast<const char*>(&s), sizeof(T));

    uint8_t has_sched = scheduler ? 1 : 0;
    f.write(reinterpret_cast<const char*>(&has_sched), sizeof(has_sched));
    if (scheduler)
        scheduler->save_state(f);
}

template<typename T>
void load_checkpoint_into(const std::string& path,
                          std::vector<Tensor<T>>& target_params,
                          std::vector<Tensor<T>>& target_opt_buffers,
                          size_t& step,
                          std::vector<T>& opt_scalars,
                          LRScheduler<T>* scheduler = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path + " for reading");

    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x4D544E53)
        throw std::runtime_error("Invalid checkpoint file (bad magic)");

    uint32_t num_params;
    f.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
    if (num_params != target_params.size())
        throw std::runtime_error("Parameter count mismatch in checkpoint");
    for (uint32_t i = 0; i < num_params; ++i) {
        auto t = read_tensor<T>(f);
        copy_into_maybe_gpu(t, target_params[i]);
    }

    uint32_t num_opt_buffers;
    f.read(reinterpret_cast<char*>(&num_opt_buffers), sizeof(num_opt_buffers));
    if (num_opt_buffers != target_opt_buffers.size())
        throw std::runtime_error("Opt buffer count mismatch in checkpoint");
    for (uint32_t i = 0; i < num_opt_buffers; ++i) {
        auto t = read_tensor<T>(f);
        copy_into_maybe_gpu(t, target_opt_buffers[i]);
    }

    uint64_t step64;
    f.read(reinterpret_cast<char*>(&step64), sizeof(step64));
    step = static_cast<size_t>(step64);

    uint32_t num_scalars;
    f.read(reinterpret_cast<char*>(&num_scalars), sizeof(num_scalars));
    if (num_scalars != opt_scalars.size())
        throw std::runtime_error("Opt scalar count mismatch in checkpoint");
    for (uint32_t i = 0; i < num_scalars; ++i)
        f.read(reinterpret_cast<char*>(&opt_scalars[i]), sizeof(T));

    uint8_t has_sched;
    f.read(reinterpret_cast<char*>(&has_sched), sizeof(has_sched));
    if (f.good() && has_sched == 1 && scheduler)
        scheduler->load_state(f);
}
