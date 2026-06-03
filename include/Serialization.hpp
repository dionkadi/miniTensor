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
// Checkpoint save/load — model params + optimizer state for training resumption
// ---------------------------------------------------------------------------

template<typename T>
struct Checkpoint {
    std::vector<Tensor<T>> params;
    std::vector<Tensor<T>> m_buffers;   // Adam first moments
    std::vector<Tensor<T>> v_buffers;   // Adam second moments
    size_t step = 0;
    T lr{}, beta1{}, beta2{}, eps{}, weight_decay{};
};

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
                     const std::vector<Tensor<T>>& m_buffers,
                     const std::vector<Tensor<T>>& v_buffers,
                     size_t step,
                     T lr, T beta1, T beta2, T eps, T weight_decay) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path + " for writing");

    uint32_t num_params = static_cast<uint32_t>(params.size());
    f.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

    // Model parameters
    for (const auto& p : params)
        write_tensor(f, p);

    // Adam m buffers
    for (const auto& m : m_buffers)
        write_tensor(f, m);

    // Adam v buffers
    for (const auto& v : v_buffers)
        write_tensor(f, v);

    // Optimizer hyperparams
    uint64_t step64 = static_cast<uint64_t>(step);
    f.write(reinterpret_cast<const char*>(&step64), sizeof(step64));
    f.write(reinterpret_cast<const char*>(&lr), sizeof(T));
    f.write(reinterpret_cast<const char*>(&beta1), sizeof(T));
    f.write(reinterpret_cast<const char*>(&beta2), sizeof(T));
    f.write(reinterpret_cast<const char*>(&eps), sizeof(T));
    f.write(reinterpret_cast<const char*>(&weight_decay), sizeof(T));
}

template<typename T>
Checkpoint<T> load_checkpoint(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path + " for reading");

    uint32_t num_params;
    f.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

    Checkpoint<T> ckpt;

    // Model parameters
    for (uint32_t i = 0; i < num_params; ++i)
        ckpt.params.push_back(read_tensor<T>(f));

    // Adam m buffers
    for (uint32_t i = 0; i < num_params; ++i)
        ckpt.m_buffers.push_back(read_tensor<T>(f));

    // Adam v buffers
    for (uint32_t i = 0; i < num_params; ++i)
        ckpt.v_buffers.push_back(read_tensor<T>(f));

    // Optimizer hyperparams
    uint64_t step64;
    f.read(reinterpret_cast<char*>(&step64), sizeof(step64));
    ckpt.step = static_cast<size_t>(step64);
    f.read(reinterpret_cast<char*>(&ckpt.lr), sizeof(T));
    f.read(reinterpret_cast<char*>(&ckpt.beta1), sizeof(T));
    f.read(reinterpret_cast<char*>(&ckpt.beta2), sizeof(T));
    f.read(reinterpret_cast<char*>(&ckpt.eps), sizeof(T));
    f.read(reinterpret_cast<char*>(&ckpt.weight_decay), sizeof(T));

    return ckpt;
}

template<typename T>
void load_checkpoint_into(const std::string& path,
                          std::vector<Tensor<T>>& target_params,
                          std::vector<Tensor<T>>& target_m,
                          std::vector<Tensor<T>>& target_v,
                          size_t& step,
                          T& lr, T& beta1, T& beta2, T& eps, T& weight_decay) {
    auto ckpt = load_checkpoint<T>(path);

    if (ckpt.params.size() != target_params.size())
        throw std::runtime_error("Parameter count mismatch in checkpoint");

    for (size_t i = 0; i < ckpt.params.size(); ++i) {
        copy_into_maybe_gpu(ckpt.params[i], target_params[i]);
        copy_into_maybe_gpu(ckpt.m_buffers[i], target_m[i]);
        copy_into_maybe_gpu(ckpt.v_buffers[i], target_v[i]);
    }

    step = ckpt.step;
    lr = ckpt.lr;
    beta1 = ckpt.beta1;
    beta2 = ckpt.beta2;
    eps = ckpt.eps;
    weight_decay = ckpt.weight_decay;
}

// ---------------------------------------------------------------------------
// Checkpoint save/load with optional scheduler state
// ---------------------------------------------------------------------------
// Format extension: after the standard checkpoint data, a uint8_t marker
// (1 = scheduler present) followed by scheduler save_state(). Old checkpoints
// without scheduler data are handled gracefully on load.
// ---------------------------------------------------------------------------

template<typename T>
void save_checkpoint(const std::string& path,
                     const std::vector<Tensor<T>>& params,
                     const std::vector<Tensor<T>>& m_buffers,
                     const std::vector<Tensor<T>>& v_buffers,
                     size_t step,
                     T lr, T beta1, T beta2, T eps, T weight_decay,
                     const LRScheduler<T>* scheduler) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path + " for writing");

    uint32_t num_params = static_cast<uint32_t>(params.size());
    f.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

    for (const auto& p : params) write_tensor(f, p);
    for (const auto& m : m_buffers) write_tensor(f, m);
    for (const auto& v : v_buffers) write_tensor(f, v);

    uint64_t step64 = static_cast<uint64_t>(step);
    f.write(reinterpret_cast<const char*>(&step64), sizeof(step64));
    f.write(reinterpret_cast<const char*>(&lr), sizeof(T));
    f.write(reinterpret_cast<const char*>(&beta1), sizeof(T));
    f.write(reinterpret_cast<const char*>(&beta2), sizeof(T));
    f.write(reinterpret_cast<const char*>(&eps), sizeof(T));
    f.write(reinterpret_cast<const char*>(&weight_decay), sizeof(T));

    // Optional scheduler state appended to file
    if (scheduler) {
        uint8_t has_sched = 1;
        f.write(reinterpret_cast<const char*>(&has_sched), sizeof(has_sched));
        scheduler->save_state(f);
    }
}

template<typename T>
void load_checkpoint_into(const std::string& path,
                          std::vector<Tensor<T>>& target_params,
                          std::vector<Tensor<T>>& target_m,
                          std::vector<Tensor<T>>& target_v,
                          size_t& step,
                          T& lr, T& beta1, T& beta2, T& eps, T& weight_decay,
                          LRScheduler<T>* scheduler) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path + " for reading");

    uint32_t num_params;
    f.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

    auto read_vec = [&]() {
        std::vector<Tensor<T>> v;
        for (uint32_t i = 0; i < num_params; ++i)
            v.push_back(read_tensor<T>(f));
        return v;
    };

    auto loaded_params = read_vec();
    auto loaded_m = read_vec();
    auto loaded_v = read_vec();

    uint64_t step64;
    f.read(reinterpret_cast<char*>(&step64), sizeof(step64));
    step = static_cast<size_t>(step64);
    f.read(reinterpret_cast<char*>(&lr), sizeof(T));
    f.read(reinterpret_cast<char*>(&beta1), sizeof(T));
    f.read(reinterpret_cast<char*>(&beta2), sizeof(T));
    f.read(reinterpret_cast<char*>(&eps), sizeof(T));
    f.read(reinterpret_cast<char*>(&weight_decay), sizeof(T));

    if (loaded_params.size() != target_params.size())
        throw std::runtime_error("Parameter count mismatch in checkpoint");

    for (size_t i = 0; i < loaded_params.size(); ++i) {
        copy_into_maybe_gpu(loaded_params[i], target_params[i]);
        copy_into_maybe_gpu(loaded_m[i], target_m[i]);
        copy_into_maybe_gpu(loaded_v[i], target_v[i]);
    }

    // Optional scheduler state (backward compatible: old checkpoints without it)
    if (scheduler) {
        uint8_t has_sched;
        f.read(reinterpret_cast<char*>(&has_sched), sizeof(has_sched));
        if (f.good() && has_sched == 1) {
            scheduler->load_state(f);
        }
    }
}
