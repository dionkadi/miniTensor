#pragma once

#include "Tensor.hpp"
#include "Broadcast.hpp"
#include "cpu_ops.hpp"

#if defined(USE_CUDA) || defined(USE_ROCM)
#include "gpu_ops.hpp"
#endif

template<typename T>
Tensor<T> add(const Tensor<T>& A, const Tensor<T>& B) {
    if (A.device() != B.device()) {
        throw std::invalid_argument("Tensors must be on the same device to perform addition.");
    }

    std::vector<size_t> out_shape = compute_broadcast_shape(A.shape(), B.shape());
    Tensor<T> expanded_A = A.expand(out_shape);
    Tensor<T> expanded_B = B.expand(out_shape);

    Tensor<T> C(out_shape, A.device());

    bool all_contiguous = expanded_A.is_contiguous() && expanded_B.is_contiguous() && C.is_contiguous();

    switch (A.device().type) {
        case DeviceType::CPU: {
            if (all_contiguous) add_cpu(expanded_A, expanded_B, C);
            else add_cpu_strided(expanded_A, expanded_B, C);
            break;
        }
        case DeviceType::CUDA: {
#if defined(USE_CUDA) || defined(USE_ROCM)
            if (all_contiguous) add_gpu(expanded_A, expanded_B, C);
            else add_gpu_strided(expanded_A, expanded_B, C);
#else
            throw std::runtime_error("Library was not compiled with GPU support!");
#endif
            break;
        }
        default:
            throw std::runtime_error("Unknown device type.");
    }

    return C;
}

template<typename T>
Tensor<T> operator+(const Tensor<T>& A, const Tensor<T>& B) {
    return add(A, B);
}

template<typename T>
Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B) {
    if (A.device() != B.device()) {
        throw std::invalid_argument("Tensors must be on the same device for matmul.");
    }

    if (A.shape().size() != 2 || B.shape().size() != 2) {
        throw std::invalid_argument("matmul currently only supports 2D tensors.");
    }

    if (A.shape()[1] != B.shape()[0]) {
        throw std::invalid_argument("Dimension mismatch: A columns must match B rows.");
    }

    size_t M = A.shape()[0];
    size_t N = B.shape()[1];

    Tensor<T> C({M, N}, A.device());

    bool all_contiguous = A.is_contiguous() && B.is_contiguous() && C.is_contiguous();

    switch (A.device().type) {
        case DeviceType::CPU: {
            if (all_contiguous) {
                matmul_cpu(A, B, C);
            } else {
                matmul_cpu_strided(A, B, C);
            }
            break;
        }
        case DeviceType::CUDA: {
#if defined(USE_CUDA) || defined(USE_ROCM)
            if (all_contiguous) {
                matmul_gpu(A, B, C);
            } else {
                matmul_gpu_strided(A, B, C);
            }
#else
            throw std::runtime_error("Library was not compiled with GPU support!");
#endif
            break;
        }
        default:
            throw std::runtime_error("Unknown device type.");
    }

    return C;
}