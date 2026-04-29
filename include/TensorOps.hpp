#pragma once

#include "Autograd.hpp"

// #############################
// #   CPU operations
// #############################
template<typename T>
void add_cpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    const T* a_ptr = A.data();
    const T* b_ptr = B.data();
    T* c_ptr = C.data();

    // Standard optimized CPU loop (could use OpenMP here later)
    for (size_t i = 0; i < A.total_elements(); ++i) {
        c_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

template<typename T>
void add_cpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    const T *A_ptr = A.data() + A.offset();
    const T *B_ptr = B.data() + B.offset();
    T *C_ptr = C.data() + C.offset();

    TensorInfo info_A(A.shape(), A.strides());
    TensorInfo info_B(B.shape(), B.strides());
    TensorInfo info_C(C.shape(), C.strides());

    collapse_dims(info_A, info_B, info_C);

    for (size_t i = 0; i < A.total_elements(); ++i) {
        size_t linear_idx = i;
        size_t phys_a = 0, phys_b = 0, phys_c = 0;

        for (int d = info_A.ndims - 1; d >= 0; --d) {
            size_t coord = linear_idx % info_A.shape[d];
            linear_idx /= info_A.shape[d];

            phys_a += coord * info_A.strides[d];
            phys_b += coord * info_B.strides[d];
            phys_c += coord * info_C.strides[d];
        }

        C_ptr[phys_c] = A_ptr[phys_a] + B_ptr[phys_b];
    }
}

template<typename T>
void matmul_cpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];

    const T* a_ptr = A.data() + A.offset();
    const T* b_ptr = B.data() + B.offset();
    T* c_ptr = C.data() + C.offset();

    // Naive O(N^3) matrix multiplication
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = 0;
            for (size_t k = 0; k < K; ++k) {
                // A[i, k] * B[k, j]
                sum += a_ptr[i * K + k] * b_ptr[k * N + j];
            }
            c_ptr[i * N + j] = sum;
        }
    }
}

template<typename T>
void matmul_cpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];

    const T* a_ptr = A.data() + A.offset();
    const T* b_ptr = B.data() + B.offset();
    T* c_ptr = C.data() + C.offset();

    // Cache strides locally for faster loop access
    size_t a_stride_m = A.strides()[0], a_stride_k = A.strides()[1];
    size_t b_stride_k = B.strides()[0], b_stride_n = B.strides()[1];
    size_t c_stride_m = C.strides()[0], c_stride_n = C.strides()[1];

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = 0;
            for (size_t k = 0; k < K; ++k) {
                // Read using exact strides
                T a_val = a_ptr[i * a_stride_m + k * a_stride_k];
                T b_val = b_ptr[k * b_stride_k + j * b_stride_n];
                sum += a_val * b_val;
            }
            c_ptr[i * c_stride_m + j * c_stride_n] = sum;
        }
    }
}

template<typename T>
Tensor<T> sum_cpu(const Tensor<T>& input, size_t axis, bool keepdims) {
    const auto& in_shape = input.shape();
    const auto& in_strides = input.strides();
    
    std::vector<size_t> out_shape = in_shape;
    if (keepdims) {
        out_shape[axis] = 1;
    } else {
        out_shape.erase(out_shape.begin() + axis);
    }

    Tensor<T> output(out_shape, input.device());
    
    const T* in_ptr = input.data() + input.offset();
    T* out_ptr = output.data();
    
    size_t axis_size = in_shape[axis];
    size_t axis_stride = in_strides[axis];
    size_t in_ndims = in_shape.size();
    
    size_t out_elements = output.total_elements();
    
    for (size_t i = 0; i < out_elements; ++i) {
        size_t linear_idx = i;
        size_t in_phys_base = 0;
        
        for (int d = in_ndims - 1; d >= 0; --d) {
            size_t coord = linear_idx % in_shape[d];
            linear_idx /= in_shape[d];

            in_phys_base += coord * in_strides[d];
        }
        
        T sum = 0;
        for (size_t j = 0; j < axis_size; ++j) {
            sum += in_ptr[in_phys_base + j * axis_stride];
        }
        out_ptr[i] = sum;
    }
    
    return output;
}

// #############################
// #   GPU operations
// #############################
#if defined(USE_CUDA) || defined(USE_ROCM)

template<typename T>
void add_gpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);

template<typename T>
void add_gpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);

template<typename T>
void matmul_gpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);

template<typename T>
void matmul_gpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);

template<typename T>
Tensor<T> sum_gpu(const Tensor<T>& input, size_t axis, bool keepdims);

#endif

// #############################
// #   Dispatcher
// #############################
std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b);

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

    if (A.requires_grad() || B.requires_grad()) {
        C.set_requires_grad(true);
        C.set_grad_fn(std::make_shared<AddBackward<T>>(A, B));
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

    if (A.requires_grad() || B.requires_grad()) {
        C.set_requires_grad(true);
        C.set_grad_fn(std::make_shared<MatmulBackward<T>>(A, B));
    }

    return C;
}

template<typename T>
Tensor<T> sum(const Tensor<T>& A, size_t axis, bool keepdims) {
    switch (A.device().type) {
        case DeviceType::CPU: {
            return sum_cpu(A, axis, keepdims);
            break;
        }
        case DeviceType::CUDA: {
#if defined(USE_CUDA) || defined(USE_ROCM)
            return sum_gpu(A, axis, keepdims);
#else
            throw std::runtime_error("Library was not compiled with GPU support!");
#endif
            break;
        }
        default:
            throw std::runtime_error("Unknown device type.");
    }
}


// #############################
// #   Broadcast
// #############################
inline std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b) {
    size_t max_dims = std::max(shape_a.size(), shape_b.size());
    std::vector<size_t> out_shape(max_dims);

    int offset_a = max_dims - shape_a.size();
    int offset_b = max_dims - shape_b.size();
    
    for (int i = max_dims - 1; i >= 0; --i) {
        size_t a_dim = (i >= offset_a) ? shape_a[i - offset_a] : 1;
        size_t b_dim = (i >= offset_b) ? shape_b[i - offset_b] : 1;
        
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::invalid_argument("Tensors cannot be broadcast together.");
        }
        out_shape[i] = std::max(a_dim, b_dim);
    }
    return out_shape;
}

template <typename T>
Tensor<T> unbroadcast(Tensor<T> grad, const std::vector<size_t>& target_shape) {
    const auto& grad_shape = grad.shape();

    if (grad_shape == target_shape) {
        return grad;
    }

    Tensor<T> result = grad;
    int ndims_grad = grad_shape.size();
    int ndims_target = target_shape.size();

    // Sum out added leading dimensions.
    // e.g., target=[3], grad=[2, 3] -> diff=1. Sum along axis 0.
    int diff = ndims_grad - ndims_target;
    for (int i = 0; i < diff; ++i) {
        // Repeatedly summing axis 0 removes the leading dimensions
        result = sum(result, 0, false); 
    }

    // SSum out expanded dimensions.
    // e.g., target=[2, 1], grad=[2, 3] -> Sum along axis 1, keep dimension.
    for (int i = 0; i < ndims_target; ++i) {
        if (target_shape[i] == 1 && result.shape()[i] > 1) {
            result = sum(result, i, true);
        }
    }

    return result;
}