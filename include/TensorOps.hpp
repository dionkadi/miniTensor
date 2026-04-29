#pragma once

#include "Autograd.hpp"
#include <cstddef>

// Macro to seamlessly compile functors for both Host (CPU) and Device (GPU)
#ifndef HD_INLINE
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__) || defined(__NVCC__)
#define HD_INLINE __host__ __device__ inline
#else
#define HD_INLINE inline
#endif
#endif

// #######################################################
// #   Functors
// #######################################################
template<typename T> struct AddOp { HD_INLINE T operator()(T a, T b) const { return a + b; } };
template<typename T> struct SubOp { HD_INLINE T operator()(T a, T b) const { return a - b; } };
template<typename T> struct MulOp { HD_INLINE T operator()(T a, T b) const { return a * b; } };
template<typename T> struct Pow2Op { HD_INLINE T operator()(T a) const { return a * a; } };
template<typename T> struct MulScalarOp { 
    T scalar;
    HD_INLINE MulScalarOp(T s) : scalar(s) {}
    HD_INLINE T operator()(T a) const { return a * scalar; } 
};

// #######################################################
// #   Generic CPU Execution Engine
// #######################################################
template<typename T, typename Op>
void binary_cpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, Op op) {
    const T* a_ptr = A.data();
    const T* b_ptr = B.data();
    T* c_ptr = C.data();
    for (size_t i = 0; i < A.total_elements(); ++i) {
        c_ptr[i] = op(a_ptr[i], b_ptr[i]);
    }
}

template<typename T, typename Op>
void binary_cpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, Op op) {
    const T *a_ptr = A.data() + A.offset();
    const T *b_ptr = B.data() + B.offset();
    T *c_ptr = C.data() + C.offset();

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
        c_ptr[phys_c] = op(a_ptr[phys_a], b_ptr[phys_b]);
    }
}

template<typename T, typename Op>
void unary_cpu(const Tensor<T>& A, Tensor<T>& C, Op op) {
    const T* a = A.data(); 
    T* c = C.data();

    for (size_t i = 0; i < A.total_elements(); ++i) {
        c[i] = op(a[i]);
    }
}

template<typename T, typename Op>
void unary_cpu_strided(const Tensor<T>& A, Tensor<T>& C, Op op) {
    const T *a_ptr = A.data() + A.offset(); 
    T *c_ptr = C.data() + C.offset();

    TensorInfo info_A(A.shape(), A.strides()); 
    TensorInfo info_C(C.shape(), C.strides());

    for (size_t i = 0; i < A.total_elements(); ++i) {
        size_t linear_idx = i;
        size_t phys_a = 0, phys_c = 0;
        for (int d = info_A.ndims - 1; d >= 0; --d) {
            size_t coord = linear_idx % info_A.shape[d];
            linear_idx /= info_A.shape[d];
            phys_a += coord * info_A.strides[d]; 
            phys_c += coord * info_C.strides[d];
        }
        c_ptr[phys_c] = op(a_ptr[phys_a]);
    }
}

// #############################
// #   CPU operations
// #############################
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

// #######################################################
// #   Generic GPU Declarations
// #######################################################
#if defined(USE_CUDA) || defined(USE_ROCM)
template<typename T, typename Op> void binary_gpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, Op op);
template<typename T, typename Op> void binary_gpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, Op op);
template<typename T, typename Op> void unary_gpu(const Tensor<T>& A, Tensor<T>& C, Op op);
template<typename T, typename Op> void unary_gpu_strided(const Tensor<T>& A, Tensor<T>& C, Op op);

template<typename T> void matmul_gpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);
template<typename T> void matmul_gpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);
template<typename T> Tensor<T> sum_gpu(const Tensor<T>& input, size_t axis, bool keepdims);
#endif

// #############################
// #   Dispatcher
// #############################
std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b);

template<typename T, typename Op, typename AutogradNodeT>
Tensor<T> dispatch_binary(const Tensor<T>& A, const Tensor<T>& B, Op op) {
    if (A.device() != B.device()) throw std::invalid_argument("Device mismatch.");
    
    std::vector<size_t> out_shape = compute_broadcast_shape(A.shape(), B.shape());
    Tensor<T> exp_A = A.expand(out_shape); 
    Tensor<T> exp_B = B.expand(out_shape);
    Tensor<T> C(out_shape, A.device());

    bool contiguous = exp_A.is_contiguous() && exp_B.is_contiguous() && C.is_contiguous();

    if (A.device().type == DeviceType::CPU) {
        if (contiguous) binary_cpu(exp_A, exp_B, C, op); 
        else binary_cpu_strided(exp_A, exp_B, C, op);
    } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (contiguous) binary_gpu(exp_A, exp_B, C, op); 
        else binary_gpu_strided(exp_A, exp_B, C, op);
#else
        throw std::runtime_error("Library was not compiled with GPU support!");
#endif
    }

    if (A.requires_grad() || B.requires_grad()) {
        C.set_requires_grad(true);
        C.set_grad_fn(std::make_shared<AutogradNodeT>(A, B));
    }
    return C;
}

template<typename T, typename Op>
void dispatch_binary_inplace(Tensor<T>& A, const Tensor<T>& B, Op op) {
    if (A.device() != B.device()) {
        throw std::invalid_argument("Device mismatch.");
    }

    // Cannot mutate a leaf variable that requires gradients
    // UNLESS you are operating under a no_grad() context
    if (GradMode::is_enabled() && A.requires_grad() && A.is_leaf()) {
        throw std::runtime_error(
            "A leaf Tensor that requires grad is being used in an in-place operation."
        );
    }

    // In-place requires B to be broadcastable to A, but A's shape cannot change!
    std::vector<size_t> expected_shape = compute_broadcast_shape(A.shape(), B.shape());
    if (A.shape() != expected_shape) {
        throw std::invalid_argument(
            "In-place operation destination tensor cannot be broadcasted to a larger shape."
        );
    }

    Tensor<T> exp_B = B.expand(A.shape());
    bool contiguous = A.is_contiguous() && exp_B.is_contiguous();

    if (A.device().type == DeviceType::CPU) {
        if (contiguous) binary_cpu(A, exp_B, A, op); 
        else binary_cpu_strided(A, exp_B, A, op);
    } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (contiguous) binary_gpu(A, exp_B, A, op); 
        else binary_gpu_strided(A, exp_B, A, op);
#else
        throw std::runtime_error("GPU not supported");
#endif
    }

    A.bump_version();
}

template<typename T, typename Op, typename AutogradNodeT, typename... Args>
Tensor<T> dispatch_unary(const Tensor<T>& A, Op op, Args&&... args) {
    Tensor<T> C(A.shape(), A.device());
    
    if (A.device().type == DeviceType::CPU) {
        if (A.is_contiguous()) unary_cpu(A, C, op); 
        else unary_cpu_strided(A, C, op);
    } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (A.is_contiguous()) unary_gpu(A, C, op); 
        else unary_gpu_strided(A, C, op);
#else
        throw std::runtime_error("Library was not compiled with GPU support!");
#endif
    }

    if (A.requires_grad()) {
        C.set_requires_grad(true);
        C.set_grad_fn(std::make_shared<AutogradNodeT>(A, std::forward<Args>(args)...));
    }
    return C;
}

template<typename T> 
Tensor<T> add(const Tensor<T>& A, const Tensor<T>& B) { 
    return dispatch_binary<T, AddOp<T>, AddBackward<T>>(A, B, AddOp<T>{}); 
}

template<typename T> 
Tensor<T> operator+(const Tensor<T>& A, const Tensor<T>& B) { 
    return add(A, B); 
}

template<typename T> 
Tensor<T> sub(const Tensor<T>& A, const Tensor<T>& B) { 
    return dispatch_binary<T, SubOp<T>, SubBackward<T>>(A, B, SubOp<T>{}); 
}

template<typename T> 
Tensor<T> operator-(const Tensor<T>& A, const Tensor<T>& B) { 
    return sub(A, B); 
}

template<typename T> 
Tensor<T> mul(const Tensor<T>& A, const Tensor<T>& B) { 
    return dispatch_binary<T, MulOp<T>, MulBackward<T>>(A, B, MulOp<T>{}); 
}

template<typename T>
Tensor<T> operator*(const Tensor<T>& A, const Tensor<T>& B) { 
    return mul(A, B); 
}

template<typename T>
Tensor<T> pow2(const Tensor<T>& A) { 
    return dispatch_unary<T, Pow2Op<T>, Pow2Backward<T>>(A, Pow2Op<T>{}); 
}

template<typename T> 
Tensor<T> mul_scalar(const Tensor<T>& A, T scalar) { 
    return dispatch_unary<T, MulScalarOp<T>, MulScalarBackward<T>>(A, MulScalarOp<T>{scalar}, scalar); 
}

template<typename T>
Tensor<T> operator*(const Tensor<T>& A, T scalar) { 
    return mul_scalar(A, scalar); 
}

template<typename T> void sub_(Tensor<T>& A, const Tensor<T>& B) { dispatch_binary_inplace(A, B, SubOp<T>{}); }
template<typename T> void add_(Tensor<T>& A, const Tensor<T>& B) { dispatch_binary_inplace(A, B, AddOp<T>{}); }
template<typename T> void mul_(Tensor<T>& A, const Tensor<T>& B) { dispatch_binary_inplace(A, B, MulOp<T>{}); }

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
    Tensor<T> output;

    switch (A.device().type) {
        case DeviceType::CPU: {
            output = sum_cpu(A, axis, keepdims);
            break;
        }
        case DeviceType::CUDA: {
#if defined(USE_CUDA) || defined(USE_ROCM)
            output = sum_gpu(A, axis, keepdims);
#else
            throw std::runtime_error("Library was not compiled with GPU support!");
#endif
            break;
        }
        default:
            throw std::runtime_error("Unknown device type.");
    }

    if (A.requires_grad()) {
        output.set_requires_grad(true);
        output.set_grad_fn(std::make_shared<SumBackward<T>>(A, axis, keepdims));
    }

    return output;
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