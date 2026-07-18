#pragma once

#include "Autograd.hpp"
#include <cstddef>
#include <algorithm>

// Macro to seamlessly compile functors for both Host (CPU) and Device (GPU)
#ifndef HD_INLINE
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__) || defined(__NVCC__)
#define HD_INLINE __host__ __device__ inline
#else
#define HD_INLINE inline
#endif
#endif

// Prevent unsigned wraparound: returns 0 instead of SIZE_MAX when input + 2*pad < kernel
inline size_t safe_out_size(size_t input_dim, size_t pad, size_t kernel, size_t stride) noexcept {
    if (input_dim + 2 * pad < kernel) return 0;
    return (input_dim + 2 * pad - kernel) / stride + 1;
}

// #######################################################
// #   Functors
// #######################################################
template<typename T> struct AddOp { HD_INLINE T operator()(T a, T b) const { return a + b; } };
template<typename T> struct SubOp { HD_INLINE T operator()(T a, T b) const { return a - b; } };
template<typename T> struct MulOp { HD_INLINE T operator()(T a, T b) const { return a * b; } };
template<typename T> struct Pow2Op { HD_INLINE T operator()(T a) const { return a * a; } };
template<typename T> struct ExpOp { HD_INLINE T operator()(T a) const { return exp(a); } };
template<typename T> struct LogOp { HD_INLINE T operator()(T a) const { return log(a); } };
template<typename T> struct DivOp { HD_INLINE T operator()(T a, T b) const { return a / b; } };
template<typename T> struct SqrtOp { HD_INLINE T operator()(T a) const { return sqrt(a); } };
template<typename T> struct AddScalarOp { 
    T scalar;
    HD_INLINE AddScalarOp(T s) : scalar(s) {}
    HD_INLINE T operator()(T a) const { return a + scalar; } 
};  
template<typename T> struct MulScalarOp { 
    T scalar;
    HD_INLINE MulScalarOp(T s) : scalar(s) {}
    HD_INLINE T operator()(T a) const { return a * scalar; } 
};

// Clamp to a maximum value: min(a, max_val). Used for gradient clipping.
// No backward needed — always used under NoGradGuard.
template<typename T> struct ClampMaxScalarOp { 
    T max_val;
    HD_INLINE ClampMaxScalarOp(T m) : max_val(m) {}
    HD_INLINE T operator()(T a) const { return a > max_val ? max_val : a; } 
};

template<typename T> struct ReLUOp { 
    HD_INLINE T operator()(T a) const { return a > (T)0.0 ? a : (T)0.0; }
};
template<typename T> struct ReLUGradOp { 
    HD_INLINE T operator()(T a, T grad_out) const { return a > (T)0.0 ? grad_out : (T)0.0; }
};

template<typename T> struct SigmoidOp { HD_INLINE T operator()(T a) const { return (T)1.0 / ((T)1.0 + exp(-a)); } };
template<typename T> struct TanhOp { HD_INLINE T operator()(T a) const { return tanh(a); } };

template<typename T> 
struct IdentityOp { 
    HD_INLINE T operator()(T a) const { return a; } 
};

// #######################################################
// #   Generic CPU Execution Engine
// #######################################################
template<typename T, typename Op>
void binary_cpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, Op op) {
    const T* a_ptr = A.data() + A.offset();
    const T* b_ptr = B.data() + B.offset();
    T* c_ptr = C.data() + C.offset();
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
    const T* a = A.data() + A.offset(); 
    T* c = C.data() + C.offset();

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
    
    size_t out_elements = output.total_elements();
    
    for (size_t i = 0; i < out_elements; ++i) {
        size_t linear_idx = i;
        size_t in_phys_base = 0;
        
        for (int d = out_shape.size() - 1; d >= 0; --d) {
            size_t coord = linear_idx % out_shape[d];
            linear_idx /= out_shape[d];

            int in_d = keepdims ? d : (d >= (int)axis ? d + 1 : d);
            in_phys_base += coord * in_strides[in_d];
        }
        
        T sum = 0;
        for (size_t j = 0; j < axis_size; ++j) {
            sum += in_ptr[in_phys_base + j * axis_stride];
        }
        out_ptr[i] = sum;
    }
    
    return output;
}

template<typename T>
void conv2d_cpu(const Tensor<T>& input, const Tensor<T>& weight, const Tensor<T>& bias, Tensor<T>& output, size_t stride, size_t padding) {
    size_t N = input.shape()[0], C_in = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t C_out = weight.shape()[0], kH = weight.shape()[2], kW = weight.shape()[3];
    size_t H_out = output.shape()[2], W_out = output.shape()[3];

    const T* in_ptr = input.data() + input.offset();
    const T* w_ptr = weight.data() + weight.offset();
    const T* b_ptr = bias.empty() ? nullptr : bias.data() + bias.offset();
    T* out_ptr = output.data() + output.offset();

    // Naive 6-nested loop
    for (size_t n = 0; n < N; ++n) {
        for (size_t c_out = 0; c_out < C_out; ++c_out) {
            for (size_t y = 0; y < H_out; ++y) {
                for (size_t x = 0; x < W_out; ++x) {
                    T val = b_ptr ? b_ptr[c_out] : (T)0.0;

                    for (size_t c_in = 0; c_in < C_in; ++c_in) {
                        for (size_t r = 0; r < kH; ++r) {
                            for (size_t s = 0; s < kW; ++s) {
                                int in_y = y * stride + r - padding;
                                int in_x = x * stride + s - padding;

                                if (in_y >= 0 && in_y < (int)H && in_x >= 0 && in_x < (int)W) {
                                    size_t in_idx = n * (C_in * H * W) + c_in * (H * W) + in_y * W + in_x;
                                    size_t w_idx = c_out * (C_in * kH * kW) + c_in * (kH * kW) + r * kW + s;
                                    val += in_ptr[in_idx] * w_ptr[w_idx];
                                }
                            }
                        }
                    }
                    size_t out_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + y * W_out + x;
                    out_ptr[out_idx] = val;
                }
            }
        }
    }
}

template<typename T>
void conv2d_cpu_strided(const Tensor<T>& input, const Tensor<T>& weight, const Tensor<T>& bias, Tensor<T>& output, size_t stride, size_t padding) {
    size_t N = input.shape()[0], C_in = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t C_out = weight.shape()[0], kH = weight.shape()[2], kW = weight.shape()[3];
    size_t H_out = output.shape()[2], W_out = output.shape()[3];

    const T* in_ptr = input.data() + input.offset();
    const T* w_ptr = weight.data() + weight.offset();
    const T* b_ptr = bias.empty() ? nullptr : bias.data() + bias.offset();
    T* out_ptr = output.data() + output.offset();

    auto in_strides = input.strides();
    auto w_strides = weight.strides();
    auto out_strides = output.strides();

    // Naive 6-nested loop
    for (size_t n = 0; n < N; ++n) {
        for (size_t c_out = 0; c_out < C_out; ++c_out) {
            for (size_t y = 0; y < H_out; ++y) {
                for (size_t x = 0; x < W_out; ++x) {
                    T val = b_ptr ? b_ptr[c_out * w_strides[0]] : (T)0.0;

                    for (size_t c_in = 0; c_in < C_in; ++c_in) {
                        for (size_t r = 0; r < kH; ++r) {
                            for (size_t s = 0; s < kW; ++s) {
                                int in_y = y * stride + r - padding;
                                int in_x = x * stride + s - padding;

                                if (in_y >= 0 && in_y < (int)H && in_x >= 0 && in_x < (int)W) {
                                    size_t in_idx = n * in_strides[0] + c_in * in_strides[1] + in_y * in_strides[2] + in_x * in_strides[3];
                                    size_t w_idx = c_out * w_strides[0] + c_in * w_strides[1] + r * w_strides[2] + s * w_strides[3];
                                    val += in_ptr[in_idx] * w_ptr[w_idx];
                                }
                            }
                        }
                    }
                    size_t out_idx = n * out_strides[0] + c_out * out_strides[1] + y * out_strides[2] + x * out_strides[3];
                    out_ptr[out_idx] = val;
                }
            }
        }
    }
}

template<typename T>
void conv2d_backward_input_cpu(const Tensor<T>& grad_output, const Tensor<T>& weight, Tensor<T>& grad_input, size_t stride, size_t padding) {
    size_t N = grad_input.shape()[0], C_in = grad_input.shape()[1],
           H_in = grad_input.shape()[2], W_in = grad_input.shape()[3];
    size_t C_out = weight.shape()[0], kH = weight.shape()[2], kW = weight.shape()[3];
    size_t H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];

    const T* go = grad_output.data() + grad_output.offset();
    const T* wt = weight.data() + weight.offset();
    T* gi = grad_input.data() + grad_input.offset();

    std::fill(gi, gi + grad_input.total_elements(), (T)0);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c_out = 0; c_out < C_out; ++c_out) {
            for (size_t y = 0; y < H_out; ++y) {
                for (size_t x = 0; x < W_out; ++x) {
                    T go_val = go[n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + y * W_out + x];
                    for (size_t c_in = 0; c_in < C_in; ++c_in) {
                        for (size_t r = 0; r < kH; ++r) {
                            for (size_t s_ = 0; s_ < kW; ++s_) {
                                int in_y = static_cast<int>(y * stride + r - padding);
                                int in_x = static_cast<int>(x * stride + s_ - padding);
                                if (in_y >= 0 && in_y < static_cast<int>(H_in) &&
                                    in_x >= 0 && in_x < static_cast<int>(W_in))
                                {
                                    size_t in_idx = n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + in_y * W_in + in_x;
                                    size_t w_idx = c_out * (C_in * kH * kW) + c_in * (kH * kW) + r * kW + s_;
                                    gi[in_idx] += go_val * wt[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename T>
void conv2d_backward_weight_cpu(const Tensor<T>& grad_output, const Tensor<T>& input, Tensor<T>& grad_weight, size_t stride, size_t padding) {
    size_t N = input.shape()[0], C_in = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t C_out = grad_weight.shape()[0], kH = grad_weight.shape()[2], kW = grad_weight.shape()[3];
    size_t H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];

    const T* go = grad_output.data() + grad_output.offset();
    const T* in = input.data() + input.offset();
    T* gw = grad_weight.data() + grad_weight.offset();

    std::fill(gw, gw + grad_weight.total_elements(), (T)0);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c_out = 0; c_out < C_out; ++c_out) {
            for (size_t c_in = 0; c_in < C_in; ++c_in) {
                for (size_t r = 0; r < kH; ++r) {
                    for (size_t s_ = 0; s_ < kW; ++s_) {
                        T sum = 0;
                        for (size_t y = 0; y < H_out; ++y) {
                            for (size_t x = 0; x < W_out; ++x) {
                                int in_y = static_cast<int>(y * stride + r - padding);
                                int in_x = static_cast<int>(x * stride + s_ - padding);
                                if (in_y >= 0 && in_y < static_cast<int>(H) &&
                                    in_x >= 0 && in_x < static_cast<int>(W))
                                {
                                    size_t in_idx = n * (C_in * H * W) + c_in * (H * W) + in_y * W + in_x;
                                    size_t go_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + y * W_out + x;
                                    sum += in[in_idx] * go[go_idx];
                                }
                            }
                        }
                        size_t w_idx = c_out * (C_in * kH * kW) + c_in * (kH * kW) + r * kW + s_;
                        gw[w_idx] += sum;
                    }
                }
            }
        }
    }
}

template<typename T>
void conv2d_backward_bias_cpu(const Tensor<T>& grad_output, Tensor<T>& grad_bias) {
    size_t N = grad_output.shape()[0], C_out = grad_output.shape()[1], H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];
    
    const T* go = grad_output.data() + grad_output.offset();
    T* gb = grad_bias.data() + grad_bias.offset();
    
    std::fill(gb, gb + C_out, (T)0);
    
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C_out; ++c) {
            T sum = 0;
            for (size_t y = 0; y < H_out; ++y) {
                for (size_t x = 0; x < W_out; ++x) {
                    sum += go[n * (C_out * H_out * W_out) + c * (H_out * W_out) + y * W_out + x];
                }
            }
            gb[c] += sum;
        }
    }
}

template<typename T>
std::pair<Tensor<T>, std::vector<size_t>>   // output tensor + flat indices
max_pool2d_cpu(const Tensor<T>& input, size_t k, size_t stride, size_t padding)
{
    size_t N = input.shape()[0], C = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t H_out = safe_out_size(H, padding, k, stride);
    size_t W_out = safe_out_size(W, padding, k, stride);

    Tensor<T> output({N, C, H_out, W_out}, input.device());
    std::vector<size_t> indices(N * C * H_out * W_out); // flat indices

    const T* in_ptr = input.data() + input.offset();
    T* out_ptr = output.data() + output.offset();

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t y = 0; y < H_out; ++y) {
                for (size_t x = 0; x < W_out; ++x) {
                    T max_val = -std::numeric_limits<T>::infinity();
                    size_t max_flat = 0;
                    for (size_t r = 0; r < k; ++r) {
                        for (size_t s_ = 0; s_ < k; ++s_) {
                            int in_y = static_cast<int>(y * stride + r - padding);
                            int in_x = static_cast<int>(x * stride + s_ - padding);
                            if (in_y >= 0 && in_y < static_cast<int>(H) &&
                                in_x >= 0 && in_x < static_cast<int>(W))
                            {
                                size_t flat = (n * C + c) * (H * W) + in_y * W + in_x;
                                T val = in_ptr[flat];
                                if (val > max_val) {
                                    max_val = val;
                                    max_flat = flat;
                                }
                            }
                        }
                    }
                    out_ptr[(n * C + c) * (H_out * W_out) + y * W_out + x] = max_val;
                    size_t idx = ((n * C + c) * H_out + y) * W_out + x;
                    indices[idx] = max_flat;
                }
            }
        }
    }
    return {output, indices};
}

template<typename T>
std::pair<Tensor<T>, std::vector<size_t>>   // output tensor + flat indices
max_pool2d_cpu_strided(const Tensor<T>& input, size_t k, size_t stride, size_t padding)
{
    size_t N = input.shape()[0], C = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t H_out = safe_out_size(H, padding, k, stride);
    size_t W_out = safe_out_size(W, padding, k, stride);

    Tensor<T> output({N, C, H_out, W_out}, input.device());
    std::vector<size_t> indices(N * C * H_out * W_out); // flat indices

    const T* in_ptr = input.data() + input.offset();
    T* out_ptr = output.data() + output.offset();

    const auto& in_strides = input.strides();

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t y = 0; y < H_out; ++y) {
                for (size_t x = 0; x < W_out; ++x) {
                    T max_val = -std::numeric_limits<T>::infinity();
                    size_t max_flat = 0;
                    for (size_t r = 0; r < k; ++r) {
                        for (size_t s_ = 0; s_ < k; ++s_) {
                            int in_y = static_cast<int>(y * stride + r - padding);
                            int in_x = static_cast<int>(x * stride + s_ - padding);
                            if (in_y >= 0 && in_y < static_cast<int>(H) &&
                                in_x >= 0 && in_x < static_cast<int>(W))
                            {
                                size_t flat = n * in_strides[0] + c * in_strides[1] + in_y * in_strides[2] + in_x * in_strides[3];
                                T val = in_ptr[flat];
                                if (val > max_val) {
                                    max_val = val;
                                    max_flat = ((n * C + c) * H + in_y) * W + in_x;
                                }
                            }
                        }
                    }
                    size_t idx = n * in_strides[0] + c * in_strides[1] + y * in_strides[2] + x * in_strides[3];
                    out_ptr[idx] = max_val;
                    indices[idx] = max_flat;
                }
            }
        }
    }
    return {output, indices};
}

template<typename T>
void max_pool2d_backward_cpu(const Tensor<T>& grad_output, Tensor<T>& grad_input, const std::vector<size_t>& indices) {
    const T* go_ptr = grad_output.data() + grad_output.offset();
    T* gi_ptr = grad_input.data() + grad_input.offset();
    
    std::fill(gi_ptr, gi_ptr + grad_input.total_elements(), (T)0);
    
    for (size_t i = 0; i < grad_output.total_elements(); ++i) {
        gi_ptr[indices[i]] += go_ptr[i];
    }
}

template<typename T>
Tensor<T> softmax_cpu(const Tensor<T>& input) {
    auto shape = input.shape();
    size_t last_dim = shape.back();
    size_t outer = input.total_elements() / last_dim;

    Tensor<T> output(input.shape(), input.device());
    const T* in_ptr = input.data() + input.offset();
    T* out_ptr = output.data() + output.offset();

    for (size_t i = 0; i < outer; ++i) {
        const T* row_in = in_ptr + i * last_dim;
        T* row_out = out_ptr + i * last_dim;

        T max_val = *std::max_element(row_in, row_in + last_dim);

        T sum = 0;
        for (size_t j = 0; j < last_dim; ++j) {
            row_out[j] = exp(row_in[j] - max_val);
            sum += row_out[j];
        }

        T inv_sum = T(1) / sum;
        for (size_t j = 0; j < last_dim; ++j) {
            row_out[j] *= inv_sum;
        }
    }
    return output;
}

template<typename T>
void copy_cpu_strided(const Tensor<T> src, T* dst) {
    size_t total_elements = src.total_elements();

    const T *src_ptr = src.data() + src.offset();

    const auto& shape = src.shape();
    const auto& strides = src.strides();
    int ndims = shape.size();
    
    for (size_t i = 0; i < total_elements; ++i) {
        size_t linear_idx = i;
        size_t src_phys_idx = 0;
        
        for (int d = ndims - 1; d >= 0; --d) {
            size_t coord = linear_idx % shape[d];
            linear_idx /= shape[d];
            src_phys_idx += coord * strides[d];
        }
        
        dst[i] = src_ptr[src_phys_idx];
    }
}

template <typename T>
Tensor<T> stack(const std::vector<Tensor<T>>& tensors, size_t dim) {
    if (tensors.empty())
        throw std::invalid_argument("stack: empty tensor list");

    const auto& first_shape = tensors[0].shape();
    for (size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i].shape() != first_shape)
            throw std::invalid_argument("stack: all tensors must have same shape");
    }

    // New shape: insert a dimension of size N at position `dim`
    size_t N = tensors.size();
    auto out_shape = first_shape;
    out_shape.insert(out_shape.begin() + dim, N);

    Device dev = tensors[0].device();
    Tensor<T> result(out_shape, dev);

    // For each input, copy its entire data into the appropriate slice of result
    for (size_t i = 0; i < N; ++i) {
        auto out_slice = result.slice(dim, i, i + 1);   // shape with 1 at `dim`
        // out_slice is a view; we need to copy the input's values into it.
        // Make both contiguous to use a simple copy, or use copy_strided.
        auto in_contig = tensors[i].contiguous();

        size_t num_elements = in_contig.total_elements();
        size_t bytes = num_elements * sizeof(T);

        if (out_slice.is_contiguous()) {
            if (dev.type == DeviceType::CPU) {
                std::copy(in_contig.data() + in_contig.offset(), in_contig.data() + in_contig.offset() + num_elements, out_slice.data() + out_slice.offset());
            } else {
    #if defined(USE_CUDA)
                GPU_CHECK(cudaMemcpy(out_slice.data() + out_slice.offset(), in_contig.data() + in_contig.offset(), bytes, cudaMemcpyDeviceToDevice));
    #elif defined(USE_ROCM)
                GPU_CHECK(hipMemcpy(out_slice.data() + out_slice.offset(), in_contig.data() + in_contig.offset(), bytes, hipMemcpyDeviceToDevice));
    #else
                throw std::runtime_error("GPU not supported");
    #endif
            }
        } else {
            if (dev.type == DeviceType::CPU) {
                unary_cpu_strided(in_contig, out_slice, IdentityOp<T>{});
            } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
                unary_gpu_strided(in_contig, out_slice, IdentityOp<T>{});
#endif
            }
        }

    }
    return result;
}

template <typename T>
Tensor<T> concat(const std::vector<Tensor<T>>& tensors, size_t dim) {
    if (tensors.empty()) throw std::invalid_argument("concat: empty list");
    const auto& first_shape = tensors[0].shape();
    size_t ndim = first_shape.size();
    for (const auto& t : tensors) {
        if (t.shape().size() != ndim)
            throw std::invalid_argument("concat: tensors must have same number of dimensions");
    }
    // verify other dims equal
    std::vector<size_t> out_shape = first_shape;
    out_shape[dim] = 0;
    for (const auto& t : tensors) {
        for (size_t d = 0; d < ndim; ++d) {
            if (d == dim)
                out_shape[d] += t.shape()[d];
            else if (t.shape()[d] != out_shape[d])
                throw std::invalid_argument("concat: shape mismatch");
        }
    }

    Device dev = tensors[0].device();
    Tensor<T> result(out_shape, dev);
    size_t offset = 0;

    for (const auto& t : tensors) {
        auto slice = result.slice(dim, offset, offset + t.shape()[dim]);
        auto contig_t = t.contiguous();

        size_t num_elements = contig_t.total_elements();
        size_t bytes = num_elements * sizeof(T);

        if (slice.is_contiguous()) {
            if (dev.type == DeviceType::CPU) {
                std::copy(contig_t.data() + contig_t.offset(), contig_t.data() + contig_t.offset() + num_elements, slice.data() + slice.offset());
            } else {
    #if defined(USE_CUDA)
                GPU_CHECK(cudaMemcpy(slice.data() + slice.offset(), contig_t.data() + contig_t.offset(), bytes, cudaMemcpyDeviceToDevice));
    #elif defined(USE_ROCM)
                GPU_CHECK(hipMemcpy(slice.data() + slice.offset(), contig_t.data() + contig_t.offset(), bytes, hipMemcpyDeviceToDevice));
    #else
                throw std::runtime_error("GPU not supported");
    #endif
            }
        } else {
            if (dev.type == DeviceType::CPU) {
                unary_cpu_strided(contig_t, slice, IdentityOp<T>{});
            } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
                unary_gpu_strided(contig_t, slice, IdentityOp<T>{});
#endif
            }
        }

        offset += t.shape()[dim];
    }
    return result;
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
template<typename T> void matmul_gpu_v2(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);
template<typename T> void matmul_gpu_bk16(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);
template<typename T> Tensor<T> sum_gpu(const Tensor<T>& input, size_t axis, bool keepdims);
template<typename T> void conv2d_gpu(const Tensor<T>& input, const Tensor<T>& weight, const Tensor<T>& bias, Tensor<T>& output, size_t stride, size_t padding);
template<typename T> void conv2d_gpu_strided(const Tensor<T>& input, const Tensor<T>& weight, const Tensor<T>& bias, Tensor<T>& output, size_t stride, size_t padding);
template<typename T> void conv2d_backward_input_gpu(const Tensor<T>& grad_output, const Tensor<T>& weight, Tensor<T>& grad_input, size_t stride, size_t padding);
template<typename T> void conv2d_backward_weight_gpu(const Tensor<T>& grad_output, const Tensor<T>& input, Tensor<T>& grad_weight, size_t stride, size_t padding);
template<typename T> void conv2d_backward_bias_gpu(const Tensor<T>& grad_output, Tensor<T>& grad_bias);
template<typename T> std::pair<Tensor<T>, Tensor<size_t>> max_pool2d_gpu(const Tensor<T>& input, size_t k, size_t stride, size_t padding);
template<typename T> std::pair<Tensor<T>, Tensor<size_t>> max_pool2d_gpu_strided(const Tensor<T>& input, size_t k, size_t stride, size_t padding);
template<typename T> void max_pool2d_backward_gpu(const Tensor<T>& grad_output, Tensor<T>& grad_input, const Tensor<size_t>& d_indices);
template<typename T> void max_pool2d_backward_gpu_strided(const Tensor<T>& grad_output, Tensor<T>& grad_input, const Tensor<size_t>& d_indices);
template<typename T> void copy_gpu_strided(const Tensor<T> src, T* dst);
template<typename T> Tensor<T> cross_entropy_fwd_gpu(const Tensor<T>&, const Tensor<T>&);
template<typename T> void cross_entropy_bwd_gpu(const Tensor<T>&, const Tensor<T>&, const Tensor<T>&, Tensor<T>&);
template<typename T> void adam_step_gpu(Tensor<T>& param, const Tensor<T>& grad, Tensor<T>& m, Tensor<T>& v, T lr, T beta1, T beta2, T eps, T bias_correction1, T bias_correction2, T weight_decay);
template<typename T> void adamw_step_gpu(Tensor<T>& param, const Tensor<T>& grad, Tensor<T>& m, Tensor<T>& v, T lr, T beta1, T beta2, T eps, T bias_correction1, T bias_correction2, T weight_decay);
template<typename T> Tensor<T> softmax_gpu(const Tensor<T>& input);
template<typename T> void fill_gpu(T* data, T val, size_t N);
template<typename T> void add_relu_gpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);
template<typename T> void bn_fwd_gpu(const Tensor<T>& input, Tensor<T>& mean, Tensor<T>& var);
template<typename T> void bn_relu_fwd_gpu(const Tensor<T>& input, Tensor<T>& output, const Tensor<T>& mean, const Tensor<T>& var, const Tensor<T>& gamma, const Tensor<T>& beta, T eps);
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

    if (GradMode::is_enabled() && (A.requires_grad() || B.requires_grad())) {
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

    if (GradMode::is_enabled() && A.requires_grad()) {
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
Tensor<T> div(const Tensor<T>& A, const Tensor<T>& B) { 
    return dispatch_binary<T, DivOp<T>, DivBackward<T>>(A, B, DivOp<T>{}); 
}

template<typename T>
Tensor<T> operator/(const Tensor<T>& A, const Tensor<T>& B) { 
    return div(A, B); 
}

template<typename T>
Tensor<T> pow2(const Tensor<T>& A) { 
    return dispatch_unary<T, Pow2Op<T>, Pow2Backward<T>>(A, Pow2Op<T>{}); 
}

template<typename T>
Tensor<T> exp(const Tensor<T>& A) { 
    return dispatch_unary<T, ExpOp<T>, ExpBackward<T>>(A, ExpOp<T>{}); 
}

template<typename T>
Tensor<T> log(const Tensor<T>& A) { 
    return dispatch_unary<T, LogOp<T>, LogBackward<T>>(A, LogOp<T>{}); 
}

template<typename T>
Tensor<T> sqrt(const Tensor<T>& A) { 
    return dispatch_unary<T, SqrtOp<T>, Pow2Backward<T>>(A, SqrtOp<T>{}); 
}

template<typename T> 
Tensor<T> add_scalar(const Tensor<T>& A, T scalar) { 
    return dispatch_unary<T, AddScalarOp<T>, AddScalarBackward<T>>(A, AddScalarOp<T>{scalar}, scalar); 
}

template<typename T>
Tensor<T> operator+(const Tensor<T>& A, T scalar) { 
    return add_scalar(A, scalar); 
}

template<typename T> 
Tensor<T> mul_scalar(const Tensor<T>& A, T scalar) { 
    return dispatch_unary<T, MulScalarOp<T>, MulScalarBackward<T>>(A, MulScalarOp<T>{scalar}, scalar); 
}

template<typename T>
Tensor<T> operator*(const Tensor<T>& A, T scalar) { 
    return mul_scalar(A, scalar); 
}

// Clamp each element to a maximum value: min(a, max_val).
// Used for gradient norm clipping (always under NoGradGuard).
template<typename T>
Tensor<T> clamp_max_scalar(const Tensor<T>& A, T max_val) { 
    return dispatch_unary<T, ClampMaxScalarOp<T>, Pow2Backward<T>>(A, ClampMaxScalarOp<T>{max_val});
}

template<typename T>
Tensor<T> flatten(const Tensor<T>& x) {
    if (x.shape().empty()) return x;
    size_t batch = x.shape()[0];
    size_t rest = x.total_elements() / batch;
    Tensor<T> out = x.reshape({batch, rest});
    
    if (GradMode::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out.set_grad_fn(std::make_shared<FlattenBackward<T>>(x));
    }
    return out;
}

template<typename T> void sub_(Tensor<T>& A, const Tensor<T>& B) { dispatch_binary_inplace(A, B, SubOp<T>{}); }
template<typename T> void add_(Tensor<T>& A, const Tensor<T>& B) { dispatch_binary_inplace(A, B, AddOp<T>{}); }
template<typename T> void mul_(Tensor<T>& A, const Tensor<T>& B) { dispatch_binary_inplace(A, B, MulOp<T>{}); }

template<typename T>
Tensor<T> bmm(const Tensor<T>& A, const Tensor<T>& B) {
    // A: [B, M, K], B: [B, K, N] -> C: [B, M, N]
    if (A.device() != B.device())
        throw std::invalid_argument("Tensors must be on the same device for bmm.");
    if (A.shape().size() != 3 || B.shape().size() != 3)
        throw std::invalid_argument("bmm requires 3D tensors.");
    if (A.shape()[0] != B.shape()[0])
        throw std::invalid_argument("bmm: batch dimension mismatch.");
    if (A.shape()[2] != B.shape()[1])
        throw std::invalid_argument("bmm: inner dimension mismatch (A.K != B.K).");

    size_t B_dim = A.shape()[0];
    size_t M = A.shape()[1];
    size_t K = A.shape()[2];
    size_t N = B.shape()[2];

    Tensor<T> C({B_dim, M, N}, A.device());
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    auto C_contig = C.contiguous();

    for (size_t b = 0; b < B_dim; ++b) {
        size_t offset_A = b * M * K;
        size_t offset_B = b * K * N;
        size_t offset_C = b * M * N;

        // Create 2D views for each batch
        auto A_slice = Tensor<T>(std::make_shared<TensorImpl<T>>(
            A_contig.impl_->storage_, std::vector<size_t>{M, K},
            std::vector<size_t>{K, 1}, A_contig.offset() + offset_A
        ));
        auto B_slice = Tensor<T>(std::make_shared<TensorImpl<T>>(
            B_contig.impl_->storage_, std::vector<size_t>{K, N},
            std::vector<size_t>{N, 1}, B_contig.offset() + offset_B
        ));
        auto C_slice = Tensor<T>(std::make_shared<TensorImpl<T>>(
            C_contig.impl_->storage_, std::vector<size_t>{M, N},
            std::vector<size_t>{N, 1}, C_contig.offset() + offset_C
        ));

        // matmul the 2D slices — works on CPU and GPU
        if (A.device().type == DeviceType::CPU) {
            matmul_cpu(A_slice, B_slice, C_slice);
        } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
            matmul_gpu(A_slice, B_slice, C_slice);
#else
            throw std::runtime_error("GPU not supported");
#endif
        }
    }

    if (GradMode::is_enabled() && (A.requires_grad() || B.requires_grad())) {
        C.set_requires_grad(true);
        C.set_grad_fn(std::make_shared<MatmulBackward<T>>(A, B));
    }
    return C;
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

    if (A.requires_grad() && GradMode::is_enabled()) {
        output.set_requires_grad(true);
        output.set_grad_fn(std::make_shared<SumBackward<T>>(A, axis, keepdims));
    }

    return output;
}

template<typename T>
Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& weight, const Tensor<T>& bias, size_t stride, size_t padding) {
    if (input.device() != weight.device() || (!bias.empty() && input.device() != bias.device()))
        throw std::invalid_argument("Device mismatch.");
    
    size_t N = input.shape()[0], /* C_in = input.shape()[1] */ H = input.shape()[2], W = input.shape()[3];
    size_t C_out = weight.shape()[0], kH = weight.shape()[2], kW = weight.shape()[3];
    size_t H_out = safe_out_size(H, padding, kH, stride);
    size_t W_out = safe_out_size(W, padding, kW, stride);

    Tensor<T> output({N, C_out, H_out, W_out}, input.device());

    bool contiguous = input.is_contiguous() && weight.is_contiguous() &&
                      (bias.empty() || bias.is_contiguous()) && output.is_contiguous();

    if (input.device().type == DeviceType::CPU) {
        if (contiguous) conv2d_cpu(input, weight, bias, output, stride, padding);
        else conv2d_cpu_strided(input, weight, bias, output, stride, padding);
    } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (contiguous) conv2d_gpu(input, weight, bias, output, stride, padding);
        else conv2d_gpu_strided(input, weight, bias, output, stride, padding);
#else
        throw std::runtime_error("GPU not supported");
#endif
    }

    if (GradMode::is_enabled() && (input.requires_grad() || weight.requires_grad() ||
        (!bias.empty() && bias.requires_grad())))
    {
        output.set_requires_grad(true);
        output.set_grad_fn(std::make_shared<Conv2DBackward<T>>(input, weight, bias, stride, padding));
    }
    return output;
}

template<typename T> 
void conv2d_backward_input(const Tensor<T>& grad_output, const Tensor<T>& weight, Tensor<T>& grad_input, size_t stride, size_t padding) {
    if (grad_input.device().type == DeviceType::CPU) {
        conv2d_backward_input_cpu(grad_output, weight, grad_input, stride, padding);
    } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
        conv2d_backward_input_gpu(grad_output, weight, grad_input, stride, padding); 
#else
        throw std::runtime_error("Library was not compiled with GPU support!");
#endif
    }
}

template<typename T> 
void conv2d_backward_weight(const Tensor<T>& grad_output, const Tensor<T>& input, Tensor<T>& grad_weight, size_t stride, size_t padding) {
    if (grad_weight.device().type == DeviceType::CPU) {
        conv2d_backward_weight_cpu(grad_output, input, grad_weight, stride, padding);
    } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
        conv2d_backward_weight_gpu(grad_output, input, grad_weight, stride, padding); 
#else
        throw std::runtime_error("Library was not compiled with GPU support!");
#endif
    }
}

template<typename T> 
void conv2d_backward_bias(const Tensor<T>& grad_output, Tensor<T>& grad_bias) {
    if (grad_bias.device().type == DeviceType::CPU) {
        conv2d_backward_bias_cpu(grad_output, grad_bias);
    } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
        conv2d_backward_bias_gpu(grad_output, grad_bias); 
#else
        throw std::runtime_error("Library was not compiled with GPU support!");
#endif
    }
}

template<typename T>
Tensor<T> max_pool2d(const Tensor<T>& input, size_t kernel_size, size_t stride, size_t padding) {
    if (input.shape().size() < 4)
        throw std::invalid_argument("max_pool2d: input must have at least 4 dims [N, C, H, W]");
    size_t H = input.shape()[2], W = input.shape()[3];
    if (H + 2 * padding < kernel_size || W + 2 * padding < kernel_size)
        throw std::invalid_argument("max_pool2d: input spatial dims too small for kernel+padding");
    Tensor<T> output;
    std::vector<size_t> indices;

    if (input.device().type == DeviceType::CPU) {
        if (input.is_contiguous()) {
            auto res = max_pool2d_cpu(input, kernel_size, stride, padding);
            output = res.first;
            if (GradMode::is_enabled() && input.requires_grad()) {
                output.set_requires_grad(true);
                output.set_grad_fn(std::make_shared<MaxPool2DBackward<T>>(input, std::move(res.second)));
            }
        } else {
            auto res = max_pool2d_cpu_strided(input, kernel_size, stride, padding);
            output = res.first;
            if (GradMode::is_enabled() && input.requires_grad()) {
                output.set_requires_grad(true);
                output.set_grad_fn(std::make_shared<MaxPool2DBackward<T>>(input, std::move(res.second)));
            }
        }
    } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (input.is_contiguous()) {
            auto res = max_pool2d_gpu(input, kernel_size, stride, padding);
            output = res.first;
            if (GradMode::is_enabled() && input.requires_grad()) {
                output.set_requires_grad(true);
                output.set_grad_fn(std::make_shared<MaxPool2DBackward<T>>(input, std::move(res.second)));
            }
        } else {
            auto res = max_pool2d_gpu_strided(input, kernel_size, stride, padding);
            output = res.first;
            if (GradMode::is_enabled() && input.requires_grad()) {
                output.set_requires_grad(true);
                output.set_grad_fn(std::make_shared<MaxPool2DBackward<T>>(input, std::move(res.second)));
            }
        }
#else
        throw std::runtime_error("Library was not compiled with GPU support!");
#endif
    }
    
    return output;
}

template<typename T>
void max_pool2d_backward(const Tensor<T>& grad_output, Tensor<T>& grad_input, const std::vector<size_t>& indices) {
    if (grad_input.device().type != DeviceType::CPU)
        throw std::runtime_error("max_pool2d_backward(vector) for GPU: use Tensor<size_t> overload");
    max_pool2d_backward_cpu(grad_output, grad_input, indices);
}

template<typename T>
void max_pool2d_backward(const Tensor<T>& grad_output, Tensor<T>& grad_input, const Tensor<size_t>& indices) {
#if defined(USE_CUDA) || defined(USE_ROCM)
    max_pool2d_backward_gpu(grad_output, grad_input, indices); 
#else
    throw std::runtime_error("GPU not supported");
#endif
}

template<typename T> 
Tensor<T> relu(const Tensor<T>& A) { 
    return dispatch_unary<T, ReLUOp<T>, ReLUBackward<T>>(A, ReLUOp<T>{}); 
}

template<typename T>
Tensor<T> relu_grad(const Tensor<T>& a, const Tensor<T>& grad_output) {
    // Dispatch using a dummy AutogradNode. Since we use NoGradGuard in Autograd.hpp, 
    // the node will be ignored and no graph will be built.
    return dispatch_binary<T, ReLUGradOp<T>, AddBackward<T>>(a, grad_output, ReLUGradOp<T>{}); 
}

template<typename T> 
Tensor<T> sigmoid(const Tensor<T>& A) { 
    return dispatch_unary<T, SigmoidOp<T>, SigmoidBackward<T>>(A, SigmoidOp<T>{}); 
}

template<typename T> 
Tensor<T> tanh(const Tensor<T>& A) { 
    return dispatch_unary<T, TanhOp<T>, TanhBackward<T>>(A, TanhOp<T>{}); 
}

template<typename T>
Tensor<T> softmax(const Tensor<T>& input) {
    Tensor<T> output;

    switch (input.device().type) {
        case DeviceType::CPU: {
            output = softmax_cpu(input);
            break;
        }
        case DeviceType::CUDA: {
#if defined(USE_CUDA) || defined(USE_ROCM)
            output = softmax_gpu(input);
#else
            throw std::runtime_error("Library was not compiled with GPU support!");
#endif
            break;
        }
        default:
            throw std::runtime_error("Unknown device type.");
    }

    if (GradMode::is_enabled() && input.requires_grad()) {
        output.set_requires_grad(true);
        output.set_grad_fn(std::make_shared<SoftmaxBackward<T>>(output));
    }

    return output;
}

template<typename T> 
void copy_strided(const Tensor<T> src, T* dst) {
    if (src.device() == DeviceType::CPU) {
        copy_cpu_strided(src, dst);
    } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
        copy_gpu_strided(src, dst);
#else
        throw std::runtime_error("Library was not compiled with GPU support!");
#endif
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
    NoGradGuard guard;

    const auto& grad_shape = grad.shape();

    if (grad_shape.empty()) {
        if (target_shape.empty()) {
            return grad;
        }
        throw std::invalid_argument("Cannot unbroadcast a scalar to a non‑scalar shape");
    }

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

    if (!result.is_contiguous()) {
        result = result * (T)1.0;
    }

    return result;
}

// ---------------------------------------------------------------
// Autograd‑aware view operations
// ---------------------------------------------------------------

template<typename T>
Tensor<T> view_slice(const Tensor<T>& x, size_t dim, size_t start, size_t end) {
    Tensor<T> result = x.slice(dim, start, end);
    if (GradMode::is_enabled() && x.requires_grad()) {
        result.set_grad_fn(std::make_shared<SliceBackward<T>>(x, dim, start, end));
    }
    return result;
}

template<typename T>
Tensor<T> view_transpose(const Tensor<T>& x, size_t dim0, size_t dim1) {
    Tensor<T> result = x.transpose(dim0, dim1);
    if (GradMode::is_enabled() && x.requires_grad()) {
        result.set_grad_fn(std::make_shared<TransposeBackward<T>>(x, dim0, dim1));
    }
    return result;
}

template<typename T>
Tensor<T> view_reshape(const Tensor<T>& x, const std::vector<size_t>& new_shape) {
    Tensor<T> result = x.reshape(new_shape);
    if (GradMode::is_enabled() && x.requires_grad()) {
        result.set_grad_fn(std::make_shared<ReshapeBackward<T>>(x));
    }
    return result;
}

template<typename T>
Tensor<T> view_expand(const Tensor<T>& x, const std::vector<size_t>& target_shape) {
    Tensor<T> result = x.expand(target_shape);
    if (GradMode::is_enabled() && x.requires_grad()) {
        result.set_grad_fn(std::make_shared<ExpandBackward<T>>(x, x.shape()));
    }
    return result;
}

template<typename T>
Tensor<T> add_relu(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.device() != b.device())
        throw std::invalid_argument("Device mismatch for add_relu.");
    auto out_shape = compute_broadcast_shape(a.shape(), b.shape());
    Tensor<T> A = a.expand(out_shape);
    Tensor<T> B = b.expand(out_shape);
    Tensor<T> C(out_shape, a.device());

    if (a.device().type == DeviceType::CPU) {
        T* c_ptr = C.data();
        const T* a_ptr = A.data();
        const T* b_ptr = B.data();
        for (size_t i = 0; i < C.total_elements(); ++i) {
            T val = a_ptr[i] + b_ptr[i];
            c_ptr[i] = val > T(0) ? val : T(0);
        }
    } else {
#if defined(USE_CUDA) || defined(USE_ROCM)
        add_relu_gpu(A, B, C);
#else
        throw std::runtime_error("GPU not supported for add_relu");
#endif
    }

    if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
        C.set_requires_grad(true);
        C.set_grad_fn(std::make_shared<AddReLUBackward<T>>(A, B));
    }
    return C;
}