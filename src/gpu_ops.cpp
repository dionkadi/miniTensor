#include "TensorOps.hpp"
#include "GpuUtils.hpp"
#include "TensorImpl.hpp"
#include <cstddef>

// #######################################################
// #   Generic GPU Kernels
// #######################################################
template<typename T, typename Op>
__global__ void binary_kernel(const T* a, const T* b, T* c, size_t size, Op op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = op(a[idx], b[idx]);
    }
}

template<typename T, typename Op>
__global__ void binary_kernel_strided(
    const T *a, const T *b, T *c, 
    TensorInfo info_A, TensorInfo info_B, TensorInfo info_C,
    size_t size,
    size_t offset_a, size_t offset_b, size_t offset_c,
    Op op
) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        size_t linear_idx = idx;
        size_t phys_a = 0, phys_b = 0, phys_c = 0;

        for (int d = info_A.ndims - 1; d >= 0; --d) {
            size_t coord = linear_idx % info_A.shape[d];
            linear_idx /= info_A.shape[d];
            phys_a += coord * info_A.strides[d];
            phys_b += coord * info_B.strides[d];
            phys_c += coord * info_C.strides[d];
        }
        c[offset_c + phys_c] = op(a[offset_a + phys_a], b[offset_b + phys_b]);
    }
}

template<typename T, typename Op>
__global__ void unary_kernel(const T* a, T* c, size_t size, Op op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = op(a[idx]);
    }
}

template<typename T, typename Op>
__global__ void unary_kernel_strided(
    const T *a, T *c, 
    TensorInfo info_A, TensorInfo info_C,
    size_t size, size_t offset_a, size_t offset_c,
    Op op
) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        size_t linear_idx = idx;
        size_t phys_a = 0, phys_c = 0;

        for (int d = info_A.ndims - 1; d >= 0; --d) {
            size_t coord = linear_idx % info_A.shape[d];
            linear_idx /= info_A.shape[d];
            phys_a += coord * info_A.strides[d];
            phys_c += coord * info_C.strides[d];
        }
        c[offset_c + phys_c] = op(a[offset_a + phys_a]);
    }
}

// #######################################################
// #   Generic GPU Execution Setup
// #######################################################
template<typename T, typename Op>
void binary_gpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, Op op) {
    size_t total_elements = A.total_elements();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    binary_kernel<<<blocks, threads>>>(A.data(), B.data(), C.data(), total_elements, op);
    
#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError()); GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
#endif
}

template<typename T, typename Op>
void binary_gpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, Op op) {
    size_t total_elements = A.total_elements();
    size_t threads = 256;
    size_t blocks = (total_elements + threads - 1) / threads;

    TensorInfo info_A(A.shape(), A.strides());
    TensorInfo info_B(B.shape(), B.strides());
    TensorInfo info_C(C.shape(), C.strides());
    collapse_dims(info_A, info_B, info_C);

    binary_kernel_strided<<<blocks, threads>>>(
        A.data(), B.data(), C.data(), info_A, info_B, info_C,
        total_elements, A.offset(), B.offset(), C.offset(), op
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError()); GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
#endif
}

template<typename T, typename Op>
void unary_gpu(const Tensor<T>& A, Tensor<T>& C, Op op) {
    size_t total_elements = A.total_elements();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    unary_kernel<<<blocks, threads>>>(A.data(), C.data(), total_elements, op);
    
#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError()); GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
#endif
}

template<typename T, typename Op>
void unary_gpu_strided(const Tensor<T>& A, Tensor<T>& C, Op op) {
    size_t total_elements = A.total_elements();
    size_t threads = 256;
    size_t blocks = (total_elements + threads - 1) / threads;

    TensorInfo info_A(A.shape(), A.strides());
    TensorInfo info_C(C.shape(), C.strides());

    unary_kernel_strided<<<blocks, threads>>>(
        A.data(), C.data(), info_A, info_C,
        total_elements, A.offset(), C.offset(), op
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError()); GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
#endif
}

// #######################################################
// #   Non-Generic Ops
// #######################################################
template<typename T>
__global__ void matmul_kernel(
    const T *a, const T *b, T *c,
    size_t M, size_t K, size_t N,
    size_t offset_a, size_t offset_b, size_t offset_c
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (size_t k = 0; k < K; ++k) {
            sum += a[offset_a + row * K + k] * b[offset_b + k * N + col];
        }
        c[offset_c + row * N + col] = sum;
    }
}

template<typename T>
__global__ void matmul_kernel_strided(
    const T* a, const T* b, T* c, 
    size_t M, size_t K, size_t N,
    size_t stride_a_m, size_t stride_a_k,
    size_t stride_b_k, size_t stride_b_n,
    size_t stride_c_m, size_t stride_c_n,
    size_t offset_a, size_t offset_b, size_t offset_c
) { 
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (size_t k = 0; k < K; ++k) {
            T a_val = a[offset_a + row * stride_a_m + k * stride_a_k];
            T b_val = b[offset_b + k * stride_b_k + col * stride_b_n];
            sum += a_val * b_val;
        }
        c[offset_c + row * stride_c_m + col * stride_c_n] = sum;
    }
}

template<typename T>
void matmul_gpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];

    // Standard 16x16 thread block for 2D grid
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, 
                (M + threads.y - 1) / threads.y);

    matmul_kernel<<<blocks, threads>>>(
        A.data(), B.data(), C.data(),
        M, K, N,
        A.offset(), B.offset(), C.offset()
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif
}

template<typename T>
void matmul_gpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, 
                (M + threads.y - 1) / threads.y);

    matmul_kernel_strided<<<blocks, threads>>>(
        A.data(), B.data(), C.data(),
        M, K, N,
        A.strides()[0], A.strides()[1],
        B.strides()[0], B.strides()[1],
        C.strides()[0], C.strides()[1],
        A.offset(), B.offset(), C.offset()
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif
}

template<typename T>
__global__ void sum_kernel(
    const T *in, T *out,
    size_t total_elements,
    TensorInfo in_info,
    TensorInfo out_info,
    size_t reduce_axis,
    bool keepdims,
    size_t offset_in, 
    size_t offset_out
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        size_t temp_idx = idx;
        size_t in_phys_offset = 0;
        size_t out_phys_offset = 0;

        // Map linear index back to N-dimensional coordinates
        for (int d = in_info.ndims - 1; d >= 0; --d) {
            size_t coord = temp_idx % in_info.shape[d];
            temp_idx /= in_info.shape[d];

            in_phys_offset += coord * in_info.strides[d];

            // Determine where this coordinate falls in the output tensor
            if (d == reduce_axis) {
                // We are reducing along this axis, so it maps to index 0 in the output
                if (keepdims) {
                    // The dimension still exists (size 1), so its stride is used but coord is 0
                    out_phys_offset += 0 * out_info.strides[d];
                }
                // If not keepdims, this dimension doesn't exist in the output, so we add nothing.
            } else {
                // Find the corresponding dimension in the output tensor
                int out_d = keepdims ? d : (d > reduce_axis ? d - 1 : d);
                out_phys_offset += coord * out_info.strides[out_d];
            }
        }

        atomicAdd(&out[offset_out + out_phys_offset], in[offset_in + in_phys_offset]);
    }
}

template<typename T>
Tensor<T> sum_gpu(const Tensor<T>& input, size_t axis, bool keepdims) {
    const auto& in_shape = input.shape();
    
    if (axis >= in_shape.size()) {
        throw std::invalid_argument("Reduction axis is out of bounds.");
    }

    std::vector<size_t> out_shape = in_shape;
    if (keepdims) {
        out_shape[axis] = 1;
    } else {
        out_shape.erase(out_shape.begin() + axis);
    }

    Tensor<T> output(out_shape, input.device());

    size_t bytes = output.total_elements() * sizeof(T);
#if defined(USE_CUDA)
    GPU_CHECK(cudaMemset(output.data(), 0, bytes));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMemset(output.data(), 0, bytes));
#endif

    size_t total_elements = input.total_elements();
    size_t threads = 256;
    size_t blocks = (total_elements + threads - 1) / threads;

    TensorInfo in_info(input.shape(), input.strides());
    TensorInfo out_info(output.shape(), output.strides());

    sum_kernel<<<blocks, threads>>>(
        input.data(), output.data(),
        total_elements,
        in_info, out_info,
        axis, keepdims,
        input.offset(), output.offset()
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif

    return output;
}

template<typename T>
__global__ void conv2d_forward_kernel(
    const T* __restrict__ in, const T* __restrict__ wt, const T* __restrict__ bs,
    T* __restrict__ out,
    size_t N, size_t C_in, size_t H, size_t W,
    size_t C_out, size_t kH, size_t kW,
    size_t H_out, size_t W_out,
    size_t stride, size_t padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C_out * H_out * W_out;
    if (idx >= total_out) return;

    // Decompose linear index
    int n = idx / (C_out * H_out * W_out);
    int rem = idx % (C_out * H_out * W_out);
    int c_out = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int y = rem / W_out;
    int x = rem % W_out;

    T val = (bs != nullptr) ? bs[c_out] : (T)0.0;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int r = 0; r < kH; ++r) {
            for (int s_ = 0; s_ < kW; ++s_) {
                int in_y = static_cast<int>(y * stride + r - padding);
                int in_x = static_cast<int>(x * stride + s_ - padding);
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    size_t in_idx = (n * C_in + c_in) * (H * W) + in_y * W + in_x;
                    size_t w_idx = ((c_out * C_in + c_in) * kH + r) * kW + s_;
                    val += in[in_idx] * wt[w_idx];
                }
            }
        }
    }
    out[idx] = val;
}

template<typename T>
void conv2d_gpu(
    const Tensor<T>& input, const Tensor<T>& weight,
    const Tensor<T>& bias, Tensor<T>& output,
    size_t stride, size_t padding
) {
    size_t N = input.shape()[0], C_in = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t C_out = weight.shape()[0], kH = weight.shape()[2], kW = weight.shape()[3];
    size_t H_out = output.shape()[2], W_out = output.shape()[3];
    size_t total_out = N * C_out * H_out * W_out;

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

    conv2d_forward_kernel<<<blocks, threads>>>(
        input.data(), weight.data(), bias.empty() ? nullptr : bias.data(),
        output.data(),
        N, C_in, H, W, C_out, kH, kW, H_out, W_out, stride, padding);

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif
}

template<typename T>
__global__ void conv2d_forward_kernel_strided(
    const T* __restrict__ in, const T* __restrict__ wt, const T* __restrict__ bs,
    T* __restrict__ out,
    TensorInfo info_in, TensorInfo info_wt, TensorInfo info_out,
    size_t offset_in, size_t offsete_wt, size_t offset_out,
    size_t N, size_t C_in, size_t H, size_t W,
    size_t C_out, size_t kH, size_t kW,
    size_t H_out, size_t W_out,
    size_t stride, size_t padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C_out * H_out * W_out;
    if (idx >= total_out) return;

    // Decompose linear index
    int n = idx / (C_out * H_out * W_out);
    int rem = idx % (C_out * H_out * W_out);
    int c_out = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int y = rem / W_out;
    int x = rem % W_out;

    T val = (bs != nullptr) ? bs[c_out * info_wt.strides[0]] : (T)0.0;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int r = 0; r < kH; ++r) {
            for (int s_ = 0; s_ < kW; ++s_) {
                int in_y = static_cast<int>(y * stride + r - padding);
                int in_x = static_cast<int>(x * stride + s_ - padding);
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    size_t in_idx = n * info_in.strides[0] + c_in * info_in.strides[1] + in_y * info_in.strides[2] + in_x * info_in.strides[3];
                    size_t w_idx = c_out * info_wt.strides[0] + c_in * info_wt.strides[1] + r * info_wt.strides[2] + s_ * info_wt.strides[3];
                    val += in[in_idx + offset_in] * wt[w_idx + offsete_wt];
                }
            }
        }
    }
    out[idx + offset_out] = val;
}

template<typename T>
void conv2d_gpu_strided(
    const Tensor<T>& input, const Tensor<T>& weight,
    const Tensor<T>& bias, Tensor<T>& output,
    size_t stride, size_t padding
) {
    size_t N = input.shape()[0], C_in = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t C_out = weight.shape()[0], kH = weight.shape()[2], kW = weight.shape()[3];
    size_t H_out = output.shape()[2], W_out = output.shape()[3];
    size_t total_out = N * C_out * H_out * W_out;

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

    TensorInfo info_in(input.shape(), input.strides());
    TensorInfo info_wt(weight.shape(), weight.strides());
    TensorInfo info_out(output.shape(), output.strides());

    conv2d_forward_kernel_strided<<<blocks, threads>>>(
        input.data(), weight.data(), bias.empty() ? nullptr : bias.data(),
        output.data(),
        info_in, info_wt, info_out,
        input.offset(), weight.offset(), output.offset(),
        N, C_in, H, W, C_out, kH, kW, H_out, W_out, stride, padding);

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif
}

template<typename T>
__global__ void conv2d_backward_input_kernel(
    const T* go, const T* wt, T* gi,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int H_out, int W_out,
    int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H_in * W_in;
    if (idx >= total) return;

    // decompose idx
    int n = idx / (C_in * H_in * W_in);
    int rem = idx % (C_in * H_in * W_in);
    int c_in = rem / (H_in * W_in);
    rem %= (H_in * W_in);
    int in_y = rem / W_in;
    int in_x = rem % W_in;

    T sum = 0;
    for (int c_out = 0; c_out < C_out; ++c_out) {
        for (int r = 0; r < kH; ++r) {
            int y_out = (in_y + padding - r);  // reversed mapping
            if (y_out < 0 || y_out >= H_out || y_out % stride != 0) continue;
            y_out /= stride;
            for (int s_ = 0; s_ < kW; ++s_) {
                int x_out = (in_x + padding - s_);
                if (x_out < 0 || x_out >= W_out || x_out % stride != 0) continue;
                x_out /= stride;
                // weight index: [c_out, c_in, r, s_]
                int w_idx = ((c_out * C_in + c_in) * kH + r) * kW + s_;
                int go_idx = ((n * C_out + c_out) * H_out + y_out) * W_out + x_out;
                sum += go[go_idx] * wt[w_idx];
            }
        }
    }
    gi[idx] = sum;
}

// each thread computes one weight element; needs atomicAdd because many input/output pairs contribute
// use a separate kernel with grid-stride loops and atomicAdd
template<typename T>
__global__ void conv2d_backward_weight_kernel(
    const T* in, const T* go, T* gw,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int H_out, int W_out,
    int stride, int padding)
{
    // each thread loops over a subset of (n, y, x) and atomically adds product to appropriate weight element
    // Implemented as a grid-stride loop over all (n, y, x) output positions
    int total_out_positions = N * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_loop = gridDim.x * blockDim.x;

    for (int pos = idx; pos < total_out_positions; pos += stride_loop) {
        int n = pos / (H_out * W_out);
        int rem = pos % (H_out * W_out);
        int y = rem / W_out;
        int x = rem % W_out;

        for (int c_out = 0; c_out < C_out; ++c_out) {
            T go_val = go[(n * C_out + c_out) * H_out * W_out + y * W_out + x];
            for (int c_in = 0; c_in < C_in; ++c_in) {
                for (int r = 0; r < kH; ++r) {
                    int in_y = y * stride + r - padding;
                    if (in_y < 0 || in_y >= H) continue;
                    for (int s_ = 0; s_ < kW; ++s_) {
                        int in_x = x * stride + s_ - padding;
                        if (in_x < 0 || in_x >= W) continue;
                        T in_val = in[(n * C_in + c_in) * H * W + in_y * W + in_x];
                        int w_idx = ((c_out * C_in + c_in) * kH + r) * kW + s_;
                        atomicAdd(&gw[w_idx], in_val * go_val);
                    }
                }
            }
        }
    }
}

template<typename T>
__global__ void conv2d_backward_bias_kernel(
    const T* go, T* gb,
    int N, int C_out, int H_out, int W_out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C_out) return;

    T sum = 0;
    for (int n = 0; n < N; ++n)
        for (int y = 0; y < H_out; ++y)
            for (int x = 0; x < W_out; ++x)
                sum += go[(n * C_out + c) * H_out * W_out + y * W_out + x];
    gb[c] = sum;
}

template<typename T> 
void conv2d_backward_input_gpu(const Tensor<T>& grad_output, const Tensor<T>& weight, Tensor<T>& grad_input, size_t stride, size_t padding) {
    size_t N = grad_input.shape()[0], C_in = grad_input.shape()[1], H = grad_input.shape()[2], W = grad_input.shape()[3];
    size_t C_out = weight.shape()[0], kH = weight.shape()[2], kW = weight.shape()[3];
    size_t H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];
    size_t total_out = N * C_out * H_out * W_out;
    size_t total_in = grad_input.total_elements();

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

#if defined(USE_CUDA)
    GPU_CHECK(cudaMemset(grad_input.data(), 0, total_in * sizeof(T)));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMemset(grad_input.data(), 0, total_in * sizeof(T)));
#endif

    conv2d_backward_input_kernel<<<blocks, threads>>>(
        grad_output.data(), weight.data(), grad_input.data(), 
        N, C_in, H, W, C_out, kH, kW, H_out, W_out, stride, padding);

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif
}

template<typename T> 
void conv2d_backward_weight_gpu(const Tensor<T>& grad_output, const Tensor<T>& input, Tensor<T>& grad_weight, size_t stride, size_t padding) {
    size_t N = input.shape()[0], C_in = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t C_out = grad_weight.shape()[0], kH = grad_weight.shape()[2], kW = grad_weight.shape()[3];
    size_t H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];
    size_t total_out = N * C_out * H_out * W_out;
    size_t total_wt = grad_weight.total_elements();

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

#if defined(USE_CUDA)
    GPU_CHECK(cudaMemset(grad_weight.data(), 0, total_wt * sizeof(T)));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMemset(grad_weight.data(), 0, total_wt * sizeof(T)));
#endif

    conv2d_backward_weight_kernel<<<blocks, threads>>>(
        input.data(), grad_output.data(), grad_weight.data(),
        N, C_in, H, W, C_out, kH, kW, H_out, W_out, stride, padding);

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif
}

template<typename T> 
void conv2d_backward_bias_gpu(const Tensor<T>& grad_output, Tensor<T>& grad_bias) {
    size_t N = grad_output.shape()[0], C_out = grad_output.shape()[1], H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];
    size_t total_out = N * C_out * H_out * W_out;
    size_t total_bias = grad_bias.total_elements();

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

#if defined(USE_CUDA)
    GPU_CHECK(cudaMemset(grad_bias.data(), 0, total_bias * sizeof(T)));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMemset(grad_bias.data(), 0, total_bias * sizeof(T)));
#endif

    conv2d_backward_bias_kernel<<<blocks, threads>>>(
        grad_output.data(), grad_bias.data(),
        N, C_out, H_out, W_out
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif
}

template<typename T>
__global__ void maxpool2d_kernel(
    const T* in, T* out, size_t* indices,
    size_t N, size_t C, size_t H, size_t W,
    size_t H_out, size_t W_out,
    size_t k, size_t stride, size_t padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * H_out * W_out;
    if (idx >= total_out) return;

    int n = idx / (C * H_out * W_out);
    int rem = idx % (C * H_out * W_out);
    int c = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int y = rem / W_out;
    int x = rem % W_out;

    T max_val = -(1e30);
    size_t max_flat = 0;
    for (int r = 0; r < k; ++r) {
        int in_y = y * stride + r - padding;
        if (in_y < 0 || in_y >= H) continue;
        for (int s_ = 0; s_ < k; ++s_) {
            int in_x = x * stride + s_ - padding;
            if (in_x < 0 || in_x >= W) continue;
            size_t flat = (n * C + c) * (H * W) + in_y * W + in_x;
            T val = in[flat];
            if (val > max_val) {
                max_val = val;
                max_flat = flat;
            }
        }
    }
    out[idx] = max_val;
    indices[idx] = max_flat;
}

template<typename T>
__global__ void maxpool2d_kernel_strided(
    const T* in, T* out, size_t* indices,
    TensorInfo info_in,
    size_t offset_in, size_t offset_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t H_out, size_t W_out,
    size_t k, size_t stride, size_t padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * H_out * W_out;
    if (idx >= total_out) return;

    int n = idx / (C * H_out * W_out);
    int rem = idx % (C * H_out * W_out);
    int c = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int y = rem / W_out;
    int x = rem % W_out;

    T max_val = -(1e30);
    size_t max_logical_idx = 0; 

    for (int r = 0; r < k; ++r) {
        int in_y = y * stride + r - padding;
        if (in_y < 0 || in_y >= H) continue;
        for (int s_ = 0; s_ < k; ++s_) {
            int in_x = x * stride + s_ - padding;
            if (in_x < 0 || in_x >= W) continue;
            
            size_t phys_flat = n * info_in.strides[0] + 
                               c * info_in.strides[1] + 
                               in_y * info_in.strides[2] + 
                               in_x * info_in.strides[3];
                               
            T val = in[offset_in + phys_flat];
            
            if (val > max_val) {
                max_val = val;
                max_logical_idx = ((n * C + c) * H + in_y) * W + in_x;
            }
        }
    }
    out[offset_out + idx] = max_val;
    indices[idx] = max_logical_idx;
}

template<typename T>
__global__ void maxpool2d_backward_kernel(
    const T* grad_output, 
    T* grad_input, 
    const size_t* indices, 
    size_t total_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_out) {
        size_t max_flat_idx = indices[idx];
        atomicAdd(&grad_input[max_flat_idx], grad_output[idx]);
    }
}

template<typename T>
std::pair<Tensor<T>, std::vector<size_t>> max_pool2d_gpu(
    const Tensor<T>& input, size_t k, size_t stride, size_t padding
) {
    size_t N = input.shape()[0], C = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t H_out = (H + 2 * padding - k) / stride + 1;
    size_t W_out = (W + 2 * padding - k) / stride + 1;
    size_t total_out = N * C * H_out * W_out;

    Tensor<T> output({N, C, H_out, W_out}, input.device());

    // Allocate temporary device memory for the indices
    size_t* d_indices;
#if defined(USE_CUDA)
    GPU_CHECK(cudaMalloc(&d_indices, total_out * sizeof(size_t)));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMalloc(&d_indices, total_out * sizeof(size_t)));
#endif

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

    maxpool2d_kernel<<<blocks, threads>>>(
        input.data(), output.data(), d_indices,
        N, C, H, W, H_out, W_out, k, stride, padding
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif

    std::vector<size_t> h_indices(total_out);
#if defined(USE_CUDA)
    GPU_CHECK(cudaMemcpy(h_indices.data(), d_indices, total_out * sizeof(size_t), cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaFree(d_indices));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMemcpy(h_indices.data(), d_indices, total_out * sizeof(size_t), hipMemcpyDeviceToHost));
    GPU_CHECK(hipFree(d_indices));
#endif

    return {output, h_indices};
}

template<typename T>
std::pair<Tensor<T>, std::vector<size_t>> max_pool2d_gpu_strided(
    const Tensor<T>& input, size_t k, size_t stride, size_t padding
) {
    size_t N = input.shape()[0], C = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t H_out = (H + 2 * padding - k) / stride + 1;
    size_t W_out = (W + 2 * padding - k) / stride + 1;
    size_t total_out = N * C * H_out * W_out;

    Tensor<T> output({N, C, H_out, W_out}, input.device());

    size_t* d_indices;
#if defined(USE_CUDA)
    GPU_CHECK(cudaMalloc(&d_indices, total_out * sizeof(size_t)));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMalloc(&d_indices, total_out * sizeof(size_t)));
#endif

    TensorInfo info_in(input.shape(), input.strides());

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

    maxpool2d_kernel_strided<<<blocks, threads>>>(
        input.data(), output.data(), d_indices,
        info_in, input.offset(), output.offset(),
        N, C, H, W, H_out, W_out, k, stride, padding
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif

    std::vector<size_t> h_indices(total_out);
#if defined(USE_CUDA)
    GPU_CHECK(cudaMemcpy(h_indices.data(), d_indices, total_out * sizeof(size_t), cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaFree(d_indices));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMemcpy(h_indices.data(), d_indices, total_out * sizeof(size_t), hipMemcpyDeviceToHost));
    GPU_CHECK(hipFree(d_indices));
#endif

    return {output, h_indices};
}

template<typename T>
void max_pool2d_backward_gpu(
    const Tensor<T>& grad_output, 
    Tensor<T>& grad_input, 
    const std::vector<size_t>& h_indices
)  {
    size_t total_out = grad_output.total_elements();
    size_t total_in = grad_input.total_elements();

#if defined(USE_CUDA)
    GPU_CHECK(cudaMemset(grad_input.data(), 0, total_in * sizeof(T)));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMemset(grad_input.data(), 0, total_in * sizeof(T)));
#endif

    size_t* d_indices;
#if defined(USE_CUDA)
    GPU_CHECK(cudaMalloc(&d_indices, total_out * sizeof(size_t)));
    GPU_CHECK(cudaMemcpy(d_indices, h_indices.data(), total_out * sizeof(size_t), cudaMemcpyHostToDevice));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMalloc(&d_indices, total_out * sizeof(size_t)));
    GPU_CHECK(hipMemcpy(d_indices, h_indices.data(), total_out * sizeof(size_t), hipMemcpyHostToDevice));
#endif

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

    maxpool2d_backward_kernel<<<blocks, threads>>>(
        grad_output.data(), grad_input.data(), d_indices, total_out
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif

#if defined(USE_CUDA)
    GPU_CHECK(cudaFree(d_indices));
#elif defined(USE_ROCM)
    GPU_CHECK(hipFree(d_indices));
#endif
}

// #######################################################
// #   Explicit Instantiations
// #######################################################
template void binary_gpu<float, AddOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, AddOp<float>);
template void binary_gpu_strided<float, AddOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, AddOp<float>);
template void binary_gpu<float, SubOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, SubOp<float>);
template void binary_gpu_strided<float, SubOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, SubOp<float>);
template void binary_gpu<float, MulOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, MulOp<float>);
template void binary_gpu_strided<float, MulOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, MulOp<float>);
template void binary_gpu<float, ReLUGradOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, ReLUGradOp<float>);
template void binary_gpu_strided<float, ReLUGradOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, ReLUGradOp<float>);

template void unary_gpu<float, Pow2Op<float>>(const Tensor<float>&, Tensor<float>&, Pow2Op<float>);
template void unary_gpu_strided<float, Pow2Op<float>>(const Tensor<float>&, Tensor<float>&, Pow2Op<float>);
template void unary_gpu<float, MulScalarOp<float>>(const Tensor<float>&, Tensor<float>&, MulScalarOp<float>);
template void unary_gpu_strided<float, MulScalarOp<float>>(const Tensor<float>&, Tensor<float>&, MulScalarOp<float>);
template void unary_gpu<float, ReLUOp<float>>(Tensor<float> const&, Tensor<float>&, ReLUOp<float>);
template void unary_gpu_strided<float, ReLUOp<float>>(Tensor<float> const&, Tensor<float>&, ReLUOp<float>);
template void unary_gpu<float, SigmoidOp<float>>(Tensor<float> const&, Tensor<float>&, SigmoidOp<float>);
template void unary_gpu_strided<float, SigmoidOp<float>>(Tensor<float> const&, Tensor<float>&, SigmoidOp<float>);
template void unary_gpu<float, TanhOp<float>>(Tensor<float> const&, Tensor<float>&, TanhOp<float>);
template void unary_gpu_strided<float, TanhOp<float>>(Tensor<float> const&, Tensor<float>&, TanhOp<float>);

template void matmul_gpu<float>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&);
template void matmul_gpu_strided<float>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&);
template Tensor<float> sum_gpu<float>(const Tensor<float>&, size_t, bool);

template void conv2d_gpu<float>(Tensor<float> const&, Tensor<float> const&, Tensor<float> const&, Tensor<float>&, unsigned long, unsigned long);
template void conv2d_gpu_strided<float>(Tensor<float> const&, Tensor<float> const&, Tensor<float> const&, Tensor<float>&, unsigned long, unsigned long);
template std::pair<Tensor<float>, std::vector<unsigned long, std::allocator<unsigned long>>> max_pool2d_gpu<float>(Tensor<float> const&, unsigned long, unsigned long, unsigned long);
template std::pair<Tensor<float>, std::vector<unsigned long, std::allocator<unsigned long>>> max_pool2d_gpu_strided<float>(Tensor<float> const&, unsigned long, unsigned long, unsigned long);
template void conv2d_backward_input_gpu<float>(Tensor<float> const&, Tensor<float> const&, Tensor<float>&, unsigned long, unsigned long);
template void conv2d_backward_weight_gpu<float>(Tensor<float> const&, Tensor<float> const&, Tensor<float>&, unsigned long, unsigned long);
template void conv2d_backward_bias_gpu<float>(Tensor<float> const&, Tensor<float>&);
template void max_pool2d_backward_gpu<float>(Tensor<float> const&, Tensor<float>&, std::vector<unsigned long, std::allocator<unsigned long>> const&);