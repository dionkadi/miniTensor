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

// #######################################################
// #   Explicit Instantiations
// #######################################################
template void binary_gpu<float, AddOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, AddOp<float>);
template void binary_gpu_strided<float, AddOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, AddOp<float>);
template void binary_gpu<float, SubOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, SubOp<float>);
template void binary_gpu_strided<float, SubOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, SubOp<float>);
template void binary_gpu<float, MulOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, MulOp<float>);
template void binary_gpu_strided<float, MulOp<float>>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&, MulOp<float>);

template void unary_gpu<float, Pow2Op<float>>(const Tensor<float>&, Tensor<float>&, Pow2Op<float>);
template void unary_gpu_strided<float, Pow2Op<float>>(const Tensor<float>&, Tensor<float>&, Pow2Op<float>);
template void unary_gpu<float, MulScalarOp<float>>(const Tensor<float>&, Tensor<float>&, MulScalarOp<float>);
template void unary_gpu_strided<float, MulScalarOp<float>>(const Tensor<float>&, Tensor<float>&, MulScalarOp<float>);

template void matmul_gpu<float>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&);
template void matmul_gpu_strided<float>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&);
template Tensor<float> sum_gpu<float>(const Tensor<float>&, size_t, bool);