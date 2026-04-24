#include "gpu_ops.hpp"
#include "gpu_utils.hpp"
#include "TensorInfo.hpp"
#include <cstddef>

template<typename T>
__global__ void add_kernel(const T* a, const T* b, T* c, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__global__ void add_kernel_strided(
    const T *a, const T *b, T *c, 
    TensorInfo info_A, TensorInfo info_B, TensorInfo info_C,
    size_t size,
    size_t offset_a, size_t offset_b, size_t offset_c
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

        c[offset_c + phys_c] = a[offset_a + phys_a] + b[offset_b + phys_b];
    }
}

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
void add_gpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    size_t total_elements = A.total_elements();

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(A.data(), B.data(), C.data(), total_elements);
    
#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif
}

template<typename T>
void add_gpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    size_t total_elements = A.total_elements();

    size_t threads = 256;
    size_t blocks = (total_elements + threads - 1) / threads;

    TensorInfo info_A(A.shape(), A.strides());
    TensorInfo info_B(B.shape(), B.strides());
    TensorInfo info_C(C.shape(), C.strides());

    collapse_dims(info_A, info_B, info_C);

    add_kernel_strided<<<blocks, threads>>>(
        A.data(), B.data(), C.data(),
        info_A, info_B, info_C,
        total_elements,
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

template void add_gpu<float>(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C);
template void add_gpu_strided<float>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&);

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

template void matmul_gpu<float>(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C);
template void matmul_gpu_strided<float>(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C);