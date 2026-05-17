#include "TensorOps.hpp"
#include "GpuUtils.hpp"
#include "TensorImpl.hpp"
#include <cstddef>

constexpr int WARP_SIZE = 32;
constexpr int PAD = 1;

// #######################################################
// #   Helper Functions
// #######################################################
template<typename T, typename V>
__device__ constexpr T& cast(const V* val) {
    return reinterpret_cast<T*>(const_cast<V*>(val))[0];
}

template<size_t kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (size_t mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(~0ULL, val, mask);        
    }
    return val;
}

template<size_t kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (size_t mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(~0ULL, val, mask));
    }
    return val;
}

template<size_t BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[BLOCK_SIZE / 32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum<32>(val);

    // Write warp results to shared memory
    if (lane == 0) shared[wid] = val;

    __syncthreads();

    // Read from shared memory and do final warp reduction
    // Assuming BLOCK_SIZE <= 1024, so we have at most 32 warps
    val = (threadIdx.x < (BLOCK_SIZE / 32)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum<32>(val);
    }
    return val;
}
// #######################################################
// #   Generic GPU Kernels
// #######################################################
template<typename T, typename Op>
__global__ void elementwise_unary_kernel(
    const T *a, T *c, size_t N, Op op
) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        c[idx] = op(a[idx]);
    }
}

template<typename Op>
__global__ void elementwise_unary_vec_kernel(
    const float *a, float *c, size_t N, Op op
) {
    size_t idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if ((idx + 3) < N) {
        float4 reg_a = cast<float4>(&a[idx]);
        float4 reg_c;
        reg_c.x = op(reg_a.x);
        reg_c.y = op(reg_a.y);
        reg_c.z = op(reg_a.z);
        reg_c.w = op(reg_a.w);
        cast<float4>(&c[idx]) = reg_c;
    } else if (idx < N) {
        for (size_t i = 0; (idx + i) < N; ++i) {
            c[idx + i] = op(a[idx + i]);
        }
    }
}

template<typename T, typename Op>
__global__ void elementwise_binary_kernel(
    const T *a, const T *b, T *c, size_t N, Op op
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = op(a[idx], b[idx]);
    }
}

template<typename Op>
__global__ void elementwise_binary_vec_kernel(
    const float *a, const float *b, float *c, size_t N, Op op
) {
    size_t idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if ((idx + 3) < N) {
        float4 reg_a = cast<float4>(&a[idx]);
        float4 reg_b = cast<float4>(&b[idx]);
        float4 reg_c;
        reg_c.x = op(reg_a.x, reg_b.x);
        reg_c.y = op(reg_a.y, reg_b.y);
        reg_c.z = op(reg_a.z, reg_b.z);
        reg_c.w = op(reg_a.w, reg_b.w);
        cast<float4>(&c[idx]) = reg_c;
    } else if (idx < N) {
        for (size_t i = 0; (idx + i) < N; ++i) {
            c[idx + i] = op(a[idx + i], b[idx + i]);
        }
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
    int threads = 256;
    size_t total_elements = A.total_elements();
    
    if constexpr (std::is_same_v<T, float>) {
        int blocks = (total_elements + threads * 4 - 1) / (threads * 4);
        elementwise_binary_vec_kernel<<<blocks, threads>>>(
            A.data(), B.data(), C.data(), total_elements, op
        );
    } else {
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        elementwise_binary_kernel<<<blocks, threads>>>(A.data(), B.data(), C.data(), total_elements, op);
    }
    
#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
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
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif
}

template<typename T, typename Op>
void unary_gpu(const Tensor<T>& A, Tensor<T>& C, Op op) {
    size_t total_elements = A.total_elements();
    int threads = 256;
    
    if constexpr (std::is_same_v<T, float>) {
        int blocks = (total_elements + threads * 4 - 1) / (threads * 4);
        elementwise_unary_vec_kernel<<<blocks, threads>>>(A.data(), C.data(), total_elements, op);
    } else {
        int blocks = (total_elements + threads - 1) / threads;
        elementwise_unary_kernel<<<blocks, threads>>>(A.data(), C.data(), total_elements, op);
    }
    
#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
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
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif
}

// #######################################################
// #   Non-Generic Ops
// #######################################################
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_kernel(
    const float *a, const float *b, float *c, const int M, const int N, const int K
) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    float r_load_a[TM / 2];
    float r_load_b[TN / 2];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0f};

    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid / 32;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        if (load_a_gmem_m < M && load_a_gmem_k + 3 < K) {
            cast<float4>(&r_load_a[0]) = cast<float4>(&a[load_a_gmem_addr]);
        } else if (load_a_gmem_m < M && load_a_gmem_k < K) {
            for (int j = 0; j < 4 && load_a_gmem_k + j < K; ++j)
                r_load_a[j] = a[load_a_gmem_addr + j];
            for (int j = K - load_a_gmem_k; j < 4; ++j)
                r_load_a[j] = 0.0f;
        } else {
            r_load_a[0] = 0.0f; r_load_a[1] = 0.0f; r_load_a[2] = 0.0f; r_load_a[3] = 0.0f;
        }
        if (load_b_gmem_k < K && load_b_gmem_n + 3 < N) {
            cast<float4>(&r_load_b[0]) = cast<float4>(&b[load_b_gmem_addr]);
        } else if (load_b_gmem_k < K && load_b_gmem_n < N) {
            for (int j = 0; j < 4 && load_b_gmem_n + j < N; ++j)
                r_load_b[j] = b[load_b_gmem_addr + j];
            for (int j = N - load_b_gmem_n; j < 4; ++j)
                r_load_b[j] = 0.0f;
        } else {
            r_load_b[0] = 0.0f; r_load_b[1] = 0.0f; r_load_b[2] = 0.0f; r_load_b[3] = 0.0f;
        }

        s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        cast<float4>(&s_b[0][load_b_smem_k][load_b_smem_n]) = cast<float4>(&r_load_b[0]);
    }

    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        if (load_a_gmem_m < M && load_a_gmem_k + 3 < K) {
            cast<float4>(&r_load_a[0]) = cast<float4>(&a[load_a_gmem_addr]);
        } else if (load_a_gmem_m < M && load_a_gmem_k < K) {
            for (int j = 0; j < 4 && load_a_gmem_k + j < K; ++j)
                r_load_a[j] = a[load_a_gmem_addr + j];
            for (int j = K - load_a_gmem_k; j < 4; ++j)
                r_load_a[j] = 0.0f;
        } else {
            r_load_a[0] = 0.0f; r_load_a[1] = 0.0f; r_load_a[2] = 0.0f; r_load_a[3] = 0.0f;
        }
        if (load_b_gmem_k < K && load_b_gmem_n + 3 < N) {
            cast<float4>(&r_load_b[0]) = cast<float4>(&b[load_b_gmem_addr]);
        } else if (load_b_gmem_k < K && load_b_gmem_n < N) {
            for (int j = 0; j < 4 && load_b_gmem_n + j < N; ++j)
                r_load_b[j] = b[load_b_gmem_addr + j];
            for (int j = N - load_b_gmem_n; j < 4; ++j)
                r_load_b[j] = 0.0f;
        } else {
            r_load_b[0] = 0.0f; r_load_b[1] = 0.0f; r_load_b[2] = 0.0f; r_load_b[3] = 0.0f;
        }

        #pragma unroll
        for (int tk = 0; tk < BK; ++tk) {
            cast<float4>(&r_comp_a[0]) = cast<float4>(&s_a[smem_sel][tk][ty * TM / 2]);
            cast<float4>(&r_comp_a[4]) = cast<float4>(&s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            cast<float4>(&r_comp_b[0]) = cast<float4>(&s_b[smem_sel][tk][tx * TN / 2]);
            cast<float4>(&r_comp_b[4]) = cast<float4>(&s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        cast<float4>(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = cast<float4>(&r_load_b[0]);

        __syncthreads();
    }

    int smem_sel_last = ((K + BK - 1) / BK - 1) & 1;

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        cast<float4>(&r_comp_a[0]) = cast<float4>(&s_a[smem_sel_last][tk][ty * TM / 2]);
        cast<float4>(&r_comp_a[4]) = cast<float4>(&s_a[smem_sel_last][tk][ty * TM / 2 + BM / 2]);
        cast<float4>(&r_comp_b[0]) = cast<float4>(&s_b[smem_sel_last][tk][tx * TN / 2]);
        cast<float4>(&r_comp_b[4]) = cast<float4>(&s_b[smem_sel_last][tk][tx * TN / 2 + BN / 2]);

        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        if (store_c_gmem_m < M && store_c_gmem_n < N) {
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            if (store_c_gmem_n + 3 < N) {
                cast<float4>(&c[store_c_gmem_addr]) = cast<float4>(&r_c[i][0]);
            } else {
                for (int j = 0; j < 4 && store_c_gmem_n + j < N; ++j)
                    c[store_c_gmem_addr + j] = r_c[i][j];
            }
        }
        if (store_c_gmem_m < M && store_c_gmem_n + BN / 2 < N) {
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            if (store_c_gmem_n + BN / 2 + 3 < N) {
                cast<float4>(&c[store_c_gmem_addr + BN / 2]) = cast<float4>(&r_c[i][4]);
            } else {
                for (int j = 0; j < 4 && store_c_gmem_n + BN / 2 + j < N; ++j)
                    c[store_c_gmem_addr + BN / 2 + j] = r_c[i][4 + j];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        if (store_c_gmem_m < M && store_c_gmem_n < N) {
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            if (store_c_gmem_n + 3 < N) {
                cast<float4>(&c[store_c_gmem_addr]) = cast<float4>(&r_c[i + TM / 2][0]);
            } else {
                for (int j = 0; j < 4 && store_c_gmem_n + j < N; ++j)
                    c[store_c_gmem_addr + j] = r_c[i + TM / 2][j];
            }
        }
        if (store_c_gmem_m < M && store_c_gmem_n + BN / 2 < N) {
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            if (store_c_gmem_n + BN / 2 + 3 < N) {
                cast<float4>(&c[store_c_gmem_addr + BN / 2]) = cast<float4>(&r_c[i + TM / 2][4]);
            } else {
                for (int j = 0; j < 4 && store_c_gmem_n + BN / 2 + j < N; ++j)
                    c[store_c_gmem_addr + BN / 2 + j] = r_c[i + TM / 2][4 + j];
            }
        }
    }
}

// Batch GEMM: same tiled sgemm but across N batch items in one 3D launch
// 3D grid: (N/BN, M/BM, B) where B=batch count, BM/BN=128, BK=8, TM/TN=8
// a (weight, shared): [M, K], b (col, per-batch): [B, K, N], c (out, per-batch): [B, M, N]
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 1>
__global__ void batched_sgemm_kernel(
    const float *a, const float *b, float *c,
    const int M, const int N, const int K, const int batch_stride_b, const int batch_stride_c,
    const float* bias, const int bias_N
) {
    int bx = blockIdx.x, by = blockIdx.y, batch = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0f};

    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid / (BN / 4);
    int load_b_smem_n = (tid & ((BN / 4) - 1)) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    const float* b_batch = b + batch * batch_stride_b;
    float* c_batch = c + batch * batch_stride_c;

    // Preload first tile
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        if (load_a_gmem_m < M && load_a_gmem_k + 3 < K) {
            cast<float4>(&r_load_a[0]) = cast<float4>(&a[load_a_gmem_addr]);
        } else if (load_a_gmem_m < M && load_a_gmem_k < K) {
            for (int j = 0; j < 4 && load_a_gmem_k + j < K; ++j)
                r_load_a[j] = a[load_a_gmem_addr + j];
            for (int j = K - load_a_gmem_k; j < 4; ++j) r_load_a[j] = 0.0f;
        } else {
            r_load_a[0]=0;r_load_a[1]=0;r_load_a[2]=0;r_load_a[3]=0;
        }
        if (load_b_gmem_k < K && load_b_gmem_n + 3 < N) {
            cast<float4>(&r_load_b[0]) = cast<float4>(&b_batch[load_b_gmem_addr]);
        } else if (load_b_gmem_k < K && load_b_gmem_n < N) {
            for (int j = 0; j < 4 && load_b_gmem_n + j < N; ++j)
                r_load_b[j] = b_batch[load_b_gmem_addr + j];
            for (int j = N - load_b_gmem_n; j < 4; ++j) r_load_b[j] = 0.0f;
        } else {
            r_load_b[0]=0;r_load_b[1]=0;r_load_b[2]=0;r_load_b[3]=0;
        }
        s_a[0][load_a_smem_k+0][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k+1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k+2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k+3][load_a_smem_m] = r_load_a[3];
        cast<float4>(&s_b[0][load_b_smem_k][load_b_smem_n]) = cast<float4>(&r_load_b[0]);
    }
    __syncthreads();

    // Main loop over K tiles
    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        if (load_a_gmem_m < M && load_a_gmem_k + 3 < K) {
            cast<float4>(&r_load_a[0]) = cast<float4>(&a[load_a_gmem_addr]);
        } else if (load_a_gmem_m < M && load_a_gmem_k < K) {
            for (int j = 0; j < 4 && load_a_gmem_k + j < K; ++j)
                r_load_a[j] = a[load_a_gmem_addr + j];
            for (int j = K - load_a_gmem_k; j < 4; ++j) r_load_a[j] = 0.0f;
        } else { r_load_a[0]=0;r_load_a[1]=0;r_load_a[2]=0;r_load_a[3]=0; }
        if (load_b_gmem_k < K && load_b_gmem_n + 3 < N) {
            cast<float4>(&r_load_b[0]) = cast<float4>(&b_batch[load_b_gmem_addr]);
        } else if (load_b_gmem_k < K && load_b_gmem_n < N) {
            for (int j = 0; j < 4 && load_b_gmem_n + j < N; ++j)
                r_load_b[j] = b_batch[load_b_gmem_addr + j];
            for (int j = N - load_b_gmem_n; j < 4; ++j) r_load_b[j] = 0.0f;
        } else { r_load_b[0]=0;r_load_b[1]=0;r_load_b[2]=0;r_load_b[3]=0; }

        #pragma unroll
        for (int tk = 0; tk < BK; ++tk) {
            cast<float4>(&r_comp_a[0]) = cast<float4>(&s_a[smem_sel][tk][ty * TM / 2]);
            cast<float4>(&r_comp_a[4]) = cast<float4>(&s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            cast<float4>(&r_comp_b[0]) = cast<float4>(&s_b[smem_sel][tk][tx * TN / 2]);
            cast<float4>(&r_comp_b[4]) = cast<float4>(&s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);
            for (int tm = 0; tm < TM; ++tm)
                for (int tn = 0; tn < TN; ++tn)
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }

        s_a[smem_sel_next][load_a_smem_k+0][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k+1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k+2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k+3][load_a_smem_m] = r_load_a[3];
        cast<float4>(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = cast<float4>(&r_load_b[0]);
        __syncthreads();
    }

    // Final K tile computation
    int smem_sel_last = ((K + BK - 1) / BK - 1) & 1;
    for (int tk = 0; tk < BK; tk++) {
        cast<float4>(&r_comp_a[0]) = cast<float4>(&s_a[smem_sel_last][tk][ty * TM / 2]);
        cast<float4>(&r_comp_a[4]) = cast<float4>(&s_a[smem_sel_last][tk][ty * TM / 2 + BM / 2]);
        cast<float4>(&r_comp_b[0]) = cast<float4>(&s_b[smem_sel_last][tk][tx * TN / 2]);
        cast<float4>(&r_comp_b[4]) = cast<float4>(&s_b[smem_sel_last][tk][tx * TN / 2 + BN / 2]);
        for (int tm = 0; tm < TM; tm++)
            for (int tn = 0; tn < TN; tn++)
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        if (store_c_gmem_m < M && store_c_gmem_n < N) {
            int addr = store_c_gmem_m * N + store_c_gmem_n;
            if (bias) {
                float b_val = bias[store_c_gmem_m];
                if (store_c_gmem_n + 3 < N) {
                    float4 tmp = cast<float4>(&r_c[i][0]);
                    tmp.x += b_val; tmp.y += b_val; tmp.z += b_val; tmp.w += b_val;
                    cast<float4>(&c_batch[addr]) = tmp;
                } else {
                    for (int j = 0; j < 4 && store_c_gmem_n + j < N; ++j)
                        c_batch[addr + j] = r_c[i][j] + b_val;
                }
            } else {
                if (store_c_gmem_n + 3 < N)
                    cast<float4>(&c_batch[addr]) = cast<float4>(&r_c[i][0]);
                else
                    for (int j = 0; j < 4 && store_c_gmem_n + j < N; ++j)
                        c_batch[addr + j] = r_c[i][j];
            }
        }
        if (store_c_gmem_m < M && store_c_gmem_n + BN / 2 < N) {
            int addr = store_c_gmem_m * N + store_c_gmem_n;
            if (bias) {
                float b_val = bias[store_c_gmem_m];
                if (store_c_gmem_n + BN / 2 + 3 < N) {
                    float4 tmp = cast<float4>(&r_c[i][4]);
                    tmp.x += b_val; tmp.y += b_val; tmp.z += b_val; tmp.w += b_val;
                    cast<float4>(&c_batch[addr + BN / 2]) = tmp;
                } else {
                    for (int j = 0; j < 4 && store_c_gmem_n + BN / 2 + j < N; ++j)
                        c_batch[addr + BN / 2 + j] = r_c[i][4 + j] + b_val;
                }
            } else {
                if (store_c_gmem_n + BN / 2 + 3 < N)
                    cast<float4>(&c_batch[addr + BN / 2]) = cast<float4>(&r_c[i][4]);
                else
                    for (int j = 0; j < 4 && store_c_gmem_n + BN / 2 + j < N; ++j)
                        c_batch[addr + BN / 2 + j] = r_c[i][4 + j];
            }
        }
    }
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        if (store_c_gmem_m < M && store_c_gmem_n < N) {
            int addr = store_c_gmem_m * N + store_c_gmem_n;
            if (bias) {
                float b_val = bias[store_c_gmem_m];
                if (store_c_gmem_n + 3 < N) {
                    float4 tmp = cast<float4>(&r_c[i + TM / 2][0]);
                    tmp.x += b_val; tmp.y += b_val; tmp.z += b_val; tmp.w += b_val;
                    cast<float4>(&c_batch[addr]) = tmp;
                } else {
                    for (int j = 0; j < 4 && store_c_gmem_n + j < N; ++j)
                        c_batch[addr + j] = r_c[i + TM / 2][j] + b_val;
                }
            } else {
                if (store_c_gmem_n + 3 < N)
                    cast<float4>(&c_batch[addr]) = cast<float4>(&r_c[i + TM / 2][0]);
                else
                    for (int j = 0; j < 4 && store_c_gmem_n + j < N; ++j)
                        c_batch[addr + j] = r_c[i + TM / 2][j];
            }
        }
        if (store_c_gmem_m < M && store_c_gmem_n + BN / 2 < N) {
            int addr = store_c_gmem_m * N + store_c_gmem_n;
            if (bias) {
                float b_val = bias[store_c_gmem_m];
                if (store_c_gmem_n + BN / 2 + 3 < N) {
                    float4 tmp = cast<float4>(&r_c[i + TM / 2][4]);
                    tmp.x += b_val; tmp.y += b_val; tmp.z += b_val; tmp.w += b_val;
                    cast<float4>(&c_batch[addr + BN / 2]) = tmp;
                } else {
                    for (int j = 0; j < 4 && store_c_gmem_n + BN / 2 + j < N; ++j)
                        c_batch[addr + BN / 2 + j] = r_c[i + TM / 2][4 + j] + b_val;
                }
            } else {
                if (store_c_gmem_n + BN / 2 + 3 < N)
                    cast<float4>(&c_batch[addr + BN / 2]) = cast<float4>(&r_c[i + TM / 2][4]);
                else
                    for (int j = 0; j < 4 && store_c_gmem_n + BN / 2 + j < N; ++j)
                        c_batch[addr + BN / 2 + j] = r_c[i + TM / 2][4 + j];
            }
        }
    }
}

// Fused backward: batched_sgemm (W^T x dL/dY) -> col2im in one pass
// Eliminates the d_grad_col intermediate write+read traffic.
// Each thread handles one (n, c_in, h, w) pixel in grad_input.
template<typename T>
__global__ void fused_backward_input_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ weight_T,
    T* __restrict__ grad_input,
    int N, int C_in, int C_out, int H, int W,
    int kH, int kW, int pad, int stride,
    int H_out, int W_out
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % C_in;
    int n = blockIdx.z / C_in;

    if (h >= H || w >= W) return;

    T val = (T)0;

    int h_out_start = (h + pad < kH) ? 0 : (h + pad - kH) / stride + 1;
    int h_out_end   = min((h + pad) / stride + 1, H_out);
    int w_out_start = (w + pad < kW) ? 0 : (w + pad - kW) / stride + 1;
    int w_out_end   = min((w + pad) / stride + 1, W_out);

    int batch_offset = n * C_out * H_out * W_out;
    int wt_offset = c * kH * kW;

    for (int h_out = h_out_start; h_out < h_out_end; ++h_out) {
        int h_k = h + pad - h_out * stride;
        for (int w_out = w_out_start; w_out < w_out_end; ++w_out) {
            int w_k = w + pad - w_out * stride;

            int channel_idx = wt_offset + h_k * kW + w_k;

            for (int co = 0; co < C_out; ++co) {
                val += weight_T[channel_idx * C_out + co]
                     * grad_output[batch_offset + co * H_out * W_out + h_out * W_out + w_out];
            }
        }
    }

    grad_input[((n * C_in + c) * H + h) * W + w] = val;
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

    constexpr int BM = 128, BN = 128, BK = 8;
    constexpr int TM = 8, TN = 8;
    dim3 threads(16, 16);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_kernel<BM, BN, BK, TM, TN, 1><<<blocks, threads>>>(
        A.data(), B.data(), C.data(), M, N, K
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
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
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif
}

// #######################################################
// #   Batched GEMM for Conv
// #######################################################
// Computes: out[b, m, s] = sum_k a[m, k] * b[b, k, s] for all batch items.
// a = [M, K] (shared across batch), b = [B, K, N_mat], c = [B, M, N_mat].
// The output layout [B, M, N_mat] is compatible with conv output tensors.
template<typename T>
__global__ void batched_gemm_kernel(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
    int B, int M, int K, int N_mat
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * M * N_mat;
    if (idx >= total) return;

    int n = idx / (M * N_mat);
    int r = (idx % (M * N_mat)) / N_mat;
    int s = idx % N_mat;

    T sum = 0;
    const T* b_row = b + n * K * N_mat + s;
    for (int k = 0; k < K; ++k) {
        sum += a[r * K + k] * b_row[k * N_mat];
    }
    c[idx] = sum;
}

// #######################################################
// #   Non-Generic Ops
// #######################################################
__global__ void reduce_axis_kernel(
    const float* __restrict__ a, 
    float* __restrict__ y,
    const int outer_size, 
    const int reduce_size, 
    const int inner_size  // > 1
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = outer_size * inner_size;

    if (tid < N) {
        int outer_idx = tid / inner_size;
        int inner_idx = tid % inner_size;

        float sum = 0.0f;

        for (int i = 0; i < reduce_size; ++i) {
            int input_dix = (outer_idx * reduce_size + i) * inner_size + inner_idx;
            sum += a[input_dix];
        }

        y[tid] = sum;
    }
}

template<size_t BLOCK_SIZE>
__global__ void reduce_axis_inner_kernel(
    const float* __restrict__ a, 
    float* __restrict__ y,
    const int outer_size, 
    const int reduce_size
    // const int inner_size == 1
) {
    int row = blockIdx.x;
    if (row >= outer_size) return;

    int tid = threadIdx.x;
    float sum = 0.0f;

    int row_offset = row * reduce_size;
    for (int i = tid; i < reduce_size; i += BLOCK_SIZE) {
        sum += a[row_offset + i];
    }

    sum = block_reduce_sum<BLOCK_SIZE>(sum);

    if (tid == 0) {
        y[row] = sum;
    }
}

template<typename T>
Tensor<T> sum_gpu(const Tensor<T>& input, size_t axis, bool keepdims) {
    Tensor<T> in = input.contiguous();
    const auto& in_shape = in.shape();
    size_t outer = 1, reduce = in_shape[axis], inner = 1;
    for (size_t i = 0; i < axis; ++i) outer *= in_shape[i];
    for (size_t i = axis+1; i < in_shape.size(); ++i) inner *= in_shape[i];

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

    if (inner > 1) {
        int threads = 256;
        int blocks = (outer * inner + threads - 1) / threads;
        reduce_axis_kernel<<<blocks, threads>>>(in.data(), output.data(),
                                                outer, reduce, inner);
    } else {
        int threads = 256;
        int blocks = outer;   // one block per row
        reduce_axis_inner_kernel<256><<<blocks, threads>>>(in.data(), output.data(),
                                                           outer, reduce);
    }

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif

    return output;
}

template<typename T>
__global__ void mat_transpose_kernel(
    const T *x, T *y, int row, int col
) {
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    const int STRIDE = WARP_SIZE;

    __shared__ T tile[WARP_SIZE][WARP_SIZE + PAD];

    // All threads participate in loading — fixes __syncthreads divergence bug
    if (global_y < row && global_x < col) {
        tile[local_y][local_x] = x[global_y * col + global_x];
    } else {
        tile[local_y][local_x] = T(0);
    }

    __syncthreads();

    // Diagonal-style coalesced write
    if (global_y < row && global_x < col) {
        float smem_val = tile[local_y % STRIDE][local_x + local_y / STRIDE];
        int bid_y = blockIdx.y;
        int out_y = global_x + local_y / STRIDE;
        int out_x = (local_y % STRIDE) + bid_y * STRIDE;
        if (out_y < col && out_x < row) {
            y[out_y * row + out_x] = smem_val;
        }
    }
}

__global__ void mat_transpose_vec_kernel(
    float *x, float *y, int row, int col
) {
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    const int STRIDE = WARP_SIZE / 4;
    __shared__ float tile[WARP_SIZE][WARP_SIZE * 4 + PAD];

    // All threads participate in loading
    if (global_y * 4 + 3 < row && global_x < col) {
        float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
        tile[local_y][local_x * 4] = x_val.x;
        tile[local_y][local_x * 4 + 1] = x_val.y;
        tile[local_y][local_x * 4 + 2] = x_val.z;
        tile[local_y][local_x * 4 + 3] = x_val.w;
    } else {
        tile[local_y][local_x * 4] = 0.0f;
        tile[local_y][local_x * 4 + 1] = 0.0f;
        tile[local_y][local_x * 4 + 2] = 0.0f;
        tile[local_y][local_x * 4 + 3] = 0.0f;
    }

    __syncthreads();

    // Diagonal-style coalesced write
    if (global_y * 4 + 3 < row && global_x < col) {
        float4 smem_val;
        smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
        smem_val.y = tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
        smem_val.z = tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
        smem_val.w = tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];

        int bid_y = blockIdx.y;
        int out_y = global_x * 4 + local_y / STRIDE;
        int out_x = (local_y % STRIDE) * 4 + bid_y * WARP_SIZE;
        if (out_y < col && out_x < row) {
            reinterpret_cast<float4 *>(y)[out_y * row + out_x] = cast<float4>(&smem_val);
        }
    }
}

template<typename T>
__global__ void add_bias_kernel(T* out, const T* bias, int N, int C_out, int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C_out * H_out * W_out) {
        int c = (idx / (H_out * W_out)) % C_out;
        out[idx] += bias[c];
    }
}

// Grid: dim3(ceil(W_out/TILE_W), ceil(H_out/TILE_H), N * C)
// Block: dim3(TILE_W, TILE_H, 1)
// flatten to [N, C * kH * kW, H_out * W_out]
template<typename T>
__global__ void im2col_kernel(
    const T* __restrict__ data_im, 
    T* __restrict__ data_col,
    int N, int C, int H, int W,
    int kH, int kW, int pad, int stride,
    int H_out, int W_out
) {
    int w_out_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;

    int c_in = blockIdx.z % C;
    int n    = blockIdx.z / C;

    int in_tile_w = (blockDim.x * 4 - 1) * stride + kW;
    int in_tile_h = (blockDim.y - 1) * stride + kH;

    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T* img_smem = reinterpret_cast<T*>(smem);

    // top-left
    int block_start_w_in = blockIdx.x * (blockDim.x * 4) * stride - pad;
    int block_start_h_in = blockIdx.y * blockDim.y * stride - pad;

    int flat_dix = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    int total_elements = in_tile_h * in_tile_w;
    for (int i = flat_dix; i < total_elements; i += num_threads) {
        int smem_w = i % in_tile_w;
        int smem_h = i / in_tile_w;

        int im_w = block_start_w_in + smem_w;
        int im_h = block_start_h_in + smem_h;

        if (im_h >= 0 && im_h < H && im_w >= 0 && im_w < W) {
            img_smem[i] = data_im[((n * C + c_in) * H + im_h) * W + im_w];
        } else {
            img_smem[i] = 0.0f;
        }
    }

    __syncthreads();

    if (h_out < H_out) {
        for (int h_k = 0; h_k < kH; ++h_k) {
            for (int w_k = 0; w_k < kW; ++w_k) {
                int smem_h = threadIdx.y * stride + h_k;

                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    int w_out = w_out_base + v;
                    
                    if (w_out < W_out) {
                        int smem_w = (threadIdx.x * 4 + v) * stride + w_k;
                        float val = img_smem[smem_h * in_tile_w + smem_w];

                        int channel_kernel_idx = c_in * (kH * kW) + h_k * kW + w_k;
                        int spatial_out_idx = h_out * W_out + w_out;
                        int spatial_out_area = H_out * W_out;

                        int col_idx = n * (C * kH * kW * spatial_out_area) + 
                                    channel_kernel_idx * spatial_out_area + 
                                    spatial_out_idx;

                        data_col[col_idx] = val;
                    }
                }
            }
        }
    }
}

// Grid: dim3(ceil(W / TILE_W), ceil(H / TILE_H), N * C)
// Block: dim3(TILE_W, TILE_H, 1)
// Output: reconstructs data_im of shape [N, C, H, W]
template<typename T>
__global__ void col2im_kernel(
    const T* __restrict__ data_col, 
    T* __restrict__ data_im,
    int N, int C, int H, int W,
    int kH, int kW, int pad, int stride,
    int H_out, int W_out
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % C;
    int n = blockIdx.z / C;

    if (h >= H || w >= W) return;

    T val = (T)0.0f;

    int h_out_start = (h + pad < kH) ? 0 : (h + pad - kH) / stride + 1;
    int h_out_end   = min((h + pad) / stride + 1, H_out);
    int w_out_start = (w + pad < kW) ? 0 : (w + pad - kW) / stride + 1;
    int w_out_end   = min((w + pad) / stride + 1, W_out);

    int spatial_area = H_out * W_out;
    int batch_channel_offset = n * (C * kH * kW * spatial_area);

    for (int h_out = h_out_start; h_out < h_out_end; ++h_out) {
        int h_k = (h + pad) - h_out * stride;
        for (int w_out = w_out_start; w_out < w_out_end; ++w_out) {
            int w_k = (w + pad) - w_out * stride;
            
            int channel_idx = c * (kW * kH) + h_k * kW + w_k;
            int spatial_out_idx = h_out * W_out + w_out;

            int col_idx = batch_channel_offset + channel_idx * spatial_area + spatial_out_idx;

            val += data_col[col_idx];
        }
    }

    data_im[((n * C + c) * H + h) * W + w] = val;
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
    
    size_t M = C_out;
    size_t K = C_in * kH * kW;
    size_t N_mat = H_out * W_out;

    T* d_data_col;
    size_t col_size = N * K * N_mat * sizeof(T);
    d_data_col = static_cast<T*>(MemoryPool::get().allocate(col_size, Device{DeviceType::CUDA}));

    dim3 threads(16, 16);
    dim3 blocks((W_out + threads.x * 4 - 1) / (threads.x * 4), 
                (H_out + threads.y - 1) / threads.y, 
                N * C_in);
    size_t smem_size = ((threads.y - 1) * stride + kH) * ((threads.x * 4 - 1) * stride + kW) * sizeof(T);
    
    im2col_kernel<T><<<blocks, threads, smem_size>>>(
        input.data(), d_data_col, N, C_in, H, W, kH, kW, padding, stride, H_out, W_out
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif

    // Tiled batched GEMM: all N batch items in one 3D launch
    // weight [M, K] (shared), col [N, K, N_mat], output [N, M, N_mat]
    if constexpr (std::is_same_v<T, float>) {
        dim3 batched_threads(16, 16);
        dim3 batched_blocks((N_mat + 127) / 128, (M + 127) / 128, N);
        batched_sgemm_kernel<128, 128, 8, 8, 8><<<batched_blocks, batched_threads>>>(
            reinterpret_cast<const float*>(weight.data()),
            reinterpret_cast<const float*>(d_data_col),
            reinterpret_cast<float*>(output.data()),
            M, N_mat, K, K * N_mat, M * N_mat,
            bias.empty() ? nullptr : reinterpret_cast<const float*>(bias.data()),
            C_out
        );
    } else {
        for (size_t n = 0; n < N; ++n) {
            T* col_ptr = d_data_col + n * (K * N_mat);
            T* out_ptr = output.data() + n * (M * N_mat);
            dim3 mm_threads(16, 16);
            dim3 strided_blocks((N_mat + 15) / 16, (M + 15) / 16);
            matmul_kernel_strided<T><<<strided_blocks, mm_threads>>>(
                weight.data(), col_ptr, out_ptr,
                M, K, N_mat, K, 1, N_mat, 1, N_mat, 1, 0, 0, 0
            );
        }
    }

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif

    MemoryPool::get().free(d_data_col, col_size, Device{DeviceType::CUDA});
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

    T val = (bs != nullptr) ? bs[c_out] : (T)0.0;

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
    if (blocks == 0) return;

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
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
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
    
    size_t K = C_out;
    size_t M = C_in * kH * kW;

    T* d_weight_T;
    size_t weight_T_size = K * M * sizeof(T);
    d_weight_T = static_cast<T*>(MemoryPool::get().allocate(weight_T_size, Device{DeviceType::CUDA}));
#if defined(USE_CUDA)
    GPU_CHECK(cudaMemset(grad_input.data(), 0, grad_input.total_elements() * sizeof(T)));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMemset(grad_input.data(), 0, grad_input.total_elements() * sizeof(T)));
#endif

    // Transpose the weights (W^T)
    dim3 tp_threads(WARP_SIZE, WARP_SIZE);
    dim3 tp_blocks((M + WARP_SIZE - 1) / WARP_SIZE, (K + WARP_SIZE - 1) / WARP_SIZE);
    mat_transpose_kernel<T><<<tp_blocks, tp_threads>>>(weight.data(), d_weight_T, K, M);

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif

    // Fused backward: single kernel does W^T x dY GEMM + col2im in one pass
    {
        dim3 fwd_threads(16, 16);
        dim3 fwd_blocks((W + 15) / 16, (H + 15) / 16, N * C_in);
        fused_backward_input_kernel<T><<<fwd_blocks, fwd_threads>>>(
            grad_output.data(), d_weight_T, grad_input.data(),
            N, C_in, C_out, H, W, kH, kW, (int)padding, (int)stride, H_out, W_out
        );
    }

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif
    MemoryPool::get().free(d_weight_T, weight_T_size, Device{DeviceType::CUDA});
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
    if (blocks == 0) return;

    conv2d_backward_weight_kernel<<<blocks, threads>>>(
        input.data(), grad_output.data(), grad_weight.data(),
        N, C_in, H, W, C_out, kH, kW, H_out, W_out, stride, padding);

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
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
    if (blocks == 0) return;

    conv2d_backward_bias_kernel<<<blocks, threads>>>(
        grad_output.data(), grad_bias.data(),
        N, C_out, H_out, W_out
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
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
    size_t H_out = safe_out_size(H, padding, k, stride);
    size_t W_out = safe_out_size(W, padding, k, stride);
    size_t total_out = N * C * H_out * W_out;

    Tensor<T> output({N, C, H_out, W_out}, input.device());

    // Allocate temporary device memory for the indices
    size_t* d_indices;
    size_t idx_size = total_out * sizeof(size_t);
    if (total_out == 0) d_indices = nullptr;
    else d_indices = static_cast<size_t*>(MemoryPool::get().allocate(idx_size, Device{DeviceType::CUDA}));

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks == 0) {
        return {std::move(output), std::vector<size_t>()};
    }

    maxpool2d_kernel<<<blocks, threads>>>(
        input.data(), output.data(), d_indices,
        N, C, H, W, H_out, W_out, k, stride, padding
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif

    std::vector<size_t> h_indices(total_out);
#if defined(USE_CUDA)
    GPU_CHECK(cudaMemcpy(h_indices.data(), d_indices, idx_size, cudaMemcpyDeviceToHost));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMemcpy(h_indices.data(), d_indices, idx_size, hipMemcpyDeviceToHost));
#endif
    MemoryPool::get().free(d_indices, idx_size, Device{DeviceType::CUDA});

    return {output, h_indices};
}

template<typename T>
std::pair<Tensor<T>, std::vector<size_t>> max_pool2d_gpu_strided(
    const Tensor<T>& input, size_t k, size_t stride, size_t padding
) {
    size_t N = input.shape()[0], C = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t H_out = safe_out_size(H, padding, k, stride);
    size_t W_out = safe_out_size(W, padding, k, stride);
    size_t total_out = N * C * H_out * W_out;

    Tensor<T> output({N, C, H_out, W_out}, input.device());

    size_t* d_indices;
    size_t idx_size = total_out * sizeof(size_t);
    if (total_out == 0) d_indices = nullptr;
    else d_indices = static_cast<size_t*>(MemoryPool::get().allocate(idx_size, Device{DeviceType::CUDA}));

    TensorInfo info_in(input.shape(), input.strides());

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks == 0) {
        return {std::move(output), std::vector<size_t>()};
    }

    maxpool2d_kernel_strided<<<blocks, threads>>>(
        input.data(), output.data(), d_indices,
        info_in, input.offset(), output.offset(),
        N, C, H, W, H_out, W_out, k, stride, padding
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif

    std::vector<size_t> h_indices(total_out);
#if defined(USE_CUDA)
    GPU_CHECK(cudaMemcpy(h_indices.data(), d_indices, idx_size, cudaMemcpyDeviceToHost));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMemcpy(h_indices.data(), d_indices, idx_size, hipMemcpyDeviceToHost));
#endif
    MemoryPool::get().free(d_indices, idx_size, Device{DeviceType::CUDA});

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
    size_t idx_size = total_out * sizeof(size_t);
    d_indices = static_cast<size_t*>(MemoryPool::get().allocate(idx_size, Device{DeviceType::CUDA}));
#if defined(USE_CUDA)
    GPU_CHECK(cudaMemcpy(d_indices, h_indices.data(), idx_size, cudaMemcpyHostToDevice));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMemcpy(d_indices, h_indices.data(), idx_size, hipMemcpyHostToDevice));
#endif

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks == 0) return;

    maxpool2d_backward_kernel<<<blocks, threads>>>(
        grad_output.data(), grad_input.data(), d_indices, total_out
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif

    MemoryPool::get().free(d_indices, idx_size, Device{DeviceType::CUDA});
}

template<typename T>
__global__ void copy_kernel_strided(
    const T *src, T *dst, 
    TensorInfo info,
    size_t size,
    size_t offset_src, size_t offset_dst
) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        size_t linear_idx = idx;
        size_t phys_src = 0, phys_dst = 0;

        for (int d = info.ndims - 1; d >= 0; --d) {
            size_t coord = linear_idx % info.shape[d];
            linear_idx /= info.shape[d];
            phys_src += coord * info.strides[d];
        }

        dst[offset_dst + idx] = src[phys_src + offset_src];
    }
}

template<typename T>
void copy_gpu_strided(const Tensor<T> src, T* dst) {
    size_t total_elements = src.total_elements();

    size_t threads = 256;
    size_t blocks = (total_elements + threads - 1) / threads;
    if (blocks == 0) return;

    TensorInfo info(src.shape(), src.strides());

    copy_kernel_strided<<<blocks, threads>>>(
        src.data(), dst, info, total_elements, src.offset(), 0
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif
}

// #######################################################
// #   Fused Ops: cross-entropy forward/backward
// #######################################################
template<typename T>
__global__ void cross_entropy_fwd_kernel(
    const T* logits, const T* targets, T* per_batch_loss,
    int B, int C
) {
    int bid = blockIdx.x;
    if (bid >= B) return;

    const T* l = logits + bid * C;
    const T* t = targets + bid * C;

    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T* shared = reinterpret_cast<T*>(smem);

    T max_val = -1e30f;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        max_val = fmax(max_val, l[i]);
    shared[threadIdx.x] = max_val;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            shared[threadIdx.x] = fmax(shared[threadIdx.x], shared[threadIdx.x + s]);
    }
    __syncthreads();
    T batch_max = shared[0];
    __syncthreads();

    T sum_exp = 0;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        sum_exp += exp(l[i] - batch_max);
    shared[threadIdx.x] = sum_exp;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            shared[threadIdx.x] += shared[threadIdx.x + s];
    }
    __syncthreads();
    T total_exp = shared[0];
    __syncthreads();

    T loss = 0;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        T p = exp(l[i] - batch_max) / total_exp;
        loss += t[i] * log(fmax(p, 1e-30f));
    }
    shared[threadIdx.x] = -loss;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            shared[threadIdx.x] += shared[threadIdx.x + s];
    }
    __syncthreads();

    if (threadIdx.x == 0)
        per_batch_loss[bid] = shared[0];
}

template<typename T>
__global__ void mean_kernel(const T* data, T* result, int N) {
    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T* shared = reinterpret_cast<T*>(smem);

    T sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += data[i];
    shared[threadIdx.x] = sum;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            shared[threadIdx.x] += shared[threadIdx.x + s];
    }
    __syncthreads();
    if (threadIdx.x == 0)
        result[0] = shared[0] / T(N);
}

template<typename T>
__global__ void softmax_fwd_kernel(T* output, const T* input, int C) {
    int bid = blockIdx.x;
    const T* row_in = input + bid * C;
    T* row_out = output + bid * C;

    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T* shared = reinterpret_cast<T*>(smem);

    T max_val = -1e30f;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        max_val = fmax(max_val, row_in[i]);
    shared[threadIdx.x] = max_val;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            shared[threadIdx.x] = fmax(shared[threadIdx.x], shared[threadIdx.x + s]);
    }
    __syncthreads();
    T batch_max = shared[0];
    __syncthreads();

    T sum_exp = 0;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        T e = exp(row_in[i] - batch_max);
        row_out[i] = e;
        sum_exp += e;
    }
    shared[threadIdx.x] = sum_exp;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            shared[threadIdx.x] += shared[threadIdx.x + s];
    }
    __syncthreads();
    T total_exp = shared[0];
    __syncthreads();

    T inv_sum = T(1) / total_exp;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        row_out[i] *= inv_sum;
}

template<typename T>
__global__ void cross_entropy_bwd_kernel(
    const T* logits, const T* targets, const T* grad_output,
    T* grad_logits,
    int B, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C;
    if (idx >= total) return;

    int bid = idx / C;
    int cid = idx % C;

    const T* l = logits + bid * C;

    T max_val = l[0];
    for (int i = 1; i < C; ++i) max_val = fmax(max_val, l[i]);
    T sum_exp = 0;
    for (int i = 0; i < C; ++i) sum_exp += exp(l[i] - max_val);
    T p = exp(l[cid] - max_val) / sum_exp;

    grad_logits[idx] = (p - targets[idx]) * grad_output[0] / T(B);
}

template<typename T>
Tensor<T> cross_entropy_fwd_gpu(const Tensor<T>& logits, const Tensor<T>& targets) {
    int B = logits.shape()[0];
    int C = logits.shape()[1];

    Tensor<T> per_batch_loss({static_cast<size_t>(B)}, logits.device());
    int threads = min(C, 256);
    // round to power of 2 for tree reduction
    int pow2 = 1;
    while (pow2 < threads) pow2 <<= 1;
    threads = min(pow2, 256);
    size_t smem = (threads + 1) * sizeof(T);

    if (B == 0) {
        Tensor<T> result({1}, logits.device());
#if defined(USE_CUDA)
        GPU_CHECK(cudaMemset(result.data(), 0, sizeof(T)));
#elif defined(USE_ROCM)
        GPU_CHECK(hipMemset(result.data(), 0, sizeof(T)));
#endif
        return result;
    }
    cross_entropy_fwd_kernel<T><<<B, threads, smem>>>(
        logits.data(), targets.data(), per_batch_loss.data(), B, C
    );

    // Second kernel: reduce per-batch losses to scalar mean
    Tensor<T> result({1}, logits.device());
    int r_threads = min(B, 256);
    pow2 = 1;
    while (pow2 < r_threads) pow2 <<= 1;
    r_threads = min(pow2, 256);
    smem = (r_threads + 1) * sizeof(T);
    mean_kernel<T><<<1, r_threads, smem>>>(
        per_batch_loss.data(), result.data(), B
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif
    return result;
}

template<typename T>
void cross_entropy_bwd_gpu(
    const Tensor<T>& grad_output, const Tensor<T>& logits,
    const Tensor<T>& targets, Tensor<T>& grad_logits
) {
    int B = logits.shape()[0];
    int C = logits.shape()[1];
    int total = B * C;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks == 0) return;

    cross_entropy_bwd_kernel<T><<<blocks, threads>>>(
        logits.data(), targets.data(), grad_output.data(),
        grad_logits.data(), B, C
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif
}

template<typename T>
Tensor<T> softmax_gpu(const Tensor<T>& input) {
    auto shape = input.shape();
    int B = input.total_elements() / shape.back();
    int C = shape.back();

    Tensor<T> output(shape, input.device());

    if (B == 0) return output;

    int threads = min(C, 256);
    int pow2 = 1;
    while (pow2 < threads) pow2 <<= 1;
    threads = min(pow2, 256);
    size_t smem = (threads + 1) * sizeof(T);

    softmax_fwd_kernel<T><<<B, threads, smem>>>(
        output.data(), input.data(), C
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
#endif
    return output;
}

// #######################################################
// #   Fused Ops: Adam step
// #######################################################
template<typename T>
__global__ void adam_step_kernel(
    T* param, const T* grad, T* m, T* v,
    size_t N,
    T lr, T beta1, T beta2, T eps,
    T bias_correction1, T bias_correction2,
    T weight_decay
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    T g = grad[idx];
    if (weight_decay > T(0))
        g = g + param[idx] * weight_decay;

    T m_new = beta1 * m[idx] + (T(1.0) - beta1) * g;
    T v_new = beta2 * v[idx] + (T(1.0) - beta2) * g * g;

    m[idx] = m_new;
    v[idx] = v_new;

    T m_hat = m_new / bias_correction1;
    T v_hat = v_new / bias_correction2;

    param[idx] -= lr * m_hat / (sqrt(v_hat) + eps);
}

template<typename T>
void adam_step_gpu(
    Tensor<T>& param, const Tensor<T>& grad, Tensor<T>& m, Tensor<T>& v,
    T lr, T beta1, T beta2, T eps,
    T bias_correction1, T bias_correction2, T weight_decay
) {
    size_t N = param.total_elements();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks == 0) return;

    adam_step_kernel<T><<<blocks, threads>>>(
        param.data(), grad.data(), m.data(), v.data(),
        N, lr, beta1, beta2, eps, bias_correction1, bias_correction2, weight_decay
    );

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
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
template void unary_gpu_strided<float, IdentityOp<float>>(Tensor<float> const&, Tensor<float>&, IdentityOp<float>);
template void unary_gpu<float, ExpOp<float>>(Tensor<float> const&, Tensor<float>&, ExpOp<float>);
template void unary_gpu_strided<float, ExpOp<float>>(Tensor<float> const&, Tensor<float>&, ExpOp<float>);
template void binary_gpu<float, DivOp<float>>(Tensor<float> const&, Tensor<float> const&, Tensor<float>&, DivOp<float>);
template void binary_gpu_strided<float, DivOp<float>>(Tensor<float> const&, Tensor<float> const&, Tensor<float>&, DivOp<float>);
template void unary_gpu<float, LogOp<float>>(Tensor<float> const&, Tensor<float>&, LogOp<float>);
template void unary_gpu_strided<float, LogOp<float>>(Tensor<float> const&, Tensor<float>&, LogOp<float>);
template void unary_gpu<float, AddScalarOp<float>>(Tensor<float> const&, Tensor<float>&, AddScalarOp<float>);
template void unary_gpu_strided<float, AddScalarOp<float>>(Tensor<float> const&, Tensor<float>&, AddScalarOp<float>);
template void unary_gpu<float, SqrtOp<float>>(Tensor<float> const&, Tensor<float>&, SqrtOp<float>);
template void unary_gpu_strided<float, SqrtOp<float>>(Tensor<float> const&, Tensor<float>&, SqrtOp<float>);

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
template void copy_gpu_strided<float>(const Tensor<float> src, float* dst);

template Tensor<float> cross_entropy_fwd_gpu<float>(const Tensor<float>&, const Tensor<float>&);
template void cross_entropy_bwd_gpu<float>(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&, Tensor<float>&);
template void adam_step_gpu<float>(Tensor<float>&, const Tensor<float>&, Tensor<float>&, Tensor<float>&, float, float, float, float, float, float, float);
template Tensor<float> softmax_gpu<float>(const Tensor<float>&);