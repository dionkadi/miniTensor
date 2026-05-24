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
        elementwise_binary_vec_kernel<<<blocks, threads, 0, active_stream()>>>(
            A.data(), B.data(), C.data(), total_elements, op
        );
    } else {
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        elementwise_binary_kernel<<<blocks, threads, 0, active_stream()>>>(A.data(), B.data(), C.data(), total_elements, op);
    }
    
#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
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

    binary_kernel_strided<<<blocks, threads, 0, active_stream()>>>(
        A.data(), B.data(), C.data(), info_A, info_B, info_C,
        total_elements, A.offset(), B.offset(), C.offset(), op
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
#endif
}

template<typename T, typename Op>
void unary_gpu(const Tensor<T>& A, Tensor<T>& C, Op op) {
    size_t total_elements = A.total_elements();
    int threads = 256;
    
    if constexpr (std::is_same_v<T, float>) {
        int blocks = (total_elements + threads * 4 - 1) / (threads * 4);
        elementwise_unary_vec_kernel<<<blocks, threads, 0, active_stream()>>>(A.data(), C.data(), total_elements, op);
    } else {
        int blocks = (total_elements + threads - 1) / threads;
        elementwise_unary_kernel<<<blocks, threads, 0, active_stream()>>>(A.data(), C.data(), total_elements, op);
    }
    
#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
#endif
}

template<typename T, typename Op>
void unary_gpu_strided(const Tensor<T>& A, Tensor<T>& C, Op op) {
    size_t total_elements = A.total_elements();
    size_t threads = 256;
    size_t blocks = (total_elements + threads - 1) / threads;

    TensorInfo info_A(A.shape(), A.strides());
    TensorInfo info_C(C.shape(), C.strides());

    unary_kernel_strided<<<blocks, threads, 0, active_stream()>>>(
        A.data(), C.data(), info_A, info_C,
        total_elements, A.offset(), C.offset(), op
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
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
    sgemm_kernel<BM, BN, BK, TM, TN, 1><<<blocks, threads, 0, active_stream()>>>(
        A.data(), B.data(), C.data(), M, N, K
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
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

    matmul_kernel_strided<<<blocks, threads, 0, active_stream()>>>(
        A.data(), B.data(), C.data(),
        M, K, N,
        A.strides()[0], A.strides()[1],
        B.strides()[0], B.strides()[1],
        C.strides()[0], C.strides()[1],
        A.offset(), B.offset(), C.offset()
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
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
    GPU_CHECK(active_stream() ? cudaMemsetAsync(output.data(), 0, bytes, active_stream()) : cudaMemset(output.data(), 0, bytes));
#elif defined(USE_ROCM)
    GPU_CHECK(active_stream() ? hipMemsetAsync(output.data(), 0, bytes, active_stream()) : hipMemset(output.data(), 0, bytes));
#endif

    if (inner > 1) {
        int threads = 256;
        int blocks = (outer * inner + threads - 1) / threads;
        reduce_axis_kernel<<<blocks, threads, 0, active_stream()>>>(in.data(), output.data(),
                                                outer, reduce, inner);
    } else {
        int threads = 256;
        int blocks = outer;   // one block per row
        reduce_axis_inner_kernel<256><<<blocks, threads, 0, active_stream()>>>(in.data(), output.data(),
                                                           outer, reduce);
    }

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
#endif

    return output;
}

template<typename T>
__global__ void bn_mean_var_kernel(
    const T* __restrict__ x,
    T* __restrict__ mean_out,
    T* __restrict__ var_out,
    int N, int C, int H, int W)
{
    int c = blockIdx.x;
    if (c >= C) return;
    int tid = threadIdx.x;
    int tile = blockDim.x;
    int total = N * H * W;

    T sum_x = T(0), sum_x2 = T(0);
    int sample_stride = C * H * W;
    for (int i = tid; i < total; i += tile) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        T v = x[n * sample_stride + c * H * W + h * W + w];
        sum_x += v;
        sum_x2 += v * v;
    }

    extern __shared__ __align__(sizeof(T)) unsigned char smem_buf[];
    T* smem = reinterpret_cast<T*>(smem_buf);
    T* smem_x = smem;
    T* smem_x2 = smem + tile;

    smem_x[tid] = sum_x;
    smem_x2[tid] = sum_x2;
    __syncthreads();

    for (int s = tile >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem_x[tid] += smem_x[tid + s];
            smem_x2[tid] += smem_x2[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        T spatial_size = T(N * H * W);
        T mean_val = smem_x[0] / spatial_size;
        mean_out[c] = mean_val;
        var_out[c] = smem_x2[0] / spatial_size - mean_val * mean_val;
    }
}

template<typename T>
void bn_fwd_gpu(const Tensor<T>& input, Tensor<T>& mean, Tensor<T>& var) {
    size_t C = input.shape()[1];
    int threads = 256;
    size_t smem_bytes = 2 * threads * sizeof(T);
    bn_mean_var_kernel<T><<<(uint32_t)C, threads, smem_bytes, active_stream()>>>(
        input.data(), mean.data(), var.data(),
        (int)input.shape()[0], (int)C, (int)input.shape()[2], (int)input.shape()[3]);
    GPU_CHECK(get_last_error_capture_safe());
}

// Fused BN + ReLU forward: y = max(0, (x-mean)/sqrt(var+eps)*gamma + beta)
// Combines 4 elementwise ops (sub, div, mul, add) + relu into one kernel.
// Saves 4 kernel launches + 2 intermediate buffers per BN+ReLU pair.
template<typename T>
__global__ void bn_relu_fwd_kernel(
    const T* __restrict__ x, T* __restrict__ y,
    const T* __restrict__ mean, const T* __restrict__ var,
    const T* __restrict__ gamma, const T* __restrict__ beta,
    T eps, int N, int C, int H, int W, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = (idx / (H * W)) % C;
    T inv_std = T(1) / sqrt(var[c] + eps);
    T val = (x[idx] - mean[c]) * inv_std * gamma[c] + beta[c];
    y[idx] = val > T(0) ? val : T(0);
}

template<>
__global__ void bn_relu_fwd_kernel<float>(
    const float* __restrict__ x, float* __restrict__ y,
    const float* __restrict__ mean, const float* __restrict__ var,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float eps, int N, int C, int H, int W, int total)
{
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 3 < total) {
        float4 vx = reinterpret_cast<const float4*>(x)[idx / 4];
        int c0 = (idx / (H * W)) % C;
        int c1 = ((idx + 1) / (H * W)) % C;
        int c2 = ((idx + 2) / (H * W)) % C;
        int c3 = ((idx + 3) / (H * W)) % C;
        float s0 = gamma[c0] / sqrt(var[c0] + eps);
        float b0 = beta[c0] - mean[c0] * s0;
        float s1 = gamma[c1] / sqrt(var[c1] + eps);
        float b1 = beta[c1] - mean[c1] * s1;
        float s2 = gamma[c2] / sqrt(var[c2] + eps);
        float b2 = beta[c2] - mean[c2] * s2;
        float s3 = gamma[c3] / sqrt(var[c3] + eps);
        float b3 = beta[c3] - mean[c3] * s3;
        float4 vy;
        vy.x = fmaxf(vx.x * s0 + b0, 0.0f);
        vy.y = fmaxf(vx.y * s1 + b1, 0.0f);
        vy.z = fmaxf(vx.z * s2 + b2, 0.0f);
        vy.w = fmaxf(vx.w * s3 + b3, 0.0f);
        reinterpret_cast<float4*>(y)[idx / 4] = vy;
    } else {
        for (int i = idx; i < total; ++i) {
            int c = (i / (H * W)) % C;
            float inv_std = 1.0f / sqrt(var[c] + eps);
            y[i] = fmaxf((x[i] - mean[c]) * inv_std * gamma[c] + beta[c], 0.0f);
        }
    }
}

template<typename T>
void bn_relu_fwd_gpu(
    const Tensor<T>& input, Tensor<T>& output,
    const Tensor<T>& mean, const Tensor<T>& var,
    const Tensor<T>& gamma, const Tensor<T>& beta,
    T eps)
{
    size_t N = input.shape()[0], C = input.shape()[1];
    size_t H = input.shape()[2], W = input.shape()[3];
    size_t total = N * C * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks == 0) blocks = 1;
    bn_relu_fwd_kernel<T><<<blocks, threads, 0, active_stream()>>>(
        input.data(), output.data(),
        mean.data(), var.data(), gamma.data(), beta.data(),
        eps, (int)N, (int)C, (int)H, (int)W, (int)total);
    GPU_CHECK(get_last_error_capture_safe());
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

// #######################################################
// #   Winograd F(2x2, 3x3) GPU Kernels
// #   F(2x2, 3x3): input tile 4x4, filter 3x3, output 2x2
// #   Matrices from Lavin & Gray (2016):
// #     B^T = [1,0,-1,0; 0,1,1,0; 0,-1,1,0; 0,1,0,-1]
// #     G   = [1,0,0; 1/2,1/2,1/2; 1/2,-1/2,1/2; 0,0,1]
// #     A^T = [1,1,1,0; 0,1,-1,-1]
// #######################################################
template<typename T>
__global__ void winograd_weight_kernel(
    const T* __restrict__ w, // [C_out, C_in, 3, 3]
    T* __restrict__ u         // [C_out, C_in, 4, 4]
) {
    // One block per (cout, cin): U = G * w * G^T
    // G[0]=[1,0,0]; G[1]=[.5,.5,.5]; G[2]=[.5,-.5,.5]; G[3]=[0,0,1]
    int cout = blockIdx.x, cin = blockIdx.y;
    int base = (cout * gridDim.y + cin);
    int tid = threadIdx.x;

    // Load weight 3x3 into shared memory
    __shared__ T w_flat[9];
    if (tid < 9) {
        int r = tid / 3, c = tid % 3;
        w_flat[tid] = w[(base * 3 + r) * 3 + c];
    }
    __syncthreads();

    // Stage 1: temp = G * w  (4x3)
    // temp[0][j] = w[0][j]                                  (G row 0: [1,0,0])
    // temp[1][j] = (w[0][j] + w[1][j] + w[2][j]) / 2       (G row 1: [.5,.5,.5])
    // temp[2][j] = (w[0][j] - w[1][j] + w[2][j]) / 2       (G row 2: [.5,-.5,.5])
    // temp[3][j] = w[2][j]                                  (G row 3: [0,0,1])
    T w00 = w_flat[0], w01 = w_flat[1], w02 = w_flat[2];
    T w10 = w_flat[3], w11 = w_flat[4], w12 = w_flat[5];
    T w20 = w_flat[6], w21 = w_flat[7], w22 = w_flat[8];
    T temp[4][3];
    temp[0][0] = w00;                     temp[0][1] = w01;                     temp[0][2] = w02;
    temp[1][0] = (w00 + w10 + w20) / T(2); temp[1][1] = (w01 + w11 + w21) / T(2); temp[1][2] = (w02 + w12 + w22) / T(2);
    temp[2][0] = (w00 - w10 + w20) / T(2); temp[2][1] = (w01 - w11 + w21) / T(2); temp[2][2] = (w02 - w12 + w22) / T(2);
    temp[3][0] = w20;                     temp[3][1] = w21;                     temp[3][2] = w22;

    // Stage 2: U = temp * G^T  (4x4)
    // U[i][0] = temp[i][0]
    // U[i][1] = (temp[i][0] + temp[i][1] + temp[i][2]) / 2
    // U[i][2] = (temp[i][0] - temp[i][1] + temp[i][2]) / 2
    // U[i][3] = temp[i][2]
    if (tid < 16) {
        int i = tid / 4, j = tid % 4;
        T val;
        if (j == 0)      val = temp[i][0];
        else if (j == 1) val = (temp[i][0] + temp[i][1] + temp[i][2]) / T(2);
        else if (j == 2) val = (temp[i][0] - temp[i][1] + temp[i][2]) / T(2);
        else             val = temp[i][2];
        u[base * 16 + i * 4 + j] = val;
    }
}

// Input transform: V = B^T * d * B for each (n, cin, tile)
template<typename T>
__global__ void winograd_input_kernel(
    const T* __restrict__ x,  // [N, C_in, H, W]
    T* __restrict__ v,         // [N, tiles_h, tiles_w, C_in, 4, 4]
    int N, int C_in, int H, int W,
    int tiles_h, int tiles_w,
    int pad)
{
    int tw = blockIdx.x, th = blockIdx.y;
    int cin = blockIdx.z % C_in;
    int n   = blockIdx.z / C_in;
    int tid = threadIdx.x; // 0..15 for 4x4

    // Extract 4x4 tile from input (overlapping, stride 2 in tile space)
    int h_start = th * 2 - pad;
    int w_start = tw * 2 - pad;

    int in_row = tid / 4, in_col = tid % 4;
    int h_in = h_start + in_row;
    int w_in = w_start + in_col;

    T d_val = T(0);
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        d_val = x[((n * C_in + cin) * H + h_in) * W + w_in];
    }

    // B^T * d * B where:
    // B^T[0] = [1,  0, -1,  0]
    // B^T[1] = [0,  1,  1,  0]
    // B^T[2] = [0, -1,  1,  0]
    // B^T[3] = [0,  1,  0, -1]
    // B = B^T^T = [1,0,0,0; 0,1,-1,1; -1,1,1,0; 0,0,0,-1]
    
    // Compute V[in_row][in_col] = B^T[in_row] @ d @ B[:,in_col]
    // B^T and B each have 2 non-zeros per row/col → at most 4 MACs
    __shared__ T d_smem[16];
    d_smem[tid] = d_val;
    __syncthreads();

    T val = T(0);

    // B^T[in_row] non-zero column indices: p0, p1, and signs: s0, s1
    int p0, p1; T s0, s1;
    if (in_row == 0)        { p0 = 0; s0 = +1; p1 = 2; s1 = -1; }
    else if (in_row == 1)   { p0 = 1; s0 = +1; p1 = 2; s1 = +1; }
    else if (in_row == 2)   { p0 = 1; s0 = -1; p1 = 2; s1 = +1; }
    else                    { p0 = 1; s0 = +1; p1 = 3; s1 = -1; }

    // B[:,in_col] non-zero row indices: q0, q1, and signs: t0, t1
    int q0, q1; T t0, t1;
    if (in_col == 0)        { q0 = 0; t0 = +1; q1 = 2; t1 = -1; }
    else if (in_col == 1)   { q0 = 1; t0 = +1; q1 = 2; t1 = +1; }
    else if (in_col == 2)   { q0 = 1; t0 = -1; q1 = 2; t1 = +1; }
    else                    { q0 = 1; t0 = +1; q1 = 3; t1 = -1; }

    // Direct 2×2 = 4-element sparse matrix-matrix multiply
    val += (s0 * t0) * d_smem[p0 * 4 + q0];
    val += (s0 * t1) * d_smem[p0 * 4 + q1];
    val += (s1 * t0) * d_smem[p1 * 4 + q0];
    val += (s1 * t1) * d_smem[p1 * 4 + q1];

    // Store V
    int tile_idx = (n * tiles_h + th) * tiles_w + tw;
    v[((tile_idx * C_in + cin) * 4 + in_row) * 4 + in_col] = val;
}

// Simple bias-add kernel: broadcasts [C] over [N, C, H, W]
// Avoids collapse_dims bug when using binary_gpu_strided with reshape+dims>1
template<typename T>
__global__ void bias_add_kernel(T* data, const T* bias, int N, int C, int H, int W, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = (idx / (H * W)) % C;
        data[idx] += bias[c];
    }
}

// Fused multiply-accumulate + output transform
// M[4x4] = sum_cin U[cout][cin][4x4] ⊙ V[tile][cin][4x4]
// Y[2x2] = A^T * M * A
template<typename T>
__global__ void winograd_forward_kernel(
    const T* __restrict__ u,   // [C_out, C_in, 4, 4]
    const T* __restrict__ v,   // [N, tiles_h, tiles_w, C_in, 4, 4]
    T* __restrict__ y,          // [N, C_out, H_out, W_out]
    int N, int C_in, int C_out,
    int tiles_h, int tiles_w,
    int H_out, int W_out)
{
    int tw = blockIdx.x, th = blockIdx.y;
    int cout = blockIdx.z % C_out;
    int n    = blockIdx.z / C_out;
    int tid = threadIdx.x; // 0..15

    if (tw >= tiles_w || th >= tiles_h) return;

    int tile_idx = (n * tiles_h + th) * tiles_w + tw;

    // Accumulate M[4x4] over C_in into shared memory (visible to all 16 threads)
    __shared__ T M_smem[16];
    // Initialize to zero (thread 0 does it once)
    if (tid < 16) M_smem[tid] = T(0);
    __syncthreads();

    int vr = tid / 4, vc = tid % 4;
    for (int cin = 0; cin < C_in; ++cin) {
        int v_off = ((tile_idx * C_in + cin) * 4 + vr) * 4 + vc;
        T v_val = v[v_off];
        T u_val = u[((cout * C_in + cin) * 4 + vr) * 4 + vc];
        M_smem[tid] += u_val * v_val;
    }
    __syncthreads();

    // Output transform: Y[2x2] = A^T * M * A
    // All 16 threads cooperate: each loads full M from smem into registers
    T M[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            M[i][j] = M_smem[i * 4 + j];
    __syncthreads();

    // mid[2x4] = A^T[2x4] @ M[4x4]
    T mid[4]; // mid[threadRow][0..3]
    int row = tid / 4;  // 0..3, but only rows 0,1 are valid (A^T is 2x4)
    if (row < 2) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            mid[j] = T(0);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                T aik;
                if (row == 0) aik = (k < 3) ? T(1) : T(0);
                else          aik = (k == 0) ? T(0) : (k == 3) ? T(-1) : (k == 1) ? T(1) : T(-1);
                mid[j] += aik * M[k][j];
            }
        }
        // out[2x2] = mid[2x4] @ A[4x2]
        int col = tid % 4;
        if (col < 2) {
            int h_out = th * 2 + row;
            int w_out = tw * 2 + col;
            if (h_out < H_out && w_out < W_out) {
                T out_val = T(0);
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    T akj;
                    if (col == 0) akj = (k < 3) ? T(1) : T(0);
                    else          akj = (k == 0) ? T(0) : (k == 3) ? T(-1) : (k == 1) ? T(1) : T(-1);
                    out_val += mid[k] * akj;
                }
                y[((n * C_out + cout) * H_out + h_out) * W_out + w_out] = out_val;
            }
        }
    }
}

template<typename T>
bool conv2d_winograd_gpu(
    const Tensor<T>& input, const Tensor<T>& weight,
    const Tensor<T>& bias, Tensor<T>& output)
{
    // Winograd F(2x2,3x3) for 3x3 stride=1 convolutions
    size_t kH = weight.shape()[2], kW = weight.shape()[3];
    if (kH != 3 || kW != 3) return false; // not supported

    size_t N = input.shape()[0], C_in = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t C_out = output.shape()[1], H_out = output.shape()[2], W_out = output.shape()[3];

    // Only 3x3 stride=1 (pad=1)
    // H_out = H + 2*pad - 2 (for stride=1, kernel=3)
    bool exact = (H_out == H && W_out == W);     // pad=1 for H=W
    // More generally, check H_out = (H + 2*pad - 2) / 1 + 1 == H + 2*pad - 1
    // For the general case, just check if stride=1 and kernel=3
    // Let caller verify applicability: stride=1, kH=kW=3

    int tiles_h = (H_out + 1) / 2;
    int tiles_w = (W_out + 1) / 2;

    // Allocate transformed weight buffer [C_out, C_in, 4, 4]
    size_t u_size = C_out * C_in * 16;
    T* u_buf = static_cast<T*>(MemoryPool::get().allocate(u_size * sizeof(T), Device{DeviceType::CUDA}));

    // Transform weights
    {
        dim3 wt_block(16);
        dim3 wt_grid((uint32_t)C_out, (uint32_t)C_in);
        winograd_weight_kernel<T><<<wt_grid, wt_block, 0, active_stream()>>>(
            weight.data(), u_buf);
        GPU_CHECK(get_last_error_capture_safe());
    }

    // Allocate transformed input buffer [N, tiles_h, tiles_w, C_in, 4, 4]
    size_t v_size = N * tiles_h * tiles_w * C_in * 16;
    T* v_buf = static_cast<T*>(MemoryPool::get().allocate(v_size * sizeof(T), Device{DeviceType::CUDA}));

    // Transform input tiles
    {
        dim3 inp_block(16);
        dim3 inp_grid((uint32_t)tiles_w, (uint32_t)tiles_h, (uint32_t)(N * C_in));
        winograd_input_kernel<T><<<inp_grid, inp_block, 0, active_stream()>>>(
            input.data(), v_buf,
            (int)N, (int)C_in, (int)H, (int)W,
            tiles_h, tiles_w, 1);
        GPU_CHECK(get_last_error_capture_safe());
    }

    // Fused multiply-accumulate + output transform
    {
        dim3 fwd_block(16);
        dim3 fwd_grid((uint32_t)tiles_w, (uint32_t)tiles_h, (uint32_t)(N * C_out));
        winograd_forward_kernel<T><<<fwd_grid, fwd_block, 0, active_stream()>>>(
            u_buf, v_buf, output.data(),
            (int)N, (int)C_in, (int)C_out,
            tiles_h, tiles_w, (int)H_out, (int)W_out);
        GPU_CHECK(get_last_error_capture_safe());
    }

    // Add bias if present — simple broadcast kernel avoids collapse_dims bug
    if (!bias.empty()) {
        size_t total = N * C_out * H_out * W_out;
        size_t bt = 256;
        size_t bg = (total + bt - 1) / bt;
        bias_add_kernel<T><<<bg, bt, 0, active_stream()>>>(output.data(), bias.data(), (int)N, (int)C_out, (int)H_out, (int)W_out, (int)total);
        GPU_CHECK(get_last_error_capture_safe());
    }

    MemoryPool::get().free(v_buf, v_size * sizeof(T), Device{DeviceType::CUDA});
    MemoryPool::get().free(u_buf, u_size * sizeof(T), Device{DeviceType::CUDA});
    return true;
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

    if constexpr (std::is_same_v<T, float>) {
        if (kH == 3 && kW == 3 && stride == 1 && padding == 1) {
            bool success = conv2d_winograd_gpu(input, weight, bias, output);
            GPU_CHECK(get_last_error_capture_safe());
            if (success) return;
        }
    }

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
    
    im2col_kernel<T><<<blocks, threads, smem_size, active_stream()>>>(
        input.data(), d_data_col, N, C_in, H, W, kH, kW, padding, stride, H_out, W_out
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
#endif

    // Tiled batched GEMM: all N batch items in one 3D launch
    // weight [M, K] (shared), col [N, K, N_mat], output [N, M, N_mat]
    if constexpr (std::is_same_v<T, float>) {
        dim3 batched_threads(16, 16);
        dim3 batched_blocks((N_mat + 127) / 128, (M + 127) / 128, N);
        batched_sgemm_kernel<128, 128, 8, 8, 8><<<batched_blocks, batched_threads, 0, active_stream()>>>(
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
            matmul_kernel_strided<T><<<strided_blocks, mm_threads, 0, active_stream()>>>(
                weight.data(), col_ptr, out_ptr,
                M, K, N_mat, K, 1, N_mat, 1, N_mat, 1, 0, 0, 0
            );
        }
    }

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
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

    conv2d_forward_kernel_strided<<<blocks, threads, 0, active_stream()>>>(
        input.data(), weight.data(), bias.empty() ? nullptr : bias.data(),
        output.data(),
        info_in, info_wt, info_out,
        input.offset(), weight.offset(), output.offset(),
        N, C_in, H, W, C_out, kH, kW, H_out, W_out, stride, padding);

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
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
    GPU_CHECK(active_stream() ? cudaMemsetAsync(grad_input.data(), 0, grad_input.total_elements() * sizeof(T), active_stream()) : cudaMemset(grad_input.data(), 0, grad_input.total_elements() * sizeof(T)));
#elif defined(USE_ROCM)
    GPU_CHECK(active_stream() ? hipMemsetAsync(grad_input.data(), 0, grad_input.total_elements() * sizeof(T), active_stream()) : hipMemset(grad_input.data(), 0, grad_input.total_elements() * sizeof(T)));
#endif

    // Transpose the weights (W^T)
    dim3 tp_threads(WARP_SIZE, WARP_SIZE);
    dim3 tp_blocks((M + WARP_SIZE - 1) / WARP_SIZE, (K + WARP_SIZE - 1) / WARP_SIZE);
    mat_transpose_kernel<T><<<tp_blocks, tp_threads, 0, active_stream()>>>(weight.data(), d_weight_T, K, M);

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
#endif

    // Fused backward: single kernel does W^T x dY GEMM + col2im in one pass
    {
        dim3 fwd_threads(16, 16);
        dim3 fwd_blocks((W + 15) / 16, (H + 15) / 16, N * C_in);
        fused_backward_input_kernel<T><<<fwd_blocks, fwd_threads, 0, active_stream()>>>(
            grad_output.data(), d_weight_T, grad_input.data(),
            N, C_in, C_out, H, W, kH, kW, (int)padding, (int)stride, H_out, W_out
        );
    }

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
#endif
    MemoryPool::get().free(d_weight_T, weight_T_size, Device{DeviceType::CUDA});
}

template<typename T>
__global__ void conv2d_bwd_weight_direct_kernel(
    const T* __restrict__ dY,
    const T* __restrict__ X,
    T* __restrict__ dW,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int H_out, int W_out,
    int stride, int padding)
{
    // Each block handles one weight element dW[cout][cin][kh][kw]
    // blockIdx.x = cout, blockIdx.y = cin, blockIdx.z = kh * kW + kw
    // Threads within the block reduce over N * H_out * W_out

    int cout = blockIdx.x;
    int cin  = blockIdx.y;
    int kw_i = blockIdx.z % kW;
    int kh_i = blockIdx.z / kW;

    if (cout >= C_out || cin >= C_in || kh_i >= kH || kw_i >= kW) return;

    int K = C_in * kH * kW;
    int HW = H_out * W_out;
    int total = N * HW;

    // Block-wise strided reduction over N*HW
    int tile_size = blockDim.x;
    int tid = threadIdx.x;

    // Shared memory for reduction
    extern __shared__ __align__(sizeof(T)) unsigned char smem_buf[];
    T* smem = reinterpret_cast<T*>(smem_buf);

    T sum = T(0);
    int dY_batch_stride = C_out * HW;
    int X_sample_stride = C_in * H * W;

    // Map linear position to (n, h_out, w_out)
    for (int pos = tid; pos < total; pos += tile_size) {
        int n  = pos / HW;
        int hw = pos % HW;
        int h_out = hw / W_out;
        int w_out = hw % W_out;

        int h_in = h_out * stride + kh_i - padding;
        int w_in = w_out * stride + kw_i - padding;

        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            sum += dY[n * dY_batch_stride + cout * HW + hw]
                 * X[n * X_sample_stride + cin * H * W + h_in * W + w_in];
        }
    }

    // Tree reduction in shared memory
    smem[tid] = sum;
    __syncthreads();

#pragma unroll
    for (int s = tile_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int w_idx = (cout * C_in + cin) * kH * kW + kh_i * kW + kw_i;
        dW[w_idx] = smem[0];
    }
}

template<typename T> 
void conv2d_backward_weight_gpu(const Tensor<T>& grad_output, const Tensor<T>& input, Tensor<T>& grad_weight, size_t stride, size_t padding) {
    size_t N = input.shape()[0], C_in = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t C_out = grad_weight.shape()[0], kH = grad_weight.shape()[2], kW = grad_weight.shape()[3];
    size_t H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];

    int threads = 256;
    size_t smem_bytes = threads * sizeof(T);
    dim3 blocks((uint32_t)C_out, (uint32_t)C_in, (uint32_t)(kH * kW));

    conv2d_bwd_weight_direct_kernel<<<blocks, threads, smem_bytes, active_stream()>>>(
        grad_output.data(), input.data(), grad_weight.data(),
        (int)N, (int)C_in, (int)H, (int)W, (int)C_out, (int)kH, (int)kW,
        (int)H_out, (int)W_out, (int)stride, (int)padding);
    GPU_CHECK(get_last_error_capture_safe());
}

template<typename T> 
void conv2d_backward_bias_gpu(const Tensor<T>& grad_output, Tensor<T>& grad_bias) {
    size_t N = grad_output.shape()[0], C_out = grad_output.shape()[1], H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];
    size_t total_out = N * C_out * H_out * W_out;
    size_t total_bias = grad_bias.total_elements();

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

#if defined(USE_CUDA)
    GPU_CHECK(active_stream() ? cudaMemsetAsync(grad_bias.data(), 0, total_bias * sizeof(T), active_stream()) : cudaMemset(grad_bias.data(), 0, total_bias * sizeof(T)));
#elif defined(USE_ROCM)
    GPU_CHECK(active_stream() ? hipMemsetAsync(grad_bias.data(), 0, total_bias * sizeof(T), active_stream()) : hipMemset(grad_bias.data(), 0, total_bias * sizeof(T)));
#endif
    if (blocks == 0) return;

    conv2d_backward_bias_kernel<<<blocks, threads, 0, active_stream()>>>(
        grad_output.data(), grad_bias.data(),
        N, C_out, H_out, W_out
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
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
std::pair<Tensor<T>, Tensor<size_t>> max_pool2d_gpu(
    const Tensor<T>& input, size_t k, size_t stride, size_t padding
) {
    size_t N = input.shape()[0], C = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t H_out = safe_out_size(H, padding, k, stride);
    size_t W_out = safe_out_size(W, padding, k, stride);
    size_t total_out = N * C * H_out * W_out;

    Tensor<T> output({N, C, H_out, W_out}, input.device());
    // Keep indices as a device-side tensor (no host copy)
    Tensor<size_t> indices({total_out}, Device{DeviceType::CUDA});

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks == 0) {
        return {std::move(output), std::move(indices)};
    }

    maxpool2d_kernel<<<blocks, threads, 0, active_stream()>>>(
        input.data(), output.data(), indices.data(),
        N, C, H, W, H_out, W_out, k, stride, padding
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
#endif

    return {output, indices};
}

template<typename T>
std::pair<Tensor<T>, Tensor<size_t>> max_pool2d_gpu_strided(
    const Tensor<T>& input, size_t k, size_t stride, size_t padding
) {
    size_t N = input.shape()[0], C = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    size_t H_out = safe_out_size(H, padding, k, stride);
    size_t W_out = safe_out_size(W, padding, k, stride);
    size_t total_out = N * C * H_out * W_out;

    Tensor<T> output({N, C, H_out, W_out}, input.device());
    Tensor<size_t> indices({total_out}, Device{DeviceType::CUDA});

    TensorInfo info_in(input.shape(), input.strides());

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks == 0) {
        return {std::move(output), std::move(indices)};
    }

    maxpool2d_kernel_strided<<<blocks, threads, 0, active_stream()>>>(
        input.data(), output.data(), indices.data(),
        info_in, input.offset(), output.offset(),
        N, C, H, W, H_out, W_out, k, stride, padding
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
#endif

    return {output, indices};
}

template<typename T>
void max_pool2d_backward_gpu(
    const Tensor<T>& grad_output, 
    Tensor<T>& grad_input, 
    const Tensor<size_t>& d_indices
)  {
    size_t total_out = grad_output.total_elements();
    size_t total_in = grad_input.total_elements();

#if defined(USE_CUDA)
    GPU_CHECK(active_stream() ? cudaMemsetAsync(grad_input.data(), 0, total_in * sizeof(T), active_stream()) : cudaMemset(grad_input.data(), 0, total_in * sizeof(T)));
#elif defined(USE_ROCM)
    GPU_CHECK(active_stream() ? hipMemsetAsync(grad_input.data(), 0, total_in * sizeof(T), active_stream()) : hipMemset(grad_input.data(), 0, total_in * sizeof(T)));
#endif

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks == 0) return;

    maxpool2d_backward_kernel<<<blocks, threads, 0, active_stream()>>>(
        grad_output.data(), grad_input.data(), d_indices.data(), total_out
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
#endif
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

    copy_kernel_strided<<<blocks, threads, 0, active_stream()>>>(
        src.data(), dst, info, total_elements, src.offset(), 0
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
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
        GPU_CHECK(active_stream() ? cudaMemsetAsync(result.data(), 0, sizeof(T), active_stream()) : cudaMemset(result.data(), 0, sizeof(T)));
#elif defined(USE_ROCM)
        GPU_CHECK(active_stream() ? hipMemsetAsync(result.data(), 0, sizeof(T), active_stream()) : hipMemset(result.data(), 0, sizeof(T)));
#endif
        return result;
    }
    cross_entropy_fwd_kernel<T><<<B, threads, smem, active_stream()>>>(
        logits.data(), targets.data(), per_batch_loss.data(), B, C
    );

    // Second kernel: reduce per-batch losses to scalar mean
    Tensor<T> result({1}, logits.device());
    int r_threads = min(B, 256);
    pow2 = 1;
    while (pow2 < r_threads) pow2 <<= 1;
    r_threads = min(pow2, 256);
    smem = (r_threads + 1) * sizeof(T);
    mean_kernel<T><<<1, r_threads, smem, active_stream()>>>(
        per_batch_loss.data(), result.data(), B
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
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

    cross_entropy_bwd_kernel<T><<<blocks, threads, 0, active_stream()>>>(
        logits.data(), targets.data(), grad_output.data(),
        grad_logits.data(), B, C
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
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

    softmax_fwd_kernel<T><<<B, threads, smem, active_stream()>>>(
        output.data(), input.data(), C
    );

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
#endif
    return output;
}

// #######################################################
// #   Fused Ops: add+relu
// #######################################################
template<typename T>
__global__ void add_relu_kernel(const T* a, const T* b, T* c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    T val = a[idx] + b[idx];
    c[idx] = val > T(0) ? val : T(0);
}

template<typename Op>
__global__ void add_relu_vec_kernel(
    const float* a, const float* b, float* c, size_t N, Op op
) {
    size_t vec_idx = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t idx = vec_idx * 4;
    if (idx >= N) return;

    float4 a_val = reinterpret_cast<const float4*>(a)[vec_idx];
    float4 b_val = reinterpret_cast<const float4*>(b)[vec_idx];

    float4 c_val;
    c_val.x = op(a_val.x + b_val.x);
    c_val.y = op(a_val.y + b_val.y);
    c_val.z = op(a_val.z + b_val.z);
    c_val.w = op(a_val.w + b_val.w);

    reinterpret_cast<float4*>(c)[vec_idx] = c_val;
}

template<typename T>
void add_relu_gpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    size_t N = A.total_elements();
    int threads = 256;

    if constexpr (sizeof(T) == 4) {
        size_t vec_n = N / 4;
        if (vec_n > 0) {
            int vec_blocks = (vec_n + threads - 1) / threads;
            add_relu_vec_kernel<ReLUOp<float>><<<vec_blocks, threads, 0, active_stream()>>>(
                A.data(), B.data(), C.data(), N, ReLUOp<float>{}
            );
        }
        size_t rem = N % 4;
        if (rem > 0) {
            size_t offset = N - rem;
            int rem_blocks = (rem + threads - 1) / threads;
            if (rem_blocks == 0) rem_blocks = 1;
            add_relu_kernel<T><<<rem_blocks, threads, 0, active_stream()>>>(
                A.data() + offset, B.data() + offset, C.data() + offset, rem
            );
        }
    } else {
        int blocks = (N + threads - 1) / threads;
        if (blocks == 0) return;
        add_relu_kernel<T><<<blocks, threads, 0, active_stream()>>>(
            A.data(), B.data(), C.data(), N
        );
    }

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
#endif
}

// #######################################################
// #   Fused Ops: Adam step
// #######################################################
// Float4 vectorized adam step: 4 elements per thread
template<typename T>
__global__ void adam_step_vec_kernel(
    T* param, const T* grad, T* m, T* v,
    size_t N,
    T lr, T beta1, T beta2, T eps,
    T bias_correction1, T bias_correction2,
    T weight_decay
) {
    size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = vec_idx * 4;
    if (idx >= N) return;

    // Load 4 elements via float4
    float4 g_vec = reinterpret_cast<const float4*>(grad)[vec_idx];
    float4 p_vec = reinterpret_cast<const float4*>(param)[vec_idx];
    float4 m_vec = reinterpret_cast<const float4*>(m)[vec_idx];
    float4 v_vec = reinterpret_cast<const float4*>(v)[vec_idx];

    float4 g_wd;
    g_wd.x = g_vec.x + (weight_decay > T(0) ? p_vec.x * weight_decay : T(0));
    g_wd.y = g_vec.y + (weight_decay > T(0) ? p_vec.y * weight_decay : T(0));
    g_wd.z = g_vec.z + (weight_decay > T(0) ? p_vec.z * weight_decay : T(0));
    g_wd.w = g_vec.w + (weight_decay > T(0) ? p_vec.w * weight_decay : T(0));

    float4 m_out, v_out;
    m_out.x = fma(beta1, m_vec.x, (T(1.0) - beta1) * g_wd.x);
    v_out.x = fma(beta2, v_vec.x, (T(1.0) - beta2) * g_wd.x * g_wd.x);
    m_out.y = fma(beta1, m_vec.y, (T(1.0) - beta1) * g_wd.y);
    v_out.y = fma(beta2, v_vec.y, (T(1.0) - beta2) * g_wd.y * g_wd.y);
    m_out.z = fma(beta1, m_vec.z, (T(1.0) - beta1) * g_wd.z);
    v_out.z = fma(beta2, v_vec.z, (T(1.0) - beta2) * g_wd.z * g_wd.z);
    m_out.w = fma(beta1, m_vec.w, (T(1.0) - beta1) * g_wd.w);
    v_out.w = fma(beta2, v_vec.w, (T(1.0) - beta2) * g_wd.w * g_wd.w);

    reinterpret_cast<float4*>(m)[vec_idx] = m_out;
    reinterpret_cast<float4*>(v)[vec_idx] = v_out;

    float inv_bc1 = T(1.0) / bias_correction1;
    float inv_bc2 = T(1.0) / bias_correction2;

    float4 p_out;
    p_out.x = p_vec.x - lr * (m_out.x * inv_bc1) / (sqrt(v_out.x * inv_bc2) + eps);
    p_out.y = p_vec.y - lr * (m_out.y * inv_bc1) / (sqrt(v_out.y * inv_bc2) + eps);
    p_out.z = p_vec.z - lr * (m_out.z * inv_bc1) / (sqrt(v_out.z * inv_bc2) + eps);
    p_out.w = p_vec.w - lr * (m_out.w * inv_bc1) / (sqrt(v_out.w * inv_bc2) + eps);

    reinterpret_cast<float4*>(param)[vec_idx] = p_out;
}

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

    if constexpr (sizeof(T) == 4) {
        // Vectorized kernel for bulk (4 elements per thread)
        size_t vec_n = N / 4;
        if (vec_n > 0) {
            int vec_blocks = (vec_n + threads - 1) / threads;
            adam_step_vec_kernel<T><<<vec_blocks, threads, 0, active_stream()>>>(
                param.data(), grad.data(), m.data(), v.data(),
                N, lr, beta1, beta2, eps,
                bias_correction1, bias_correction2, weight_decay
            );
        }
        // Scalar kernel for remaining elements
        size_t rem = N % 4;
        if (rem > 0) {
            size_t offset = N - rem;
            int rem_blocks = (rem + threads - 1) / threads;
            if (rem_blocks == 0) rem_blocks = 1;
            adam_step_kernel<T><<<rem_blocks, threads, 0, active_stream()>>>(
                param.data() + offset, grad.data() + offset,
                m.data() + offset, v.data() + offset,
                rem, lr, beta1, beta2, eps,
                bias_correction1, bias_correction2, weight_decay
            );
        }
    } else {
        int blocks = (N + threads - 1) / threads;
        if (blocks == 0) return;
        adam_step_kernel<T><<<blocks, threads, 0, active_stream()>>>(
            param.data(), grad.data(), m.data(), v.data(),
            N, lr, beta1, beta2, eps,
            bias_correction1, bias_correction2, weight_decay
        );
    }

#if defined(USE_CUDA)
    GPU_CHECK(get_last_error_capture_safe());
#elif defined(USE_ROCM)
    GPU_CHECK(get_last_error_capture_safe());
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
template void add_relu_gpu<float>(const Tensor<float>&, const Tensor<float>&, Tensor<float>&);
template void bn_fwd_gpu<float>(const Tensor<float>&, Tensor<float>&, Tensor<float>&);
template void bn_relu_fwd_gpu<float>(const Tensor<float>&, Tensor<float>&, const Tensor<float>&, const Tensor<float>&, const Tensor<float>&, const Tensor<float>&, float);

template void conv2d_gpu<float>(Tensor<float> const&, Tensor<float> const&, Tensor<float> const&, Tensor<float>&, unsigned long, unsigned long);
template void conv2d_gpu_strided<float>(Tensor<float> const&, Tensor<float> const&, Tensor<float> const&, Tensor<float>&, unsigned long, unsigned long);
template std::pair<Tensor<float>, Tensor<size_t>> max_pool2d_gpu<float>(Tensor<float> const&, unsigned long, unsigned long, unsigned long);
template std::pair<Tensor<float>, Tensor<size_t>> max_pool2d_gpu_strided<float>(Tensor<float> const&, unsigned long, unsigned long, unsigned long);
template void conv2d_backward_input_gpu<float>(Tensor<float> const&, Tensor<float> const&, Tensor<float>&, unsigned long, unsigned long);
template void conv2d_backward_weight_gpu<float>(Tensor<float> const&, Tensor<float> const&, Tensor<float>&, unsigned long, unsigned long);
template void conv2d_backward_bias_gpu<float>(Tensor<float> const&, Tensor<float>&);
template void max_pool2d_backward_gpu<float>(Tensor<float> const&, Tensor<float>&, Tensor<size_t> const&);
template void copy_gpu_strided<float>(const Tensor<float> src, float* dst);

template Tensor<float> cross_entropy_fwd_gpu<float>(const Tensor<float>&, const Tensor<float>&);
template void cross_entropy_bwd_gpu<float>(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&, Tensor<float>&);
template void adam_step_gpu<float>(Tensor<float>&, const Tensor<float>&, Tensor<float>&, Tensor<float>&, float, float, float, float, float, float, float);
template Tensor<float> softmax_gpu<float>(const Tensor<float>&);