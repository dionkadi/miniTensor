#include <cstddef>

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#elif defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#endif

constexpr float MAX_EXP_F32 = 88.3762626647949f;
constexpr float MIN_EXP_F32 = -88.3762626647949f;
const half MAX_EXP_F16 = __float2half(11.089866488461016f);
const half MIN_EXP_F16 = __float2half(-9.704060527839234f);

constexpr int WARP_SIZE = 32;
constexpr int PAD = 1;

template<typename T>
__device__ constexpr T clamp(T x, T hi, T lo) {
    return fmin(fmax(x, lo), hi);
}

template<>
__device__ constexpr half clamp(half x, half hi, half lo) {
    return __hmin(__hmax(x, lo), hi);
}

template<typename T, typename V>
__device__ constexpr T cast(V val) {
    return reinterpret_cast<T*>(&val)[0];
}

template<typename T, typename Op>
__global__ void elementwise_unary_kernel(
    T *a, T *c, size_t N, Op op
) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        c[idx] = op(a[idx]);
    }
}

template<typename Op>
__global__ void elementwise_unary_vec_kernel(
    float *a, float *c, size_t N, Op op
) {
    size_t idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if ((idx + 3) < N) {
        float4 reg_a = cast<float4>(a[idx]);
        float4 reg_c;
        reg_c.x = op(reg_a.x);
        reg_c.y = op(reg_a.y);
        reg_c.z = op(reg_a.z);
        reg_c.w = op(reg_a.w);
        cast<float4>(c[idx]) = reg_c;
    } else if (idx < N) {
        for (size_t i = 0; (idx + i) < N; ++i) {
            c[idx + i] = op(a[idx + i]);
        }
    }
}

template<typename T, typename Op>
__global__ void elementwise_binary_kernel(
    T *a, T *b, T *c, size_t N, Op op
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = op(a[idx], b[idx]);
    }
}

template<typename Op>
__global__ void elementwise_binary_vec_kernel(
    float *a, float *b, float *c, size_t N, Op op
) {
    size_t idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if ((idx + 3) < N) {
        float4 reg_a = cast<float4>(a[idx]);
        float4 reg_b = cast<float4>(b[idx]);
        float4 reg_c;
        reg_c.x = op(reg_a.x, reg_b.x);
        reg_c.y = op(reg_a.y, reg_b.y);
        reg_c.z = op(reg_a.z, reg_b.z);
        reg_c.w = op(reg_a.w, reg_b.w);
        cast<float4>(c[idx]) = reg_c;
    } else if (idx < N) {
        for (size_t i = 0; (idx + i) < N; ++i) {
            c[idx + i] = op(a[idx + i], b[idx + i]);
        }
    }
}

__global__ void elementwise_add_kernel(
    float *a, float *b, float *c, size_t N
) {
    size_t idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if ((idx + 3) < N) {
        float4 reg_a = cast<float4>(a[idx]);
        float4 reg_b = cast<float4>(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        cast<float4>(c[idx]) = reg_c;
    } else if (idx < N) {
        for (size_t i = 0; (idx + i) < N; ++i) {
            c[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}

__global__ void elementwise_add_f16x8_kernel(
    half *a, half *b, half *c, size_t N
) {
    size_t idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    if ((idx + 7) < N) {
        half pack_a[8], pack_b[8], pack_c[8];
        cast<float4>(pack_a[0]) = cast<float4>(a[idx]);
        cast<float4>(pack_b[0]) = cast<float4>(b[idx]);

#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            cast<half2>(pack_c[i]) = __hadd2(cast<half2>(pack_a[i]), cast<half2>(pack_b[i]));
        }

        cast<float4>(c[idx]) = cast<float4>(pack_c[0]);
    } else if (idx < N) {
        for (size_t i = 0; (idx + i) < N; ++i) {
            c[idx + i] = __hadd(a[idx + i], b[idx + i]);
        }
    }
}

__global__ void histogram_kernel(
    int *a, int *y, size_t N
) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    atomicAdd(&(y[idx]), 1);
}

__global__ void histogram_vec_kernel(
    int *a, int *y, size_t N
) {
    size_t idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if ((idx + 4) < N) {
        int4 reg_a = cast<int4>(a[idx]);
        atomicAdd(&(y[reg_a.x]), 1);
        atomicAdd(&(y[reg_a.y]), 1);
        atomicAdd(&(y[reg_a.z]), 1);
        atomicAdd(&(y[reg_a.w]), 1);
    } else if (idx < N) {
        for (int i = 0; (idx + i) < N; ++i) {
            atomicAdd(&(y[a[idx + i]]), 1);
        }
    }
}

__global__ void sigmoid_kernel(
    float *x, float *y, size_t N
) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < N) {
        float reg_x = x[idx], reg_y;
        reg_x = clamp(reg_x, MAX_EXP_F32, MIN_EXP_F32);
        reg_y = 1.0f / (1.0f + expf(-reg_x));
        y[idx] = reg_y;
    }
}

__global__ void sigmoid_vec_kernel(
    float *x, float *y, size_t N
) {
    size_t idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    
    if ((idx + 3) < N) {
        float4 reg_x = cast<float4>(x[idx]);
        float4 reg_y;
    
        reg_x.x = clamp(reg_x.x, MAX_EXP_F32, MIN_EXP_F32);
        reg_x.y = clamp(reg_x.y, MAX_EXP_F32, MIN_EXP_F32);
        reg_x.z = clamp(reg_x.z, MAX_EXP_F32, MIN_EXP_F32);
        reg_x.w = clamp(reg_x.w, MAX_EXP_F32, MIN_EXP_F32);
    
        reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
        reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
        reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
        reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
        cast<float4>(y[idx]) = reg_y;
    }
}

__global__ void sigmoid_f16_kernel(
    half *x, half *y, size_t N
) {
    size_t idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    
    if ((idx + 7) < N) {
        half pack_x[8], pack_y[8];
        const half one = __float2half(1.0f);
        cast<float4>(pack_x[0]) = cast<float4>(x[idx]);
    
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half v = clamp(pack_x[i], MAX_EXP_F16, MIN_EXP_F16);
            pack_y[i] = one / (one + hexp(-v));
        }
        cast<float4>(y[idx]) = cast<float4>(pack_y[0]);
    }
}

__global__ void relu_f32_kernel(
    float *x, float *y, size_t N
) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < N) {
        y[idx] = fmaxf(x[idx], 0.0f);
    }
}

__global__ void relu_f32_vec_kernel(
    float *x, float *y, size_t N
) {
    size_t idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    
    if ((idx + 3) < N) {
        float4 reg_x = cast<float4>(x[idx]);
        float4 reg_y;
    
        reg_y.x = fmaxf(reg_x.x, 0.0f);
        reg_y.y = fmaxf(reg_x.y, 0.0f);
        reg_y.z = fmaxf(reg_x.z, 0.0f);
        reg_y.w = fmaxf(reg_x.w, 0.0f);

        cast<float4>(y[idx]) = reg_y;
    }
}

__global__ void relu_f16_kernel(
    half *x, half *y, size_t N
) {
    size_t idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};

    if ((idx + 7) < N) {
        half pack_x[8], pack_y[8];
        cast<float4>(pack_x[0]) = cast<float4>(x[idx]);
    
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            cast<half2>(pack_y[i]) = __hmax2(cast<half2>(pack_x[i]), z2);
        }
        cast<float4>(y[idx]) = cast<float4>(pack_y[0]);
    }
}

__global__ void elu_f32_kernel(
    float *x, float *y, size_t N
) {
    size_t idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    const float ALPHA = 1.0f;
    if ((idx + 3) < N) {
        float4 reg_x = cast<float4>(x[idx]);
        float4 reg_y;
        
        reg_x.x = clamp(reg_x.x, MAX_EXP_F32, MIN_EXP_F32);
        reg_x.y = clamp(reg_x.y, MAX_EXP_F32, MIN_EXP_F32);
        reg_x.z = clamp(reg_x.z, MAX_EXP_F32, MIN_EXP_F32);
        reg_x.w = clamp(reg_x.w, MAX_EXP_F32, MIN_EXP_F32);
    
        reg_y.x = reg_x.x > 0.0f ? reg_x.x : ALPHA * (expf(reg_x.x) - 1.0f);
        reg_y.y = reg_x.y > 0.0f ? reg_x.y : ALPHA * (expf(reg_x.y) - 1.0f);
        reg_y.z = reg_x.z > 0.0f ? reg_x.z : ALPHA * (expf(reg_x.z) - 1.0f);
        reg_y.w = reg_x.w > 0.0f ? reg_x.w : ALPHA * (expf(reg_x.w) - 1.0f);
        cast<float4>(y[idx]) = reg_y;
    }
}

__global__ void elu_f16_kernel(
    half *x, half *y, size_t N
) {
    size_t idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    const half ALPHA = __float2half(1.0f);   
    const half one = __float2half(1.0f);
    if ((idx + 7) < N) {
        half pack_x[8], pack_y[8];
        cast<float4>(pack_x[0]) = cast<float4>(x[idx]);
    
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half v = clamp(pack_x[i], MAX_EXP_F16, MIN_EXP_F16);
            pack_y[i] = v > 0.0f ? v : ALPHA * (__hexp(v) - one);
        }
        cast<float4>(y[idx]) = cast<float4>(pack_y[0]);
    }
}

__global__ void mat_transpose_kernel(
    float *x, float *y, int row, int col
) {
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    const int STRIDE = WARP_SIZE;
    __shared__ float tile[WARP_SIZE][WARP_SIZE + PAD];
    if (global_y < row && global_x < col) {
        float x_val = x[global_y * col + global_x];
        tile[local_y][local_x] = x_val;

        __syncthreads();
        
        float smem_val = tile[local_y % STRIDE][local_x + local_y / STRIDE];;
        
        int bid_y = blockIdx.y;
        int out_y = global_x + local_y / STRIDE;
        int out_x = (local_y % STRIDE) + bid_y;
        y[out_y * row + out_x] = smem_val;
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
    if (global_y * 4 < row && global_x < col) {
        float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
        tile[local_y][local_x * 4] = x_val.x;
        tile[local_y][local_x * 4 + 1] = x_val.y;
        tile[local_y][local_x * 4 + 2] = x_val.z;
        tile[local_y][local_x * 4 + 3] = x_val.w;
        __syncthreads();
        
        float4 smem_val;
        smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
        smem_val.y = tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
        smem_val.z = tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
        smem_val.w = tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
        
        int bid_y = blockIdx.y;
        int out_x = global_x * 4 + local_y / STRIDE;
        int out_y = (local_y % STRIDE) * 4 + bid_y;
        reinterpret_cast<float4 *>(y)[out_y * row + out_x] = cast<float4>(smem_val);
    }
}

template<size_t kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<size_t kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
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

template<size_t NUM_THREADS = 256>
__global__ void reduce_sum_kernel(
    float *a, float *y, size_t N
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ sum_smem[NUM_WARPS];

    float sum = (idx < N) ? a[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        sum_smem[warp] = sum;
    } 

    __syncthreads();

    sum = (lane < NUM_WARPS) ? sum_smem[lane] : 0.0f;
    if (warp == 0) {
        sum = warp_reduce_sum<NUM_WARPS>(sum);
    }
    if (tid == 0) {
        atomicAdd(y, sum);
    }
}

template<size_t NUM_THREADS = 256 / 4>
__global__ void reduce_sum_vec_kernel(
    float *a, float *y, size_t N
) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ sum_smem[NUM_WARPS];

    float4 reg_a = cast<float4>(a[idx]);
    float sum = (idx < N) ? reg_a.x + reg_a.y + reg_a.z + reg_a.w : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        sum_smem[warp] = sum;
    } 

    __syncthreads();

    sum = (lane < NUM_WARPS) ? sum_smem[lane] : 0.0f;
    if (warp == 0) {
        sum = warp_reduce_sum<NUM_WARPS>(sum);
    }
    if (tid == 0) {
        atomicAdd(y, sum);
    }
}

// outer_size: The product of all dimensions before the reduction axis.
// reduce_size: The size of the reduction axis.
// inner_size: The product of all dimensions after the reduction axis.
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
    const int reduce_size, 
    // const int inner_size == 1
) {
    int row = blockIdx.x;
    if (row >= outer_size) return;

    int tid = threadIdx.x;
    float sum = 0.0f;

    int row_offset = row * outer_size;
    for (int i = tid; i < reduce_size; i += BLOCK_SIZE) {
        sum += a[row_offset + i];
    }

    sum = block_reduce_sum<BLOCK_SIZE>(sum);

    if (tid == 0) {
        y[row] = sum;
    }
}

template<size_t NUM_THREADS = 256>
__global__ void dot_product_kernel(
    float *a, float *b, float *y, size_t N
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ product_smem[NUM_WARPS];

    float product = (idx < N) ? a[idx] * b[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    
    product = warp_reduce_sum(product);
    if (lane == 0) {
        product_smem[warp] = product;
    } 

    __syncthreads();

    product = (lane < NUM_WARPS) ? product_smem[lane] : 0.0f;
    if (warp == 0) {
        product = warp_reduce_sum<NUM_WARPS>(product);
    }
    if (tid == 0) {
        atomicAdd(y, product);
    }
}

template<size_t NUM_THREADS = 256 / 4>
__global__ void dot_product_vec_kernel(
    float *a, float *b, float *y, size_t N
) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ product_smem[NUM_WARPS];

    float4 reg_a = cast<float4>(a[idx]);
    float4 reg_b = cast<float4>(b[idx]);
    float product = (idx < N) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y + reg_a.z * reg_b.z + reg_a.w * reg_b.w) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    
    product = warp_reduce_sum(product);
    if (lane == 0) {
        product_smem[warp] = product;
    } 

    __syncthreads();

    product = (lane < NUM_WARPS) ? product_smem[lane] : 0.0f;
    if (warp == 0) {
        product = warp_reduce_sum<NUM_WARPS>(product);
    }
    if (tid == 0) {
        atomicAdd(y, product);
    }
}

template<size_t NUM_THREADS = 256>
__global__ void softmax_kernel(
    float *x, float *y, size_t N
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float reg_x = (idx + 1 < N) ? x[idx] : -FLT_MAX;
    float val_max = warp_reduce_max<NUM_THREADS>(reg_x);

    float reg_exp = (idx + 1 < N) ? fexp(reg_x - val_max) : 0.0f;
    
    float exp_val = reg_exp;
    float exp_sum = warp_reduce_sum<NUM_THREADS>(exp_val);

    if (idx < N) {
        y[idx] = exp_val / exp_sum;
    }
}

template<size_t NUM_THREADS = 256 / 4>
__global__ void softmax_vec_kernel(
    float *x, float *y, size_t N
) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + tid) * 4;
    
    float4 reg_x = cast<float4>(x[idx]);
    reg_x.x = (idx + 1 < N) ? reg_x.x : -FLT_MAX;
    reg_x.y = (idx + 1 < N) ? reg_x.y : -FLT_MAX;
    reg_x.z = (idx + 1 < N) ? reg_x.z : -FLT_MAX;
    reg_x.w = (idx + 1 < N) ? reg_x.w : -FLT_MAX;

    float val = reg_x.x;
    val = fmaxf(val, reg_x.y);
    val = fmaxf(val, reg_x.z);
    val = fmaxf(val, reg_x.w);
    float val_max = warp_reduce_max<NUM_THREADS>(val);

    float4 reg_exp;
    reg_exp.x = (idx + 1 < N) ? fexp(reg_x.x - val_max) : 0.0f;
    reg_exp.y = (idx + 1 < N) ? fexp(reg_x.y - val_max) : 0.0f;
    reg_exp.z = (idx + 1 < N) ? fexp(reg_x.z - val_max) : 0.0f;
    reg_exp.w = (idx + 1 < N) ? fexp(reg_x.w - val_max) : 0.0f;
    
    float exp_val = reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w;
    float exp_sum = warp_reduce_sum<NUM_THREADS>(exp_val);

    if ((idx + 3) < N) {
        float4 reg_y;
        reg_y.x = exp_val.x / exp_sum;
        reg_y.y = exp_val.y / exp_sum;
        reg_y.z = exp_val.z / exp_sum;
        reg_y.w = exp_val.w / exp_sum;
        cast<float4>(y[idx]) = reg_y;
    }
}

__global__ void sgemv_kernel(
    float *a, float *x, float *y, int M, int K
) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x;
    int lane = tx % WARP_SIZE;
    int m = bx * blockDim.y + ty;

    if (m < M) {
        float sum = 0.0f;
        const int NUM_WARPS = ((K + WARP_SIZE - 1) / WARP_SIZE + 4 - 1) / 4;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = (w * WARP_SIZE + lane) * 4;
            float4 reg_x = cast<float4>(x[k]);
            float4 reg_a = cast<float4>(a[m * K + k]);
            sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.z * reg_x.z + reg_a.w * reg_x.w); 
        }

        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            y[m] = sum;
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_kernel(
    float *a, float *b, float *c, const int M, const int N, const int K
) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ s_a[2][BK][BM + OFFSET];
    __shared__ s_b[2][BK][BN + OFFSET];

    float r_load_a[TM / 2];
    float r_load_b[TN / 2];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN];

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
        cast<float4>(r_load_a[0]) = cast<float4>(a[load_a_gmem_addr]);
        cast<float4>(r_load_b[0]) = cast<float4>(b[load_b_gmem_addr]);

        s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        cast<float4>(s_b[0][load_b_smem_k][load_b_smem_n]) = cast<float4>(r_load_b[0]);
    }

    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        cast<float4>(r_load_a[0]) = cast<float4>(a[load_a_gmem_addr]);
        cast<float4>(r_load_b[0]) = cast<float4>(b[load_b_gmem_addr]);

        #pragma unroll
        for (int tk = 0; tk < BK; ++tk) {
            cast<float4>(r_comp_a[0]) = cast<float4>(s_a[smem_sel][tk][ty * TM / 2]);
            cast<float4>(r_comp_a[4]) = cast<float4>(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            cast<float4>(r_comp_b[0]) = cast<float4>(s_b[smem_sel][tk][ty * TN / 2]);
            cast<float4>(r_comp_b[4]) = cast<float4>(s_b[smem_sel][tk][ty * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], tc[tm][tn]);
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        cast<float4>(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = cast<float4>(r_load_b[0]);

        __syncthreads();
    }

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        cast<float4>(r_comp_a[0]) = cast<float4>(s_a[1][tk][ty * TM / 2]);
        cast<float4>(r_comp_a[4]) = cast<float4>(s_a[1][tk][ty * TM / 2 + BM / 2]);
        cast<float4>(r_comp_b[0]) = cast<float4>(s_b[1][tk][tx * TN / 2]);
        cast<float4>(r_comp_b[4]) = cast<float4>(s_b[1][tk][tx * TN / 2 + BN / 2]);

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
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        cast<float4>(c[store_c_gmem_addr]) = cast<float4>(r_c[i][0]);
        cast<float4>(c[store_c_gmem_addr + BN / 2]) = cast<float4>(r_c[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        cast<float4>(c[store_c_gmem_addr]) = cast<float4>(r_c[i + TM / 2][0]);
        cast<float4>(c[store_c_gmem_addr + BN / 2]) = cast<float4>(r_c[i + TM / 2][4]);
    }
}

// Grid: dim3(ceil(W_out/TILE_W), ceil(H_out/TILE_H), N * C)
// Block: dim3(TILE_W, TILE_H, 1)
// flatten to [N, C * kH * kW, H_out * W_out]
template <int TILE_W, int TILE_H, int kW, int kH, int stride>
__global__ void im2col_kernel(
    const float* __restrict__ data_im, 
    float* __restrict__ data_col,
    int N, int C, int H, int W,
    int kH, int kW, int pad, int stride,
    int H_out, int W_out
) {
    int w_out_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;

    int c_in = blockIdx.z % C;
    int n    = blockIdx.z / C;

    constexpr int IN_TILE_W = (TILE_W * 4 - 1) * stride + kW;
    constexpr int IN_TILE_H = (TILE_H - 1) * stride + kH;
    constexpr int SMEM_SIZE = IN_TILE_W * IN_TILE_H;

    __shared__ float img_smem[SMEM_SIZE];

    // top-left
    int block_start_w_in = blockIdx.x * (blockDim.x * 4) * stride - pad;
    int block_start_h_in = blockIdx.y * blockDim.y * stride - pad;

    int flat_dix = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    int total_elements = IN_TILE_H * IN_TILE_W;
    for (int i = flat_dix; i < total_elements; i += num_threads) {
        int smem_w = i % IN_TILE_W;
        int smem_h = i / IN_TILE_W;

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
                        float val = img_smem[smem_h * IN_TILE_W + smem_w];

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
__global__ void col2im_kernel(
    const float* __restrict__ data_col, 
    float* __restrict__ data_im,
    int N, int C, int H, int W,
    int kH, int kW, int pad, int stride,
    int H_out, int W_out
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % C;
    int n = blockIdx.z / C;

    if (h >= H || w >= W) return;

    float val = 0.0f;

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
            int spatial_out_idx = h_out * H_out + w_out;

            int col_idx = batch_channel_offset + channel_idx * spatial_area + spatial_out_idx;

            val += data_col[col_idx];
        }
    }

    data_im[((n * C + c) * H + h) * W + w] = val;
}


// --------------------------------------
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
    GPU_CHECK(cudaGetLastError()); GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
#endif
}

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
    GPU_CHECK(cudaGetLastError()); GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
#endif
}

template<typename T>
void matmul_gpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];

    if constexpr (std::is_same_v<T, float>) {
        constexpr int BM = 128, BN = 128, BK = 8;
        constexpr int TM = 8, TN = 8;
        dim3 threads(16, 16);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        sgemm_kernel<BM, BN, BK, TM, TN, 1><<<blocks, threads>>>(
            A.data(), B.data(), C.data(), M, N, K
        );
    } else {
        constexpr int BM = 128, BN = 128, BK = 8;
        constexpr int TM = 8, TN = 8;
        dim3 threads(16, 16);
        dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
        sgemm_kernel<BM, BN, BK, TM, TN, 1><<<blocks, threads>>>(
            A.data(), B.data(), C.data(), M, N, K
        );
    }

#if defined(USE_CUDA)
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
#endif
}