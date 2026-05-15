# GPU Kernel Optimization Guide

## Kernel Inventory

All GPU kernels in `src/gpu_ops.cpp`, their optimization status, and techniques used.

## Element-Wise Kernels

### `elementwise_unary_kernel<T, Op>` (line 59)
**Status**: Optimal.  
**Techniques**: Coalesced global access, 1D grid-stride loop.  
Each thread processes one element with linear indexing. The vec variant `elementwise_unary_vec_kernel<float, Op>` (line 69) uses **float4 vectorization** — 4× fewer loads/stores, 128-byte aligned access.

### `elementwise_binary_kernel<T, Op>` (line 89)
**Status**: Optimal. Same as unary but with two inputs.

### `elementwise_binary_vec_kernel<float, Op>` (line 99)
**Status**: Optimal. Float4 vectorized with edge-case scalar fallback for remainder elements.

### `binary_kernel_strided<T, Op>` (line 120)
**Status**: Optimal for purpose. Handles arbitrary strides with dimension collapse optimization (TensorInfo + collapse_dims).

### `unary_kernel_strided<T, Op>` (line 144)
**Status**: Optimal for purpose. Same strided access pattern as binary version.

## Matrix Operations

### `sgemm_kernel` (line 259)
**Status**: Highly optimized.  
**Techniques**:
- **Shared memory tiling** (BM=128, BN=128, BK=8) — 3D register blocking (TM=8, TN=8)
- **Double buffering** — ping-pong between smem banks to overlap load/compute
- **Vectorized loads/stores** — float4 for all global memory access
- **FMA instructions** — `__fmaf_rn` for fused multiply-add
- **Bank conflict avoidance** — PAD=1 padding on shared memory columns
- **Dynamic smem bank selection** — `smem_sel_last = ((K+BK-1)/BK - 1) & 1` avoids reading uninitialized smem when K ≤ BK
- **Partial float4 handling** — scalar fallback when last tile doesn't fill a float4
- **Bounds-checked stores** — prevents OOB writes when M < BM or N < BN
- **Warp-level thread mapping** — `tid = ty*blockDim.x + tx` maps to load_a/b patterns

### `matmul_kernel_strided<T>` (line 442)
**Status**: Fallback kernel for non-contiguous matrices. Not optimized for throughput (used when sgemm_kernel cannot be used).

## Convolution Kernels

### `im2col_kernel<T>` (line 721)
**Status**: Well-optimized.  
**Techniques**:
- **Shared memory tiling** — loads input tile into smem, then reads from smem in output loop
- **Cooperative load** — all threads load the tile cooperatively
- **Dynamic shared memory** — `extern __shared__` with size computed from threads × kernel dims
- **Float4 output stride** — writes 4 consecutive output columns per thread

### `col2im_kernel<T>` (line 796)
**Status**: Not optimized (gather pattern is inherently hard to tile). Each thread reads from scattered global memory positions. For kH/kW ≤ 3, overhead is minimal.

### `conv2d_forward_kernel_strided<T>` (line 912)
**Status**: Fallback for non-contiguous tensors. Uses per-thread index decomposition.

### `conv2d_backward_input_kernel<T>` (line 987)
**Status**: Not optimized. Each thread computes one input gradient element by iterating over (c_out, kH, kW). Could use shared memory tiling for large kH/kW but current kH=kW=3 makes it acceptable.

### `conv2d_backward_weight_kernel<T>` (line 1029)
**Status**: Uses grid-stride loop + `atomicAdd`. High contention for large C_out/C_in. Shared memory reduction would require ~72KB for worst case (exceeds 48KB limit). Acceptable for current model sizes.

### `conv2d_backward_bias_kernel<T>` (line 1068)
**Status**: Simple reduction. Each thread sums over (N, H_out, W_out) for one output channel.

## Reduction Kernels

### `reduce_axis_kernel` (line 546)
**Status**: Optimal. Each thread processes one output element with a loop over the reduction axis.

### `reduce_axis_inner_kernel<BLOCK_SIZE>` (line 571)
**Status**: Well-optimized.  
**Techniques**:
- **Shared memory reduction** — tree reduction in smem (warp reduce → shared → final warp)
- **Cooperative loading** — `for (int i = tid; i < reduce_size; i += BLOCK_SIZE)` with block reduction
- **`warp_reduce_sum`** — shuffle-based warp reduction (no shared memory for warp-level)

## Transpose Kernels

### `mat_transpose_kernel<T>` (line 643)
**Status**: Fixed.  
**Techniques**:
- **Shared memory tiling** (WARP_SIZE × (WARP_SIZE+PAD)) — PAD=1 avoids bank conflicts
- **Diagonal-style coalesced write** — reorganizes thread-to-output mapping for coalesced global writes
- **`__syncthreads` divergence fix** — all threads participate in load with zero-fill

### `mat_transpose_vec_kernel` (line 670)
**Status**: Fixed. Float4 version transposes 4 rows per thread.

## Pooling Kernels

### `maxpool2d_kernel<T>` (line 1226)
**Status**: Optimal. Each thread computes one output element.

### `maxpool2d_kernel_strided<T>` (line 1264)
**Status**: Optimal for strided case.

### `maxpool2d_backward_kernel<T>` (line 1311)
**Status**: Optimal. Uses `atomicAdd` for gradient scatter.

## Miscellaneous

### `add_bias_kernel<T>` (line 700)
**Status**: Optimal. Simple element-wise add with broadcasting.

### `copy_kernel_strided<T>` (line 1433)
**Status**: Optimal for strided copy.

### `cross_entropy_fwd_kernel<T>` (line 1551)
**Status**: Well-optimized. Uses shared memory tree reduction for softmax denominator and loss.

### `cross_entropy_bwd_kernel<T>` (line 1609)
**Status**: Acceptable. Each thread recomputes softmax for its element.

### `mean_kernel<T>` (line 1579)
**Status**: Simple shared memory reduction.

### `adam_step_kernel<T>` (line 1660)
**Status**: Element-wise fused kernel. Each thread updates one parameter element through all Adam steps.

## Key Fixes Applied

| Kernel | Bug/Optimization | Impact |
|--------|-----------------|--------|
| `mat_transpose_kernel` | `__syncthreads()` inside divergent branch (UB) | Random hangs/crashes |
| `mat_transpose_kernel` | `out_x = ty + bid_y` missing `* STRIDE` | Wrong output for multi-block (K>32) |
| `mat_transpose_vec_kernel` | Same two bugs | Same |
| `sgemm_kernel` | Hardcoded `s_a[1]` when K ≤ BK | Read uninitialized data |
| All vec kernels via `cast()` | Stack garbage read + write no-op | All element-wise ops broken |
