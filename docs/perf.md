# Performance Analysis & Optimization History

## Current State

MNIST training (batch_size=64, 3-conv + 2-linear network):
- **Before optimization** (Debug, `-O0`, per-sample sgemm): **~1366s/epoch**
- **After optimization** (Debug, `-O2`, tiled batched GEMM, fused kernels, pinned memory): **~72s/epoch**
- **Speedup: ~19×**
- Final accuracy: **98.5-98.8%** test accuracy

## Bottleneck Analysis (Root Causes Found)

### Root Cause #1: `-O0` Compilation for GPU Device Code (BIGGEST)

**Problem**: Both CPU code and HIP device code compiled with `-O0` (no optimization). On ROCm, `-O0` means GPU kernels have no register optimization → register spilling to slow local memory, no instruction scheduling, no inlining → **50-100× slower kernel execution**.

**Evidence**: 
- `compile_commands.json` showed `-O0` for all compilation units
- The HIP compilation (`-x hip`) had NO optimization flags (defaults to `-O0`)
- Perf profiling showed ~57ms CPU launch time per batch → actual GPU compute still dominated

**Fix**: Changed `CMakeLists.txt`:
```cmake
# CPU code: -O0 → -O2
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2 -g ...")
# HIP device code: added -O2
set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG} -O2 -g")
```

**Impact**: From ~5× to ~10× speedup (Debug mode was unusably slow before).

### Root Cause #2: Per-Sample sgemm Loop (384 launches → 6)

**Location**: `conv2d_gpu()` and `conv2d_backward_input_gpu()` in `src/gpu_ops.cpp`

**Problem**: Conv2D forward and backward each looped over N=64 batch items, launching a separate `sgemm_kernel` per sample. At 64 samples × 3 conv layers × 2 (fwd+bwd) = **384 kernel launches per iteration**. On ROCm, kernel launch latency is ~300μs each, so 384 × 300μs = **115ms of launch overhead**.

```cpp
for (size_t n = 0; n < N; ++n) {
    sgemm_kernel<<<mm_blocks, mm_threads>>>(weight, col_ptr, out_ptr, M, N_mat, K);
}
```

**Initial Attempt (failed)**: Replaced with naive `batched_gemm_kernel` (one thread per output element, triple loop). This had **terrible memory coalescing** — adjacent threads in a warp accessed global memory with 64-256 byte strides. Uncoalesced accesses on ROCm cost ~100 cycles vs ~4 cycles for coalesced → **10-50× slower compute**.

**Real Fix**: Created a **tiled batched `batched_sgemm_kernel`** with shared memory (128×128 tiles, BK=8, TM/TN=8 register blocking, 3D grid with batch as Z dimension):
```cpp
dim3 batched_blocks((N_mat+127)/128, (M+127)/128, N);
batched_sgemm_kernel<128,128,8,8,8><<<batched_blocks, batched_threads>>>(
    weight.data(), d_data_col, output.data(), M, N_mat, K, K*N_mat, M*N_mat);
```

**Impact**: 384 launches → 6 launches. Launch overhead: ~115ms → ~1.8ms. ~2-3× total training speedup.

### Root Cause #3: 46 Redundant `hipDeviceSynchronize()` Calls

**Location**: Every GPU function in `src/gpu_ops.cpp` (46 call sites)

**Problem**: Every function called `hipDeviceSynchronize()` after every kernel launch. On the default stream, GPU kernels execute in-order automatically — explicit syncs are redundant. Each sync:
1. Blocks the CPU until GPU drains all work
2. Prevents CPU-GPU overlap (kernel launches stack up on the command queue)
3. Costs ~50-200μs each on ROCm

**Impact**: With ~140 syncs per iteration (forward + backward + optimizer), total sync overhead: ~140 × 100μs = ~14ms → ~20% of CPU launch time.

**Fix**: Removed ALL `hipDeviceSynchronize()` / `cudaDeviceSynchronize()` calls from all GPU functions. Kept `hipGetLastError()` for launch error detection (catches invalid launch parameters synchronously).

**Impact**: Reduced CPU-side wait time, enabled overlap between CPU launch work and GPU execution.

### Root Cause #4: Decomposed Adam Step (100 launches → 1)

**Location**: `Adam::step()` in `include/Optimizer.hpp`

**Problem**: The Adam optimizer decomposed the parameter update into ~10 separate elementwise operations (add, mul, sqrt, div, sub) per parameter group. With ~10 parameter groups, that's **~100 kernel launches + syncs per step**.

A fused `adam_step_gpu` kernel already existed at `src/gpu_ops.cpp` but was DEAD CODE — never called from the optimizer.

**Fix**: Wired `adam_step_gpu` into `Adam::step()`. Single fused kernel does all Adam operations in one launch.

**Impact**: 100+ launches → 1. Saves ~3ms per step.

### Root Cause #5: DataLoader `stack` vs `concat` (5D Tensors)

**Location**: `default_collate()` in `include/Dataset.hpp`

**Problem**: `default_collate` used `stack()` instead of `concat()`. Each MNIST image was [1, 1, 28, 28] (from slice). `stack()` adds a NEW dimension, producing [N, 1, 1, 28, 28] — a **5D tensor**. Conv2D read `H = shape()[2] = 1` (the inserted dimension), which cascaded through subsequent conv layers producing H=0, triggering unsigned wraparound in the maxpool formula.

**Fix**: Changed to `concat()` which preserves existing dimensions.

### Root Cause #6: Buffer Overflows in Transpose Kernels

**Location**: `mat_transpose_kernel` and `mat_transpose_vec_kernel` in `src/gpu_ops.cpp`

**Problem**: 
- `mat_transpose_kernel`: The diagonal-style write computes `out_y = global_x + local_y / STRIDE` which can exceed `col` when the matrix row count is not a multiple of WARP_SIZE. **448 bytes corrupted per training iteration** across all 3 conv layers.
- `mat_transpose_vec_kernel`: Float4 loads used guard `global_y*4 < row` instead of `global_y*4 + 3 < row`, reading 1-3 floats past buffer when `row % 4 ≠ 0`.

**Impact**: Silent GPU memory corruption accumulates across iterations, eventually producing wrong gradients or NaN → crashes. This caused the "late in training" crash pattern.

**Fix**: Added output bounds check (`if (out_y < col && out_x < row)`) in `mat_transpose_kernel`. Fixed load guard in `mat_transpose_vec_kernel`.

### Root Cause #7: Missing Kernel Launch Guards (UB on zero blocks)

**Location**: Multiple kernel launch sites in `src/gpu_ops.cpp`

**Problem**: 9+ kernel launch sites computed `blocks = (total + threads - 1) / threads` without checking `total == 0`. If a zero-dim tensor reached these sites, blocks=0 → invalid grid dimension → privileged instruction crash.

**Fix**: Added `if (blocks == 0) return;` (or proper empty-tensor return) at all launch sites.

### Root Cause #8: Unsigned Wraparound in Spatial Formulas

**Location**: `max_pool2d_gpu()`, `conv2d()` and related functions

**Problem**: `size_t H_out = (H + 2*padding - k) / stride + 1` — when `H + 2*padding < k`, the subtraction wraps to `SIZE_MAX`, producing a huge H_out that overflows the total_elements computation to 0.

**Fix**: Created `safe_out_size()` helper:
```cpp
inline size_t safe_out_size(size_t input_dim, size_t pad, size_t kernel, size_t stride) noexcept {
    if (input_dim + 2 * pad < kernel) return 0;
    return (input_dim + 2 * pad - kernel) / stride + 1;
}
```
Applied to all 10+ spatial formula locations.

### Root Cause #9: `slice()` Zero-Dim Defect

**Location**: `Tensor::slice()` in `include/Tensor.hpp`

**Problem**: Validation checked `start > end` but not `start == end`. Calling `slice(dim, i, i)` produced a tensor with a zero dimension.

**Fix**: Changed `start > end` → `start >= end`.

## Performance Optimizations Applied

| # | Optimization | File(s) | Impact | Est. Speedup |
|---|-------------|---------|--------|-------------|
| 1 | **`-O0` → `-O2`** for CPU + HIP | CMakeLists.txt | GPU kernel perf 10-50× better | ~5-10× |
| 2 | **Tiled batched_sgemm_kernel** — 3D grid, shared memory tiling | gpu_ops.cpp | 384 → 6 launches | ~2-3× |
| 3 | **Removed 46 `hipDeviceSynchronize()`** — redundant on default stream | gpu_ops.cpp | Eliminates ~14ms sync/iter | ~20% |
| 4 | **Fused `adam_step_gpu`** — 1 launch instead of 100 decomposed | Optimizer.hpp | ~3ms/iter | Moderate |
| 5 | **Fix DataLoader `stack` → `concat`** — was creating 5D tensors | Dataset.hpp | **CRASH FIX** | Critical |
| 6 | **Fix transpose buffer overflow** — 448 bytes/iter corruption | gpu_ops.cpp | **CORRECTNESS** | Critical |
| 7 | **Fix unsigned wraparound** — `safe_out_size` helper | TensorOps.hpp, gpu_ops.cpp | **CRASH FIX** | Critical |
| 8 | **Add zero-block launch guards** — 9+ sites | gpu_ops.cpp | **CRASH FIX** | Critical |
| 9 | **Fix `slice()` zero-dim validation** — `start >= end` | Tensor.hpp | **CORRECTNESS** | Low |
| 10 | **Pinned host memory** — `hipHostMalloc` for CPU tensors | MemoryPool.hpp | 2-5× faster memcpy | Moderate |
| 11 | **Fuse bias into batched_sgemm** — saves 3 launches | gpu_ops.cpp | ~1ms/iter | Low-Med |
| 12 | **Fused backward kernel** — sgemm+col2im in one | gpu_ops.cpp | Eliminates 57MB intermediate | Moderate |
| 13 | **Double-buffered DataLoader** — overlap CPU batch construction with GPU compute | mnist.cpp | ~0.5ms hidden/iter | Low |

## Performance Pipeline Breakdown

Profiling data (Debug, -O2, steady-state after warmup):

| Phase | Time per batch | % of total | Notes |
|-------|---------------|-----------|-------|
| **Data copy** `to(gpu)` | **0.087ms** | 0.1% | Pinned memory helps |
| **Forward pass** | **15.3ms** | 27% | ~15 ops, batched GEMM, fused bias |
| **Backward pass** | **42.4ms** | 73% | ⚠️ Dominant cost — fused backward kernel |
| **Optimizer (Adam)** | **0.024ms** | 0.1% | Fused kernel |
| **GPU execution wait** | ~28ms | — | Hidden inside `loss.to(cpu)` sync |
| **CPU launch overhead** | ~57ms | — | HIP runtime + launch latency |

Remaining bottlenecks are dominated by **GPU kernel execution time** (~3ms per kernel on ROCm), likely due to:
- Small grid sizes not saturating GPU compute units
- Custom kernels (im2col, col2im) with scatter memory patterns
- HIP runtime overhead per launch

## Fixes Applied (Chronological)

### Phase 1: Crash Fixes (Correctness)
1. `default_collate`: `stack` → `concat` (5D tensor bug)
2. `safe_out_size` helper for all spatial formulas (unsigned wraparound)
3. Zero-block launch guards at all kernel sites (blocks=0 UB)
4. `slice()` zero-dim validation fix
5. MaxPool2D input validation (dim ≥ 4, spatial size check)

### Phase 2: Correctness (GPU Memory Corruption)
6. `mat_transpose_kernel` output bounds check
7. `mat_transpose_vec_kernel` load guard fix
8. `col2im_kernel` spatial index fix (`h_out * H_out` → `h_out * W_out`)

### Phase 3: Performance Optimization
9. CPU + HIP `-O0` → `-O2`
10. Removed 46 `hipDeviceSynchronize()` calls
11. Tiled `batched_sgemm_kernel` (3D grid, shared memory tiling)
12. Fused `adam_step_gpu` (1 launch vs 100+ decomposed)
13. Pinned host memory (`hipHostMalloc`)
14. Bias fusion into `batched_sgemm_kernel`
15. Fused backward kernel (sgemm + col2im in one)
16. Double-buffered DataLoader (pre-load next batch)

## Remaining Potential Optimizations

| # | Idea | Complexity | Est. Gain | Notes |
|---|------|-----------|-----------|-------|
| 1 | **Use rocBLAS** instead of custom sgemm | Medium | 20-40% | rocBLAS has arch-tuned kernels |
| 2 | **rocBLAS batched GEMM** for all batch GEMMs | Medium | 20-30% | Single API call, no custom kernel |
| 3 | **Fuse im2col + sgemm** — eliminate d_data_col intermediate | High | 10-20% | Complex thread mapping |
| 4 | **Use HIP streams** for async data transfer | Low | 5-10% | Overlap data loading with compute |
| 5 | **Benchmark different tile sizes** per layer | Low | 5-15% | BM=64 vs 128, BK=16 vs 8 per layer type |

## Summary Table

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Epoch time (Debug) | 1366s | ~72s | **~19×** |
| Per-batch time | ~1.46s | ~82ms | **~18×** |
| GPU kernel launches/iter | ~500 | ~30 | **~17×** |
| `hipDeviceSynchronize`/iter | ~142 | ~0 | Eliminated |
| Adam launches/iter | ~100 | ~1 | **~100×** |
| sgemm launches/conv | 64 | 1 | **~64×** |
| Data copy | pageable | pinned | 2-5× faster |
| Compilation flags | `-O0` | `-O2` | 5-10× kernel perf |
