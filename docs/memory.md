# GPU Memory Fault Root-Cause Analysis

## Fault Signature

```
Memory access fault by GPU node-1 (Agent handle: 0x55d481dfc090)
on address 0x7f6b3d659000. Reason: Page not present or supervisor privilege.
```

The faulting address `0x7f6b3d659000` is in the **host** address range (`0x7f…`),
not GPU device memory. A GPU kernel was dereferencing a pointer that resolved
to host-side unmapped memory.

## Root Cause #1: `cast()` write-failure (PRIMARY)

**File**: `src/gpu_ops.cpp`, line 12-15 (original)

```cpp
template<typename T, typename V>
__device__ constexpr T cast(V val) {           // BUG: takes value, not pointer
    return reinterpret_cast<T*>(&val)[0];      // &val = stack address, not data
}
```

Two critical flaws:

1. **By-value parameter**: `&val` takes the address of the stack copy, not the
   original data. `cast<float4>(a[idx])` reads 1 float from memory, then 3
   garbage floats from the stack.

2. **Return by value**: `cast<float4>(&c[idx]) = reg_c` assigns to a
   **temporary** return value — the store to GPU memory silently fails.

This breaks EVERY vectorized kernel:
- `elementwise_unary_vec_kernel` (ReLU, sigmoid, tanh, exp, log, pow2)
- `elementwise_binary_vec_kernel` (add, sub, mul, div, ReLUGrad)
- `sgemm_kernel` (matmul, conv2d GEMM)

### Fix

Changed to pointer parameter + reference return:

```cpp
template<typename T, typename V>
__device__ constexpr T& cast(const V* val) {
    return reinterpret_cast<T*>(const_cast<V*>(val))[0];
}
```

All 23 call sites updated from `cast<float4>(arr[idx])` to `cast<float4>(&arr[idx])`.

## Root Cause #2: `sgemm_kernel` Missing Bounds Checks

**File**: `src/gpu_ops.cpp`, lines 285-371

The tiled SGEMM kernel assumes tiles fit within matrix bounds (M ≥ 128, N ≥
128). When M or N is smaller, threads load/store beyond allocations.

**Example**: Conv3 backward input (M=288, N_mat=196). Blocks cover rows [0,384)
but M=288. Threads computing rows ≥288 write past the 288×196 output buffer,
into unmapped host-address-range memory → "Page not present" fault.

### Fix

Added bounds checks at three points in `sgemm_kernel`:

1. **Initial global loads** — check `load_a_gmem_m < M && load_a_gmem_k+3 < K`
   and `load_b_gmem_k < K && load_b_gmem_n+3 < N`; zero-fill on miss.

2. **Main loop global loads** — same bounds checks.

3. **Store operations** — check both `store_c_gmem_m < M` and
   `store_c_gmem_n+3 < N` (and `+BN/2+3 < N` for the split store) before writing.

Additionally, added launcher-level fallback in `matmul_gpu`: if M < 128 or N <
128, use `matmul_gpu_strided` (which has per-element bounds checking) instead of
`sgemm_kernel`. Same fallback added in `conv2d_gpu` and
`conv2d_backward_input_gpu`.

## Root Cause #3: `reduce_axis_inner_kernel` Wrong Offset

**File**: `src/gpu_ops.cpp`, line 490 (original)

```cpp
int row_offset = row * outer_size;  // BUG: should be row * reduce_size
```

For a contiguous [outer × reduce] matrix, row `r` starts at index `r * reduce`,
not `r * outer`. This produces wrong sum results (overlapping/wrong data) but
does not cause GPU memory faults since the contiguous buffer is large enough.

### Fix

```cpp
int row_offset = row * reduce_size;
```

## Root Cause #4: `conv2d_forward_kernel_strided` Bias Stride

**File**: `src/gpu_ops.cpp`, line 841 (original)

```cpp
T val = (bs != nullptr) ? bs[c_out * info_wt.strides[0]] : (T)0.0;
```

Uses weight's stride (`info_wt.strides[0]` = `C_in*kH*kW`, e.g. 288) to index
the bias tensor. Bias is 1D `[C_out]` with stride `[1]`. Accesses `bs[c_out *
288]` — far beyond the C_out-element allocation.

This path is dormant in normal training (only hit for non-contiguous tensors)
but would produce wrong results or a fault if activated.

### Fix

```cpp
T val = (bs != nullptr) ? bs[c_out] : (T)0.0;
```

## The MemoryPool Factor

The `MemoryPool` caches freed GPU pointers by exact byte size and reissues them.
This means an out-of-bounds write can corrupt a *different* tensor's data that
happened to receive the same-size allocation. In the fault observed, the
out-of-bounds access landed on unmapped memory rather than another allocation,
producing the clean "Page not present" error rather than silent data corruption.

## Summary of All Changes

| Bug | Location | Impact | Fix |
|-----|----------|--------|-----|
| `cast()` stack-read / write-as-temporary | `src/gpu_ops.cpp:12-15` | All vec kernels broken | Pointer param + ref return |
| `sgemm_kernel` OOB loads/stores | `src/gpu_ops.cpp:285-371` | GPU memory fault | Bounds checks + launcher fallback |
| `reduce_axis_inner_kernel` wrong offset | `src/gpu_ops.cpp:490` | Wrong sum results | `outer_size` → `reduce_size` |
| `conv2d_forward_kernel_strided` bias stride | `src/gpu_ops.cpp:841` | OOB bias access (dormant) | `info_wt.strides[0]` → direct index |
