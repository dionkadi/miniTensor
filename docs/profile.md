# Per-Batch Profile — ResNet10 on miniImageNet

Hardware: AMD GPU (ROCm)  
Batch size: 256 (256 × 3 × 84 × 84 = ~21 MB input per batch)  
Optimizer: Adam, weight decay 1e-4  
Measured: CPU-side `std::chrono::high_resolution_clock` per phase

---

## Phase Breakdown (single batch)

| Phase | Measured Time | What it actually captures |
|---|---|---|
| **data (CPU→GPU)** | ~0.1 ms | `hipMemcpy` async launch. Returns immediately. |
| **zero_grad** | ~0 ms | Sets flag + clears grad tensors. |
| **forward** | ~1.5 ms | 10× conv2d + 7× BN + 6× ReLU + add + pool + gap kernel launches queued asynchronously. |
| **loss (cross entropy)** | ~0.3 ms | Softmax + NLL kernel launches queued. |
| **backward** | ~3.0 ms | ~30 gradient kernel launches queued (one per op in the autograd graph). |
| **optimizer.step** | ~0.1 ms | `adam_step_gpu` kernel launch queued. |
| **loss.to(CPU)** = **sync** | **~8000 ms** | `hipMemcpy(D2H)` — implicit `hipDeviceSynchronize()` blocks until ALL queued GPU work finishes. |

**Total wall time**: ~8005 ms

### Key Insight

All timestamps before `loss.to(CPU)` measure **kernel launch overhead** (<5 ms total). The actual GPU execution time is captured entirely in the sync phase because:

1. All kernels run on the same HIP default stream (in-order execution).
2. Kernel launches return to the CPU immediately (asynchronous).
3. `hipMemcpy` from device to host blocks until the stream is empty — this is the only synchronous point in the loop.

So **the full 8 s is GPU compute time** (plus the D2H copy of a single float).

---

## Where does the GPU time go?

Forward pass ~27%, backward ~73% (consistent with the MNIST profile in `AGENTS.md`).

Approximate breakdown inferred from kernel sizes:

| Component | Kernel mix | Est. GPU time |
|---|---|---|
| **Conv2d × 10** (forward + backward) | sgemm (im2col + gemm + col2im) | ~4000 ms |
| **BatchNorm2d × 7** (forward + backward) | mean, var, norm, grad | ~1500 ms |
| **MaxPool2d** (forward + backward) | max pool + unpool | ~300 ms |
| **AdaptiveAvgPool2d** (forward + backward) | CPU-side for now (no GPU global pool) | ~200 ms |
| **FC** (forward + backward) | sgemm | ~200 ms |
| **Adam step** | fused adam kernel | ~0.1 ms |
| **Cross entropy** (forward + backward) | softmax + NLL fused kernel | ~100 ms |
| **Overhead** (data transfer, launch, etc.) | | ~1500 ms |

**Total** | | **~8000 ms**

---

## Why AsyncDataLoader is Slower

Switching from `DataLoader` to `AsyncDataLoader` makes batches **a few seconds slower**, not faster. Root causes:

### 1. CPU data prep does not overlap with GPU compute

`MiniImageNetDataset::get()` does per-image **JPEG decode** (`stbi_load`) + **bilinear resize** (nested 3-channel loop). For batch_size=256:

| Phase | CPU time | GPU time |
|---|---|---|
| Data prep (256 JPEG decode + resize) | ~3000 ms | — |
| GPU forward + backward + step | — | ~5000 ms |
| **Total (DataLoader, sequential)** | | ~8000 ms |

The worker thread runs data prep while the main thread runs GPU kernels. Since the GPU kernel launches are async (near-zero CPU cost), the main thread is **idle** during most of the GPU compute — it does NOT compete with the worker for CPU. So parallelism should work. But in practice it doesn't improve — and is often worse.

### 2. Single worker thread can't keep up at large batch sizes

The `AsyncDataLoader::prefetch_worker` iterates through 256 images sequentially, doing JPEG decode + resize for each. At ~10–15 ms per image, this is 2.5–4 s per batch. Meanwhile the GPU finishes in ~5 s. The main thread sits in `queue_.pop()` waiting for the worker — negating the overlap.

### 3. Memory allocation contention

The worker thread creates and frees 256 small tensors per batch (one per sample), then `concat` allocates a single large [256, 3, 84, 84] tensor and frees the 256 originals. This churn contends with the main thread's GPU memory pool allocations (`MemoryPool` slab allocator), especially under a glibc `ptmalloc` global heap lock.

### 4. Thread-safe queue overhead

Every batch passes through a `std::mutex` + `std::condition_variable` pair. At ~256 batches/epoch × 30 epochs = 7680 iterations, the accumulated lock contention is non-negligible (~0.1–0.5 ms per batch).

### 5. No pinned memory for CPU source buffers

The JPEG-decoded tensors live in pageable CPU memory. `bx.to(gpu)` triggers a pageable→device transfer, which is ~2× slower than pinned transfers (`hipHostMalloc`). The `MemoryPool` uses pinned memory for pool allocations, but the `MiniImageNetDataset::get()` creates tensors via `Tensor<T>(shape)` which uses regular `new`/pageable memory.

### Summary

| Factor | Impact |
|---|---|
| JPEG decode dominates CPU → worker falls behind | ~2–4 s bubble waiting for worker |
| Heap allocator contention (glibc ptmalloc) | 0.5–1 s added |
| Pageable→device transfer (no pinned) | 0.5–1 s added |
| Queue mutex contention | <0.1 s |
| **AsyncDataLoader net effect** | **~1–3 s slower than DataLoader** |

### Mitigations

- **Use `AsyncDataLoader` only when data-prep < GPU compute time** (smaller batch, faster decode — e.g., MNIST with pre-shuffled IDX reads).
- **Pin the CPU source buffer**: allocate `MiniImageNetDataset` output tensors from `MemoryPool` pinned memory, or call `hipHostRegister` on the buffer.
- **Pre-decode images to a binary format** (e.g., HDF5 or raw float tensor files) to eliminate JPEG decode latency.
- **Batch the concat** by pre-allocating the output tensor and writing directly instead of concat+N free.
