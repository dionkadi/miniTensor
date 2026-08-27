# miniTensor

A from-scratch C++20 tensor library with automatic differentiation, a PyTorch-style
`nn` module API, and CPU + GPU (CUDA/ROCm) backends. Built as an educational project:
no BLAS, no cuDNN, no external tensor library underneath — the kernels, the autograd
engine, and the training stack are all hand-written.

## Features

- **Tensors & views** — `Tensor<T>` with row-major layout, strides, NumPy-style
  broadcasting, and views (`slice`, `transpose`, `reshape`, `expand`) with
  version-counter integrity checking for in-place mutation
- **Autograd** — 26 backward nodes, topological-sort `backward()`, `NoGradGuard`,
  supports branching (residual connections)
- **nn API** — `Linear`, `Conv2D`, `BatchNorm2d` (full training-mode backward),
  `LayerNorm`, `Dropout`, `Sequential`, activations (`ReLU`, `Tanh`, `Sigmoid`, ...),
  plus a functional API in namespace `F`
- **Training stack** — `SGD`, `Adam`, `AdamW`, `GradientScaler` (mixed-precision
  infrastructure), learning-rate schedulers (`StepLR`, `CosineAnnealingLR`,
  `ReduceLROnPlateau`, ...)
- **Losses** — MSE, cross-entropy with optional label smoothing (fused softmax+NLL
  kernel on GPU)
- **Data** — `AsyncDataLoader` with background-thread batch prefetching
- **GPU backends** — CUDA (`sm_75`, `sm_80`) and ROCm/HIP (`gfx1102`, `gfx1103`);
  power-of-2 binned memory pool, RAII `GpuStream`/`GpuEvent`, fused kernels
  (cross-entropy, BN+ReLU, Adam step), CUDA/HIP graph capture via `GraphExecutor`
- **Serialization & export** — binary state dict save/load, ONNX-like graph
  DAG export/import, graph-mode JIT tracing/replay (`GraphMode`)

## Requirements

- CMake 4.0+
- clang++ (the ROCm fork when building HIP)
- C++20 (`-std=c++20` is forced; debug builds use `-Wall -Wextra -Werror`)
- System packages: **GTest** and **fmt** (`find_package(GTest REQUIRED)` /
  `find_package(fmt REQUIRED)` — install them via your distro before configuring)

## Build

```sh
cmake -B build -G Ninja                        # auto-detect GPU; falls back to CPU-only
cmake -B build -G Ninja -DUSE_ROCM=ON          # AMD GPU (gfx1102, gfx1103)
cmake -B build -G Ninja -DUSE_CUDA=ON          # NVIDIA GPU (sm_75, sm_80)
cmake --build build
```

- CPU-only builds need no GPU driver: `GPU_CHECK(call)` expands to nothing and GPU
  functions return early.
- Debug builds are usable for performance work (host `-O2 -g`; HIP device `-O3 -g`
  — HIP at `-O0` is 50–100× slower).
- Don't enable `USE_CUDA` and `USE_ROCM` at the same time (CMake errors).
- `CMakePresets.json` exists but builds into `build/debug` / `build/release`;
  `clangd` reads `build/compile_commands.json`, so prefer the plain `cmake -B build` flow.

## Quick start

```cpp
#include <iostream>
#include "Tensor.hpp"
#include "TensorOps.hpp"

int main() {
    // Autograd: y = 2x² + x, d(y)/dx = 4x + 1
    Tensor<float> x({2, 3});
    x.set_requires_grad(true);

    Tensor<float> y = pow2(x) * 2.0f + x;
    Tensor<float> loss = sum(sum(y, 1, true), 0, true);  // reduce to a scalar
    loss.backward();

    std::cout << x.grad().data()[0] << "\n";  // 1.0
    return 0;
}
```

Training a small MLP:

```cpp
#include <iostream>
#include "Tensor.hpp"
#include "Module.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"

int main() {
    Sequential<float> model({
        std::make_shared<Linear<float>>(2, 16),
        std::make_shared<Tanh<float>>(),
        std::make_shared<Linear<float>>(16, 1),
        std::make_shared<Sigmoid<float>>()
    });
    SGD<float> optim(model.parameters(), 0.1f);

    Tensor<float> X({4, 2});
    X.fill(1.0f);
    Tensor<float> Y({4, 1});
    Y.fill(0.0f);

    for (int epoch = 0; epoch < 100; ++epoch) {
        Tensor<float> preds = model(X);
        Tensor<float> loss = mse_loss(preds, Y);
        optim.zero_grad();
        loss.backward();
        optim.step();
        std::cout << "epoch " << epoch << " loss " << loss.data()[0] << "\n";
    }
    return 0;
}
```

Using the GPU:

```cpp
Tensor<float> a({64, 64}, Device{DeviceType::CUDA});   // allocate on GPU
Tensor<float> b = a.to(Device{DeviceType::CPU});       // copy to CPU
```

Notes:

- `cross_entropy(logits, targets, smoothing)` expects **one-hot targets** of shape
  `[N, C]`, not class indices (PyTorch style).
- He initialization uses a fixed seed (`42`) for reproducibility.

## Tests

```sh
./build/tensor_tests
```

Only **one test file is compiled at a time** — switch by editing the commented-out
list inside `add_executable(tensor_tests ...)` in `CMakeLists.txt`, then rebuild:

| File | What it exercises |
|---|---|
| `test/benchmark.cpp` | GPU kernel benchmark suite (currently active) |
| `test/resnet.cpp` | ResNet50 on miniImageNet (Bottleneck blocks, label smoothing) |
| `test/mnist.cpp` | MNIST CNN training |
| `test/engien_test.cpp` | Autograd engine GTest suite (CPU+GPU, parametrized; filename misspelling is intentional) |
| `test/{tensor,spatial,func,dataset}_test.cpp` | GTest suites (GPU sections behind `USE_CUDA`/`USE_ROCM`) |
| `test/view_test.cpp`, `test/nn_test.cpp`, `test/scheduler_test.cpp` | Standalone `main()` tests |
| `test/linear_regression_test.cpp` | On disk but not wired into CMake |

`test/tensor_test.cpp` and `src/gpu_ops.cpp` are compiled with the CUDA/HIP language
based on the selected backend. MNIST data lives in `data/`.

## Benchmarks

`test/benchmark.cpp` (currently the active test target) runs the GPU kernel suite:
build it, then `./build/tensor_tests`. The embedded correctness checks (v2 vs strided
matmul) pass. Numbers below are from a single run — device clocks/power state can
move individual rows ±20% (especially the small ones), so treat them as indicative.

Reference run: AMD Radeon RX 7600M XT (gfx1102) via ROCm/HIP, default build flags
(host `-O2`, HIP device `-O3`):

```
==========================================================================================
Kernel                      Problem                         Time(ms)      GB/s    TFLOPS
------------------------------------------------------------------------------------------
binary/add                  1048576 elems                      0.032     399.4         -
binary/add                  4194304 elems                      0.222     226.8         -
binary/add                  16777216 elems                     0.856     235.2         -
binary/mul                  1048576 elems                      0.032     394.5         -
binary/mul                  4194304 elems                      0.222     226.7         -
unary/relu                  1048576 elems                      0.024     354.0         -
unary/sigmoid               1048576 elems                      0.029     285.1         -
unary/tanh                  1048576 elems                      0.029     293.9         -
unary/exp                   1048576 elems                      0.025     340.4         -
unary/sqrt                  1048576 elems                      0.025     336.1         -
unary/log                   1048576 elems                      0.024     351.9         -
add_relu                    1048576 elems                      0.032     397.2         -
add_relu                    4194304 elems                      0.222     226.6         -
matmul                      32x32 * 32x32                      0.016       0.8      0.00
matmul                      63x4096 * 4096x4096                4.972      13.9      0.43
matmul                      512x512 * 512x512                  0.349       9.0      0.77
matmul                      1024x1024 * 1024x1024              2.513       5.0      0.85
matmul                      2048x2048 * 2048x2048             17.422       2.9      0.99
matmul                      4096x4096 * 4096x4096            125.514       1.6      1.10
matmul                      1x4096 * 4096x4096                 3.526      19.0      0.01
matmul                      4096x1 * 1x4096                    1.808      37.1      0.02
matmul_bk16_vec             512x512 * 512x512                  0.239      13.2      1.12
matmul_bk16_vec             1024x1024 * 1024x1024              1.702       7.4      1.26
matmul_bk16_vec             2048x2048 * 2048x2048             12.362       4.1      1.39
matmul_bk16_vec             4096x4096 * 4096x4096             89.421       2.3      1.54
matmul_v2                   32x32 * 32x32                      0.019       0.6      0.00
matmul_v2                   63x4096 * 4096x4096                2.538      27.3      0.83
matmul_v2                   512x512 * 512x512                  0.195      16.1      1.37
matmul_v2                   1024x1024 * 1024x1024              1.370       9.2      1.57
matmul_v2                   2048x2048 * 2048x2048             11.488       4.4      1.50
matmul_v2                   4096x4096 * 4096x4096             91.439       2.2      1.50
matmul_v2                   1x4096 * 4096x4096                 2.634      25.5      0.01
matmul_v2                   4096x1 * 1x4096                    1.834      36.6      0.02
matmul_v2                   257x383 * 383x511                  0.148      11.5      0.68
matmul_strided              512x512 * 512x512                  0.707       4.5      0.38
matmul_strided              1024x1024 * 1024x1024              6.391       2.0      0.34
sum(axis=1)                 1048576 elems                      0.024     175.8         -
sum(axis=1)                 4194304 elems                      0.072     233.5         -
sum(axis=1)                 67108864 elems                     1.220     220.2         -
bn_mean_var                 32x64x56x56                        0.235     109.4         -
bn_mean_var                 8x256x14x14                        0.031      51.1         -
bn_relu_fwd                 32x64x56x56                        0.409     125.7      0.02
bn_relu_fwd                 8x256x14x14                        0.032     101.0      0.01
softmax                     1x1000                             0.014       0.6         -
softmax                     128x1000                           0.024      43.3         -
softmax                     1024x1000                          0.095      86.4         -
cross_entropy_fwd           128x1000                           0.033      31.1         -
cross_entropy_bwd           128x1000                           0.022      45.7         -
cross_entropy_fwd           64x10000                           0.075      68.4         -
cross_entropy_bwd           64x10000                           0.066      78.0         -
conv2d_fwd                  1x3x224x224 w64 k3 s1 p1           1.097      12.3      0.16
conv2d_fwd                  32x64x56x56 w64 k3 s1 p1          13.678       3.8      0.54
conv2d_fwd                  8x256x14x14 w512 k3 s1 p1          4.354       2.2      0.85
conv2d_fwd                  8x64x28x28 w64 k5 s1 p2            3.206       1.1      0.40
conv2d_fwd                  32x64x56x56 w128 k3 s2 p1          5.166       7.5      0.72
conv2d_bwd_input            1x3->64 k3 s1                      1.084      12.4         -
conv2d_bwd_weight           1x3->64 k3 s1                      4.026       3.3      0.04
conv2d_bwd_bias             1x3->64 k3 s1                      3.946       3.3         -
conv2d_bwd_input            8x64->64 k3 s1                     8.738       1.5         -
conv2d_bwd_weight           8x64->64 k3 s1                     9.958       1.3      0.19
conv2d_bwd_bias             8x64->64 k3 s1                     1.597       4.0         -
conv2d_bwd_input            8x256->512 k3 s1                  18.195       0.5         -
conv2d_bwd_weight           8x256->512 k3 s1                   3.483       2.7      1.06
conv2d_bwd_bias             8x256->512 k3 s1                   0.248      12.9         -
maxpool_fwd                 1x64x112x112 k3 s2                 0.040     263.2         -
maxpool_bwd                 1x64x112x112 k3 s2                 0.057      69.9         -
maxpool_fwd                 32x64x56x56 k3 s2                  0.305     274.1         -
maxpool_bwd                 32x64x56x56 k3 s2                  0.467      68.8         -
maxpool_fwd                 8x256x14x14 k3 s2                  0.030     171.4         -
maxpool_bwd                 8x256x14x14 k3 s2                  0.040      50.1         -
adam_step                   1048576 elems                      0.069     243.1         -
adam_step                   4194304 elems                      0.562     119.5         -
adam_step                   16777216 elems                     2.292     117.1         -
copy_strided                1048576 elems                      0.328      12.8         -
copy_strided                16777216 elems                     5.067      13.2         -
==========================================================================================
```

Notable: the 3×3 s=1 p=1 conv rows use the im2col+GEMM path — the Winograd kernel in
`src/gpu_ops.cpp` measured slower (0.2–0.3 vs 0.5–0.9 TFLOPS on these shapes) and is
disabled; see the comment in `conv2d_gpu` before re-enabling it.

## Repository layout

```
include/    all headers (header-only CPU; GPU wrappers dispatch to src/gpu_ops.cpp)
src/        gpu_ops.cpp — GPU kernels + launchers (compiled as CUDA or HIP)
test/       test programs, one compiled at a time
data/       MNIST IDX files, miniImageNet class dirs
docs/       design notes: perf, kernels, memory, features, optimizations, scheduler, ...
```

Key headers: `Tensor.hpp` (public API), `TensorImpl.hpp` (storage + views),
`TensorOps.hpp` (ops), `Autograd.hpp` (backward nodes), `Module.hpp`,
`Functional.hpp`, `Optimizer.hpp`, `Loss.hpp`, `Scheduler.hpp`, `Serialization.hpp`,
`GraphExport.hpp`, `GraphMode.hpp`, `GraphExecutor.hpp`, `Dataset.hpp`,
`MemoryPool.hpp`, `GpuUtils.hpp`, `Dispatch.hpp`.

## GPU notes

- Only `float` instantiations exist for GPU kernels; `Float16`/`BFloat16` are enum
  placeholders.
- ROCm reuses `DeviceType::CUDA` at the C++ level (kernel code is shared via
  HIP/CUDA macros).
- Some launchers guard against zero-dim inputs (`if (blocks == 0) return;`) but not
  all — watch out when adding new kernels.
- A Winograd conv kernel exists in `src/gpu_ops.cpp` but is intentionally disabled
  (slower than the im2col+GEMM path); see the comment before re-enabling it.

## Limitations

- CPU ops are single-threaded (no OpenMP/TBB); the only background thread is the
  data loader's prefetcher.
- No multi-GPU support, no dtype promotion, no fp16/bfloat16 compute yet.
- No CI, no formatter, no pre-commit hooks.

## Further reading

The `docs/` directory contains design notes on performance (`perf.md`), kernels
(`kernels.md`), memory (`memory.md`), features (`features.md`), optimizations
(`optimizations.md`), the scheduler (`Scheduler.md`), and more. `TODO.md` is a
completed feature checklist that doubles as a capability reference.
