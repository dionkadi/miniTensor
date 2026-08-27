# miniTensor — agent instructions

From-scratch C++20 tensor library with CPU + GPU (CUDA/ROCm) backends, autograd,
nn module API, serialization, graph export, and graph mode (JIT). Header-only CPU
ops; GPU kernels live in `src/gpu_ops.cpp` compiled as CUDA or HIP.

## Build

```sh
cmake -B build -G Ninja -DUSE_ROCM=ON          # AMD GPU (gfx1102,gfx1103)
cmake -B build -G Ninja -DUSE_CUDA=ON          # NVIDIA GPU (sm_75,sm_80)
cmake -B build -G Ninja                        # auto-detect GPU, falls back CPU-only
cmake --build build
```

- CMake 4.0+ required. Compiler: clang++ (ROCm fork for HIP). C++20, strict (`-std=c++20`).
- Debug flags: host `-O2 -g`, HIP device `-O3 -g` (HIP at `-O0` is 50-100× slower).
- CPU-only builds: `GPU_CHECK(call)` expands to nothing, GPU functions return early. No GPU driver needed.
- VS Code settings force `-DUSE_ROCM=ON`; clangd reads `build/compile_commands.json`.
- `CMakePresets.json` defines `debug`/`release` presets but they build into `build/debug` / `build/release`, so clangd (which reads `build/`) won't see them — prefer the plain `cmake -B build` flow.
- `.clangd` adds `--rocm-path=/opt/rocm` and `-D__HIP_PLATFORM_AMD__=1` for clangd in ROCm builds.
- `include/Kernels.cpp` is **dead code** — commented out in CMakeLists.txt, never compiled.
- `config.json` at root is OpenCode agent config, not part of the library.

## Run / Test

```sh
./build/tensor_tests
```

Only one test file is compiled at a time. Edit the commented-out list inside `add_executable(tensor_tests ...)` in CMakeLists.txt to switch:
- `test/benchmark.cpp` — GPU kernel benchmark suite (own `main()`, currently active)
- `test/resnet.cpp` — ResNet50 on miniImageNet, Bottleneck blocks + label smoothing (own `main()`)
- `test/mnist.cpp` — MNIST CNN training (own `main()`)
- `test/view_test.cpp` — View/autograd tests (own `main()`)
- `test/engien_test.cpp` — Autograd engine tests (GTest parametrized CPU+GPU via `TEST_P`; note the filename misspelling — that's intentional, not a typo)
- `test/{tensor,spatial,func,dataset}_test.cpp` — GTest; GPU sections guarded by `#if defined(USE_CUDA) || defined(USE_ROCM)`
- `test/nn_test.cpp` — standalone (`main()`), manual test of Sequential+SGD
- `test/scheduler_test.cpp` — Scheduler unit tests (own `main()`)
- `test/linear_regression_test.cpp` — exists on disk but **not** wired into CMakeLists.txt
- `test/tensor_test.cpp` and `src/gpu_ops.cpp` get `LANGUAGE` set to CUDA or HIP depending on backend selected.
- Fixed seed `42` for determinism. Used in He init for `Linear`/`Conv2D` and test random generation.

**Test helpers**: `get_val(t, idx=0)` extracts scalar via `to(CPU)` (in `test/engien_test.cpp`). `expect_tensor_data(t, expected)` copies to CPU then checks element-by-element (duplicated in `test/func_test.cpp` and `test/dataset_test.cpp`).

## Architecture

```
include/       → all headers (CPU-only)
src/           → gpu_ops.cpp (GPU kernels + launcher wrappers)
test/          → test files (one compiled at a time; includes stb_image.h for JPEG)
data/          → MNIST IDX, miniImageNet class dirs with JPEGs
build/         → out-of-tree (ninja)
docs/          → perf, memory, kernels, features, optimizations, scheduler, etc.
```

Key headers beyond `Tensor.hpp` (public API), `TensorImpl.hpp` (storage + view), and `Defines.hpp` (MAX_DIMS, Dtype, DeviceType):
- `TensorOps.hpp` — CPU kernels + GPU declarations; `safe_out_size()` prevents unsigned wraparound
- `Autograd.hpp` — 26 backward nodes inheriting `AutogradNode<T>`, `SavedTensor`, `NoGradGuard`
- `Module.hpp` — `Linear`, `Conv2D`, `BatchNorm2d`, `LayerNorm`, `Dropout`, `Sequential`, etc.
- `Functional.hpp` — `F::relu`, `softmax`, `conv2d`, etc.
- `Optimizer.hpp` — `SGD`, `Adam`, `AdamW`, plus `GradientScaler` for mixed precision
- `Loss.hpp` — `cross_entropy_loss`, `mse_loss`
- `Scheduler.hpp` — `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, etc.
- `Serialization.hpp` — `save_state_dict` / `load_state_dict` (binary)
- `GraphExport.hpp` — ONNX-like graph DAG export/import
- `GraphMode.hpp` — `ComputationGraph` + `GraphTracer` for JIT recording/replay
- `GraphExecutor.hpp` — CUDA/HIP graph capture and replay for fused training loops
- `Dispatch.hpp` — `OpRegistry<T>` vtable-style runtime dispatch
- `MemoryPool.hpp` — power-of-2 binned GPU allocator
- `GpuUtils.hpp` — `GpuStream`, `GpuEvent` RAII wrappers, `active_stream()` thread-local override
- `Activation.hpp` — `Activation<T>` base, `ReLU`, standalone `relu()` 
- `Dataset.hpp` — `AsyncDataLoader` with threaded prefetching

## GPU kernels (`src/gpu_ops.cpp`)

- `cast<T>(ptr)`: returns `T&` from pointer (by-reference return). Historical bug: by-value broke all vectorized writes.
- **All `hipDeviceSynchronize()`/`cudaDeviceSynchronize()` removed** — default stream is ordered implicitly.
- Error checking uses `GPU_CHECK(get_last_error_capture_safe())` after each launch. This tolerates `cudaErrorStreamCaptureUnsupported` during graph capture (warmup runs verify valid args before capture).
- Each launcher computes `blocks = (total + threads - 1) / threads` with `int threads = 256`. Some launchers use a vectorized variant `(total + threads*4 - 1) / (threads*4)`. A subset (not all) have `if (blocks == 0) return;` to prevent zero-dim crashes — add this guard when writing new launchers.
- Template functions explicitly instantiated for `float` only at bottom of file. No other type works on GPU.
- `active_stream()` thread-local override used in kernel launch wrappers and memcpys for graph capture. Set before capture, `nullptr` otherwise.
- Fused kernels (do NOT decompose): `cross_entropy_fwd_gpu`/`cross_entropy_bwd_gpu` (softmax+NLLLoss+label smoothing — both take a `smoothing` param), `adam_step_gpu` (weight decay + moment + bias correction), `bn_relu_fwd_gpu`.
- `conv2d_winograd_gpu` exists but is **intentionally disabled** (commented out in `conv2d_forward_gpu`): benchmarks show 0.22-0.33 TFLOPS vs 1.1+ for the im2col+GEMM path on 3x3 s1 p1 because it re-transforms weights and allocates buffers per call. Don't re-enable; fix it to cache weight transforms first.

## Autograd quirks

- No separate `Variable` class. `Tensor<T>` carries `requires_grad_`, `grad_fn_`, `grad_` directly in its shared `TensorImpl<T>`.
- `backward()`: seeds grad with ones if empty, DFS topological sort, reverse apply.
- `NoGradGuard`: RAII, disables `GradMode::is_enabled()`. Always use inside backward `apply()`.
- `SavedTensor` has version counter — throws if tensor mutated in-place since saving.
- In-place ops (`add_`, `sub_`, `mul_`) bump version counter and reject leaf `requires_grad` tensors.
- `unbroadcast(grad, original_shape)` applied in backward nodes after binary ops.
- `BatchNorm2d` has full training-mode backward (`BatchNormBackward` node in Autograd.hpp; gamma/beta are trainable) — not inference-only.
- `TODO.md` is a completed feature checklist (all items checked, not a plan) — useful reference for library capabilities.

## Conventions

- Row-major (C-style) layout with strides.
- Broadcast: stride=0 expansion in `TensorImpl::expand()`.
- `DeviceType` enum: `CPU`, `CUDA`. ROCm reuses `DeviceType::CUDA` at C++ level.
- `MAX_DIMS = 8`.
- `HD_INLINE` macro: `__host__ __device__ inline` under CUDA/HIP, plain `inline` otherwise.
- All ops functors (e.g., `AddOp<T>`, `MulOp<T>`) use `HD_INLINE operator()` for dual CPU/GPU use.
- `Dtype` enum: `Float32`, `Float16` (placeholder), `BFloat16` (placeholder), `Int64`, `Int8`.

## What is NOT present

- No CI (no `.github/`), no formatter (no `.clang-format`), no pre-commit hooks.
- No multi-GPU support, no tensor type promotion.
- Only `float` template instantiations exist for GPU. `Float16`/`BFloat16` are enum placeholders only.
- `gpu_ops_lib.cpp` does not exist on disk (notable because the pattern is easy to guess wrong).
- No OpenMP, no TBB — CPU ops are single-threaded. The only multi-threading is `AsyncDataLoader`'s prefetch thread.
