# Features

## Train/Eval Modes

`Module<T>` base class now has:
- `bool is_training_` — `true` by default (training mode)
- `void train()` — sets `is_training_ = true`, propagates to sub-modules in `Sequential`
- `void eval()`  — sets `is_training_ = false`, propagates to sub-modules in `Sequential`

**Dependency for**: `Dropout`, `BatchNorm2d`, and any future stochastic or stateful layers.

**File**: `include/Module.hpp`

---

## LayerNorm

Standard Layer Normalization along the last dimension:
- `LayerNorm<T>(normalized_shape, eps=1e-5, device)`
- Learnable `gamma` (init 1), `beta` (init 0)
- `forward(x)`: computes `x_hat = (x - mean) / sqrt(var + eps)`, returns `gamma * x_hat + beta`
- Full autograd support via existing ops (`sum`, `pow2`, `sqrt`, `add`, `mul`, `div`) — no custom backward node needed
- Same behavior in train and eval modes

```cpp
auto ln = LayerNorm<float>({128});
auto out = ln.forward(x);
```

**File**: `include/Module.hpp`

---

## BatchNorm2d

2D Batch Normalization for conv net outputs `[N, C, H, W]`:
- `BatchNorm2d<T>(num_features, eps=1e-5, momentum=0.9, device)`
- Learnable `gamma` (init 1), `beta` (init 0)
- Running statistics: `running_mean` (init 0), `running_var` (init 1)
- **Training**: normalizes using batch statistics (mean/var over `N, H, W` per channel), updates running stats via `running = momentum * running + (1-momentum) * batch_stat`
- **Eval**: normalizes using `running_mean` / `running_var`
- Full autograd support via existing ops

```cpp
auto bn = BatchNorm2d<float>(32);  // 32 channels
auto out = bn.forward(x);          // x: [N, 32, H, W]
```

**File**: `include/Module.hpp`

---

## Dropout

Standard dropout regularization:
- `Dropout<T>(p=0.5)` where `p` is the drop probability
- **Training**: generates Bernoulli(1-p) mask, scales output by `1/(1-p)`, saves mask for backward
- **Eval**: identity function (no-op)
- Custom `DropoutBackward<T>` autograd node: `grad_input = grad_output * mask / (1-p)`

```cpp
auto dropout = Dropout<float>(0.5);
auto out = dropout.forward(x);  // Randomly drops ~50% of activations
```

**File**: `include/Module.hpp`, `include/Autograd.hpp` (backward node)

---

## Softmax

Numerically stable softmax along the last dimension:
- `softmax(input)` — standalone function
- Forward: `exp(x_i - max(x)) / sum(exp(x_j - max(x)))` along last dim
- Backward via `SoftmaxBackward<T>`: `grad_input = s * (grad_output - sum(grad_output * s))` where `s = softmax(x)`
- CPU: `softmax_cpu` — row-by-row max + exp + sum + normalize
- GPU: `softmax_fwd_kernel` — shared memory reduction for max + sum-exp, one block per row

```cpp
auto probs = softmax(logits);  // shape preserved, values in [0,1]
```

**Files**: `include/TensorOps.hpp`, `include/Autograd.hpp`, `src/gpu_ops.cpp`

---

## Batch Matrix Multiplication (BMM)

Batched matrix multiplication `[B, M, K] @ [B, K, N] → [B, M, N]`:
- `bmm(A, B)` — standalone function
- Both A and B can have separate per-batch values (unlike `batched_sgemm_kernel` which assumes shared A)
- Implementation: loops over batch dimension, calls `matmul` (2D) per sample
- CPU: uses `matmul_cpu` per batch
- GPU: uses `matmul_gpu` (tiled sgemm) per batch
- Full autograd support via `MatmulBackward<T>` (reuses existing backward)

```cpp
auto C = bmm(A, B);  // C.shape() = [B, M, N]
```

**File**: `include/TensorOps.hpp`

---

## Model Serialization

Binary state dict save/load:
- `save_state_dict<T>(path, params)` — writes params to binary file
- `load_state_dict<T>(path)` — reads and returns tensor list
- `load_state_dict_into<T>(path, target_params)` — loads into existing parameter tensors (handles CPU and GPU device transfer)
- **Format**: `[num_params: u32] [for each: ndim: u32, shape[ndim]: u64[], raw_bytes: T[]]`
- Lightweight, portable, no dependencies

```cpp
// Save
save_state_dict("model.bin", model->parameters());

// Load (training a different copy)
auto params = load_state_dict<float>("model.bin");

// Load into existing model (handles GPU targets)
load_state_dict_into("model.bin", model->parameters());
```

**File**: `include/Serialization.hpp`

---

## Functional API

Namespace `F` with convenience wrappers matching PyTorch's `torch.nn.functional`:
- `F::relu(x)`, `F::sigmoid(x)`, `F::tanh(x)`, `F::softmax(x)`
- `F::flatten(x)`, `F::dropout(x, p)`
- `F::conv2d(input, weight, bias, stride, padding)`
- `F::max_pool2d(x, k, stride, padding)`
- `F::batch_norm(x, gamma, beta, running_mean, running_var, training, eps, momentum)`

Branching is supported natively by the autograd engine (topological sort handles DAGs):

```cpp
struct ResBlock : Module<float> {
    Conv2D<float> conv1, conv2;
    Tensor<float> forward(const Tensor<float>& x) override {
        auto y = conv1.forward(x);
        y = F::relu(y);
        y = conv2.forward(y);
        return y + x;  // Skip connection — autograd handles the branch
    }
};
```

**File**: `include/Functional.hpp`

---

## Async DataLoader

Threaded prefetching data loader:
- `AsyncDataLoader<T>(dataset, batch_size, shuffle=true, prefetch=2)`
- Spawns a background thread that prefetches batches into a `ThreadSafeQueue`
- Main thread pops ready batches without waiting for CPU-side batch construction
- Clean shutdown via destructor (joins worker thread)
- Compatible with any `Dataset<T>` implementation

```cpp
AsyncDataLoader<float> loader(train_ds, 64, true, 3);
for (auto [x, y] : loader) {
    // x, y are already prepared — no wait for CPU batch construction
    auto pred = model->forward(x.to(gpu));
    ...
}
```

**File**: `include/Dataset.hpp`

---

## Broadcasting (existing — documented here for completeness)

Broadcasting follows NumPy rules (stride=0 expansion trick in `TensorImpl::expand()`). Two dispatch paths:
- **Contiguous**: `binary_gpu`/`binary_cpu` — simple flat loop (fastest path, used when all dims match)
- **Strided**: `binary_gpu_strided`/`binary_cpu_strided` — handles stride=0 broadcast dims via per-element coordinate computation

GPU strided kernel (`binary_kernel_strided`) uses sized grid stride loops + TensorInfo for source indexing. The strided path is correct but bandwidth-limited for small tensors with many broadcast dims.

**Future optimization**: Specialized broadcast kernel that computes source indices via `flat_idx % shape` modulo math instead of the coordinate decomposition loop in `binary_kernel_strided`. This would improve performance for common patterns like bias addition `[N,C,H,W] + [C,1,1]`.

---

## View Semantics & Autograd Integrity

Views (slice, transpose, reshape, expand) properly propagate `requires_grad` and version counters:

- **Version counter**: `TensorStorage::version_` is shared by all views — any in-place modification bumps it for all views. `SavedTensor::unpack()` throws if the version changed since saving.
- **requires_grad propagation**: `slice()` now passes `requires_grad()` to the view's impl (was previously defaulting to `false`). `transpose()`, `expand()`, `reshape()` already propagated via `TensorImpl`.
- **Backward nodes**: Four backward nodes (`SliceBackward`, `TransposeBackward`, `ReshapeBackward`, `ExpandBackward`) correctly map the gradient back to the original tensor shape.
- **Autograd-aware wrappers**: `view_slice()`, `view_transpose()`, `view_reshape()`, `view_expand()` in `TensorOps.hpp` wire up the appropriate backward node when `GradMode` is enabled.

```cpp
auto A = Tensor<float>({3, 4}, CPU);
A.set_requires_grad(true);

// view_slice tracks gradients — original gets grads in the sliced region only
auto A_view = view_slice(A, 0, 0, 2);  // [2, 4] view of rows 0:1
auto loss = sum(A_view, 1, false);
loss = sum(loss, 0, false);
loss.backward();
// A.grad() is [[1,1,1,1], [1,1,1,1], [0,0,0,0]]

// In-place mutation on ANY view is detected by ALL SavedTensor instances
auto view = A.slice(0, 0, 1);
view.set_requires_grad(false);
add_(view, view);  // bumps version — any SavedTensor holding A will throw on unpack
```

**Files**: `include/Autograd.hpp` (backward nodes), `include/TensorOps.hpp` (autograd-aware wrappers), `include/Tensor.hpp` (requires_grad fix)

---

## Memory Pool with Power-of-2 Binning

GPU memory allocations are rounded to the next power of 2 to prevent fragmentation from near-miss sizes (e.g., 1024B and 1028B no longer create separate bins):

- `next_pow2(n)` helper: bit-twiddle for O(1) power-of-2 ceiling
- `allocate()` rounds up for GPU paths, exact for CPU (pinned memory)
- `free()` rounds up identically so keys match the allocation bin
- Reduces pool fragmentation by collapsing similar-sized allocations into shared power-of-2 bins
- CPU pinned memory uses the exact-size pool unchanged

**Waste trade-off**: up to 2× memory overhead in the worst case (just above a power of 2). For production use, a full caching allocator with segment sub-allocation and coalescing would eliminate this overhead.

```cpp
// 1024B and 1028B both hit the same 2048B bin → no fragmentation
auto a = Tensor<float>({256}, gpu);   // 1024 bytes
auto b = Tensor<float>({257}, gpu);   // 1028 bytes -> reuses 2048B pool slot
```

**File**: `include/MemoryPool.hpp`

---

## Type Dispatch Infrastructure

Preparation for multi-dtype support without template explosion:

- **`Dtype` enum**: `Float32`, `Float16`, `BFloat16`, `Int64`, `Int8` with `dtype_name()`, `dtype_size()`, `dtype_of<T>()` helpers
- **`dtype_t<Dtype>` trait**: maps enum back to C++ type
- **`Tensor<T>::dtype()`**: runtime dtype query (returns `Dtype::Float32` for `float`)
- **`OpRegistry<T>`**: vtable-style dispatch registry mapping `OpType` → unary/binary function — allows type-erased code to look up typed kernels
- **`AutoRegistrar<T>`**: RAII helper for populating the registry at static-init time

```cpp
auto t = Tensor<float>({128}, CPU);
Dtype d = t.dtype();                            // Dtype::Float32
size_t elem_size = dtype_size(d);               // 4

OpRegistry<float>::instance().register_unary(OpType::Relu,
    [](const Tensor<float>& a, Tensor<float>& o) { o = relu(a); });
```

**Files**: `include/Defines.hpp`, `include/Dispatch.hpp`

---

## GPU Streams & Events

RAII wrappers for GPU streams and events for multi-stream execution:

- **`GpuStream`**: creates a non-blocking GPU stream, auto-synchronizes and destroys on destruction. Move-only.
- **`GpuEvent`**: creates a GPU event (timing disabled), supports `record(stream)`, `wait(stream)`, `synchronize()`, `query()`. Move semantics via RAII.
- Streams can be passed to kernel launchers to overlap computation with data transfers.

```cpp
GpuStream compute_stream;
// Launch kernels on compute_stream: kernel<<<grid, block, 0, compute_stream.get()>>>(...);
// Meanwhile, queue data transfer on default stream
GpuEvent done;
done.record(compute_stream.get());
done.wait();  // default stream waits for compute to finish
```

**File**: `include/GpuUtils.hpp`

---

## ONNX-like Graph Export

Model architecture export/import as a binary graph DAG:

- **`GraphExport<T>`**: extracts graph from Module tree via `extract_graph()`, saves/loads binary format
- **`GraphExport<T>::NodeInfo`**: per-node type name, parameter shapes, child list
- **`build_from_graph<T>()`**: reconstructs module tree from NodeInfo (parameter values loaded separately via `Serialization`)
- **Format**: `[name_len: u32][name_bytes][n_params: u32]{[ndim: u32][dim: u64...]}[n_children: u32][children...]`
- **Module `name()`**: every module class overrides `name()` via `const char* name() const override` (Sequential, Linear, Conv2D, ReLU, LayerNorm, etc.)

```cpp
auto model = Sequential<float>{...};
GraphExport<float>::save_graph("model.graph", model);
auto info = GraphExport<float>::load_graph("model.graph");
auto rebuilt = build_from_graph<float>(info, gpu);
load_state_dict_into("model.bin", rebuilt->parameters());
```

**Files**: `include/GraphExport.hpp`, `include/Module.hpp` (`name()`, `modules()`, `add()`)

---

## Graph Tracer (JIT Preview)

Graph recording and replay infrastructure:

- **`ComputationGraph<T>`**: holds a list of `GraphNode<T>` (Input/Op) and their concrete tensor outputs
- **`GraphTracer<T>::trace()`**: runs a Module forward pass, recording each op as a graph node
- **`GraphTracer<T>::replay_graph()`**: replays a recorded graph using a caller-provided executor
- **`ComputationGraph::print()`**: prints the graph in SSA-style format (`%id = op_name(%input,...)`)

```cpp
GraphTracer<float> tracer;
auto graph = tracer.trace(*model, input);
graph.print();  // %0 = input(1,28,28)\n%1 = Conv2D(%0)\n...
```

**Future**: Dead-code elimination, layout transformations, and operator fusion run on the recorded graph.

**File**: `include/GraphMode.hpp`

---

## Operator Fusion (Dispatch Pattern)

While kernel fusion requires a graph compiler and per-op fused CUDA/HIP kernels, the dispatch pattern is established:

- Fused kernel example: `adam_step_gpu` fuses weight decay, moment updates, bias correction, and parameter update into one kernel
- Fused kernel example: `fused_backward_input_kernel` combines `batched_sgemm` + `col2im` (eliminates ~57MB intermediate tensor)
- Pattern: fused ops dispatch through `TensorOps.hpp` — CPU falls back to decomposed ops; GPU uses a single kernel launch

For `Linear → ReLU` fusion, a fused `linear_relu_kernel` would replace separate `sgemm_kernel` + `binary_kernel` launches. This is documented as a known optimization pathway (see `docs/kernels.md`).

**Files**: `src/gpu_ops.cpp` (fused kernels), `docs/kernels.md`

---

## Mixed Precision Training (Infrastructure)

Gradient scaler for FP16 training — prevents gradient underflow by dynamically scaling loss:

- **`GradientScaler<T>`**: scales loss, tracks overflows, unscales gradients before optimizer step
- `scale_loss(loss)` → multiply loss by current scale before backward
- `unscale_gradients(optimizer)` → divide all gradients by scale before `optimizer.step()`
- `update(overflow)` → if overflow detected: scale *= 0.5, skip step. Otherwise: scale *= 2 every N steps
- Default init scale: 65536 (2^16), growth factor: 2, backoff factor: 0.5

```cpp
GradientScaler<float> scaler;
scaled_loss = scaler.scale_loss(loss);
scaled_loss.backward();
scaler.unscale_gradients(adam);
bool ok = scaler.update(/*overflow*/ false);
if (ok) adam.step();
```

**File**: `include/Optimizer.hpp`
