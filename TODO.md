- [x] Model Serialization (Save/Load): Implement a `save_state_dict` and `load_state_dict` feature. A simple binary format that writes the shape, stride, and raw bytes of each parameter tensor to disk is a great start. For a modern touch, consider implementing the Hugging Face `safetensors` format, which is safe, fast, and cross-language.

- [x] Train vs. Eval Modes: Add a `bool is_training_ = true;` flag to base `Module` class, along with `.train()` and `.eval()` methods. This is a prerequisite for stochastic/stateful layers like Dropout and Batch Normalization.

- [x] Dropout Layer: Implement `Dropout`. During training, generate a random mask tensor using a Bernoulli distribution, scale the output by `1 / (1 - p)`, and save the mask for the backward pass. During evaluation, it acts as an identity function.

- [x] Normalization Layers: Implement `BatchNorm2d` and `LayerNorm`. Batch Normalization will require maintaining moving averages of the mean and variance, which also relies on the Train/Eval modes mentioned above.

- [x] Batch Matrix Multiplication (BMM): Implement `bmm` (e.g., `[B, M, K] @ [B, K, N] -> [B, M, N]`) is the primary blocker for implementing Self-Attention and Transformers.

- [x] Functional API & Branching: Autograd engine uses topological sort, which naturally supports branching (e.g., $y = F(x) + x$). However, to build a ResNet cleanly, the library might need to expose a functional API (like `torch.nn.functional`) alongside `Module` classes so users can write custom `forward` passes easily without wrapping every single addition in a module.

- [x] Softmax Op: Implement a standalone `softmax` operation (with its corresponding backward pass) is required for Attention mechanisms and multi-class inference probability outputs.

- [x] Asynchronous DataLoader: Implement double-buffering logic into the `DataLoader`. Use `std::thread` and a thread-safe queue to prefetch and decode batches in the background so the GPU never has to wait for the disk or the CPU normalize loop.

- [x] Broadcasting Improvements: `compute_broadcast_shape` works well, but expanding tensors physically (or virtually via strides with 0) before binary ops can be optimized. Implement specialized broadcast kernels on the GPU that calculate the modulo math on the fly, saving memory bandwidth.

- [x] The Memory Fragmentation Trap: As implementing dynamic sequence lengths (e.g., Transformers) or complex branching architectures, the memory pool will fill up with blocks of slightly different sizes (e.g., 1024 bytes, 1028 bytes). GPU memory is ran out of, because of fragmentation. A caching allocator (like PyTorch’s) that allocates large blocks (e.g., 20MB chunks) from the OS/CUDA/ROCM, and then sub-allocates, splits, and merges those blocks internally, should be implemented.

- [x] The Operator Dispatch Explosion: Right now, C++ templates (`template<typename T>`) are used to handle data types, and simple if/else statements to switch between CPU and GPU execution. Binary size will explode when building support for float32, float16, bfloat16, int64, and int8 across CPU, CUDA, ROCM, and maybe Apple Metal (MPS). A dynamic dispatching mechanism is needed. Frameworks handle this by type-erasing the C++ templates at the highest level, passing around an opaque "Tensor" object that holds metadata (dtype, device, layout), and using a v-table or registry pattern to look up and invoke the correct specialized kernel at runtime.

- [x] View Semantics vs. Autograd Integrity: Things get incredibly thorny when introducing more advanced "views". If to slice a tensor (`A_view = A.slice(...)`), and then modify the view in-place (`A_view += 1`), the underlying storage of `A` changes. If `A` was saved for a backward pass by a completely different operator, the autograd engine needs to know that `A_view`'s mutation corrupted A. A system should be implemented, where tensors sharing the same `TensorStorage` share or correctly propagate version counters and requires_grad lineage.

- [x] GPU Synchronization and Streams: When implementing Multi-GPU training (DDP) or trying to overlap gradient communication with backward pass computation, the default stream acts as a massive bottleneck because it serializes operations. Explicit Stream management is needed. Pass `GpuStream_t` to every single one of GPU kernels and use Events to synchronize memory transfers without blocking the CPU thread.

- [x] ONNX Export/Import: Implement the ability to export models to the Open Neural Network Exchange (ONNX) format, or load ONNX models. This instantly gives the library interoperability with the rest of the machine learning world, allowing user to run models trained in PyTorch or TensorFlow.

- [x] Graph Mode / JIT Compilation: Instead of executing operations immediately, trace them to build a computational graph. Once you have the graph, you can optimize it before execution (e.g., dead code elimination, layout transformations).

- [x] Operator Fusion: If you have a graph, you can fuse kernels. Instead of launching three separate memory-bound kernels for `Linear -> ReLU -> Dropout`, a graph compiler can fuse them into a single custom CUDA/HIP kernel, drastically reducing GPU memory bandwidth bottlenecking.

- [x] Mixed Precision Training (FP16 / BF16): Standard training uses 32-bit floats. Implementing 16-bit precision requires handling dynamic loss scaling (to prevent gradients from underflowing to zero) and maintaining master copies of FP32 weights, but it effectively doubles your VRAM capacity and compute throughput on modern GPUs (via Tensor Cores).