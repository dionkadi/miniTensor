- [ ] Model Serialization (Save/Load): Implement a `save_state_dict` and `load_state_dict` feature. A simple binary format that writes the shape, stride, and raw bytes of each parameter tensor to disk is a great start. For a modern touch, consider implementing the Hugging Face `safetensors` format, which is safe, fast, and cross-language.

- [ ] Train vs. Eval Modes: Add a `bool is_training_ = true;` flag to base `Module` class, along with `.train()` and `.eval()` methods. This is a prerequisite for stochastic/stateful layers like Dropout and Batch Normalization.

- [ ] Dropout Layer: Implement `Dropout`. During training, generate a random mask tensor using a Bernoulli distribution, scale the output by `1 / (1 - p)`, and save the mask for the backward pass. During evaluation, it acts as an identity function.

- [ ] Normalization Layers: Implement `BatchNorm2d` and `LayerNorm`. Batch Normalization will require maintaining moving averages of the mean and variance, which also relies on the Train/Eval modes mentioned above.

- [ ] Batch Matrix Multiplication (BMM): Implement `bmm` (e.g., `[B, M, K] @ [B, K, N] -> [B, M, N]`) is the primary blocker for implementing Self-Attention and Transformers.

- [ ] Functional API & Branching: Autograd engine uses topological sort, which naturally supports branching (e.g., $y = F(x) + x$). However, to build a ResNet cleanly, the library might need to expose a functional API (like `torch.nn.functional`) alongside `Module` classes so users can write custom `forward` passes easily without wrapping every single addition in a module.

- [ ] Softmax Op: Implement a standalone `softmax` operation (with its corresponding backward pass) is required for Attention mechanisms and multi-class inference probability outputs.

- [ ] Asynchronous DataLoader: Implement double-buffering logic into the `DataLoader`. Use `std::thread` and a thread-safe queue to prefetch and decode batches in the background so the GPU never has to wait for the disk or the CPU normalize loop.

- [ ] Broadcasting Improvements: `compute_broadcast_shape` works well, but expanding tensors physically (or virtually via strides with 0) before binary ops can be optimized. Implement specialized broadcast kernels on the GPU that calculate the modulo math on the fly, saving memory bandwidth.
