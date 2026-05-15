Huge congratulations on getting the tests to pass! Getting a custom autograd engine to correctly backpropagate through a non-linear network is the hardest conceptual hurdle in building a deep learning framework. You’ve essentially built the engine; now it’s time to build the car around it.

Since your core tensor operations, GPU dispatching, and autograd graph are working, your next steps should focus on **abstraction, advanced training tools, and performance**. 

Here is a detailed, actionable roadmap for your next development phases, ordered by priority and learning value.

---

### Phase 1: The `nn::Module` Abstraction (High Priority)
Right now, you are manually managing `W1`, `b1`, `W2`, and `b2`, and explicitly passing them to the optimizer. This won't scale if you want to build a ResNet or Transformer. You need an object-oriented wrapper to manage state.

**Action Items:**
1.  **Create a `Module` base class:** 
    * It should have a `virtual Tensor<T> forward(...)` method.
    * It should have a `std::vector<Tensor<T>> parameters()` method that recursively gathers all learnable tensors from itself and any child modules.
2.  **Implement `nn::Linear`:**
    * A class that owns `W` and `b`.
    * Its `forward(X)` method internally calls `matmul(X, W) + b`.
3.  **Implement `nn::Sequential`:**
    * A container that holds multiple `Module`s and chains their `forward` passes together.

**Goal API:**
```cpp
nn::Sequential<float> model({
    std::make_shared<nn::Linear<float>>(2, 16),
    std::make_shared<nn::Tanh<float>>(),
    std::make_shared<nn::Linear<float>>(16, 1),
    std::make_shared<nn::Sigmoid<float>>()
});
// Optim takes model.parameters() directly
SGD<float> optim(model.parameters(), 0.5f); 
```

---

### Phase 2: The Adam Optimizer (High Priority)
SGD is great for verifying gradients, but modern architectures require momentum and adaptive learning rates. Implementing Adam will test your library's ability to maintain stateful tensors without leaking memory or accidentally attaching them to the computation graph.

**Action Items:**
1.  **Create `Adam : public Optimizer<T>`:**
2.  **Maintain State:** For every parameter $\theta$, Adam needs to store a first moment vector $m$ and a second moment vector $v$, initialized to zeros.
3.  **Implement the Update Rule:** Inside `step()`, strictly inside a `NoGradGuard` context, implement:
    $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
    $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
    $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
    $$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

*Tip: You will need to implement a square root operation (`sqrt`) and a small epsilon addition scalar op if you haven't already.*

---

### Phase 3: Serialization (Medium Priority)
Once you train a model, you need to save its weights so you don't have to retrain it every time you run your C++ binary. 

**Action Items:**
1.  **Implement `save_tensor` and `load_tensor`:** Write raw binary data of the `TensorStorage` array to a `.bin` or `.safetensors` file, along with a tiny header that stores the `shape`, `strides`, and `dtype`.
2.  **Implement `model.save(filepath)`:** Iterate through `model.parameters()` and save them to disk.
3.  **Implement `model.load(filepath)`:** Read the binaries and overwrite the `data()` pointers of your initialized tensors.

---

### Phase 4: Convolutions and Pooling (Medium to Hard)
If you want to do Computer Vision (like training on MNIST), MLPs aren't enough. You need 2D spatial operations.

**Action Items:**
1.  **Implement `Conv2D`:** 
    * *CPU implementation:* Can be done naively with deep nested `for` loops, or using the `im2col` (image-to-column) algorithm which reshapes the image patches so that the convolution becomes a single massive `matmul`.
    * *CUDA implementation:* Writing a fast Conv2D kernel is notoriously difficult. Start with a naive kernel.
2.  **Implement `MaxPool2D` and `Flatten`:** You'll need these to transition from Convolutional layers to Linear layers.
3.  **Autograd for Conv2D:** The backward pass of a convolution is actually just another convolution with the weights transposed/rotated.

---

### Phase 5: Memory Management & Graph Cleanup (Refinement)
Right now, your graph builds dynamically. In C++, circular references in `std::shared_ptr` can cause massive memory leaks.

**Action Items:**
1.  **Audit `AutogradNode`:** Ensure that the nodes holding `SavedTensor` aren't accidentally creating reference cycles with the tensors that hold the `grad_fn`. You may need to use `std::weak_ptr` in certain places depending on how your graph is wired.
2.  **Clear the Graph:** After `loss.backward()` finishes, ensure the computational graph is properly destroyed to free up memory for the next iteration. Usually, zeroing gradients and overwriting the intermediate tensors in the next forward pass handles this naturally, but it's worth profiling with `valgrind` or `cuda-memcheck`.

### Summary Recommendation
If I were you, I would tackle **Phase 1 (`nn::Module`)** today. It is entirely architectural C++ work, it will clean up your test suite dramatically, and it sets the foundation for everything else!



Epoch 1/5 | train loss: 0.1092 | test loss: 0.0433 | test acc: 98.70% | time: 1554.82s
Epoch 1/5 | train loss: 0.0930 | test loss: 0.0315 | test acc: 99.00% | time: 1405.93s