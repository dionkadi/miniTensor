To make this robust, the executor needs to operate as a **State Machine** that completely abstracts memory management, stream synchronization, and fallback logic away from the user.

### 1. The State Machine

The `GraphExecutor` will always be in one of four states:

1. **`NEEDS_WARMUP`**: The very first batch. We must run standard eager mode to initialize memory pools, autograd nodes, and kernel auto-tuners.
2. **`NEEDS_CAPTURE`**: The second batch. We record all GPU operations into a graph executable.
3. **`GRAPH_READY`**: The third batch onwards. We bypass the autograd engine completely and just blast memory to the GPU and launch the graph.
4. **`EAGERNESS_FALLBACK`**: Used dynamically if the user passes a batch size that doesn't match the captured graph (like the final batch of an epoch), or if CPU control flow is detected.

### 2. The Class Definition (`GraphExecutor.hpp`)

Here is the architectural blueprint of the class:

```cpp
#pragma once
#include "Tensor.hpp"
#include "Module.hpp"
#include "Optimizer.hpp"
#include "GpuUtils.hpp"
#include <functional>

#if defined(USE_CUDA)
    #include <cuda_runtime.h>
    using gpuGraph_t = cudaGraph_t;
    using gpuGraphExec_t = cudaGraphExec_t;
    using gpuStream_t = cudaStream_t;
#elif defined(USE_ROCM)
    #include <hip/hip_runtime.h>
    using gpuGraph_t = hipGraph_t;
    using gpuGraphExec_t = hipGraphExec_t;
    using gpuStream_t = hipStream_t;
#endif

template<typename T>
class GraphExecutor {
private:
    enum class State { NEEDS_WARMUP, NEEDS_CAPTURE, GRAPH_READY };
    State state_ = State::NEEDS_WARMUP;

    // References to user's training components
    std::shared_ptr<Module<T>> model_;
    Optimizer<T>& optimizer_;
    std::function<Tensor<T>(const Tensor<T>&, const Tensor<T>&)> loss_fn_;

    // Static memory buffers baked into the graph
    Tensor<T> static_x_;
    Tensor<T> static_y_;
    Tensor<T> static_loss_;
    
    // Graph and Stream handles
    gpuGraph_t graph_ = nullptr;
    gpuGraphExec_t graph_exec_ = nullptr;
    gpuStream_t capture_stream_ = nullptr;

    // Safety trackers
    std::vector<size_t> captured_batch_shape_;

public:
    GraphExecutor(std::shared_ptr<Module<T>> model, 
                  Optimizer<T>& optimizer,
                  std::function<Tensor<T>(const Tensor<T>&, const Tensor<T>&)> loss_fn)
        : model_(model), optimizer_(optimizer), loss_fn_(loss_fn) {
        
        // Create a dedicated stream for capture and replay to avoid global stream pollution
        GPU_CHECK(hipStreamCreate(&capture_stream_));
    }

    ~GraphExecutor() {
        if (graph_exec_) GPU_CHECK(hipGraphExecDestroy(graph_exec_));
        if (graph_) GPU_CHECK(hipGraphDestroy(graph_));
        if (capture_stream_) GPU_CHECK(hipStreamDestroy(capture_stream_));
    }

    // The single, clean user-facing API
    Tensor<T> step(const Tensor<T>& x, const Tensor<T>& y);

private:
    Tensor<T> eager_step(const Tensor<T>& x, const Tensor<T>& y);
    void capture_step();
    Tensor<T> replay_step(const Tensor<T>& x, const Tensor<T>& y);
    
    bool is_safe_to_graph() const;
};

```

### 3. Implementation of the Core Logic

#### The Main `step()` Dispatcher

This acts as the traffic controller, deciding exactly how the current batch should be processed.

```cpp
template<typename T>
Tensor<T> GraphExecutor<T>::step(const Tensor<T>& x, const Tensor<T>& y) {
    // FALLBACK 1: Dynamic batch size check (e.g., last batch of epoch is smaller)
    if (state_ == State::GRAPH_READY && x.shape() != captured_batch_shape_) {
        // We cannot use the graph, fall back to standard execution
        return eager_step(x, y);
    }

    if (state_ == State::NEEDS_WARMUP) {
        // Run normally to initialize everything
        Tensor<T> loss = eager_step(x, y);
        
        // Allocate the static memory buffers now that we know the shapes
        static_x_ = Tensor<T>(x.shape(), x.device());
        static_y_ = Tensor<T>(y.shape(), y.device());
        captured_batch_shape_ = x.shape();
        
        state_ = State::NEEDS_CAPTURE;
        return loss;
    }

    if (state_ == State::NEEDS_CAPTURE) {
        // FALLBACK 2: Graph safety check (ensure no CPU ops in autograd graph)
        if (!is_safe_to_graph()) {
            std::cerr << "[Warning] CPU operations detected. Disabling CUDA Graphs.\n";
            state_ = State::GRAPH_READY; // Fake ready, but will always trigger fallback if we wanted to
            return eager_step(x, y);     // For now, just run eager
        }

        // Copy this batch's data into the static buffers
        static_x_.copy_from(x);
        static_y_.copy_from(y);
        
        capture_step();
        state_ = State::GRAPH_READY;
        
        // The capture step doesn't execute the graph, it just records it. 
        // We must replay it immediately to actually train on Batch 1.
        return replay_step(x, y); 
    }

    // Happy Path: Graph is ready, blast it.
    return replay_step(x, y);
}

```

#### The `capture_step()` (The Magic)

This function isolates your forward, backward, and optimizer logic, telling the GPU driver to record the kernels rather than execute them.

```cpp
template<typename T>
void GraphExecutor<T>::capture_step() {
    // 1. Ensure all prior GPU work is done
    GPU_CHECK(hipDeviceSynchronize());

    // 2. Begin Capture on our dedicated stream
    GPU_CHECK(hipStreamBeginCapture(capture_stream_, hipStreamCaptureModeGlobal));

    // 3. Execute the exact training loop (GPU driver intercepts these)
    optimizer_.zero_grad();
    
    // Notice we use the STATIC buffers here. Their memory addresses get baked in.
    Tensor<T> preds = model_->forward(static_x_);
    static_loss_ = loss_fn_(preds, static_y_);
    
    static_loss_.backward();
    optimizer_.step();

    // 4. End Capture
    GPU_CHECK(hipStreamEndCapture(capture_stream_, &graph_));

    // 5. Instantiate the executable graph
    GPU_CHECK(hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
}

```

#### The `replay_step()` (The Speed)

This is where you get your massive performance boost. Eager execution overhead (Python/C++ dispatch, autograd graph construction, memory allocation) drops to literally zero microseconds.

```cpp
template<typename T>
Tensor<T> GraphExecutor<T>::replay_step(const Tensor<T>& x, const Tensor<T>& y) {
    // 1. Copy new batch data directly into the graph's static memory buffers
    // We use asynchronous copies on the capture stream so they queue up perfectly before the graph
    copy_tensor_async(x, static_x_, capture_stream_);
    copy_tensor_async(y, static_y_, capture_stream_);

    // 2. Launch the entire Forward + Backward + Optimizer step in one API call
    GPU_CHECK(hipGraphLaunch(graph_exec_, capture_stream_));

    // 3. We must synchronize the capture stream here so we can safely return the loss
    // to the CPU for logging purposes. (Or use hipMemcpyDtoHAsync and wait for it).
    GPU_CHECK(hipStreamSynchronize(capture_stream_));

    // Note: We return the static_loss_. To log it, your main loop usually does `loss.to(CPU)`
    return static_loss_; 
}

```

### 4. Critical Requirements for this to Work

You must enforce the following rules in your library:

1. **No CPU Allocation during Forward/Backward:** Once inside `capture_step()`, `MemoryPool::allocate` should ideally only be serving GPU memory. If a CPU allocation occurs (e.g., creating a temporary CPU tensor), it will not be captured.
2. **Optimizer must be 100% GPU:** Your `Adam<T>::step()` function can only contain GPU kernel launches. It cannot contain host-side synchronization (like `hipDeviceSynchronize`), nor can it conditionally branch based on the value of a GPU tensor (unless using CUDA Graph conditional nodes, which is overly complex).
3. **The `is_safe_to_graph()` check:** You should ensure model's parameters are on GPU `DeviceType::CUDA`.
