#pragma once

#include "Tensor.hpp"
#include "Module.hpp"
#include "Optimizer.hpp"
#include "GpuUtils.hpp"
#include <functional>
#include <memory>
#include <vector>

#if defined(USE_CUDA)
    #include <cuda_runtime.h>
    using gpuGraph_t     = cudaGraph_t;
    using gpuGraphExec_t = cudaGraphExec_t;
#elif defined(USE_ROCM)
    #include <hip/hip_runtime.h>
    using gpuGraph_t     = hipGraph_t;
    using gpuGraphExec_t = hipGraphExec_t;
#endif

#if defined(USE_CUDA) || defined(USE_ROCM)
template<typename T>
class GraphExecutor {
public:
    using LossFn = std::function<Tensor<T>(const Tensor<T>&, const Tensor<T>&)>;

    GraphExecutor(std::shared_ptr<Module<T>> model,
                  Optimizer<T>&           optimizer,
                  LossFn                  loss_fn)
        : model_(model), optimizer_(optimizer), loss_fn_(loss_fn)
    {
#if defined(USE_CUDA)
        GPU_CHECK(cudaStreamCreate(&capture_stream_));
#elif defined(USE_ROCM)
        GPU_CHECK(hipStreamCreate(&capture_stream_));
#endif
    }

    ~GraphExecutor() {
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (graph_exec_) {
#if defined(USE_CUDA)
            (void)cudaGraphExecDestroy(graph_exec_);
#elif defined(USE_ROCM)
            (void)hipGraphExecDestroy(graph_exec_);
#endif
        }
        if (graph_) {
#if defined(USE_CUDA)
            (void)cudaGraphDestroy(graph_);
#elif defined(USE_ROCM)
            (void)hipGraphDestroy(graph_);
#endif
        }
        if (capture_stream_) {
#if defined(USE_CUDA)
            (void)cudaStreamDestroy(capture_stream_);
#elif defined(USE_ROCM)
            (void)hipStreamDestroy(capture_stream_);
#endif
        }
#endif
    }

    GraphExecutor(const GraphExecutor&) = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;

    Tensor<T> step(const Tensor<T>& x, const Tensor<T>& y) {
        if (state_ == State::GRAPH_READY && x.shape() != captured_batch_shape_)
            return eager_step(x, y);

        if (state_ == State::NEEDS_WARMUP) {
            Tensor<T> loss = eager_step(x, y);
            static_x_ = Tensor<T>(x.shape(), x.device());
            static_y_ = Tensor<T>(y.shape(), y.device());
            captured_batch_shape_ = x.shape();
            state_ = State::NEEDS_CAPTURE;
            return loss;
        }

        if (state_ == State::NEEDS_CAPTURE) {
#if defined(USE_CUDA)
            GPU_CHECK(cudaMemcpyAsync(static_x_.data(), x.data(),
                            x.total_elements() * sizeof(T),
                            cudaMemcpyDeviceToDevice, capture_stream_));
            GPU_CHECK(cudaMemcpyAsync(static_y_.data(), y.data(),
                            y.total_elements() * sizeof(T),
                            cudaMemcpyDeviceToDevice, capture_stream_));
            (void)cudaStreamSynchronize(capture_stream_);
            (void)cudaDeviceSynchronize();
#elif defined(USE_ROCM)
            GPU_CHECK(hipMemcpyAsync(static_x_.data(), x.data(),
                            x.total_elements() * sizeof(T),
                            hipMemcpyDeviceToDevice, capture_stream_));
            GPU_CHECK(hipMemcpyAsync(static_y_.data(), y.data(),
                            y.total_elements() * sizeof(T),
                            hipMemcpyDeviceToDevice, capture_stream_));
            (void)hipStreamSynchronize(capture_stream_);
            (void)hipDeviceSynchronize();
#endif
            capture_graph();
            // Device sync after capture: cudaStreamEndCapture removes the captured
            // operations from the stream, so syncing the stream returns immediately.
            // The captured GPU work (forward/backward/step) may still be in-flight
            // on the device — a device-wide sync ensures static_loss_'s data and all
            // parameter/velocity updates are visible before the caller reads them.
#if defined(USE_CUDA)
            (void)cudaDeviceSynchronize();
#elif defined(USE_ROCM)
            (void)hipDeviceSynchronize();
#endif
            state_ = State::GRAPH_READY;
            // Don't replay here — capture_graph() already executed the full
            // forward→backward→step. Replaying would double-apply the step,
            // overfitting params to this batch and exploding loss on the next.
            return static_loss_;
        }

        return replay(x, y);
    }

private:
    enum class State { NEEDS_WARMUP, NEEDS_CAPTURE, GRAPH_READY };
    State state_ = State::NEEDS_WARMUP;

    std::shared_ptr<Module<T>> model_;
    Optimizer<T>&              optimizer_;
    LossFn                     loss_fn_;

    Tensor<T>         static_x_;
    Tensor<T>         static_y_;
    Tensor<T>         static_loss_;
    std::vector<size_t> captured_batch_shape_;

    gpuGraph_t         graph_       = nullptr;
    gpuGraphExec_t     graph_exec_  = nullptr;
    GpuStream_t        capture_stream_ = nullptr;

    Tensor<T> eager_step(const Tensor<T>& x, const Tensor<T>& y) {
        std::cout << "Eager...\n";
        model_->train();
        optimizer_.zero_grad();
        Tensor<T> preds = model_->forward(x);
        Tensor<T> loss  = loss_fn_(preds, y);
        loss.backward();
        optimizer_.step();
        return loss;
    }

    void capture_graph() {
        std::cout << "Capture...\n";
#if defined(USE_CUDA)
        (void)cudaDeviceSynchronize();
        GPU_CHECK(cudaStreamBeginCapture(
            capture_stream_, cudaStreamCaptureModeRelaxed));

        active_stream() = capture_stream_;
        optimizer_.zero_grad();
        Tensor<T> preds = model_->forward(static_x_);
        static_loss_    = loss_fn_(preds, static_y_);
        static_loss_.backward();
        optimizer_.step();
        active_stream() = nullptr;

        GPU_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));
        GPU_CHECK(cudaGraphInstantiate(
            &graph_exec_, graph_, nullptr, nullptr, 0));
#elif defined(USE_ROCM)
        (void)hipDeviceSynchronize();
        GPU_CHECK(hipStreamBeginCapture(
            capture_stream_, hipStreamCaptureModeRelaxed));

        active_stream() = capture_stream_;
        optimizer_.zero_grad();
        Tensor<T> preds = model_->forward(static_x_);
        static_loss_    = loss_fn_(preds, static_y_);
        static_loss_.backward();
        optimizer_.step();
        active_stream() = nullptr;

        GPU_CHECK(hipStreamEndCapture(capture_stream_, &graph_));
        GPU_CHECK(hipGraphInstantiate(
            &graph_exec_, graph_, nullptr, nullptr, 0));
#endif
    }

    Tensor<T> replay(const Tensor<T>& x, const Tensor<T>& y) {
        // std::cout << "Replay...\n";
#if defined(USE_CUDA)
        GPU_CHECK(cudaMemcpyAsync(static_x_.data(), x.data(),
                        x.total_elements() * sizeof(T),
                        cudaMemcpyDeviceToDevice, capture_stream_));
        GPU_CHECK(cudaMemcpyAsync(static_y_.data(), y.data(),
                        y.total_elements() * sizeof(T),
                        cudaMemcpyDeviceToDevice, capture_stream_));

        GPU_CHECK(cudaGraphLaunch(graph_exec_, capture_stream_));
        (void)cudaStreamSynchronize(capture_stream_);
#elif defined(USE_ROCM)
        GPU_CHECK(hipMemcpyAsync(static_x_.data(), x.data(),
                        x.total_elements() * sizeof(T),
                        hipMemcpyDeviceToDevice, capture_stream_));
        GPU_CHECK(hipMemcpyAsync(static_y_.data(), y.data(),
                        y.total_elements() * sizeof(T),
                        hipMemcpyDeviceToDevice, capture_stream_));

        GPU_CHECK(hipGraphLaunch(graph_exec_, capture_stream_));
        (void)hipStreamSynchronize(capture_stream_);
#endif

        model_->train();
        return static_loss_;
    }
};
#endif // defined(USE_CUDA) || defined(USE_ROCM)
