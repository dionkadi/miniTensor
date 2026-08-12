// GPU kernel benchmark suite for miniTensor.
// Measures throughput (GB/s or TFLOPS) for kernels defined in gpu_ops.cpp.
//
// Build: only meaningful under USE_CUDA or USE_ROCM.
// Run:   ./build/tensor_tests
//
// Each benchmark runs N_warmup warmup iterations then N_trial timed iterations
// with device synchronization before/after. Results are reported in a table.

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#include <functional>
#include <algorithm>

#include "Defines.hpp"
#include "Tensor.hpp"
#include "TensorOps.hpp"

// ============================================================================
// Guard: GPU-only benchmark
// ============================================================================
#if !defined(USE_CUDA) && !defined(USE_ROCM)
int main() {
    std::cout << "GPU benchmark requires USE_CUDA or USE_ROCM. Rebuild with GPU support.\n";
    return 0;
}
#else

// ============================================================================
// Device synchronization helper
// ============================================================================
static void sync_device() {
#if defined(USE_CUDA)
    GPU_CHECK(cudaDeviceSynchronize());
#elif defined(USE_ROCM)
    GPU_CHECK(hipDeviceSynchronize());
#endif
}

// ============================================================================
// Random GPU tensor factory
// ============================================================================
Tensor<float> make_gpu_tensor(const std::vector<size_t>& shape) {
    Tensor<float> cpu(shape, Device{DeviceType::CPU});
    float* ptr = cpu.data() + cpu.offset();
    for (size_t i = 0; i < cpu.total_elements(); ++i)
        ptr[i] = static_cast<float>((i * 131) % 997) / 997.0f; // deterministic pseudo-random
    return cpu.to(Device{DeviceType::CUDA});
}

Tensor<float> make_gpu_tensor_val(const std::vector<size_t>& shape, float val) {
    Tensor<float> t(shape, Device{DeviceType::CUDA});
    t.fill(val);
    return t;
}

Tensor<float> make_gpu_tensor_ones(const std::vector<size_t>& shape) {
    return make_gpu_tensor_val(shape, 1.0f);
}

// ============================================================================
// Timing harness
// ============================================================================
struct BenchResult {
    std::string name;
    std::string label;        // size / problem description
    double      time_ms;      // average kernel time in ms
    double      gb_s;         // bandwidth (0 if N/A)
    double      tflops;       // compute throughput (0 if N/A)
};

static BenchResult run_bench(
    const std::string& name,
    const std::string& label,
    int warmup,
    int trials,
    std::function<void()> kernel_fn,
    double bytes_per_call = 0.0,   // bytes moved (for GB/s)
    double flops_per_call = 0.0    // FLOPs performed (for TFLOPS)
) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        kernel_fn();
        sync_device();
    }

    // Timed runs
    sync_device();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; ++i) {
        kernel_fn();
    }
    sync_device();
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / static_cast<double>(trials);

    double gb_s = 0.0, tflops = 0.0;
    if (avg_ms > 0) {
        if (bytes_per_call > 0)
            gb_s = (bytes_per_call / 1.0e9) / (avg_ms / 1.0e3);
        if (flops_per_call > 0)
            tflops = (flops_per_call / 1.0e12) / (avg_ms / 1.0e3);
    }

    return {name, label, avg_ms, gb_s, tflops};
}

// Forward declaration
static void print_result(const BenchResult& r);

// ============================================================================
// Benchmarks
// ============================================================================

// --- Elementwise ---

static void bench_unary(const std::string& op_name,
                        const std::vector<size_t>& shape,
                        std::function<void(Tensor<float>&, Tensor<float>&)> fn)
{
    size_t N = 1;
    for (auto d : shape) N *= d;
    auto a = make_gpu_tensor(shape);
    auto c = make_gpu_tensor(shape);
    double bytes = 2.0 * N * sizeof(float);   // read a, write c

    auto r = run_bench("unary/" + op_name,
                       std::to_string(N) + " elems",
                       /*warmup=*/5, /*trials=*/20,
                       [&]() { fn(a, c); },
                       bytes, 0);
    print_result(r);
}

static void bench_binary(const std::string& op_name,
                         const std::vector<size_t>& shape,
                         std::function<void(Tensor<float>&, Tensor<float>&, Tensor<float>&)> fn)
{
    size_t N = 1;
    for (auto d : shape) N *= d;
    auto a = make_gpu_tensor(shape);
    auto b = make_gpu_tensor(shape);
    auto c = make_gpu_tensor(shape);
    double bytes = 3.0 * N * sizeof(float);   // read a, b; write c

    auto r = run_bench("binary/" + op_name,
                       std::to_string(N) + " elems",
                       5, 20,
                       [&]() { fn(a, b, c); },
                       bytes, 0);
    print_result(r);
}

// --- Matmul ---

static void bench_matmul(int M, int N, int K) {
    auto a = make_gpu_tensor({static_cast<size_t>(M), static_cast<size_t>(K)});
    auto b = make_gpu_tensor({static_cast<size_t>(K), static_cast<size_t>(N)});
    auto c = make_gpu_tensor({static_cast<size_t>(M), static_cast<size_t>(N)});

    double bytes = static_cast<double>(M * K + K * N + M * N) * sizeof(float); // read a,b; write c
    double flops = 2.0 * M * N * K; // multiply + add per element

    // Compute the number of trials adaptively: target ~200ms total
    // Rough estimate: try with 1 trial first to gauge speed
    {
        sync_device();
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_gpu<float>(a, b, c);
        sync_device();
        auto t1 = std::chrono::high_resolution_clock::now();
        double one_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int trials = std::max(1, static_cast<int>(200.0 / std::max(one_ms, 0.01)));
        int warmup = std::min(5, trials);

        auto r = run_bench("matmul",
                           std::to_string(M) + "x" + std::to_string(K) + " * " +
                           std::to_string(K) + "x" + std::to_string(N),
                           warmup, trials,
                           [&]() { matmul_gpu<float>(a, b, c); },
                           bytes, flops);
        print_result(r);
    }
}

// --- BK=16 vectorized matmul ---

static void bench_matmul_bk16(int M, int N, int K) {
    auto a = make_gpu_tensor({static_cast<size_t>(M), static_cast<size_t>(K)});
    auto b = make_gpu_tensor({static_cast<size_t>(K), static_cast<size_t>(N)});
    auto c = make_gpu_tensor({static_cast<size_t>(M), static_cast<size_t>(N)});

    double bytes = static_cast<double>(M * K + K * N + M * N) * sizeof(float);
    double flops = 2.0 * M * N * K;

    {
        sync_device();
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_gpu_bk16<float>(a, b, c);
        sync_device();
        auto t1 = std::chrono::high_resolution_clock::now();
        double one_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int trials = std::max(1, static_cast<int>(200.0 / std::max(one_ms, 0.01)));
        int warmup = std::min(5, trials);

        auto r = run_bench("matmul_bk16_vec",
                           std::to_string(M) + "x" + std::to_string(K) + " * " +
                           std::to_string(K) + "x" + std::to_string(N),
                           warmup, trials,
                           [&]() { matmul_gpu_bk16<float>(a, b, c); },
                           bytes, flops);
        print_result(r);
    }
}

// --- V2 matmul (coalesced A-load, 1024 threads, reduced regs) ---

static void bench_matmul_v2(int M, int N, int K) {
    auto a = make_gpu_tensor({static_cast<size_t>(M), static_cast<size_t>(K)});
    auto b = make_gpu_tensor({static_cast<size_t>(K), static_cast<size_t>(N)});
    auto c = make_gpu_tensor({static_cast<size_t>(M), static_cast<size_t>(N)});

    double bytes = static_cast<double>(M * K + K * N + M * N) * sizeof(float);
    double flops = 2.0 * M * N * K;

    {
        sync_device();
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_gpu_v2<float>(a, b, c);
        sync_device();
        auto t1 = std::chrono::high_resolution_clock::now();
        double one_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int trials = std::max(1, static_cast<int>(200.0 / std::max(one_ms, 0.01)));
        int warmup = std::min(5, trials);

        auto r = run_bench("matmul_v2",
                           std::to_string(M) + "x" + std::to_string(K) + " * " +
                           std::to_string(K) + "x" + std::to_string(N),
                           warmup, trials,
                           [&]() { matmul_gpu_v2<float>(a, b, c); },
                           bytes, flops);
        print_result(r);
    }
}

// --- Strided matmul ---

static void bench_matmul_strided(int M, int N, int K) {
    auto a = make_gpu_tensor({static_cast<size_t>(M), static_cast<size_t>(K)});
    auto b = make_gpu_tensor({static_cast<size_t>(K), static_cast<size_t>(N)});
    auto c = make_gpu_tensor({static_cast<size_t>(M), static_cast<size_t>(N)});

    double bytes = static_cast<double>(M * K + K * N + M * N) * sizeof(float);
    double flops = 2.0 * M * N * K;

    {
        sync_device();
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_gpu_strided<float>(a, b, c);
        sync_device();
        auto t1 = std::chrono::high_resolution_clock::now();
        double one_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int trials = std::max(1, static_cast<int>(200.0 / std::max(one_ms, 0.01)));
        int warmup = std::min(5, trials);

        auto r = run_bench("matmul_strided",
                           std::to_string(M) + "x" + std::to_string(K) + " * " +
                           std::to_string(K) + "x" + std::to_string(N),
                           warmup, trials,
                           [&]() { matmul_gpu_strided<float>(a, b, c); },
                           bytes, flops);
        print_result(r);
    }
}

// --- Reduction ---

static void bench_sum(const std::vector<size_t>& shape, size_t axis) {
    auto a = make_gpu_tensor(shape);
    size_t N = a.total_elements();

    // compute output size
    std::vector<size_t> out_shape = shape;
    out_shape.erase(out_shape.begin() + axis);

    size_t out_elems = 1;
    for (auto d : out_shape) out_elems *= d;

    double bytes = static_cast<double>(N + out_elems) * sizeof(float); // read N, write out

    auto r = run_bench("sum(axis=" + std::to_string(axis) + ")",
                       std::to_string(N) + " elems",
                       5, 20,
                       [&]() { sum_gpu<float>(a, axis, false); },
                       bytes, 0);
    print_result(r);
}

// --- Conv2D forward ---

static void bench_conv2d(int N, int C_in, int C_out, int H, int W, int k, int stride, int pad) {
    auto input  = make_gpu_tensor({static_cast<size_t>(N), static_cast<size_t>(C_in),
                                   static_cast<size_t>(H), static_cast<size_t>(W)});
    auto weight = make_gpu_tensor({static_cast<size_t>(C_out), static_cast<size_t>(C_in),
                                   static_cast<size_t>(k), static_cast<size_t>(k)});
    auto bias   = make_gpu_tensor({static_cast<size_t>(C_out)});

    int H_out = (H + 2 * pad - k) / stride + 1;
    int W_out = (W + 2 * pad - k) / stride + 1;
    auto output = make_gpu_tensor({static_cast<size_t>(N), static_cast<size_t>(C_out),
                                   static_cast<size_t>(H_out), static_cast<size_t>(W_out)});

    // Calculate bytes: input, weight, bias read; output write
    double bytes = static_cast<double>(
        N * C_in * H * W +
        C_out * C_in * k * k +
        C_out +
        N * C_out * H_out * W_out
    ) * sizeof(float);

    // FLOPs: each output element is 2 * C_in * k * k MACs
    double flops = 2.0 * N * C_out * H_out * W_out * C_in * k * k;

    std::string label = std::to_string(N) + "x" + std::to_string(C_in) + "x" +
                        std::to_string(H) + "x" + std::to_string(W) +
                        " w" + std::to_string(C_out) + " k" + std::to_string(k) +
                        " s" + std::to_string(stride) + " p" + std::to_string(pad);

    auto r = run_bench("conv2d_fwd", label,
                       3, 10,
                       [&]() { conv2d_gpu<float>(input, weight, bias, output, stride, pad); },
                       bytes, flops);
    print_result(r);
}

// --- Conv2D backward ---

static void bench_conv2d_bwd(int N, int C_in, int C_out, int H, int W, int k, int stride, int pad) {
    int H_out = (H + 2 * pad - k) / stride + 1;
    int W_out = (W + 2 * pad - k) / stride + 1;

    auto input  = make_gpu_tensor({static_cast<size_t>(N), static_cast<size_t>(C_in),
                                   static_cast<size_t>(H), static_cast<size_t>(W)});
    auto weight = make_gpu_tensor({static_cast<size_t>(C_out), static_cast<size_t>(C_in),
                                   static_cast<size_t>(k), static_cast<size_t>(k)});
    auto grad_output = make_gpu_tensor({static_cast<size_t>(N), static_cast<size_t>(C_out),
                                        static_cast<size_t>(H_out), static_cast<size_t>(W_out)});
    auto grad_input  = make_gpu_tensor({static_cast<size_t>(N), static_cast<size_t>(C_in),
                                        static_cast<size_t>(H), static_cast<size_t>(W)});
    auto grad_weight = make_gpu_tensor({static_cast<size_t>(C_out), static_cast<size_t>(C_in),
                                        static_cast<size_t>(k), static_cast<size_t>(k)});
    auto grad_bias   = make_gpu_tensor({static_cast<size_t>(C_out)});

    std::string label = std::to_string(N) + "x" + std::to_string(C_in) + "->" +
                        std::to_string(C_out) + " k" + std::to_string(k) +
                        " s" + std::to_string(stride);

    // grad_input: reads grad_output + weight; writes grad_input
    {
        size_t grad_in_bytes = (N * C_out * H_out * W_out + C_out * C_in * k * k + N * C_in * H * W) * sizeof(float);
        auto r = run_bench("conv2d_bwd_input", label,
                           3, 10,
                           [&]() { conv2d_backward_input_gpu<float>(grad_output, weight, grad_input, stride, pad); },
                           static_cast<double>(grad_in_bytes), 0);
        print_result(r);
    }

    // grad_weight: reads grad_output + input; writes grad_weight
    {
        size_t grad_w_bytes = (N * C_out * H_out * W_out + N * C_in * H * W + C_out * C_in * k * k) * sizeof(float);
        double grad_w_flops = 2.0 * N * C_out * C_in * k * k * H_out * W_out;
        auto r = run_bench("conv2d_bwd_weight", label,
                           3, 5,  // few trials — weight kernel can be slow with atomics
                           [&]() { conv2d_backward_weight_gpu<float>(grad_output, input, grad_weight, stride, pad); },
                           static_cast<double>(grad_w_bytes), grad_w_flops);
        print_result(r);
    }

    // grad_bias
    {
        size_t grad_b_bytes = (N * C_out * H_out * W_out + C_out) * sizeof(float);
        auto r = run_bench("conv2d_bwd_bias", label,
                           3, 20,
                           [&]() { conv2d_backward_bias_gpu<float>(grad_output, grad_bias); },
                           static_cast<double>(grad_b_bytes), 0);
        print_result(r);
    }
}

// --- MaxPool ---

static void bench_maxpool(int N, int C, int H, int W, int k, int stride, int pad) {
    int H_out = (H + 2 * pad - k) / stride + 1;
    int W_out = (W + 2 * pad - k) / stride + 1;

    auto input = make_gpu_tensor({static_cast<size_t>(N), static_cast<size_t>(C),
                                  static_cast<size_t>(H), static_cast<size_t>(W)});

    std::string label = std::to_string(N) + "x" + std::to_string(C) + "x" +
                        std::to_string(H) + "x" + std::to_string(W) +
                        " k" + std::to_string(k) + " s" + std::to_string(stride);

    // Forward
    {
        double bytes = static_cast<double>(N * C * H * W + N * C * H_out * W_out + N * C * H_out * W_out * sizeof(size_t)) * sizeof(float);
        auto r = run_bench("maxpool_fwd", label,
                           3, 20,
                           [&]() { max_pool2d_gpu<float>(input, k, stride, pad); },
                           bytes, 0);
        print_result(r);
    }

    // Backward
    {
        auto fwd_result = max_pool2d_gpu<float>(input, k, stride, pad);
        auto& output = fwd_result.first;
        auto& indices = fwd_result.second;
        auto grad_output = make_gpu_tensor(output.shape());
        auto grad_input  = make_gpu_tensor(input.shape());

        double bwd_bytes = static_cast<double>(N * C * H_out * W_out + N * C * H * W) * sizeof(float);
        auto r = run_bench("maxpool_bwd", label,
                           3, 20,
                           [&]() { max_pool2d_backward_gpu<float>(grad_output, grad_input, indices); },
                           bwd_bytes, 0);
        print_result(r);
    }
}

// --- Batch Norm forward (mean/var) ---

static void bench_bn_fwd(int N, int C, int H, int W) {
    auto input = make_gpu_tensor({static_cast<size_t>(N), static_cast<size_t>(C),
                                  static_cast<size_t>(H), static_cast<size_t>(W)});
    auto mean = make_gpu_tensor({static_cast<size_t>(C)});
    auto var  = make_gpu_tensor({static_cast<size_t>(C)});

    double bytes = static_cast<double>(N * C * H * W + 2 * C) * sizeof(float);
    std::string label = std::to_string(N) + "x" + std::to_string(C) + "x" +
                        std::to_string(H) + "x" + std::to_string(W);

    auto r = run_bench("bn_mean_var", label,
                       5, 20,
                       [&]() { bn_fwd_gpu<float>(input, mean, var); },
                       bytes, 0);
    print_result(r);
}

// --- Fused BN+ReLU ---

static void bench_bn_relu(int N, int C, int H, int W) {
    auto input  = make_gpu_tensor({static_cast<size_t>(N), static_cast<size_t>(C),
                                   static_cast<size_t>(H), static_cast<size_t>(W)});
    auto output = make_gpu_tensor({static_cast<size_t>(N), static_cast<size_t>(C),
                                   static_cast<size_t>(H), static_cast<size_t>(W)});
    auto mean   = make_gpu_tensor_val({static_cast<size_t>(C)}, 0.0f);
    auto var    = make_gpu_tensor_val({static_cast<size_t>(C)}, 1.0f); // pre-computed
    auto gamma  = make_gpu_tensor_val({static_cast<size_t>(C)}, 1.0f);
    auto beta   = make_gpu_tensor_val({static_cast<size_t>(C)}, 0.0f);

    double bytes = static_cast<double>(N * C * H * W + N * C * H * W + 4 * C) * sizeof(float);
    std::string label = std::to_string(N) + "x" + std::to_string(C) + "x" +
                        std::to_string(H) + "x" + std::to_string(W);

    auto r = run_bench("bn_relu_fwd", label,
                       5, 20,
                       [&]() { bn_relu_fwd_gpu<float>(input, output, mean, var, gamma, beta, 1e-5f); },
                       bytes, static_cast<double>(N * C * H * W));
    print_result(r);
}

// --- Softmax ---

static void bench_softmax(int B, int C) {
    auto input = make_gpu_tensor({static_cast<size_t>(B), static_cast<size_t>(C)});
    double bytes = 2.0 * B * C * sizeof(float); // read input, write output

    auto r = run_bench("softmax",
                       std::to_string(B) + "x" + std::to_string(C),
                       5, 20,
                       [&]() { softmax_gpu<float>(input); },
                       bytes, 0);
    print_result(r);
}

// --- Cross-entropy ---

static void bench_cross_entropy(int B, int C) {
    auto logits  = make_gpu_tensor({static_cast<size_t>(B), static_cast<size_t>(C)});
    // One-hot targets from soft labels
    Tensor<float> targets_cpu({static_cast<size_t>(B), static_cast<size_t>(C)}, Device{DeviceType::CPU});
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < C; ++j)
            targets_cpu.data()[i * C + j] = (j == (i % C)) ? 1.0f : 0.0f;
    auto targets = targets_cpu.to(Device{DeviceType::CUDA});

    double bytes = static_cast<double>(B * C + B * C + 1) * sizeof(float);

    // Forward
    {
        auto r = run_bench("cross_entropy_fwd",
                           std::to_string(B) + "x" + std::to_string(C),
                           5, 20,
                           [&]() { cross_entropy_fwd_gpu<float>(logits, targets, 0.0f); },
                           bytes, 0);
        print_result(r);
    }

    // Backward
    {
        auto loss = cross_entropy_fwd_gpu<float>(logits, targets, 0.0f);
        auto grad_logits = make_gpu_tensor({static_cast<size_t>(B), static_cast<size_t>(C)});

        auto r = run_bench("cross_entropy_bwd",
                           std::to_string(B) + "x" + std::to_string(C),
                           5, 20,
                           [&]() { cross_entropy_bwd_gpu<float>(loss, logits, targets, grad_logits, 0.0f); },
                           bytes, 0);
        print_result(r);
    }
}

// --- Add+ReLU ---

static void bench_add_relu(const std::vector<size_t>& shape) {
    size_t N = 1;
    for (auto d : shape) N *= d;
    auto a = make_gpu_tensor(shape);
    auto b = make_gpu_tensor(shape);
    auto c = make_gpu_tensor(shape);

    double bytes = 3.0 * N * sizeof(float); // read a,b; write c

    auto r = run_bench("add_relu",
                       std::to_string(N) + " elems",
                       5, 20,
                       [&]() { add_relu_gpu<float>(a, b, c); },
                       bytes, 0);
    print_result(r);
}

// --- Adam ---

static void bench_adam(const std::vector<size_t>& shape) {
    size_t N = 1;
    for (auto d : shape) N *= d;
    auto param  = make_gpu_tensor(shape);
    auto grad   = make_gpu_tensor(shape);
    auto m      = make_gpu_tensor(shape);
    auto v      = make_gpu_tensor(shape);

    double bytes = 4.0 * N * sizeof(float); // read param,grad,m,v; write param,m,v

    auto r = run_bench("adam_step",
                       std::to_string(N) + " elems",
                       5, 20,
                       [&]() { adam_step_gpu<float>(param, grad, m, v,
                                                      0.001f, 0.9f, 0.999f, 1e-8f,
                                                      1.0f, 1.0f, 0.0f); },
                       bytes, 0);
    print_result(r);
}

// --- Strided copy ---

static void bench_copy_strided(const std::vector<size_t>& shape) {
    size_t N = 1;
    for (auto d : shape) N *= d;
    auto src = make_gpu_tensor(shape);
    auto dst_cpu = Tensor<float>(shape, Device{DeviceType::CPU});
    float* dst_ptr = dst_cpu.data() + dst_cpu.offset();

    double bytes = static_cast<double>(N) * sizeof(float); // just read src

    auto r = run_bench("copy_strided",
                       std::to_string(N) + " elems",
                       5, 20,
                       [&]() { copy_gpu_strided<float>(src, dst_ptr); },
                       bytes, 0);
    print_result(r);
}

// ============================================================================
// Result table printer
// ============================================================================
static std::vector<BenchResult> g_results;
static void print_result(const BenchResult& r) {
    g_results.push_back(r);
}

static void print_table() {
    std::cout << "\n";
    std::cout << "==========================================================================================\n";
    std::cout << std::left << std::setw(28) << "Kernel"
              << std::left << std::setw(28) << "Problem"
              << std::right << std::setw(12) << "Time(ms)"
              << std::right << std::setw(10) << "GB/s"
              << std::right << std::setw(10) << "TFLOPS"
              << "\n";
    std::cout << "------------------------------------------------------------------------------------------\n";
    for (const auto& r : g_results) {
        std::cout << std::left << std::setw(28) << r.name
                  << std::left << std::setw(28) << r.label
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms;
        if (r.gb_s > 0)
            std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << r.gb_s;
        else
            std::cout << std::right << std::setw(10) << "-";
        if (r.tflops > 0)
            std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << r.tflops;
        else
            std::cout << std::right << std::setw(10) << "-";
        std::cout << "\n";
    }
    std::cout << "==========================================================================================\n";
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "miniTensor GPU Kernel Benchmark Suite\n";
    std::cout << "=====================================\n\n";

    // Verify GPU is accessible
    int device_count = 0;
#if defined(USE_CUDA)
    GPU_CHECK(cudaGetDeviceCount(&device_count));
#elif defined(USE_ROCM)
    GPU_CHECK(hipGetDeviceCount(&device_count));
#endif
    if (device_count == 0) {
        std::cout << "No GPU device found. Aborting.\n";
        return 1;
    }

    std::cout << "Found " << device_count << " GPU device(s).\n\n";

    // ------------------------------------------------------------------
    // Elementwise (memory-bound)
    // ------------------------------------------------------------------
    std::cout << "--- Elementwise (memory-bound) ---\n";

    std::vector<size_t> sizes_1m  = {1 << 20};       // 1M elements
    std::vector<size_t> sizes_4m  = {1 << 22};       // 4M
    std::vector<size_t> sizes_16m = {1 << 24};       // 16M

    for (auto N : {1 << 20, 1 << 22, 1 << 24}) {
        std::vector<size_t> shape = {static_cast<size_t>(N)};

        bench_binary("add", shape,
                     [](auto& a, auto& b, auto& c) { binary_gpu<float, AddOp<float>>(a, b, c, AddOp<float>{}); });
    }

    for (auto N : {1 << 20, 1 << 22}) {
        std::vector<size_t> shape = {static_cast<size_t>(N)};

        bench_binary("mul", shape,
                     [](auto& a, auto& b, auto& c) { binary_gpu<float, MulOp<float>>(a, b, c, MulOp<float>{}); });
    }

    for (auto N : {1 << 20}) {
        std::vector<size_t> shape = {static_cast<size_t>(N)};

        bench_unary("relu", shape,
                    [](auto& a, auto& c) { unary_gpu<float, ReLUOp<float>>(a, c, ReLUOp<float>{}); });
        bench_unary("sigmoid", shape,
                    [](auto& a, auto& c) { unary_gpu<float, SigmoidOp<float>>(a, c, SigmoidOp<float>{}); });
        bench_unary("tanh", shape,
                    [](auto& a, auto& c) { unary_gpu<float, TanhOp<float>>(a, c, TanhOp<float>{}); });
        bench_unary("exp", shape,
                    [](auto& a, auto& c) { unary_gpu<float, ExpOp<float>>(a, c, ExpOp<float>{}); });
        bench_unary("sqrt", shape,
                    [](auto& a, auto& c) { unary_gpu<float, SqrtOp<float>>(a, c, SqrtOp<float>{}); });
        bench_unary("log", shape,
                    [](auto& a, auto& c) { unary_gpu<float, LogOp<float>>(a, c, LogOp<float>{}); });
    }

    // ------------------------------------------------------------------
    // Fused elementwise
    // ------------------------------------------------------------------
    std::cout << "\n--- Fused elementwise ---\n";
    bench_add_relu({1 << 20});
    bench_add_relu({1 << 22});

    // ------------------------------------------------------------------
    // Correctness check: v2 vs strided (reference)
    // ------------------------------------------------------------------
    {
        auto a = make_gpu_tensor({32, 32});
        auto b = make_gpu_tensor({32, 32});
        auto cv2 = make_gpu_tensor({32, 32});
        auto cref = make_gpu_tensor({32, 32});
        matmul_gpu_v2<float>(a, b, cv2);
        matmul_gpu_strided<float>(a, b, cref);
        auto cv2_cpu = cv2.to(Device{DeviceType::CPU});
        auto cref_cpu = cref.to(Device{DeviceType::CPU});
        float max_err = 0.0f;
        for (size_t i = 0; i < 1024; ++i) {
            float err = std::abs(cv2_cpu.data()[i] - cref_cpu.data()[i]);
            if (err > max_err) max_err = err;
        }
        std::cout << "  [v2 vs strided @ 32x32] max_err=" << max_err;
        std::cout << (max_err < 1e-4f ? "  PASS\n" : "  FAIL\n");
    }
    {
        auto a = make_gpu_tensor({1, 256});
        auto b = make_gpu_tensor({256, 1});
        auto cv2 = make_gpu_tensor({1, 1});
        auto cref = make_gpu_tensor({1, 1});
        matmul_gpu_v2<float>(a, b, cv2);
        matmul_gpu_strided<float>(a, b, cref);
        auto cv2_cpu = cv2.to(Device{DeviceType::CPU});
        auto cref_cpu = cref.to(Device{DeviceType::CPU});
        float max_err = std::abs(cv2_cpu.data()[0] - cref_cpu.data()[0]);
        std::cout << "  [v2 vs strided @ 1x256 * 256x1] max_err=" << max_err;
        std::cout << (max_err < 1e-4f ? "  PASS\n" : "  FAIL\n");
    }

    // ------------------------------------------------------------------
    // Matmul (compute-bound)
    // ------------------------------------------------------------------
    std::cout << "\n--- Matmul (compute-bound) ---\n";
    bench_matmul(32,    32,    32);
    bench_matmul(63,    4096,  4096);   // M<64 → fallback to strided
    bench_matmul(512,   512,   512);
    bench_matmul(1024,  1024,  1024);
    bench_matmul(2048,  2048,  2048);
    bench_matmul(4096,  4096,  4096);
    bench_matmul(1,     4096,  4096);   // vector * matrix, M<64 → fallback
    bench_matmul(4096,  4096,  1);      // matrix * vector, N<64 → fallback

    // BK=16 vectorized variant comparison
    std::cout << "\n  -- BK=16 vectorized vs BK=8 scalar --\n";
    bench_matmul_bk16(512,   512,   512);
    bench_matmul_bk16(1024,  1024,  1024);
    bench_matmul_bk16(2048,  2048,  2048);
    bench_matmul_bk16(4096,  4096,  4096);

    // V2 coalesced A-load variant (no heuristic — always uses 128×128 tile)
    std::cout << "\n  -- V2 (always 128×128 tile, no fallback) --\n";
    bench_matmul_v2(32,    32,    32);
    bench_matmul_v2(63,    4096,  4096);
    bench_matmul_v2(512,   512,   512);
    bench_matmul_v2(1024,  1024,  1024);
    bench_matmul_v2(2048,  2048,  2048);
    bench_matmul_v2(4096,  4096,  4096);
    bench_matmul_v2(1,     4096,  4096);   // vector * matrix
    bench_matmul_v2(4096,  4096,  1);      // matrix * vector
    bench_matmul_v2(257,   511,   383);    // non-power-of-2, non-divisible

    // Strided variant (reference)
    bench_matmul_strided(512,  512,  512);
    bench_matmul_strided(1024, 1024, 1024);

    // ------------------------------------------------------------------
    // Reduction
    // ------------------------------------------------------------------
    std::cout << "\n--- Reduction ---\n";
    bench_sum({1024, 1024}, 1);
    bench_sum({1024, 4096}, 1);
    bench_sum({1024, 1024, 64}, 1);
    bench_bn_fwd(32, 64, 56, 56);
    bench_bn_fwd(8, 256, 14, 14);

    // ------------------------------------------------------------------
    // Fused BN+ReLU
    // ------------------------------------------------------------------
    std::cout << "\n--- Fused BN+ReLU ---\n";
    bench_bn_relu(32, 64, 56, 56);
    bench_bn_relu(8, 256, 14, 14);

    // ------------------------------------------------------------------
    // Softmax
    // ------------------------------------------------------------------
    std::cout << "\n--- Softmax ---\n";
    bench_softmax(1, 1000);
    bench_softmax(128, 1000);
    bench_softmax(1024, 1000);

    // ------------------------------------------------------------------
    // Cross-entropy
    // ------------------------------------------------------------------
    std::cout << "\n--- Cross-entropy ---\n";
    bench_cross_entropy(128, 1000);
    bench_cross_entropy(64, 10000);

    // ------------------------------------------------------------------
    // Conv2D forward
    // ------------------------------------------------------------------
    std::cout << "\n--- Conv2D Forward ---\n";
    // Winograd path (3x3 s=1 p=1)
    bench_conv2d(1,  3,   64,  224, 224, 3, 1, 1);
    bench_conv2d(32, 64,  64,  56,  56,  3, 1, 1);
    bench_conv2d(8,  256, 512, 14,  14,  3, 1, 1);
    // im2col path (5x5 s=1 p=2)
    bench_conv2d(8,  64,  64,  28,  28,  5, 1, 2);
    // Strided
    bench_conv2d(32, 64,  128, 56,  56,  3, 2, 1);

    // ------------------------------------------------------------------
    // Conv2D backward
    // ------------------------------------------------------------------
    std::cout << "\n--- Conv2D Backward ---\n";
    bench_conv2d_bwd(1,  3,   64,  224, 224, 3, 1, 1);
    bench_conv2d_bwd(8,  64,  64,  56,  56,  3, 1, 1);
    bench_conv2d_bwd(8,  256, 512, 14,  14,  3, 1, 1);

    // ------------------------------------------------------------------
    // MaxPool
    // ------------------------------------------------------------------
    std::cout << "\n--- MaxPool ---\n";
    bench_maxpool(1,  64,  112, 112, 3, 2, 1);
    bench_maxpool(32, 64,  56,  56,  3, 2, 1);
    bench_maxpool(8,  256, 14,  14,  3, 2, 1);

    // ------------------------------------------------------------------
    // Adam optimizer step
    // ------------------------------------------------------------------
    std::cout << "\n--- Adam Optimizer ---\n";
    bench_adam({1 << 20});
    bench_adam({1 << 22});
    bench_adam({1 << 24});

    // ------------------------------------------------------------------
    // Strided copy
    // ------------------------------------------------------------------
    std::cout << "\n--- Strided Copy ---\n";
    bench_copy_strided({1 << 20});
    bench_copy_strided({1 << 24});

    // ------------------------------------------------------------------
    // Results table
    // ------------------------------------------------------------------
    print_table();

    return 0;
}

#endif // GPU guard
