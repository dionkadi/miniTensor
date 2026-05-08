#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "Tensor.hpp"
#include "TensorOps.hpp"
#include "Module.hpp"
#include "Autograd.hpp"

// Utility to populate a tensor with sequential data for deterministic testing
template<typename T>
void fill_sequential(Tensor<T>& t, T start = 0.0, T step = 1.0) {
    T* data = t.data() + t.offset();
    for (size_t i = 0; i < t.total_elements(); ++i) {
        data[i] = start + i * step;
    }
}

void fill_random_distinct(Tensor<float>& t) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    // Move to CPU to initialize, then back to original device
    Tensor<float> temp(t.shape(), {DeviceType::CPU});
    float* ptr = temp.data();
    for (size_t i = 0; i < temp.total_elements(); ++i) {
        ptr[i] = dist(gen);
    }
    
    if (t.device().type != DeviceType::CPU) {
        t = temp.to(t.device());
    } else {
        t = temp;
    }
    t.set_requires_grad(true);
}

// Utility to read/write a single element transparently across CPU/GPU
template<typename T>
T read_element(const Tensor<T>& t, size_t flat_index) {
    T val;
    if (t.device().type == DeviceType::CPU) {
        val = t.data()[t.offset() + flat_index];
    } else {
#if defined(USE_CUDA)
        cudaMemcpy(&val, t.data() + t.offset() + flat_index, sizeof(T), cudaMemcpyDeviceToHost);
#elif defined(USE_ROCM)
        GPU_CHECK(hipMemcpy(&val, t.data() + t.offset() + flat_index, sizeof(T), hipMemcpyDeviceToHost));
#endif
    }
    return val;
}

template<typename T>
void write_element(Tensor<T>& t, size_t flat_index, T val) {
    if (t.device().type == DeviceType::CPU) {
        t.data()[t.offset() + flat_index] = val;
    } else {
#if defined(USE_CUDA)
        cudaMemcpy(t.data() + t.offset() + flat_index, &val, sizeof(T), cudaMemcpyHostToDevice);
#elif defined(USE_ROCM)
        GPU_CHECK(hipMemcpy(t.data() + t.offset() + flat_index, &val, sizeof(T), hipMemcpyHostToDevice));
#endif
    }
}

template<typename Func>
void check_gradients_numerical(
    std::vector<Tensor<float>>& all_params, // Pass all params to safely clear them
    Tensor<float>& param_to_check, 
    Func compute_loss, 
    float eps = 1e-3f, 
    float atol = 1e-3f, 
    float rtol = 1e-2f) 
{
    // 1. Zero gradients for ALL parameters involved in the graph
    for(auto& p : all_params) {
        p.zero_grad();
    }
    
    // 2. Run standard forward and backward pass
    Tensor<float> analytical_loss = compute_loss();
    analytical_loss.backward();
    
    Tensor<float> analytical_grad = param_to_check.grad();
    ASSERT_FALSE(analytical_grad.empty()) << "Analytical gradient was not computed!";

    Tensor<float> ag_cpu = analytical_grad.device().type == DeviceType::CPU 
                           ? analytical_grad 
                           : analytical_grad.to({DeviceType::CPU});
    const float* ag_ptr = ag_cpu.data() + ag_cpu.offset();

    // 3. Compute numerical gradients via finite differences
    size_t total_elements = param_to_check.total_elements();
    size_t check_limit = std::min(total_elements, (size_t)100);

    for (size_t i = 0; i < check_limit; ++i) {
        float orig_val = read_element(param_to_check, i);

        write_element(param_to_check, i, orig_val + eps);
        float pos_loss;
        {
            NoGradGuard guard; 
            Tensor<float> L_pos = compute_loss();
            pos_loss = read_element(L_pos, 0); 
        }

        write_element(param_to_check, i, orig_val - eps);
        float neg_loss;
        {
            NoGradGuard guard;
            Tensor<float> L_neg = compute_loss();
            neg_loss = read_element(L_neg, 0);
        }

        write_element(param_to_check, i, orig_val);

        float numerical_grad = (pos_loss - neg_loss) / (2.0f * eps);
        float actual_grad = ag_ptr[i];

        float allowed_error = atol + rtol * std::abs(numerical_grad);
        EXPECT_NEAR(numerical_grad, actual_grad, allowed_error) 
            << "Gradient mismatch at flat index " << i << "\n"
            << "Analytical: " << actual_grad << "\n"
            << "Numerical:  " << numerical_grad;
    }
}

// Utility to assert tensors are close
template<typename T>
void ExpectTensorClose(const Tensor<T>& a, const Tensor<T>& b, double rtol = 1e-4, double atol = 1e-5) {
    ASSERT_EQ(a.shape(), b.shape()) << "Tensor shapes do not match.";
    
    // Ensure data is on host for comparison
    Tensor<T> a_cpu = a.device().type == DeviceType::CPU ? a : a.to({DeviceType::CPU});
    Tensor<T> b_cpu = b.device().type == DeviceType::CPU ? b : b.to({DeviceType::CPU});
    
    const T* a_ptr = a_cpu.data() + a_cpu.offset();
    const T* b_ptr = b_cpu.data() + b_cpu.offset();
    
    for (size_t i = 0; i < a.total_elements(); ++i) {
        double expected = static_cast<double>(a_ptr[i]);
        double actual = static_cast<double>(b_ptr[i]);
        
        // Allowed error scales with the magnitude of the expected value
        double allowed_error = atol + rtol * std::abs(expected);
        
        EXPECT_NEAR(expected, actual, allowed_error) 
            << "Mismatch at flat index " << i << "\n"
            << "CPU: " << expected << "\n"
            << "GPU: " << actual << "\n"
            << "Allowed Error: " << allowed_error;
    }
}

TEST(MiniTensorConv2D, ForwardAnalytical) {
    Device cpu{DeviceType::CPU};
    
    // Input: [1, 1, 3, 3]
    Tensor<float> input({1, 1, 3, 3}, cpu);
    fill_sequential(input, 1.0f, 1.0f); // 1, 2, 3 ... 9
    
    // Weight: [1, 1, 2, 2]
    Tensor<float> weight({1, 1, 2, 2}, cpu);
    weight.fill(1.0f); // Simple sum filter
    
    // Bias: [1]
    Tensor<float> bias({1}, cpu);
    bias.fill(0.0f);

    // K=2, stride=1, padding=0. Output shape: [1, 1, 2, 2]
    Tensor<float> output = conv2d(input, weight, bias, 1, 0);

    ASSERT_EQ(output.shape(), (std::vector<size_t>{1, 1, 2, 2}));
    
    // Expected output:
    // Window 1: 1+2+4+5 = 12
    // Window 2: 2+3+5+6 = 16
    // Window 3: 4+5+7+8 = 24
    // Window 4: 5+6+8+9 = 28
    const float* out_ptr = output.data();
    EXPECT_FLOAT_EQ(out_ptr[0], 12.0f);
    EXPECT_FLOAT_EQ(out_ptr[1], 16.0f);
    EXPECT_FLOAT_EQ(out_ptr[2], 24.0f);
    EXPECT_FLOAT_EQ(out_ptr[3], 28.0f);
}

TEST(MiniTensorMaxPool, ForwardAnalytical) {
    Device cpu{DeviceType::CPU};
    
    // Input: [1, 1, 4, 4]
    Tensor<float> input({1, 1, 4, 4}, cpu);
    fill_sequential(input, 1.0f, 1.0f); // 1 through 16
    
    // k=2, stride=2, padding=0. Output shape: [1, 1, 2, 2]
    Tensor<float> output = max_pool2d(input, 2, 2, 0);
    
    ASSERT_EQ(output.shape(), (std::vector<size_t>{1, 1, 2, 2}));
    
    const float* out_ptr = output.data();
    // In a 4x4 filled 1-16 sequentially, the maxes in 2x2 non-overlapping blocks are:
    EXPECT_FLOAT_EQ(out_ptr[0], 6.0f);
    EXPECT_FLOAT_EQ(out_ptr[1], 8.0f);
    EXPECT_FLOAT_EQ(out_ptr[2], 14.0f);
    EXPECT_FLOAT_EQ(out_ptr[3], 16.0f);
}

#if defined(USE_CUDA) || defined(USE_ROCM)

TEST(MiniTensorDeviceEquivalence, Conv2DForward) {
    Device cpu{DeviceType::CPU};
    Device gpu{DeviceType::CUDA}; // Or ROCm based on your config

    // Random input and weights
    Tensor<float> input_cpu({4, 3, 32, 32}, cpu);
    Tensor<float> weight_cpu({16, 3, 3, 3}, cpu);
    Tensor<float> bias_cpu({16}, cpu);
    
    fill_sequential(input_cpu, 0.1f, 0.01f);
    fill_sequential(weight_cpu, -0.5f, 0.02f);
    bias_cpu.fill(0.1f);

    Tensor<float> input_gpu = input_cpu.to(gpu);
    Tensor<float> weight_gpu = weight_cpu.to(gpu);
    Tensor<float> bias_gpu = bias_cpu.to(gpu);

    auto out_cpu = conv2d(input_cpu, weight_cpu, bias_cpu, 2, 1); // stride 2, pad 1
    auto out_gpu = conv2d(input_gpu, weight_gpu, bias_gpu, 2, 1);

    ExpectTensorClose(out_cpu, out_gpu, 1e-3, 1e-4);
}

TEST(MiniTensorDeviceEquivalence, MaxPoolBackward) {
    Device cpu{DeviceType::CPU};
    Device gpu{DeviceType::CUDA};

    Tensor<float> input_cpu({2, 64, 16, 16}, cpu);
    fill_sequential(input_cpu, -10.0f, 0.5f);
    
    // We must manually trigger requires_grad setup for standalone test
    input_cpu.set_requires_grad(true);
    Tensor<float> input_gpu = input_cpu.to(gpu);
    input_gpu.set_requires_grad(true);

    auto out_cpu = max_pool2d(input_cpu, 2, 2, 0);
    auto out_gpu = max_pool2d(input_gpu, 2, 2, 0);

    // Create a dummy gradient representing upstream loss
    Tensor<float> grad_out_cpu = Tensor<float>::ones_like(out_cpu.shape(), cpu);
    Tensor<float> grad_out_gpu = Tensor<float>::ones_like(out_gpu.shape(), gpu);

    out_cpu.grad_fn()->apply(grad_out_cpu);
    out_gpu.grad_fn()->apply(grad_out_gpu);

    ExpectTensorClose(input_cpu.grad(), input_gpu.grad(), 1e-4);
}

#endif

TEST(MiniTensorStrided, Conv2DStridedExecution) {
    Device cpu{DeviceType::CPU};
    
    // Create an input where channels are in the last dimension [N, H, W, C]
    Tensor<float> nhwc_input({2, 16, 16, 3}, cpu);
    fill_sequential(nhwc_input, 0.0f, 0.1f);
    
    // Transpose to [N, C, H, W]. This creates a view with non-contiguous strides!
    Tensor<float> nchw_input = nhwc_input.transpose(1, 3).transpose(2, 3);
    
    EXPECT_FALSE(nchw_input.is_contiguous());

    Tensor<float> weight({8, 3, 3, 3}, cpu);
    weight.fill(0.5f);
    Tensor<float> bias({8}, cpu);
    bias.fill(0.0f);

    // If the strided dispatcher is broken, this will segfault or return garbage.
    Tensor<float> output = conv2d(nchw_input, weight, bias, 1, 1);
    
    // Validate shape is standard
    ASSERT_EQ(output.shape(), (std::vector<size_t>{2, 8, 16, 16}));
}

// -----------------------------------------------------------------
// Conv2D Tests
// -----------------------------------------------------------------
void run_conv2d_gradcheck(Device device) {
    Tensor<float> input({2, 3, 5, 5}, device);
    Tensor<float> weight({4, 3, 3, 3}, device);
    Tensor<float> bias({4}, device);

    fill_random_distinct(input);
    fill_random_distinct(weight);
    fill_random_distinct(bias);

    std::vector<Tensor<float>> params = {input, weight, bias};

    // The forward pass encapsulated in a lambda
    auto compute_loss = [&]() {
        Tensor<float> out = conv2d(input, weight, bias, 1, 0);
        // Reduce to scalar loss: sum of all elements
        for (int i = out.shape().size() - 1; i >= 0; --i) {
            out = sum(out, i, false);
        }
        return out;
    };

    // Check all three tensors!
    check_gradients_numerical(params, input, compute_loss);
    check_gradients_numerical(params, weight, compute_loss);
    check_gradients_numerical(params, bias, compute_loss);
}

TEST(GradCheck, Conv2D_CPU) {
    run_conv2d_gradcheck({DeviceType::CPU});
}

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(GradCheck, Conv2D_GPU) {
    run_conv2d_gradcheck({DeviceType::CUDA});
}
#endif


// -----------------------------------------------------------------
// MaxPool2D Tests
// -----------------------------------------------------------------
void run_maxpool2d_gradcheck(Device device) {
    Tensor<float> input({2, 3, 8, 8}, device);
    
    // For MaxPool, values must be distinctly separated. If two values in a 
    // pooling window are identical, adding `eps` to one might change the max element,
    // making the finite difference check unstable.
    fill_sequential(input, -1.0f, 0.001f);
    input.set_requires_grad(true);

    std::vector<Tensor<float>> params = {input};

    auto compute_loss = [&]() {
        Tensor<float> out = max_pool2d(input, 2, 2, 0); // 2x2 pool, stride 2
        
        // Let's make the loss function a bit more complex to test chain rule:
        // Loss = sum(out * out)
        out = out * out; 
        
        for (int i = out.shape().size() - 1; i >= 0; --i) {
            out = sum(out, i, false);
        }
        return out;
    };

    check_gradients_numerical(params, input, compute_loss);
}

TEST(GradCheck, MaxPool2D_CPU) {
    run_maxpool2d_gradcheck({DeviceType::CPU});
}

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(GradCheck, MaxPool2D_GPU) {
    run_maxpool2d_gradcheck({DeviceType::CUDA});
}
#endif