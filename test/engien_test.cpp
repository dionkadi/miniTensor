#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <vector>

#include "Defines.hpp"
#include "Tensor.hpp"
#include "TensorOps.hpp"
#include "Autograd.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"

// ============================================================================
// Helper Utilities
// ============================================================================

// Helper to easily extract a scalar value from any device tensor
float get_val(const Tensor<float>& t, size_t linear_idx = 0) {
    Tensor<float> cpu_t = t.to({DeviceType::CPU});
    return cpu_t.data()[cpu_t.offset() + linear_idx];
}

// Helper to initialize a random tensor (weights) on a specific device
Tensor<float> make_random_tensor(const std::vector<size_t>& shape, Device device, float stddev = 0.1f) {
    Tensor<float> cpu_t(shape, {DeviceType::CPU});
    std::mt19937 gen(42); // Fixed seed for deterministic testing
    std::normal_distribution<float> dist(0.0f, stddev);
    
    for (size_t i = 0; i < cpu_t.total_elements(); ++i) {
        cpu_t.data()[i] = dist(gen);
    }
    
    Tensor<float> final_t = cpu_t.to(device);
    final_t.set_requires_grad(true);
    return final_t;
}

// ============================================================================
// GTest Fixture Setup
// ============================================================================

// Test fixture that will be parameterized over DeviceType (CPU and CUDA)
class AutogradEngineTest : public ::testing::TestWithParam<DeviceType> {
protected:
    Device device;

    void SetUp() override {
        device = Device(GetParam());
    }
};

// ============================================================================
// Unit Tests: Primitive Math & Autograd
// ============================================================================

TEST_P(AutogradEngineTest, AddBackward) {
    Tensor<float> a({1}, device); a.fill(2.0f); a.set_requires_grad(true);
    Tensor<float> b({1}, device); b.fill(3.0f); b.set_requires_grad(true);
    
    Tensor<float> c = a + b;
    EXPECT_FLOAT_EQ(get_val(c), 5.0f);
    
    c.backward();
    EXPECT_FLOAT_EQ(get_val(a.grad()), 1.0f);
    EXPECT_FLOAT_EQ(get_val(b.grad()), 1.0f);
}

TEST_P(AutogradEngineTest, MulBackward) {
    Tensor<float> a({1}, device); a.fill(2.0f); a.set_requires_grad(true);
    Tensor<float> b({1}, device); b.fill(3.0f); b.set_requires_grad(true);
    
    Tensor<float> c = a * b;
    EXPECT_FLOAT_EQ(get_val(c), 6.0f);
    
    c.backward();
    EXPECT_FLOAT_EQ(get_val(a.grad()), 3.0f); // dC/dA = b
    EXPECT_FLOAT_EQ(get_val(b.grad()), 2.0f); // dC/dB = a
}

TEST_P(AutogradEngineTest, MatmulBackward) {
    Tensor<float> a({2, 3}, device); a.fill(2.0f); a.set_requires_grad(true);
    Tensor<float> b({3, 2}, device); b.fill(3.0f); b.set_requires_grad(true);
    
    Tensor<float> c = matmul(a, b); 
    // c is [2, 2], each element is 2*3 + 2*3 + 2*3 = 18
    EXPECT_FLOAT_EQ(get_val(c, 0), 18.0f);
    
    c.backward(); 
    
    // dL/dA = grad_C @ B^T. grad_C is ones[2,2]. B is [3,2] of 3s. B^T is [2,3] of 3s.
    // ones @ B^T = [2,2] @ [2,3] -> each elem is 1*3 + 1*3 = 6.
    EXPECT_FLOAT_EQ(get_val(a.grad(), 0), 6.0f);
    
    // dL/dB = A^T @ grad_C -> [3,2] @ [2,2] -> each elem is 2*1 + 2*1 = 4.
    EXPECT_FLOAT_EQ(get_val(b.grad(), 0), 4.0f);
}

// ============================================================================
// Unit Tests: Activations
// ============================================================================

TEST_P(AutogradEngineTest, ReLUTest) {
    Tensor<float> a({2}, device); 
    Tensor<float> cpu_a({2}, {DeviceType::CPU});
    cpu_a.data()[0] = -1.0f; 
    cpu_a.data()[1] =  2.0f;
    a = cpu_a.to(device);
    a.set_requires_grad(true);

    Tensor<float> c = relu(a);
    EXPECT_FLOAT_EQ(get_val(c, 0), 0.0f);
    EXPECT_FLOAT_EQ(get_val(c, 1), 2.0f);

    c.backward();
    EXPECT_FLOAT_EQ(get_val(a.grad(), 0), 0.0f); // grad of negative is 0
    EXPECT_FLOAT_EQ(get_val(a.grad(), 1), 1.0f); // grad of positive is 1
}

TEST_P(AutogradEngineTest, SigmoidTest) {
    Tensor<float> a({1}, device); 
    a.fill(0.0f); 
    a.set_requires_grad(true);

    Tensor<float> c = sigmoid(a);
    EXPECT_FLOAT_EQ(get_val(c), 0.5f); // sigmoid(0) = 0.5

    c.backward();
    // ds/dx = s * (1 - s) = 0.5 * 0.5 = 0.25
    EXPECT_FLOAT_EQ(get_val(a.grad()), 0.25f);
}

// ============================================================================
// Integration Test: End-to-End MLP Training (XOR Problem)
// ============================================================================

TEST_P(AutogradEngineTest, TrainXORNetwork) {
    // 1. Prepare XOR Data
    Tensor<float> X_cpu({4, 2}, {DeviceType::CPU});
    X_cpu.data()[0] = 0; X_cpu.data()[1] = 0;
    X_cpu.data()[2] = 0; X_cpu.data()[3] = 1;
    X_cpu.data()[4] = 1; X_cpu.data()[5] = 0;
    X_cpu.data()[6] = 1; X_cpu.data()[7] = 1;
    Tensor<float> X = X_cpu.to(device);
    
    Tensor<float> Y_cpu({4, 1}, {DeviceType::CPU});
    Y_cpu.data()[0] = 0; Y_cpu.data()[1] = 1; 
    Y_cpu.data()[2] = 1; Y_cpu.data()[3] = 0;
    Tensor<float> Y = Y_cpu.to(device);

    // 2. Initialize Model Parameters (Hidden dim = 16)
    Tensor<float> W1 = make_random_tensor({2, 16}, device);
    Tensor<float> b1({1, 16}, device); b1.fill(0.0f); b1.set_requires_grad(true);
    
    Tensor<float> W2 = make_random_tensor({16, 1}, device);
    Tensor<float> b2({1, 1}, device); b2.fill(0.0f); b2.set_requires_grad(true);

    // 3. Setup Optimizer
    float learning_rate = 0.5f;
    SGD<float> optim({W1, b1, W2, b2}, learning_rate);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    // 4. Training Loop
    int epochs = 2000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // --- Forward Pass ---
        // Hidden Layer
        Tensor<float> z1 = matmul(X, W1) + b1;
        Tensor<float> a1 = tanh(z1); // Tanh works better for XOR convergence than ReLU
        
        // Output Layer
        Tensor<float> z2 = matmul(a1, W2) + b2;
        Tensor<float> pred = sigmoid(z2);
        
        // Loss Computation
        Tensor<float> loss = mse_loss(pred, Y);
        
        // --- Logging ---
        float current_loss = get_val(loss);
        if (epoch == 0) initial_loss = current_loss;
        // if (epoch % 500 == 0) {
        //     std::cout << "Epoch " << epoch << " Loss: " << current_loss << std::endl;
        // }
        final_loss = current_loss;

        // --- Backward Pass ---
        optim.zero_grad();
        loss.backward();

        // std::cout << "Epoch 0 after backward:" << std::endl;
        // std::cout << "  W1.grad empty? " << W1.grad().empty() << std::endl;
        // if (!W1.grad().empty()) {
        //     std::cout << "  W1.grad[0] = " << get_val(W1.grad(), 0) << std::endl;
        //     std::cout << "  W1 data[0] before step = " << get_val(W1, 0) << std::endl;
        // }
        // std::cout << "  W2.grad empty? " << W2.grad().empty() << std::endl;
        // if (!W2.grad().empty()) {
        //     std::cout << "  W2.grad[0] = " << get_val(W2.grad(), 0) << std::endl;
        //     std::cout << "  W2 data[0] before step = " << get_val(W2, 0) << std::endl;
        // }
        
        // --- Optimizer Step ---
        optim.step();

        // std::cout << "Epoch 0 after optim.step():" << std::endl;
        // std::cout << "  W1 data[0] after step = " << get_val(W1, 0) << std::endl;
        // std::cout << "  W2 data[0] after step = " << get_val(W2, 0) << std::endl;
    }

    // 5. Verification
    // Expect the network to easily converge and drop the loss significantly
    std::cout << "Initial Loss: " << initial_loss << " | Final Loss: " << final_loss << std::endl;
    EXPECT_LT(final_loss, 0.05f); 
    EXPECT_LT(final_loss, initial_loss);

    // Check predictions
    // Forward pass one last time
    Tensor<float> z1 = matmul(X, W1) + b1;
    Tensor<float> a1 = tanh(z1);
    Tensor<float> z2 = matmul(a1, W2) + b2;
    Tensor<float> final_preds = sigmoid(z2);

    Tensor<float> preds_cpu = final_preds.to({DeviceType::CPU});
    // XOR should yield ~0, ~1, ~1, ~0
    EXPECT_LT(preds_cpu.data()[0], 0.2f);
    EXPECT_GT(preds_cpu.data()[1], 0.8f);
    EXPECT_GT(preds_cpu.data()[2], 0.8f);
    EXPECT_LT(preds_cpu.data()[3], 0.2f);
}

// ============================================================================
// Parameterize & Instantiate Test Suites
// ============================================================================

// If built with GPU backends, run all tests on BOTH CPU and GPU
#if defined(USE_CUDA) || defined(USE_ROCM)
INSTANTIATE_TEST_SUITE_P(
    AllDevices,
    AutogradEngineTest,
    ::testing::Values(DeviceType::CPU, DeviceType::CUDA)
);
#else
// Otherwise, only run on CPU
INSTANTIATE_TEST_SUITE_P(
    CpuOnly,
    AutogradEngineTest,
    ::testing::Values(DeviceType::CPU)
);
#endif