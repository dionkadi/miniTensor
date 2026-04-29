#include <gtest/gtest.h>
#include <iostream>

#include "Tensor.hpp"
#include "TensorOps.hpp"
#include "Optimizer.hpp"

TEST(IntegrationTest, LinearRegressionCPU) {
    // ---------------------------------------------------------
    // Generate Synthetic Data: y = 3x + 2
    // ---------------------------------------------------------
    size_t N = 100;
    Tensor<float> X({N, 1}); // Batch of 100, 1 feature
    Tensor<float> Y({N, 1}); // Batch of 100, 1 target
    
    for (size_t i = 0; i < N; ++i) {
        float x_val = (float)i / N; // Data scaled between 0 and 1
        X.data()[i] = x_val;
        Y.data()[i] = 3.0f * x_val + 2.0f; // Target calculation
    }

    Tensor<float> W({1, 1});
    W.data()[0] = 0.0f; // Initial terrible guess
    W.set_requires_grad(true);

    Tensor<float> b({1}); // 1D bias vector
    b.data()[0] = 0.0f; // Initial terrible guess
    b.set_requires_grad(true);

    // We use a relatively high learning rate of 0.5 because our data is small 
    // and scaled between 0 and 1.
    SGD<float> optimizer({W, b}, 0.5f); 

    int epochs = 250;
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        optimizer.zero_grad();

        // Forward Pass (pred = X @ W + b)
        Tensor<float> XW = matmul(X, W); 
        Tensor<float> pred = XW + b; // Broadcasting magic! b [1] expands to [100, 1]

        // Mean Squared Error
        Tensor<float> diff = pred - Y;
        Tensor<float> sq_diff = pow2(diff);
        
        // Sum across the batch dimension (axis 0). sq_diff is [100, 1], sum_sq becomes [1]
        Tensor<float> sum_sq = sum(sq_diff, 0, false); 
        Tensor<float> loss = mul_scalar(sum_sq, 1.0f / N);

        loss.backward();
        optimizer.step();

        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch 
                      << " | Loss: " << loss.data()[0] 
                      << " | W: " << W.data()[0] 
                      << " | b: " << b.data()[0] << std::endl;
        }
    }

    // We expect the model to have learned W ~= 3.0 and b ~= 2.0
    EXPECT_NEAR(W.data()[0], 3.0f, 0.01f);
    EXPECT_NEAR(b.data()[0], 2.0f, 0.01f);
}

TEST(IntegrationTest, LinearRegressionGPU) {
    size_t N = 100;
    Tensor<float> X_cpu({N, 1});
    Tensor<float> Y_cpu({N, 1});
    
    for (size_t i = 0; i < N; ++i) {
        float x_val = (float)i / N;
        X_cpu.data()[i] = x_val;
        Y_cpu.data()[i] = 3.0f * x_val + 2.0f;
    }

    // Set up the GPU device (Defaults to CUDA index 0)
    Device gpu_dev(DeviceType::CUDA, 0);

    // Ship data to the GPU!
    Tensor<float> X = X_cpu.to(gpu_dev);
    Tensor<float> Y = Y_cpu.to(gpu_dev);

    // ---------------------------------------------------------
    // 2. Initialize Model Parameters directly on GPU
    // ---------------------------------------------------------
    Tensor<float> W({1, 1}, gpu_dev);
    W.fill(0.0f); // Use your TensorStorage::fill to safely zero GPU memory
    W.set_requires_grad(true);

    Tensor<float> b({1}, gpu_dev);
    b.fill(0.0f); 
    b.set_requires_grad(true);

    // ---------------------------------------------------------
    // 3. Setup Optimizer
    // ---------------------------------------------------------
    SGD<float> optimizer({W, b}, 0.5f); 

    // ---------------------------------------------------------
    // 4. Training Loop (Entirely on Device)
    // ---------------------------------------------------------
    int epochs = 250;
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        optimizer.zero_grad();

        // All these operations seamlessly dispatch to your CUDA/HIP kernels
        Tensor<float> XW = matmul(X, W); 
        Tensor<float> pred = XW + b; 
        
        Tensor<float> diff = pred - Y;
        Tensor<float> sq_diff = pow2(diff);
        Tensor<float> sum_sq = sum(sq_diff, 0, false); 
        Tensor<float> loss = mul_scalar(sum_sq, 1.0f / N);

        loss.backward();
        optimizer.step();

        // Print progress every 50 epochs
        // CRITICAL: We must bring the data back to CPU to print it!
        if (epoch % 50 == 0) {
            float current_loss = loss.to(Device(DeviceType::CPU)).data()[0];
            float current_W = W.to(Device(DeviceType::CPU)).data()[0];
            float current_b = b.to(Device(DeviceType::CPU)).data()[0];

            std::cout << "Epoch " << epoch 
                      << " | GPU Loss: " << current_loss 
                      << " | W: " << current_W 
                      << " | b: " << current_b << std::endl;
        }
    }

    // ---------------------------------------------------------
    // 5. Verification
    // ---------------------------------------------------------
    // Bring the final parameters back to the host for assertions
    Tensor<float> W_final = W.to(Device(DeviceType::CPU));
    Tensor<float> b_final = b.to(Device(DeviceType::CPU));

    EXPECT_NEAR(W_final.data()[0], 3.0f, 0.01f);
    EXPECT_NEAR(b_final.data()[0], 2.0f, 0.01f);
}