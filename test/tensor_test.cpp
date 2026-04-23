#include <gtest/gtest.h>
#include "Tensor.hpp" // Your header file containing the Tensor class

// Test 1: Verify CPU Tensor initialization and stride calculation
TEST(TensorTest, CpuStridesAndMemory) {
    std::vector<size_t> shape = {3, 4}; // 3x4 matrix
    Tensor<float> t(shape);

    // Check metadata
    EXPECT_EQ(t.device().type, DeviceType::CPU);
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 4);
    
    // For contiguous 3x4, strides should be [4, 1]
    EXPECT_EQ(t.strides()[0], 4);
    EXPECT_EQ(t.strides()[1], 1);

    // Assign a value using the at() method
    t.at({1, 2}) = 42.0f;
    EXPECT_FLOAT_EQ(t.at({1, 2}), 42.0f);
}

// Test 2: Verify Host-to-Device and Device-to-Host transfers
TEST(TensorTest, CudaMemoryTransfer) {
    // 1. Create a CPU tensor and fill it with dummy data
    Tensor<float> cpu_tensor({2, 2});
    cpu_tensor.at({0, 0}) = 1.0f;
    cpu_tensor.at({0, 1}) = 2.0f;
    cpu_tensor.at({1, 0}) = 3.0f;
    cpu_tensor.at({1, 1}) = 4.0f;

    // 2. Move to GPU (Host to Device)
    Device gpu_dev(DeviceType::CUDA, 0);
    Tensor<float> gpu_tensor = cpu_tensor.to(gpu_dev);

    // Verify metadata transferred correctly
    EXPECT_EQ(gpu_tensor.device().type, DeviceType::CUDA);
    EXPECT_EQ(gpu_tensor.shape()[0], 2);

    // 3. Move BACK to CPU (Device to Host) into a NEW tensor
    Tensor<float> recovered_cpu_tensor = gpu_tensor.to(Device(DeviceType::CPU));

    // 4. Assert data integrity
    EXPECT_EQ(recovered_cpu_tensor.device().type, DeviceType::CPU);
    EXPECT_FLOAT_EQ(recovered_cpu_tensor.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(recovered_cpu_tensor.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(recovered_cpu_tensor.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(recovered_cpu_tensor.at({1, 1}), 4.0f);
}