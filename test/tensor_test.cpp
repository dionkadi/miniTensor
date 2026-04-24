#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "Dispatcher.hpp"

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

TEST(DispatchTest, CpuAddition) {
    Tensor<float> A({2, 2});
    Tensor<float> B({2, 2});

    // Populate data (we use the newly exposed .data() pointer)
    A.data()[0] = 1.0f; A.data()[1] = 2.0f; A.data()[2] = 3.0f; A.data()[3] = 4.0f;
    B.data()[0] = 5.0f; B.data()[1] = 5.0f; B.data()[2] = 5.0f; B.data()[3] = 5.0f;

    // This calls operator+ which routes through your dispatcher
    Tensor<float> C = A + B;

    // Verify metadata
    EXPECT_EQ(C.device().type, DeviceType::CPU);
    EXPECT_EQ(C.shape(), A.shape());

    // Verify math
    EXPECT_FLOAT_EQ(C.data()[0], 6.0f);
    EXPECT_FLOAT_EQ(C.data()[1], 7.0f);
    EXPECT_FLOAT_EQ(C.data()[2], 8.0f);
    EXPECT_FLOAT_EQ(C.data()[3], 9.0f);
}

// Test 2: Verify the dispatcher catches shape mismatches
TEST(DispatchTest, ShapeMismatchThrows) {
    Tensor<float> A({2, 2});
    Tensor<float> B({3, 3});

    // The dispatcher should throw an invalid_argument exception
    EXPECT_THROW({
        Tensor<float> C = A + B;
    }, std::invalid_argument);
}

#if defined(USE_CUDA) || defined(USE_ROCM)

TEST(DispatchTest, GpuAddition) {
    // 1. Setup data on CPU
    Tensor<float> cpuA({2, 2});
    Tensor<float> cpuB({2, 2});

    cpuA.data()[0] = 1.0f; cpuA.data()[1] = 2.0f; cpuA.data()[2] = 3.0f; cpuA.data()[3] = 4.0f;
    cpuB.data()[0] = 10.0f; cpuB.data()[1] = 20.0f; cpuB.data()[2] = 30.0f; cpuB.data()[3] = 40.0f;

    // 2. Move to GPU
    Device gpu_dev(DeviceType::CUDA, 0);
    Tensor<float> gpuA = cpuA.to(gpu_dev);
    Tensor<float> gpuB = cpuB.to(gpu_dev);

    // 3. Perform addition! The Dispatcher should route this to add_gpu
    Tensor<float> gpuC = gpuA + gpuB;

    EXPECT_EQ(gpuC.device().type, DeviceType::CUDA);

    // 4. Bring the result back to the CPU to verify it
    Tensor<float> result = gpuC.to(Device(DeviceType::CPU));

    EXPECT_FLOAT_EQ(result.data()[0], 11.0f);
    EXPECT_FLOAT_EQ(result.data()[1], 22.0f);
    EXPECT_FLOAT_EQ(result.data()[2], 33.0f);
    EXPECT_FLOAT_EQ(result.data()[3], 44.0f);
}

TEST(DispatchTest, DeviceMismatchThrows) {
    Tensor<float> cpuA({2, 2});
    // Create a GPU tensor directly using the chained .to() call
    Tensor<float> gpuB = Tensor<float>({2, 2}).to(Device(DeviceType::CUDA, 0));

    // The dispatcher should block adding a CPU tensor to a GPU tensor
    EXPECT_THROW({
        Tensor<float> C = cpuA + gpuB;
    }, std::invalid_argument);
}

#endif

// Test 5: Verify CPU strided addition (Contiguous + Transposed)
TEST(DispatchTest, CpuStridedAddition) {
    Tensor<float> A({2, 3});
    Tensor<float> B_raw({3, 2}); // Notice B starts as 3x2

    // Fill A (2x3)
    // [1, 2, 3]
    // [4, 5, 6]
    A.data()[0] = 1.0f; A.data()[1] = 2.0f; A.data()[2] = 3.0f;
    A.data()[3] = 4.0f; A.data()[4] = 5.0f; A.data()[5] = 6.0f;

    // Fill B_raw (3x2)
    // [10, 20]
    // [30, 40]
    // [50, 60]
    B_raw.data()[0] = 10.0f; B_raw.data()[1] = 20.0f;
    B_raw.data()[2] = 30.0f; B_raw.data()[3] = 40.0f;
    B_raw.data()[4] = 50.0f; B_raw.data()[5] = 60.0f;

    // Transpose B_raw to match A's shape (2x3)
    // Logical view of B becomes:
    // [10, 30, 50]
    // [20, 40, 60]
    Tensor<float> B = B_raw.transpose(0, 1);

    // Verify contiguity flags
    EXPECT_TRUE(A.is_contiguous());
    EXPECT_FALSE(B.is_contiguous());

    // Perform addition! Dispatcher should route to add_cpu_strided
    Tensor<float> C = A + B;

    // C should be a brand new, contiguous 2x3 tensor
    EXPECT_TRUE(C.is_contiguous());
    EXPECT_EQ(C.shape()[0], 2);
    EXPECT_EQ(C.shape()[1], 3);

    // Verify mathematical results
    EXPECT_FLOAT_EQ(C.data()[0], 11.0f); // 1 + 10
    EXPECT_FLOAT_EQ(C.data()[1], 32.0f); // 2 + 30
    EXPECT_FLOAT_EQ(C.data()[2], 53.0f); // 3 + 50
    EXPECT_FLOAT_EQ(C.data()[3], 24.0f); // 4 + 20
    EXPECT_FLOAT_EQ(C.data()[4], 45.0f); // 5 + 40
    EXPECT_FLOAT_EQ(C.data()[5], 66.0f); // 6 + 60
}

#if defined(USE_CUDA) || defined(USE_ROCM)

// Test 6: Verify GPU strided addition
TEST(DispatchTest, GpuStridedAddition) {
    Tensor<float> cpu_A({2, 3});
    Tensor<float> cpu_B_raw({3, 2});

    // Fill data on CPU
    cpu_A.data()[0] = 1.0f; cpu_A.data()[1] = 2.0f; cpu_A.data()[2] = 3.0f;
    cpu_A.data()[3] = 4.0f; cpu_A.data()[4] = 5.0f; cpu_A.data()[5] = 6.0f;

    cpu_B_raw.data()[0] = 10.0f; cpu_B_raw.data()[1] = 20.0f;
    cpu_B_raw.data()[2] = 30.0f; cpu_B_raw.data()[3] = 40.0f;
    cpu_B_raw.data()[4] = 50.0f; cpu_B_raw.data()[5] = 60.0f;

    // Move to GPU
    Device gpu_dev(DeviceType::CUDA, 0);
    Tensor<float> gpu_A = cpu_A.to(gpu_dev);
    Tensor<float> gpu_B_raw = cpu_B_raw.to(gpu_dev);

    // Transpose on GPU (This is just a metadata operation! No memory is moved)
    Tensor<float> gpu_B = gpu_B_raw.transpose(0, 1);

    EXPECT_TRUE(gpu_A.is_contiguous());
    EXPECT_FALSE(gpu_B.is_contiguous());

    // Perform strided addition on GPU
    Tensor<float> gpu_C = gpu_A + gpu_B;

    // Copy back to CPU to verify
    Tensor<float> result = gpu_C.to(Device(DeviceType::CPU));

    EXPECT_FLOAT_EQ(result.data()[0], 11.0f);
    EXPECT_FLOAT_EQ(result.data()[1], 32.0f);
    EXPECT_FLOAT_EQ(result.data()[2], 53.0f);
    EXPECT_FLOAT_EQ(result.data()[3], 24.0f);
    EXPECT_FLOAT_EQ(result.data()[4], 45.0f);
    EXPECT_FLOAT_EQ(result.data()[5], 66.0f);
}

#endif

TEST(DispatchTest, CpuBroadcasting) {
    Tensor<float> A({2, 3});
    Tensor<float> B({3}); // A 1D vector of size 3

    A.data()[0] = 1.0f; A.data()[1] = 2.0f; A.data()[2] = 3.0f;
    A.data()[3] = 4.0f; A.data()[4] = 5.0f; A.data()[5] = 6.0f;

    B.data()[0] = 10.0f; B.data()[1] = 20.0f; B.data()[2] = 30.0f;

    // B should broadcast across both rows of A
    Tensor<float> C = A + B;

    EXPECT_EQ(C.shape()[0], 2);
    EXPECT_EQ(C.shape()[1], 3);

    EXPECT_FLOAT_EQ(C.data()[0], 11.0f); // 1 + 10
    EXPECT_FLOAT_EQ(C.data()[1], 22.0f); // 2 + 20
    EXPECT_FLOAT_EQ(C.data()[2], 33.0f); // 3 + 30
    EXPECT_FLOAT_EQ(C.data()[3], 14.0f); // 4 + 10
    EXPECT_FLOAT_EQ(C.data()[4], 25.0f); // 5 + 20
    EXPECT_FLOAT_EQ(C.data()[5], 36.0f); // 6 + 30
}

TEST(DispatchTest, CpuOuterAddition) {
    Tensor<float> A({3, 1}); // Column vector
    Tensor<float> B({1, 3}); // Row vector

    A.data()[0] = 10.0f; A.data()[1] = 20.0f; A.data()[2] = 30.0f;
    B.data()[0] = 1.0f; B.data()[1] = 2.0f; B.data()[2] = 3.0f;

    // Both should broadcast to a 3x3 matrix!
    Tensor<float> C = A + B;

    EXPECT_EQ(C.shape().size(), 2);
    EXPECT_EQ(C.shape()[0], 3);
    EXPECT_EQ(C.shape()[1], 3);

    EXPECT_FLOAT_EQ(C.data()[0], 11.0f); // 10 + 1
    EXPECT_FLOAT_EQ(C.data()[1], 12.0f); // 10 + 2
    EXPECT_FLOAT_EQ(C.data()[5], 23.0f); // 20 + 3
    EXPECT_FLOAT_EQ(C.data()[8], 33.0f); // 30 + 3
}

TEST(DispatchTest, CpuMatmul) {
    Tensor<float> A({2, 3}); // 2x3
    Tensor<float> B({3, 2}); // 3x2

    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    A.data()[0] = 1; A.data()[1] = 2; A.data()[2] = 3;
    A.data()[3] = 4; A.data()[4] = 5; A.data()[5] = 6;

    // B = [[7, 8],
    //      [9, 10],
    //      [11, 12]]
    B.data()[0] = 7;  B.data()[1] = 8;
    B.data()[2] = 9;  B.data()[3] = 10;
    B.data()[4] = 11; B.data()[5] = 12;

    Tensor<float> C = matmul(A, B); // Should be 2x2

    EXPECT_EQ(C.shape()[0], 2);
    EXPECT_EQ(C.shape()[1], 2);

    EXPECT_FLOAT_EQ(C.data()[0], 58.0f);  // 1*7 + 2*9 + 3*11
    EXPECT_FLOAT_EQ(C.data()[1], 64.0f);  // 1*8 + 2*10 + 3*12
    EXPECT_FLOAT_EQ(C.data()[2], 139.0f); // 4*7 + 5*9 + 6*11
    EXPECT_FLOAT_EQ(C.data()[3], 154.0f); // 4*8 + 5*10 + 6*12
}

TEST(DispatchTest, GpuMatmul) {
    Tensor<float> A({2, 3}); // 2x3
    Tensor<float> B({3, 2}); // 3x2

    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    A.data()[0] = 1; A.data()[1] = 2; A.data()[2] = 3;
    A.data()[3] = 4; A.data()[4] = 5; A.data()[5] = 6;

    // B = [[7, 8],
    //      [9, 10],
    //      [11, 12]]
    B.data()[0] = 7;  B.data()[1] = 8;
    B.data()[2] = 9;  B.data()[3] = 10;
    B.data()[4] = 11; B.data()[5] = 12;

    Device gpu_dev(DeviceType::CUDA, 0);
    Tensor<float> gpu_A = A.to(gpu_dev);
    Tensor<float> gpu_B = B.to(gpu_dev);

    Tensor<float> gpu_C = matmul(gpu_A, gpu_B); // Should be 2x2

    Tensor<float> C = gpu_C.to(Device(DeviceType::CPU));

    EXPECT_EQ(C.shape()[0], 2);
    EXPECT_EQ(C.shape()[1], 2);

    EXPECT_FLOAT_EQ(C.data()[0], 58.0f);  // 1*7 + 2*9 + 3*11
    EXPECT_FLOAT_EQ(C.data()[1], 64.0f);  // 1*8 + 2*10 + 3*12
    EXPECT_FLOAT_EQ(C.data()[2], 139.0f); // 4*7 + 5*9 + 6*11
    EXPECT_FLOAT_EQ(C.data()[3], 154.0f); // 4*8 + 5*10 + 6*12
}