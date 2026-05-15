#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include "Tensor.hpp"
#include "TensorOps.hpp"

// --- Helper Functions ---
template<typename T>
void fill_tensor(Tensor<T>& t, T start_val = 1) {
    T* ptr = t.data() + t.offset();
    for (size_t i = 0; i < t.total_elements(); ++i) {
        ptr[i] = start_val + static_cast<T>(i);
    }
}

template<typename T>
void expect_tensor_data(const Tensor<T>& t, const std::vector<T>& expected) {
    ASSERT_EQ(t.total_elements(), expected.size()) << "Element count mismatch.";
    
    // Copy to CPU if necessary for verification
    Tensor<T> cpu_t = t.to({DeviceType::CPU});
    // Ensure we are reading contiguous data for the test comparison
    Tensor<T> contig_t = cpu_t.contiguous();
    
    const T* ptr = contig_t.data() + contig_t.offset();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(ptr[i], expected[i]) << "Mismatch at flat index " << i;
    }
}

template<typename T>
using Vec = std::vector<Tensor<T>>;

// ==========================================
// Test Suite: Concat
// ==========================================

TEST(TensorConcatTest, Exceptions) {
    Device cpu{DeviceType::CPU};
    Tensor<float> t1({2, 3}, cpu);
    Tensor<float> t2({2, 4}, cpu);
    Tensor<float> t3({3, 3}, cpu);
    Tensor<float> t4({2, 3, 1}, cpu);

    std::vector<Tensor<float>> empty_list;
    EXPECT_THROW(concat(empty_list, 0), std::invalid_argument);

    // Dimension mismatch (2D vs 3D)
    EXPECT_THROW(concat(Vec{t1, t4}, 0), std::invalid_argument);

    // Shape mismatch on non-concatenated dimension
    // Concat on dim 0, but dim 1 differs (3 vs 4)
    EXPECT_THROW(concat(Vec{t1, t2}, 0), std::invalid_argument);
    
    // Concat on dim 1, but dim 0 differs (2 vs 3)
    EXPECT_THROW(concat(Vec{t1, t3}, 1), std::invalid_argument);
}

TEST(TensorConcatTest, Basic1D) {
    Tensor<float> t1({2}); fill_tensor(t1, 1.0f); // [1, 2]
    Tensor<float> t2({3}); fill_tensor(t2, 3.0f); // [3, 4, 5]

    auto out = concat(Vec{t1, t2}, 0);

    EXPECT_EQ(out.shape(), std::vector<size_t>({5}));
    expect_tensor_data(out, {1, 2, 3, 4, 5});
}

TEST(TensorConcatTest, Basic2D_Dim0) {
    Tensor<float> t1({2, 2}); fill_tensor(t1, 1.0f); // [[1, 2], [3, 4]]
    Tensor<float> t2({1, 2}); fill_tensor(t2, 5.0f); // [[5, 6]]

    auto out = concat(Vec{t1, t2}, 0);

    EXPECT_EQ(out.shape(), std::vector<size_t>({3, 2}));
    expect_tensor_data(out, {1, 2, 3, 4, 5, 6});
}

TEST(TensorConcatTest, Basic2D_Dim1) {
    Tensor<float> t1({2, 2}); fill_tensor(t1, 1.0f); // [[1, 2], [3, 4]]
    Tensor<float> t2({2, 1}); fill_tensor(t2, 5.0f); // [[5], [6]]

    auto out = concat(Vec{t1, t2}, 1);

    EXPECT_EQ(out.shape(), std::vector<size_t>({2, 3}));
    // Expected: [[1, 2, 5], 
    //            [3, 4, 6]]
    expect_tensor_data(out, {1, 2, 5, 3, 4, 6});
}

TEST(TensorConcatTest, StridedInputs) {
    // Test concatenation of non-contiguous slices
    Tensor<float> base({4, 2}); fill_tensor(base, 1.0f); 
    // base: [[1, 2], [3, 4], [5, 6], [7, 8]]

    auto slice1 = base.slice(0, 0, 2); // [[1, 2], [3, 4]]
    auto slice2 = base.slice(0, 2, 4); // [[5, 6], [7, 8]]

    // Concatenate them back together along dim 1
    auto out = concat(Vec{slice1, slice2}, 1);

    EXPECT_EQ(out.shape(), std::vector<size_t>({2, 4}));
    // Expected: [[1, 2, 5, 6], 
    //            [3, 4, 7, 8]]
    expect_tensor_data(out, {1, 2, 5, 6, 3, 4, 7, 8});
}

// ==========================================
// Test Suite: Stack
// ==========================================

TEST(TensorStackTest, Exceptions) {
    Device cpu{DeviceType::CPU};
    Tensor<float> t1({2, 3}, cpu);
    Tensor<float> t2({2, 4}, cpu);

    std::vector<Tensor<float>> empty_list;
    EXPECT_THROW(stack(empty_list, 0), std::invalid_argument);

    // All shapes must be strictly identical for stack
    EXPECT_THROW(stack(Vec{t1, t2}, 0), std::invalid_argument);
}

TEST(TensorStackTest, Basic1D_Dim0) {
    Tensor<float> t1({3}); fill_tensor(t1, 1.0f); // [1, 2, 3]
    Tensor<float> t2({3}); fill_tensor(t2, 4.0f); // [4, 5, 6]

    auto out = stack(Vec{t1, t2}, 0);

    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 3}));
    // Expected: [[1, 2, 3], 
    //            [4, 5, 6]]
    expect_tensor_data(out, {1, 2, 3, 4, 5, 6});
}

TEST(TensorStackTest, Basic1D_Dim1) {
    Tensor<float> t1({3}); fill_tensor(t1, 1.0f); // [1, 2, 3]
    Tensor<float> t2({3}); fill_tensor(t2, 4.0f); // [4, 5, 6]

    auto out = stack(Vec{t1, t2}, 1);

    EXPECT_EQ(out.shape(), (std::vector<size_t>{3, 2}));
    // Expected: [[1, 4], 
    //            [2, 5], 
    //            [3, 6]]
    expect_tensor_data(out, {1, 4, 2, 5, 3, 6});
}

TEST(TensorStackTest, StridedInputs) {
    Tensor<float> base1({2, 2}); fill_tensor(base1, 1.0f); // [[1, 2], [3, 4]]
    Tensor<float> base2({2, 2}); fill_tensor(base2, 5.0f); // [[5, 6], [7, 8]]

    // Create strided views (transpose them)
    // t1: [[1, 3], [2, 4]]
    auto t1 = base1.transpose(0, 1); 
    // t2: [[5, 7], [6, 8]]
    auto t2 = base2.transpose(0, 1); 

    auto out = stack(Vec{t1, t2}, 0);

    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 2, 2}));
    // Expected: [[[1, 3], [2, 4]], 
    //            [[5, 7], [6, 8]]]
    expect_tensor_data(out, {1, 3, 2, 4, 5, 7, 6, 8});
}

// ==========================================
// Test Suite: GPU Integration (Conditional)
// ==========================================
#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(TensorGpuTest, ConcatAndStack) {
    Device gpu{DeviceType::CUDA}; // Or ROCm depending on your defines
    
    Tensor<float> t1({2, 2}, gpu);
    Tensor<float> t2({2, 2}, gpu);
    
    // Fill on CPU, then move to GPU (or assume fill() works on GPU directly based on your storage class)
    Tensor<float> cpu_t1({2, 2}); fill_tensor(cpu_t1, 1.0f);
    Tensor<float> cpu_t2({2, 2}); fill_tensor(cpu_t2, 5.0f);
    t1 = cpu_t1.to(gpu);
    t2 = cpu_t2.to(gpu);

    auto out_concat = concat(Vec{t1, t2}, 1);
    EXPECT_EQ(out_concat.device().type, DeviceType::CUDA);
    EXPECT_EQ(out_concat.shape(), (std::vector<size_t>{2, 4}));
    expect_tensor_data(out_concat, {1, 2, 5, 6, 3, 4, 7, 8});

    auto out_stack = stack(Vec{t1, t2}, 0);
    EXPECT_EQ(out_stack.device().type, DeviceType::CUDA);
    EXPECT_EQ(out_stack.shape(), (std::vector<size_t>{2, 2, 2}));
    expect_tensor_data(out_stack, {1, 2, 3, 4, 5, 6, 7, 8});
}
#endif