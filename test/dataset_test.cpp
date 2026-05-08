#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "Tensor.hpp"
#include "TensorOps.hpp"
#include "Dataset.hpp"

// --- Helper Functions ---
template<typename T>
void fill_tensor_sequential(Tensor<T>& t, T start_val = 0) {
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
    Tensor<T> contig_t = cpu_t.contiguous();
    
    const T* ptr = contig_t.data() + contig_t.offset();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(ptr[i], expected[i]) << "Mismatch at flat index " << i;
    }
}

// ==========================================
// Test Suite: TensorDataset (CPU)
// ==========================================

TEST(DatasetTest, TensorDatasetBasic) {
    Device cpu{DeviceType::CPU};
    
    // Create a dataset of 5 samples, each with 3 features
    Tensor<float> features({5, 3}, cpu);
    fill_tensor_sequential(features, 0.0f); // 0..14
    
    // 5 labels
    Tensor<float> labels({5, 1}, cpu);
    fill_tensor_sequential(labels, 100.0f); // 100..104

    TensorDataset<float> dataset(features, labels);

    EXPECT_EQ(dataset.size(), 5);

    // Fetch index 2
    auto sample = dataset.get(2);
    
    EXPECT_EQ(sample.first.shape(), (std::vector<size_t>{1, 3}));
    EXPECT_EQ(sample.second.shape(), (std::vector<size_t>{1, 1}));
    
    // Features for index 2 should be [6, 7, 8]
    expect_tensor_data(sample.first, {6.0f, 7.0f, 8.0f});
    // Label for index 2 should be [102]
    expect_tensor_data(sample.second, {102.0f});
}

// ==========================================
// Test Suite: default_collate (CPU)
// ==========================================

TEST(DatasetTest, DefaultCollate) {
    Device cpu{DeviceType::CPU};
    
    // Mock 3 individual samples fetched from a dataset
    Tensor<float> f1({1, 2}, cpu); fill_tensor_sequential(f1, 1.0f); // [1, 2]
    Tensor<float> l1({1, 1}, cpu); fill_tensor_sequential(l1, 10.0f); // [10]
    
    Tensor<float> f2({1, 2}, cpu); fill_tensor_sequential(f2, 3.0f); // [3, 4]
    Tensor<float> l2({1, 1}, cpu); fill_tensor_sequential(l2, 20.0f); // [20]
    
    Tensor<float> f3({1, 2}, cpu); fill_tensor_sequential(f3, 5.0f); // [5, 6]
    Tensor<float> l3({1, 1}, cpu); fill_tensor_sequential(l3, 30.0f); // [30]

    std::vector<std::pair<Tensor<float>, Tensor<float>>> batch = {
        {f1, l1}, {f2, l2}, {f3, l3}
    };

    auto collated = default_collate(batch);

    // Check stacked features
    // Note: Since each feature was already {1, 2}, stack adds a new dim at 0.
    // Result shape: {3, 1, 2}
    EXPECT_EQ(collated.first.shape(), (std::vector<size_t>{3, 1, 2}));
    expect_tensor_data(collated.first, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Check stacked labels
    EXPECT_EQ(collated.second.shape(), (std::vector<size_t>{3, 1, 1}));
    expect_tensor_data(collated.second, {10.0f, 20.0f, 30.0f});
}

// ==========================================
// Test Suite: DataLoader (CPU)
// ==========================================

TEST(DatasetTest, DataLoaderSequential) {
    Device cpu{DeviceType::CPU};
    Tensor<float> features({10, 2}, cpu); fill_tensor_sequential(features, 0.0f);
    Tensor<float> labels({10, 1}, cpu);   fill_tensor_sequential(labels, 100.0f);
    TensorDataset<float> dataset(features, labels);

    // Batch size 4 means we expect batches of sizes: 4, 4, 2
    DataLoader<float> loader(dataset, 4, /*shuffle=*/false);

    size_t batch_count = 0;
    size_t total_samples = 0;

    for (const auto& batch : loader) {
        size_t current_batch_size = batch.first.shape()[0];
        total_samples += current_batch_size;
        
        if (batch_count == 0 || batch_count == 1) {
            EXPECT_EQ(current_batch_size, 4);
        } else if (batch_count == 2) {
            EXPECT_EQ(current_batch_size, 2); // Final partial batch
        }
        batch_count++;
    }

    EXPECT_EQ(batch_count, 3);
    EXPECT_EQ(total_samples, 10);
}

TEST(DatasetTest, DataLoaderShuffle) {
    Device cpu{DeviceType::CPU};
    Tensor<float> features({20, 1}, cpu); fill_tensor_sequential(features, 0.0f);
    Tensor<float> labels({20, 1}, cpu);   fill_tensor_sequential(labels, 100.0f);
    TensorDataset<float> dataset(features, labels);

    DataLoader<float> loader(dataset, 5, /*shuffle=*/true);

    std::vector<float> seen_features;
    for (const auto& batch : loader) {
        // Copy batch to CPU and track what we've seen
        Tensor<float> cpu_features = batch.first.contiguous();
        const float* ptr = cpu_features.data() + cpu_features.offset();
        for (size_t i = 0; i < cpu_features.total_elements(); ++i) {
            seen_features.push_back(ptr[i]);
        }
    }

    EXPECT_EQ(seen_features.size(), 20);

    // Check if the data is shuffled (it shouldn't be perfectly sorted 0..19)
    bool is_sorted = true;
    for (size_t i = 1; i < seen_features.size(); ++i) {
        if (seen_features[i] < seen_features[i - 1]) {
            is_sorted = false;
            break;
        }
    }
    EXPECT_FALSE(is_sorted) << "Shuffle failed: The dataset iterated in perfect sequential order.";

    // Sort to verify all original elements are still present
    std::sort(seen_features.begin(), seen_features.end());
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_FLOAT_EQ(seen_features[i], static_cast<float>(i));
    }
}

// ==========================================
// Test Suite: GPU Integration (Conditional)
// ==========================================
#if defined(USE_CUDA) || defined(USE_ROCM)

TEST(DatasetGpuTest, TensorDatasetAndCollate) {
    Device gpu{DeviceType::CUDA}; 
    
    // 1. Create dataset entirely on GPU
    Tensor<float> cpu_f({4, 2}); fill_tensor_sequential(cpu_f, 1.0f); // [1..8]
    Tensor<float> cpu_l({4, 1}); fill_tensor_sequential(cpu_l, 10.0f); // [10..13]
    
    TensorDataset<float> dataset(cpu_f.to(gpu), cpu_l.to(gpu));

    // 2. Fetch a single GPU slice
    auto sample = dataset.get(1);
    EXPECT_EQ(sample.first.device().type, DeviceType::CUDA);
    expect_tensor_data(sample.first, {3.0f, 4.0f});

    // 3. Test Collate with GPU tensors
    std::vector<std::pair<Tensor<float>, Tensor<float>>> batch = {
        dataset.get(0), dataset.get(2)
    };

    auto collated = default_collate(batch);
    
    EXPECT_EQ(collated.first.device().type, DeviceType::CUDA);
    EXPECT_EQ(collated.first.shape(), (std::vector<size_t>{2, 1, 2}));
    expect_tensor_data(collated.first, {1.0f, 2.0f, 5.0f, 6.0f});
}

TEST(DatasetGpuTest, DataLoaderEndToEnd) {
    Device gpu{DeviceType::CUDA};
    
    Tensor<float> cpu_f({10, 2}); fill_tensor_sequential(cpu_f, 0.0f);
    Tensor<float> cpu_l({10, 1}); fill_tensor_sequential(cpu_l, 100.0f);
    
    TensorDataset<float> dataset(cpu_f.to(gpu), cpu_l.to(gpu));
    DataLoader<float> loader(dataset, 4, /*shuffle=*/false);

    size_t batch_count = 0;
    for (const auto& batch : loader) {
        EXPECT_EQ(batch.first.device().type, DeviceType::CUDA);
        EXPECT_EQ(batch.second.device().type, DeviceType::CUDA);
        
        if (batch_count == 0) {
            // First batch should have items 0, 1, 2, 3
            EXPECT_EQ(batch.first.shape(), (std::vector<size_t>{4, 1, 2}));
            expect_tensor_data(batch.first, {
                0.0f, 1.0f,  // item 0
                2.0f, 3.0f,  // item 1
                4.0f, 5.0f,  // item 2
                6.0f, 7.0f   // item 3
            });
        }
        batch_count++;
    }
    
    EXPECT_EQ(batch_count, 3);
}

#endif