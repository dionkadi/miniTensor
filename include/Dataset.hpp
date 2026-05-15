#pragma once

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>
#include <random>
#include <algorithm>

#include "TensorOps.hpp"

template<typename T> class Tensor;

template<typename T>
class Dataset {
public:
    virtual ~Dataset() = default;
    
    // Returns the total number of samples
    virtual size_t size() const = 0;
    
    // Returns a single sample (e.g., a feature tensor and a label tensor)
    virtual std::pair<Tensor<T>, Tensor<T>> get(size_t index) const = 0;
};

template<typename T>
class TensorDataset : public Dataset<T> {
private:
    Tensor<T> features_;
    Tensor<T> labels_;

public:
    TensorDataset(Tensor<T> features, Tensor<T> labels) 
        : features_(features), labels_(labels) 
    {
        assert(features_.shape()[0] == labels_.shape()[0]);
    }

    size_t size() const override {
        return features_.shape()[0];
    }

    std::pair<Tensor<T>, Tensor<T>> get(size_t index) const override {
        return { features_.slice(0, index, index + 1), labels_.slice(0, index, index + 1) };
    }
};

template<typename T>
std::pair<Tensor<T>, Tensor<T>> default_collate(const std::vector<std::pair<Tensor<T>, Tensor<T>>>& batch_items) {
    std::vector<Tensor<T>> batch_features;
    std::vector<Tensor<T>> batch_labels;
    
    for (const auto& item : batch_items) {
        batch_features.push_back(item.first);
        batch_labels.push_back(item.second);
    }
    
    // Concat along dimension 0
    return { stack(batch_features, 0), stack(batch_labels, 0) };
}


template<typename T>
class DataLoader {
private:
    const Dataset<T>& dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;

public:
    DataLoader(const Dataset<T>& dataset, size_t batch_size, bool shuffle = true)
        : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle) 
    {
        indices_.resize(dataset_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
    }

    class Iterator {
    private:
        DataLoader* loader_;
        size_t current_idx_;

    public:
        Iterator(DataLoader* loader, size_t start_idx) : loader_(loader), current_idx_(start_idx) {}

        // Overload != to know when to stop the loop
        bool operator!=(const Iterator& other) const {
            return current_idx_ < other.current_idx_;
        }

        // Overload ++ to advance the batch
        Iterator& operator++() {
            current_idx_ += loader_->batch_size_;
            return *this;
        }

        // Overload * to fetch the actual batch of data
        std::pair<Tensor<T>, Tensor<T>> operator*() const {
            size_t end_idx = std::min(current_idx_ + loader_->batch_size_, loader_->indices_.size());
            
            std::vector<std::pair<Tensor<T>, Tensor<T>>> batch_items;
            for (size_t i = current_idx_; i < end_idx; ++i) {
                size_t actual_data_index = loader_->indices_[i];
                batch_items.push_back(loader_->dataset_.get(actual_data_index));
            }
            
            return default_collate(batch_items);
        }
    };

    // Called automatically at the start of a range-based for loop
    Iterator begin() {
        if (shuffle_) {
            std::mt19937 g(std::random_device{}());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }
        return Iterator(this, 0);
    }

    // Called automatically to check loop termination
    Iterator end() {
        return Iterator(this, dataset_.size());
    }
};