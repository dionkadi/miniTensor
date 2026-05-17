#pragma once

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>
#include <random>
#include <algorithm>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

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
    
    // Concat along dimension 0 (not stack — stack adds a new dimension)
    return { concat(batch_features, 0), concat(batch_labels, 0) };
}


template<typename T>
class ThreadSafeQueue {
    std::queue<T> q_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    bool done_ = false;
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mtx_);
        q_.push(std::move(item));
        cv_.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !q_.empty() || done_; });
        if (q_.empty()) return false;
        item = std::move(q_.front());
        q_.pop();
        return true;
    }

    void done() {
        std::lock_guard<std::mutex> lock(mtx_);
        done_ = true;
        cv_.notify_all();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return q_.empty();
    }
};

template<typename T>
class AsyncDataLoader {
    const Dataset<T>& dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;
    size_t current_idx_ = 0;
    ThreadSafeQueue<std::pair<Tensor<T>, Tensor<T>>> queue_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    size_t prefetch_count_;

    void prefetch_worker() {
        while (running_ && current_idx_ < indices_.size()) {
            size_t end_idx = std::min(current_idx_ + batch_size_, indices_.size());
            std::vector<std::pair<Tensor<T>, Tensor<T>>> batch_items;
            for (size_t i = current_idx_; i < end_idx; ++i) {
                batch_items.push_back(dataset_.get(indices_[i]));
            }
            current_idx_ = end_idx;
            queue_.push(default_collate(batch_items));
        }
        queue_.done();
    }

public:
    AsyncDataLoader(const Dataset<T>& dataset, size_t batch_size,
                    bool shuffle = true, size_t prefetch = 2)
        : dataset_(dataset), batch_size_(batch_size),
          shuffle_(shuffle), prefetch_count_(prefetch) {
        indices_.resize(dataset_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
    }

    ~AsyncDataLoader() { stop(); }

    void stop() {
        running_ = false;
        queue_.done();
        if (worker_.joinable()) worker_.join();
    }

    class Iterator {
        AsyncDataLoader* loader_;
        std::pair<Tensor<T>, Tensor<T>> current_;
        bool valid_ = false;
    public:
        Iterator(AsyncDataLoader* loader, bool begin) : loader_(loader) {
            if (begin) {
                if (loader_->queue_.pop(current_)) valid_ = true;
            }
        }

        bool operator!=(const Iterator& other) const { return valid_ || other.valid_; }

        Iterator& operator++() {
            if (loader_->queue_.pop(current_)) {
                valid_ = true;
            } else {
                valid_ = false;
            }
            return *this;
        }

        std::pair<Tensor<T>, Tensor<T>> operator*() const { return current_; }
    };

    Iterator begin() {
        if (shuffle_) {
            std::mt19937 g(std::random_device{}());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }
        current_idx_ = 0;
        running_ = true;
        worker_ = std::thread(&AsyncDataLoader::prefetch_worker, this);
        return Iterator(this, true);
    }

    Iterator end() { return Iterator(this, false); }
};

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