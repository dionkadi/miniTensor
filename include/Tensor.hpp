#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "gpuUtils.hpp"

enum class DeviceType { CPU, CUDA };

struct Device {
    DeviceType type;
    uint8_t index;

    Device(DeviceType t, uint8_t i = 0): type(t), index(i) {}

    bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }
};

template<typename T>
class TensorStorage {
public:
    explicit TensorStorage(size_t size, Device device): size_(size), device_(device) {
        T *raw = nullptr;
        if (device_.type == DeviceType::CPU) {
            raw = new T[size]();
        }
        else if (device_.type == DeviceType::CUDA) {
#if defined(USE_CUDA)
            GPU_CHECK(cudaMalloc((void**)&raw, size * sizeof(T)));
#elif defined(USE_ROCM)
            GPU_CHECK(hipMalloc((void**)&raw, size * sizeof(T)));
#endif
        }

        data_ = std::shared_ptr<T>(raw, [device] (T *ptr) {
            if (device.type == DeviceType::CPU) {
                delete [] ptr;
            }
            else if (device.type == DeviceType::CUDA) {
#if defined(USE_CUDA)
                cudaFree(ptr);
#elif defined(USE_ROCM)
                hipFree(ptr);
#endif
            }
        });
    }
    TensorStorage(std::initializer_list<T> init): data_(init) {}

    TensorStorage(const TensorStorage&) = delete;
    TensorStorage& operator=(const TensorStorage&) = delete;

    T* data() const noexcept { return data_.get(); }
    Device device() const noexcept { return device_; }
    size_t size() const noexcept { return size_; }

private:
    std::shared_ptr<T> data_;
    size_t size_;
    Device device_;
};

template<typename T>
class Tensor {
public:
    using Self = Tensor<T>;

    Tensor(std::vector<size_t> shape, Device device = {DeviceType::CPU})
        : shape_(std::move(shape))
        , offset_(0)
    {
        size_t total_elements = std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>{});
        storage_ = std::make_shared<TensorStorage<T>>(total_elements, device);
        compute_strides();
    }

    T& at(const std::vector<size_t>& indices) {
        size_t index = offset_;
        for (size_t i = 0; i < indices.size(); ++i) {
            index += indices[i] * strides_[i];
        }
        if (index >= storage_->size()) {
            throw std::out_of_range("Index out of range");
        }
        return storage_->data()[index];
    }

    Self transpose(size_t dim0, size_t dim1) const {
        std::vector<size_t> new_shape = shape_;
        std::vector<size_t> new_strides = strides_;

        std::swap(new_shape[dim0], new_shape[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);

        return Tensor(storage_, new_shape, new_strides, offset_);
    }

    Self to(Device target_device) const {
        if (this->device() == target_device) {
            return *this;
        }

        size_t total_elements = storage_->size();
        auto new_storage = std::make_shared<TensorStorage<T>>(total_elements, target_device);

        size_t bytes = total_elements * sizeof(T);
        if (this->device().type == DeviceType::CPU && target_device.type == DeviceType::CUDA) {
#if defined(USE_CUDA)
            GPU_CHECK(cudaMemcpy(new_storage->data(), storage_->data(), bytes, cudaMemcpyHostToDevice));
#elif defined(USE_ROCM)
            GPU_CHECK(hipMemcpy(new_storage->data(), storage_->data(), bytes, cudaMemcpyHostToDevice));
#endif
        } 
        else if (this->device().type == DeviceType::CUDA && target_device.type == DeviceType::CPU) {
#if defined(USE_CUDA)
            GPU_CHECK(cudaMemcpy(new_storage->data(), storage_->data(), bytes, cudaMemcpyDeviceToHost));
#elif defined(USE_ROCM)
            GPU_CHECK(hipMemcpy(new_storage->data(), storage_->data(), bytes, cudaMemcpyDeviceToHost));
#endif
        }
        else if (this->device().type == DeviceType::CUDA && target_device.type == DeviceType::CUDA) {
#if defined(USE_CUDA)
            GPU_CHECK(cudaMemcpy(new_storage->data(), storage_->data(), bytes, cudaMemcpyDeviceToDevice));
#elif defined(USE_ROCM)
            GPU_CHECK(hipMemcpy(new_storage->data(), storage_->data(), bytes, cudaMemcpyDeviceToDevice));
#endif
        }

        return Self(new_storage, shape_, strides_, offset_);
    }

    const std::vector<size_t>& shape() const noexcept { return shape_; }
    const std::vector<size_t>& strides() const noexcept { return strides_; }
    Device device() const noexcept { return storage_->device(); }

private:
    Tensor(std::shared_ptr<TensorStorage<T>> storage, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset) 
        : storage_(std::move(storage))
        , shape_(std::move(shape))
        , strides_(std::move(strides))
        , offset_(offset)
    {}

    void compute_strides() {
        strides_.resize(shape_.size());
        size_t current_stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            strides_[i] = current_stride;
            current_stride *= shape_[i];
        }
    }

    std::shared_ptr<TensorStorage<T>> storage_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_;  // starting point in storage
};