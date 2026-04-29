#pragma once

#include <vector>
#include <numeric>
#include <memory>

#include "Defines.hpp"
#include "GpuUtils.hpp"

template<typename T> class Tensor;
template<typename T> struct AutogradNode;

// #############################
// #
// #############################
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
                (void)cudaFree(ptr);
#elif defined(USE_ROCM)
                (void)hipFree(ptr);
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

    void fill(const T& val) {
#if defined(USE_CUDA)
    GPU_CHECK(cudaMemset(data_.get(), val, size_));
#elif defined(USE_ROCM)
    GPU_CHECK(hipMemset(data_.get(), val, size_));
#endif
    }

private:
    std::shared_ptr<T> data_;
    size_t size_;
    Device device_;
};

// #############################
// #
// #############################
struct TensorInfo {
    size_t shape[MAX_DIMS];
    size_t strides[MAX_DIMS];
    size_t ndims;

    TensorInfo(const std::vector<size_t>& s, const std::vector<size_t>& st) {
        ndims = s.size();
        if (ndims > MAX_DIMS) throw std::runtime_error("Exceeded MAX_DIMS");
        for (size_t i = 0; i < ndims; ++i) {
            shape[i] = s[i];
            strides[i] = st[i];
        }
    }
};

inline void collapse_dims(TensorInfo& a, TensorInfo& b, TensorInfo& c) {
    if (a.ndims <= 1) return;

    TensorInfo na = a, nb = b, nc = c;
    size_t keep_idx = 0;

    for (size_t i = 1; i < a.ndims; ++i) {
        bool can_collapse = 
            (na.strides[keep_idx] == a.shape[i] * a.strides[i]) &&
            (nb.strides[keep_idx] == b.shape[i] * b.strides[i]) &&
            (nc.strides[keep_idx] == c.shape[i] * c.strides[i]);

        if (can_collapse) {
            na.shape[keep_idx] *= a.shape[i];
            na.strides[keep_idx] = a.strides[i]; // The stride becomes the innermost stride
            
            nb.shape[keep_idx] *= b.shape[i];
            nb.strides[keep_idx] = b.strides[i];
            
            nc.shape[keep_idx] *= c.shape[i];
            nc.strides[keep_idx] = c.strides[i];
        } else {
            // Cannot merge, so we must advance keep_idx and copy the dimension over
            keep_idx++;
            na.shape[keep_idx] = a.shape[i];
            na.strides[keep_idx] = a.strides[i];
            
            nb.shape[keep_idx] = b.shape[i];
            nb.strides[keep_idx] = b.strides[i];
            
            nc.shape[keep_idx] = c.shape[i];
            nc.strides[keep_idx] = c.strides[i];
        }
    }

    size_t final_ndims = keep_idx + 1;
    na.ndims = final_ndims;
    nb.ndims = final_ndims;
    nc.ndims = final_ndims;

    a = na; 
    b = nb; 
    c = nc;
}


// #############################
// #
// #############################
template<typename T>
struct TensorImpl {

    using Self = TensorImpl<T>;

    std::shared_ptr<TensorStorage<T>> storage_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_;
    
    // Autograd state
    Tensor<T> grad_; 
    std::shared_ptr<AutogradNode<T>> grad_fn_;
    bool requires_grad_;

    TensorImpl(std::shared_ptr<TensorStorage<T>> storage, 
               std::vector<size_t> shape, 
               std::vector<size_t> strides, 
               size_t offset,
               bool requires_grad = false)
        : storage_(std::move(storage))
        , shape_(std::move(shape))
        , strides_(std::move(strides))
        , offset_(offset)
        , requires_grad_(requires_grad)
    {}

    Device device() const noexcept { return storage_->device(); }
    size_t total_elements() const noexcept {
        return std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>{});
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

    Tensor<T> transpose(size_t dim0, size_t dim1) const {
        std::vector<size_t> new_shape = shape_;
        std::vector<size_t> new_strides = strides_;

        std::swap(new_shape[dim0], new_shape[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);

        return Tensor(std::make_shared<Self>(storage_, new_shape, new_strides, offset_));
    }

    Tensor<T> expand(const std::vector<size_t>& target_shape) const {
        if (shape_ == target_shape) return Tensor<T>(std::make_shared<Self>(storage_, shape_, strides_, offset_));

        std::vector<size_t> new_strides(target_shape.size(), 0);
        int offset = target_shape.size() - shape_.size();

        if (offset < 0) {
            throw std::invalid_argument("Cannot expand to a smaller number of dimensions.");
        }

        for (int i = target_shape.size() - 1; i >= 0; --i) {
            size_t current_dim = (i >= offset) ? shape_[i - offset] : 1;
            size_t current_stride = (i >= offset) ? strides_[i - offset] : 0;

            if (current_dim != target_shape[i] && current_dim != 1) {
                throw std::invalid_argument("Incompatible shapes for broadcasting.");
            }

            // The Magic: If we are stretching a dimension of size 1 to size N,
            // the stride becomes 0. Otherwise, keep the original stride.
            new_strides[i] = (current_dim == target_shape[i]) ? current_stride : 0;
        }

        return Tensor(std::make_shared<Self>(storage_, target_shape, new_strides, offset_));
    }

    Tensor<T> to(Device target_device) const {
        if (this->device() == target_device) {
            return Tensor<T>(std::make_shared<Self>(storage_, shape_, strides_, offset_));
        }

        size_t total_elements = storage_->size();
        auto new_storage = std::make_shared<TensorStorage<T>>(total_elements, target_device);

        size_t bytes = total_elements * sizeof(T);
        if (this->device().type == DeviceType::CPU && target_device.type == DeviceType::CUDA) {
#if defined(USE_CUDA)
            GPU_CHECK(cudaMemcpy(new_storage->data(), storage_->data(), bytes, cudaMemcpyHostToDevice));
#elif defined(USE_ROCM)
            GPU_CHECK(hipMemcpy(new_storage->data(), storage_->data(), bytes, hipMemcpyHostToDevice));
#endif
    } 
        else if (this->device().type == DeviceType::CUDA && target_device.type == DeviceType::CPU) {
#if defined(USE_CUDA)
            GPU_CHECK(cudaMemcpy(new_storage->data(), storage_->data(), bytes, cudaMemcpyDeviceToHost));
#elif defined(USE_ROCM)
            GPU_CHECK(hipMemcpy(new_storage->data(), storage_->data(), bytes, hipMemcpyDeviceToHost));
#endif
        }
        else if (this->device().type == DeviceType::CUDA && target_device.type == DeviceType::CUDA) {
#if defined(USE_CUDA)
            GPU_CHECK(cudaMemcpy(new_storage->data(), storage_->data(), bytes, cudaMemcpyDeviceToDevice));
#elif defined(USE_ROCM)
            GPU_CHECK(hipMemcpy(new_storage->data(), storage_->data(), bytes, hipMemcpyDeviceToDevice));
#endif
        }

        return Tensor<T>(std::make_shared<Self>(new_storage, shape_, strides_, offset_));
    }

    bool is_contiguous() const noexcept {
        size_t expected_stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            if (strides_[i] != expected_stride) return false;
            expected_stride *= shape_[i];
        }
        return true;
    }
};
