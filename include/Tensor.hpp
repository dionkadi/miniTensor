#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>
#include <unordered_set>

#include "TensorImpl.hpp"

template<typename T>
class Tensor {
public:
    using Self = Tensor<T>;

    Tensor() : impl_(nullptr) {}

    Tensor(std::vector<size_t> shape, Device device = {DeviceType::CPU}) {
        size_t total_elements = std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<size_t>{});
        auto storage = std::make_shared<TensorStorage<T>>(total_elements, device);
        auto strides = compute_strides(shape);
        impl_ = std::make_shared<TensorImpl<T>>(storage, shape, strides, 0);
    }

    Tensor(std::shared_ptr<TensorImpl<T>> impl) : impl_(std::move(impl)) {}

    static Self ones_like(const std::vector<size_t>& shape, const Device& device = {DeviceType::CPU}) {
        auto tensor = Tensor<T>(shape, device);
        tensor.fill(1U);
        return tensor;
    }

    T& at(const std::vector<size_t>& indices) { return impl_->at(indices); }
    Self transpose(size_t dim0, size_t dim1) const { return impl_->transpose(dim0, dim1); }
    Self expand(const std::vector<size_t>& target_shape) const { return impl_->expand(target_shape); }
    Self to(Device target_device) const { return impl_->to(target_device); }

    bool empty() const noexcept { return impl_ == nullptr || impl_->shape_.empty(); }
    const std::vector<size_t>& shape() const noexcept { return impl_->shape_; }
    const std::vector<size_t>& strides() const noexcept { return impl_->strides_; }
    Device device() const noexcept { return impl_->storage_->device(); }
    T* data() noexcept { return impl_->storage_->data(); }
    const T* data() const noexcept { return impl_->storage_->data(); }
    size_t total_elements() const noexcept { return impl_->total_elements(); }
    size_t offset() const noexcept { return impl_->offset_; }
    bool is_contiguous() const noexcept { return impl_->is_contiguous(); }
    void fill(const T& v) { impl_->storage_->fill(v); }

    bool requires_grad() const noexcept { return impl_->requires_grad_; }
    void set_requires_grad(bool req) noexcept { impl_->requires_grad_ = req; }
    const Self& grad() const { return impl_->grad_; }
    void accumulate_grad(const Self& gradient) {
        if (impl_->grad_.empty()) {
            impl_->grad_ = gradient;
        } else {
            impl_->grad_ = impl_->grad_ + gradient;
        }
    }
    std::shared_ptr<AutogradNode<T>> grad_fn() const noexcept { return impl_->grad_fn_; }
    void set_grad_fn(std::shared_ptr<AutogradNode<T>> fn) { impl_->grad_fn_ = fn; }

    void backward() {
        if (!requires_grad()) {
            throw std::runtime_error("Called backward on a tensor that does not require gradients.");
        }

        if (grad().empty()) {
            impl_->grad_ = Tensor<T>::ones_like(shape(), device());
        }

        std::vector<Tensor<T>> topo_order;
        std::unordered_set<TensorImpl<T>*> visited;

        std::function<void(Tensor<T>)> build_topo = [&](Tensor<T> t) {
            if (t.empty()) return;
            
            TensorImpl<T>* id = t.impl_.get(); 
            
            if (visited.count(id)) return;
            visited.insert(id);

            if (t.grad_fn()) {
                for (const auto& child : t.grad_fn()->get_inputs()) {
                    build_topo(child);
                }
            }
            topo_order.push_back(t);
        };

        build_topo(*this);

        for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
            if (it->grad_fn()) {
                // The gradient for *this* tensor was accumulated in previous iterations.
                // We pass it to apply() to distribute to its inputs.
                it->grad_fn()->apply(it->grad());
            }
        }
    }

private:
    std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size());
        size_t current_stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }
        return strides;
    }

    std::shared_ptr<TensorImpl<T>> impl_;
};