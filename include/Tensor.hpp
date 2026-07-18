#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>
#include <unordered_set>

#include "TensorImpl.hpp"

template<typename T> void add_(Tensor<T>& A, const Tensor<T>& B);

template<typename T>
class Tensor {
public:
    using Self = Tensor<T>;

    Tensor() : impl_(nullptr) {}

    Tensor(std::vector<size_t> shape, Device device = {}) {
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
    Self reshape(const std::vector<size_t>& new_shape) const { return impl_->reshape(new_shape); }
    Self to(Device target_device) const { return impl_->to(target_device); }
    Dtype dtype() const noexcept { return dtype_of<T>(); }
    Self contiguous() const {
        if (is_contiguous() && offset() == 0) {
            return *this;
        }

        Self result(this->shape(), this->device());
        copy_strided(*this, result.data());

        return result;
    }
    Tensor<T> slice(size_t dim, size_t start, size_t end) const {
        if (dim >= shape().size())
            throw std::out_of_range("slice: dimension out of range");
        if (start >= end || end > shape()[dim])
            throw std::out_of_range("slice: invalid range");
        if (start == 0 && end == shape()[dim])
            return *this;   // no-op

        std::vector<size_t> new_shape = shape();
        new_shape[dim] = end - start;

        // strides remain the same, offset advanced by start * stride[dim]
        size_t new_offset = offset() + start * strides()[dim];

        auto new_impl = std::make_shared<TensorImpl<T>>(
            impl_->storage_,
            std::move(new_shape),
            strides(),
            new_offset,
            requires_grad()
        );
        return Tensor<T>(new_impl);
    }

    bool empty() const noexcept { return impl_ == nullptr || total_elements() == 0; }
    const std::vector<size_t>& shape() const noexcept { return impl_->shape_; }
    const std::vector<size_t>& strides() const noexcept { return impl_->strides_; }
    Device device() const noexcept { return impl_->storage_->device(); }
    T* data() noexcept { return impl_->storage_->data(); }
    const T* data() const noexcept { return impl_->storage_->data(); }
    size_t total_elements() const noexcept { return impl_->total_elements(); }
    size_t offset() const noexcept { return impl_->offset_; }
    bool is_contiguous() const noexcept { return impl_->is_contiguous(); }
    void fill(const T& v) { impl_->storage_->fill(v); }
    void zero_grad() {
        if (!impl_->grad_.empty()) {
            impl_->grad_.fill(T(0)); 
        }
    }

    bool requires_grad() const noexcept { return impl_->requires_grad_; }
    void set_requires_grad(bool req) noexcept { impl_->requires_grad_ = req; }
    Self& grad() { return impl_->grad_; }
    const Self& grad() const { return impl_->grad_; }
    void accumulate_grad(const Self& gradient) {
        if (impl_->grad_.empty()) {
            impl_->grad_ = gradient;
        } else {
            add_(impl_->grad_, gradient);
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

    uint32_t version() const { return impl_ ? impl_->version() : 0; }
    void bump_version() { if (impl_) impl_->bump_version(); }
    bool is_leaf() const { return impl_ ? impl_->is_leaf() : true; }

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