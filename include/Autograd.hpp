#pragma once

#include "Tensor.hpp"

struct GradMode {
    static bool& is_enabled() {
        thread_local bool enabled = true;
        return enabled;
    }
};

class NoGradGuard {
private:
    bool prev_state_;
public:
    NoGradGuard() {
        prev_state_ = GradMode::is_enabled();
        GradMode::is_enabled() = false;
    }
    ~NoGradGuard() {
        GradMode::is_enabled() = prev_state_;
    }
    
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
};

template<typename T>
class SavedTensor {
private:
    Tensor<T> tensor_;
    uint32_t saved_version_;

public:
    SavedTensor(const Tensor<T>& t) 
        : tensor_(t), saved_version_(t.empty() ? 0 : t.version()) {}

    // Safely unpacks the tensor. Throws if mutated.
    Tensor<T> unpack() const {
        if (!tensor_.empty() && tensor_.version() != saved_version_) {
            throw std::runtime_error(
                "RuntimeError: one of the variables needed for gradient computation "
                "has been modified by an inplace operation."
            );
        }
        return tensor_;
    }
};


template<typename T>
struct AutogradNode {
    virtual ~AutogradNode() = default;
    virtual void apply(const Tensor<T>& grad_output) = 0;
    virtual std::vector<Tensor<T>> get_inputs() const = 0;
};

template<typename T>
struct AddBackward : public AutogradNode<T> {
    SavedTensor<T> saved_a;
    SavedTensor<T> saved_b;

    AddBackward(Tensor<T> a, Tensor<T> b) : saved_a(a), saved_b(b) {}

    void apply(const Tensor<T>& grad_output) override {
        Tensor<T> tensor_a = saved_a.unpack();
        Tensor<T> tensor_b = saved_b.unpack();

        // dC/dA = 1, so grad_A = grad_output * 1
        if (tensor_a.requires_grad()) {
            Tensor<T> grad_a = unbroadcast(grad_output, tensor_a.shape());
            tensor_a.accumulate_grad(grad_a);
        }
        
        // dC/dB = 1, so grad_B = grad_output * 1
        if (tensor_b.requires_grad()) {
            Tensor<T> grad_b = unbroadcast(grad_output, tensor_b.shape());
            tensor_b.accumulate_grad(grad_b);
        }
    }

    std::vector<Tensor<T>> get_inputs() const override {
        return {saved_a.unpack(), saved_b.unpack()};
    }
};

template<typename T>
struct SubBackward : public AutogradNode<T> {
    SavedTensor<T> saved_a;
    SavedTensor<T> saved_b;

    SubBackward(Tensor<T> a, Tensor<T> b) : saved_a(a), saved_b(b) {}

    void apply(const Tensor<T>& grad_output) override {
        Tensor<T> tensor_a = saved_a.unpack();
        Tensor<T> tensor_b = saved_b.unpack();

        if (tensor_a.requires_grad()) {
            tensor_a.accumulate_grad(unbroadcast(grad_output, tensor_a.shape()));
        }
        if (tensor_b.requires_grad()) {
            // dL/dB = -grad_output
            Tensor<T> neg_grad = grad_output * (T)-1.0;
            tensor_b.accumulate_grad(unbroadcast(neg_grad, tensor_b.shape()));
        }
    }

    std::vector<Tensor<T>> get_inputs() const override {
        return {saved_a.unpack(), saved_b.unpack()};
    }
};

template<typename T>
struct MatmulBackward : public AutogradNode<T> {
    SavedTensor<T> saved_a;
    SavedTensor<T> saved_b;

    MatmulBackward(Tensor<T> a, Tensor<T> b) : saved_a(a), saved_b(b) {}

    void apply(const Tensor<T>& grad_output) override {
        Tensor<T> tensor_a = saved_a.unpack();
        Tensor<T> tensor_b = saved_b.unpack();

        // C = A @ B
        // dL/dA = grad_output @ B^T
        if (tensor_a.requires_grad()) {
            Tensor<T> B_T = tensor_b.transpose(0, 1);
            tensor_a.accumulate_grad(matmul(grad_output, B_T));
        }
        
        // dL/dB = A^T @ grad_output
        if (tensor_b.requires_grad()) {
            Tensor<T> A_T = tensor_a.transpose(0, 1);
            tensor_b.accumulate_grad(matmul(A_T, grad_output));
        }
    }

    std::vector<Tensor<T>> get_inputs() const override {
        return {saved_a.unpack(), saved_b.unpack()};
    }
};

template<typename T>
struct MulScalarBackward : public AutogradNode<T> {
    SavedTensor<T> saved_a;
    T scalar;

    MulScalarBackward(Tensor<T> a, T s) : saved_a(a), scalar(s) {}

    void apply(const Tensor<T>& grad_output) override {
        Tensor<T> tensor_a = saved_a.unpack();

        if (tensor_a.requires_grad()) {
            // dL/dA = grad_output * scalar
            tensor_a.accumulate_grad(grad_output * scalar);
        }
    }

    std::vector<Tensor<T>> get_inputs() const override {
        return {saved_a.unpack()};
    }
};

template<typename T>
struct MulBackward : public AutogradNode<T> {
    SavedTensor<T> saved_a;
    SavedTensor<T> saved_b;

    MulBackward(Tensor<T> a, Tensor<T> b) : saved_a(a), saved_b(b) {}

    void apply(const Tensor<T>& grad_output) override {
        Tensor<T> tensor_a = saved_a.unpack();
        Tensor<T> tensor_b = saved_b.unpack();

        // C = A * B
        // dC/dA = B  =>  grad_A = grad_output * B
        if (tensor_a.requires_grad()) {
            Tensor<T> grad_a = unbroadcast(grad_output * tensor_b, tensor_a.shape());
            tensor_a.accumulate_grad(grad_a);
        }
        
        // dC/dB = A  =>  grad_B = grad_output * A
        if (tensor_b.requires_grad()) {
            Tensor<T> grad_b = unbroadcast(grad_output * tensor_a, tensor_b.shape());
            tensor_b.accumulate_grad(grad_b);
        }
    }

    std::vector<Tensor<T>> get_inputs() const override {
        return {saved_a.unpack(), saved_b.unpack()};
    }
};

template<typename T>
struct Pow2Backward : public AutogradNode<T> {
    SavedTensor<T> saved_a;

    Pow2Backward(Tensor<T> a) : saved_a(a) {}

    void apply(const Tensor<T>& grad_output) override {
        Tensor<T> tensor_a = saved_a.unpack();

        // C = A^2
        // dC/dA = 2 * A  =>  grad_A = grad_output * (2 * A)
        if (tensor_a.requires_grad()) {
            Tensor<T> grad_a = grad_output * (tensor_a * (T)2.0);
            tensor_a.accumulate_grad(grad_a); // No unbroadcast needed for unary ops
        }
    }

    std::vector<Tensor<T>> get_inputs() const override { return {saved_a.unpack()}; }
};

template<typename T>
struct SumBackward : public AutogradNode<T> {
    SavedTensor<T> saved_input;
    size_t axis;
    bool keepdims;

    SumBackward(Tensor<T> input, size_t axis, bool keepdims)
        : saved_input(input), axis(axis), keepdims(keepdims) {}

    void apply(const Tensor<T>& grad_output) override {
        Tensor<T> input = saved_input.unpack();
        if (!input.requires_grad()) return;

        // Gradient of sum: grad_input = grad_output expanded back to input shape
        // We need to broadcast grad_output to input's original shape.
        // The output shape of sum is input.shape with axis either removed (if !keepdims)
        // or set to 1 (if keepdims).
        std::vector<size_t> target_shape = input.shape();
        
        Tensor<T> grad_expanded = grad_output;
        
        if (!keepdims) {
            // Insert a dimension of size 1 at the reduction axis to match target_shape
            std::vector<size_t> new_shape = grad_output.shape();
            new_shape.insert(new_shape.begin() + axis, 1);
            grad_expanded = grad_expanded.reshape(new_shape);
        }
        // Now grad_expanded has same number of dimensions as input, but with size 1 on `axis`
        // Expand to full input shape (broadcast along that axis)
        Tensor<T> grad_input = grad_expanded.expand(target_shape);
        input.accumulate_grad(grad_input);
    }

    std::vector<Tensor<T>> get_inputs() const override {
        return {saved_input.unpack()};
    }
};