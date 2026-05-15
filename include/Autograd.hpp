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
        NoGradGuard guard;

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
        NoGradGuard guard;

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
        NoGradGuard guard;

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
        NoGradGuard guard;

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
struct AddScalarBackward : public AutogradNode<T> {
    SavedTensor<T> saved_a;
    T scalar; 

    AddScalarBackward(Tensor<T> a, T s) : saved_a(a), scalar(s) {}

    void apply(const Tensor<T>& grad_output) override {
        NoGradGuard guard;
        Tensor<T> tensor_a = saved_a.unpack();
        if (tensor_a.requires_grad()) {
            // d(A + scalar)/dA = 1, so gradient is just grad_output
            tensor_a.accumulate_grad(grad_output);
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
        NoGradGuard guard;

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
        NoGradGuard guard;

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
        NoGradGuard guard;

        Tensor<T> input = saved_input.unpack();
        if (!input.requires_grad()) return;

        // Gradient of sum: grad_input = grad_output expanded back to input shape
        // We need to broadcast grad_output to input's original shape.
        // The output shape of sum is input.shape with axis either removed (if !keepdims)
        // or set to 1 (if keepdims).
        std::vector<size_t> target_shape = input.shape();
        
        Tensor<T> grad_expanded = grad_output.contiguous();
        
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

template<typename T> 
struct ExpBackward : public AutogradNode<T> {
    SavedTensor<T> saved_a;
    
    ExpBackward(Tensor<T> a) : saved_a(a) {}
    
    // d/dx [e^x] = e^x
    void apply(const Tensor<T>& grad_output) override {
        NoGradGuard guard;

        Tensor<T> a = saved_a.unpack();
        if (a.requires_grad()) {
            NoGradGuard guard; // Prevent higher-order graph building
            a.accumulate_grad(grad_output * exp(a));
        }
    }

    std::vector<Tensor<T>> get_inputs() const override { return {saved_a.unpack()}; }
};

template<typename T> struct LogBackward : public AutogradNode<T> {
    SavedTensor<T> saved_a;

    LogBackward(Tensor<T> a) : saved_a(a) {}
    
    // d/dx [\ln(x)] = 1/x
    void apply(const Tensor<T>& grad_output) override {
        NoGradGuard guard;

        Tensor<T> a = saved_a.unpack();
        if (a.requires_grad()) {
            NoGradGuard guard;
            a.accumulate_grad(grad_output / a);
        }
    }

    std::vector<Tensor<T>> get_inputs() const override { return {saved_a.unpack()}; }
};

template<typename T> struct DivBackward : public AutogradNode<T> {
    SavedTensor<T> saved_a;
    SavedTensor<T> saved_b;
    
    DivBackward(Tensor<T> a, Tensor<T> b) : saved_a(a), saved_b(b) {}
    
    // d/dA [A/B] = 1/B  |  d/dB [A/B] = -A / B^2
    void apply(const Tensor<T>& grad_output) override {
        NoGradGuard guard;

        Tensor<T> a = saved_a.unpack();
        Tensor<T> b = saved_b.unpack();
        
        if (a.requires_grad()) {
            Tensor<T> grad_a = unbroadcast(grad_output / b, a.shape());
            a.accumulate_grad(grad_a);
        }
        if (b.requires_grad()) {
            Tensor<T> b_sq = b * b;
            Tensor<T> neg_a = a * (T)-1.0;
            Tensor<T> grad_b_raw = neg_a / b_sq;
            Tensor<T> grad_b = unbroadcast(grad_output * grad_b_raw, b.shape());
            b.accumulate_grad(grad_b);
        }
    }

    std::vector<Tensor<T>> get_inputs() const override { return {saved_a.unpack(), saved_b.unpack()}; }
};

template<typename T>
struct ReLUBackward : public AutogradNode<T> {
    SavedTensor<T> saved_a;

    ReLUBackward(Tensor<T> a) : saved_a(a) {}

    void apply(const Tensor<T>& grad_output) override {
        NoGradGuard guard;

        Tensor<T> a = saved_a.unpack();

        if (a.requires_grad()) {
            a.accumulate_grad(relu_grad(a, grad_output));
        }
    }

    std::vector<Tensor<T>> get_inputs() const override { return {saved_a.unpack()}; }
};

template<typename T> struct SigmoidBackward : public AutogradNode<T> {
    SavedTensor<T> saved_a;

    SigmoidBackward(Tensor<T> a) : saved_a(a) {}

    void apply(const Tensor<T>& grad_output) override {
        NoGradGuard guard;

        Tensor<T> a = saved_a.unpack();

        if (a.requires_grad()) {
            Tensor<T> s = sigmoid(a);
            // d(Sigmoid)/dx = s * (1 - s)
            Tensor<T> one = Tensor<T>::ones_like(s.shape(), s.device());
            Tensor<T> one_minus_s = one - s;
            Tensor<T> ds = s * one_minus_s;
            a.accumulate_grad(grad_output * ds);
        }
    }

    std::vector<Tensor<T>> get_inputs() const override { return {saved_a.unpack()}; }
};

template<typename T> struct TanhBackward : public AutogradNode<T> {
    SavedTensor<T> saved_a;

    TanhBackward(Tensor<T> a) : saved_a(a) {}

    void apply(const Tensor<T>& grad_output) override {
        NoGradGuard guard;

        Tensor<T> a = saved_a.unpack();

        if (a.requires_grad()) {
            Tensor<T> t = tanh(a);
            // d(Tanh)/dx = 1 - t^2
            Tensor<T> t_sq = pow2(t);
            Tensor<T> one = Tensor<T>::ones_like(t.shape(), t.device());
            Tensor<T> dt = one - t_sq;
            a.accumulate_grad(grad_output * dt);
        }
    }
    
    std::vector<Tensor<T>> get_inputs() const override { return {saved_a.unpack()}; }
};

template<typename T>
struct FlattenBackward : public AutogradNode<T> {
    SavedTensor<T> saved_input;
    std::vector<size_t> original_shape;

    FlattenBackward(Tensor<T> input) : saved_input(input), original_shape(input.shape()) {}

    void apply(const Tensor<T>& grad_output) override {
        NoGradGuard guard;
        Tensor<T> input = saved_input.unpack();
        if (input.requires_grad()) {
            input.accumulate_grad(grad_output.reshape(original_shape));
        }
    }

    std::vector<Tensor<T>> get_inputs() const override { return {saved_input.unpack()}; }
};

template<typename T>
struct MaxPool2DBackward : public AutogradNode<T> {
    SavedTensor<T> saved_input;
    std::vector<size_t> indices;

    MaxPool2DBackward(Tensor<T> input, std::vector<size_t> idxs) 
        : saved_input(input), indices(std::move(idxs)) {}

    void apply(const Tensor<T>& grad_output) override {
        NoGradGuard guard;
        Tensor<T> input = saved_input.unpack();
        auto go = grad_output.contiguous();
        if (input.requires_grad()) {
            Tensor<T> grad_input(input.shape(), input.device());
            max_pool2d_backward(go, grad_input, indices);
            input.accumulate_grad(grad_input);
        }
    }

    std::vector<Tensor<T>> get_inputs() const override { return {saved_input.unpack()}; }
};

template<typename T>
struct Conv2DBackward : public AutogradNode<T> {
    SavedTensor<T> saved_input;
    SavedTensor<T> saved_weight;
    SavedTensor<T> saved_bias;
    size_t stride;
    size_t padding;

    Conv2DBackward(Tensor<T> input, Tensor<T> weight, Tensor<T> bias, size_t s, size_t p)
        : saved_input(input), saved_weight(weight), saved_bias(bias), stride(s), padding(p) {}

    void apply(const Tensor<T>& grad_output) override {
        NoGradGuard guard;
        
        Tensor<T> input     = saved_input.unpack();
        Tensor<T> weight    = saved_weight.unpack();
        Tensor<T> bias      = saved_bias.unpack();

        auto go = grad_output.contiguous();
        auto in = input.contiguous();

        if (input.requires_grad()) {
            Tensor<T> grad_input(input.shape(), input.device());
            conv2d_backward_input(go, weight, grad_input, stride, padding);
            input.accumulate_grad(grad_input);
        }

        if (weight.requires_grad()) {
            Tensor<T> grad_weight(weight.shape(), weight.device());
            conv2d_backward_weight(go, in, grad_weight, stride, padding);
            weight.accumulate_grad(grad_weight);
        }

        if (bias.requires_grad()) {
            Tensor<T> grad_bias(bias.shape(), bias.device());
            conv2d_backward_bias(go, grad_bias);
            bias.accumulate_grad(grad_bias);
        }
    }

    std::vector<Tensor<T>> get_inputs() const override { 
        return {saved_input.unpack(), saved_weight.unpack(), saved_bias.unpack()}; 
    }
};

template<typename T>
struct CrossEntropyBackward : public AutogradNode<T> {
    SavedTensor<T> saved_logits;
    SavedTensor<T> saved_targets;

    CrossEntropyBackward(Tensor<T> logits, Tensor<T> targets)
        : saved_logits(logits), saved_targets(targets) {}

    void apply(const Tensor<T>& grad_output) override {
        NoGradGuard guard;
        Tensor<T> logits = saved_logits.unpack();
        Tensor<T> targets = saved_targets.unpack();

        if (logits.requires_grad()) {
            Tensor<T> grad_logits(logits.shape(), logits.device());
            cross_entropy_bwd_gpu(grad_output, logits, targets, grad_logits);
            logits.accumulate_grad(grad_logits);
        }
    }

    std::vector<Tensor<T>> get_inputs() const override {
        return {saved_logits.unpack()};
    }
};