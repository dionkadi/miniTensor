#pragma once

#include "Defines.hpp"
#include "TensorOps.hpp"

#include <vector>
#include <random>
#include <memory>

template<typename T> class Tensor;

template<typename T>
class Module {
public:
    bool is_training_ = true;

    virtual ~Module() = default;
    
    virtual Tensor<T> forward(const Tensor<T>& x) = 0;
    virtual std::vector<Tensor<T>> parameters() const = 0; 
    virtual void to(Device) {}
    virtual const char* name() const { return "Module"; }

    void train() { is_training_ = true; }
    void eval()  { is_training_ = false; }

    Tensor<T> operator()(const Tensor<T>& x) {
        return forward(x);
    }

    void save(const std::string& path) const;
    void load(const std::string& path);
    void export_onnx(const std::string& path);
};

template<typename T>
class ReLU : public Module<T> {
public:
    const char* name() const override { return "ReLU"; }

public:
    Tensor<T> forward(const Tensor<T>& x) override {
        return relu(x);
    }

    std::vector<Tensor<T>> parameters() const override {
        return {};
    }
};

template<typename T>
class Tanh : public Module<T> {
public:
    const char* name() const override { return "Tanh"; }

public:
    Tensor<T> forward(const Tensor<T>& x) override {
        return tanh(x);
    }

    std::vector<Tensor<T>> parameters() const override {
        return {};
    }
};

template<typename T>
class Sigmoid : public Module<T> {
public:
    const char* name() const override { return "Sigmoid"; }

public:
    Tensor<T> forward(const Tensor<T>& x) override {
        return sigmoid(x);
    }

    std::vector<Tensor<T>> parameters() const override {
        return {};
    }
};

template<typename T>
class Linear : public Module<T> {
public:
    const char* name() const override { return "Linear"; }

    Tensor<T> weight_;
    Tensor<T> bias_;

public:
    Linear(size_t in_features, size_t out_features, Device device = {}) {
        weight_ = Tensor<T>({in_features, out_features}, device);
        weight_.set_requires_grad(true);

        bias_ = Tensor<T>({1, out_features}, device);
        bias_.set_requires_grad(true);

        // Initialization
        static std::mt19937::result_type linear_seed = 1000000;
        std::mt19937 gen(linear_seed++);
        std::normal_distribution<T> dist(0.0, std::sqrt(2.0 / in_features));
        
        T* w_data = weight_.data();
        for (size_t i = 0; i < weight_.total_elements(); ++i) {
            w_data[i] = dist(gen);
        }

        bias_.fill((T)0.0);
    }

    Tensor<T> forward(const Tensor<T>& x) override {
        return matmul(x, weight_) + bias_;
    }

    std::vector<Tensor<T>> parameters() const override {
        return {weight_, bias_};
    }

    void to(Device device) override {
        weight_ = weight_.to(device);
        bias_ = bias_.to(device);
    }
};

template<typename T>
class Sequential : public Module<T> {
public:
    const char* name() const override { return "Sequential"; }

private:
    std::vector<std::shared_ptr<Module<T>>> modules_;

public:
    Sequential(std::initializer_list<std::shared_ptr<Module<T>>> modules) 
        : modules_(modules) 
    {}

    void add(std::shared_ptr<Module<T>> mod) {
        modules_.push_back(std::move(mod));
    }

    template<typename U, typename... Args>
    void add(Args&&... args) {
        modules_.push_back(std::make_shared<U>(std::forward<Args>(args)...));
    }

    Tensor<T> forward(const Tensor<T>& x) override {
        Tensor<T> current_out = x;
        for (const auto& mod : modules_) {
            current_out = mod->forward(current_out);
        }
        return current_out;
    }

    std::vector<Tensor<T>> parameters() const override {
        std::vector<Tensor<T>> all_params;
        for (const auto& mod : modules_) {
            auto mod_params = mod->parameters();
            all_params.insert(all_params.end(), mod_params.begin(), mod_params.end());
        }
        return all_params;
    }

    void to(Device device) override {
        for (auto& mod : modules_) {
            mod->to(device);
        }
    }

    void train() {
        this->is_training_ = true;
        for (auto& mod : modules_) mod->train();
    }

    void eval() {
        this->is_training_ = false;
        for (auto& mod : modules_) mod->eval();
    }

    const std::vector<std::shared_ptr<Module<T>>>& modules() const { return modules_; }
};

template<typename T>
class Flatten : public Module<T> {
public:
    const char* name() const override { return "Flatten"; }

public:
    Tensor<T> forward(const Tensor<T>& x) override {
        return flatten(x);
    }

    std::vector<Tensor<T>> parameters() const override {
        return {};
    }
};

template<typename T>
class Conv2D : public Module<T> {
public:
    const char* name() const override { return "Conv2D"; }

    Tensor<T> weight_;
    Tensor<T> bias_;  // may be empty if bias=false
    size_t stride_, padding_;

public:
    Conv2D(
        size_t in_channels, size_t out_channels, size_t kernel_size,
        size_t stride = 1, size_t padding = 0, bool bias = true,
        Device device = {}
    )
        : stride_(stride), padding_(padding)
    {
        weight_ = Tensor<T>({out_channels, in_channels, kernel_size, kernel_size}, device);
        weight_.set_requires_grad(true);

        if (bias) {
            bias_ = Tensor<T>({out_channels}, device);  // [C_out]
            bias_.set_requires_grad(true);
            bias_.fill((T)0.0);
        }

        T fan_in = static_cast<T>(in_channels * kernel_size * kernel_size);
        T std = std::sqrt(T(2.0) / fan_in);
        static std::mt19937::result_type init_seed = 42;
        std::mt19937 gen(init_seed++);
        std::normal_distribution<T> dist(0.0, std);
        T* w = weight_.data();
        for (size_t i = 0; i < weight_.total_elements(); ++i) {
            w[i] = dist(gen);
        }
    }

    Tensor<T> forward(const Tensor<T>& x) override {
        return conv2d(x, weight_, bias_, stride_, padding_);
    }

    std::vector<Tensor<T>> parameters() const override {
        if (bias_.empty()) return {weight_};
        return {weight_, bias_};
    }

    void to(Device device) override {
        weight_ = weight_.to(device);
        if (!bias_.empty()) {
            bias_ = bias_.to(device);
        }
    }
};

template<typename T>
class LayerNorm : public Module<T> {
public:
    const char* name() const override { return "LayerNorm"; }

    Tensor<T> gamma_;
    Tensor<T> beta_;
    T eps_;
public:
    LayerNorm(const std::vector<size_t>& normalized_shape, T eps = T(1e-5), Device device = {})
        : eps_(eps) {
        size_t d = normalized_shape.back();
        gamma_ = Tensor<T>({d}, device);
        gamma_.set_requires_grad(true);
        gamma_.fill(T(1));
        beta_ = Tensor<T>({d}, device);
        beta_.set_requires_grad(true);
        beta_.fill(T(0));
    }

    Tensor<T> forward(const Tensor<T>& x) override {
        size_t last_dim = x.shape().size() - 1;
        T d = T(x.shape().back());

        auto mean = sum(x, last_dim, true) / d;
        auto centered = x - mean;
        auto var = sum(pow2(centered), last_dim, true) / d;
        auto x_hat = centered / sqrt(var + eps_);
        return x_hat * gamma_ + beta_;
    }

    std::vector<Tensor<T>> parameters() const override { return {gamma_, beta_}; }

    void to(Device device) override {
        gamma_ = gamma_.to(device);
        beta_ = beta_.to(device);
    }
};

template<typename T>
class BatchNorm2d : public Module<T> {
public:
    const char* name() const override { return "BatchNorm2d"; }

    Tensor<T> gamma_;
    Tensor<T> beta_;
    Tensor<T> running_mean_;
    Tensor<T> running_var_;
    T eps_;
    T momentum_;
public:
    BatchNorm2d(size_t num_features, T eps = T(1e-5), T momentum = T(0.9), Device device = {})
        : eps_(eps), momentum_(momentum) {
        gamma_ = Tensor<T>({num_features}, device);
        gamma_.set_requires_grad(true);
        gamma_.fill(T(1));
        beta_ = Tensor<T>({num_features}, device);
        beta_.set_requires_grad(true);
        beta_.fill(T(0));
        running_mean_ = Tensor<T>({num_features}, device);
        running_mean_.fill(T(0));
        running_var_ = Tensor<T>({num_features}, device);
        running_var_.fill(T(1));
    }

    Tensor<T> forward(const Tensor<T>& x) override {
        return forward_impl(x, false);
    }

    // Fused BN + ReLU: single kernel replaces BN forward + relu call.
    // Gradient flows through separate BN backward + ReLU backward nodes
    // (autograd graph is built correctly even though forward is fused).
    Tensor<T> forward_relu(const Tensor<T>& x) {
        return forward_impl(x, true);
    }

private:
    Tensor<T> forward_impl(const Tensor<T>& x, bool apply_relu) {
        size_t N = x.shape()[0], C = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
        T spatial_size = T(N * H * W);

        Tensor<T> mean, var;
#if defined(USE_CUDA) || defined(USE_ROCM)
        if (x.device().type == DeviceType::CUDA) {
            // Fused BN forward: single kernel computes mean + var
            mean = Tensor<T>({1, C, 1, 1}, x.device());
            var  = Tensor<T>({1, C, 1, 1}, x.device());
            bn_fwd_gpu(x, mean, var);
        } else {
#endif
            mean = sum(x, 0, true);
            mean = sum(mean, 2, true);
            mean = sum(mean, 3, true);
            mean = mean * (T(1) / spatial_size);

            auto centered = x - mean;
            var = sum(pow2(centered), 0, true);
            var = sum(var, 2, true);
            var = sum(var, 3, true);
            var = var * (T(1) / spatial_size);
#if defined(USE_CUDA) || defined(USE_ROCM)
        }
#endif
        if (this->is_training_) {
            NoGradGuard guard;
            running_mean_ = running_mean_ * momentum_ + mean.reshape({C}) * (T(1) - momentum_);
            running_var_  = running_var_  * momentum_ + var.reshape({C})  * (T(1) - momentum_);
        }

#if defined(USE_CUDA) || defined(USE_ROCM)
        if (apply_relu && x.device().type == DeviceType::CUDA && !GradMode::is_enabled()) {
            Tensor<T> output(x.shape(), x.device());
            bn_relu_fwd_gpu(x, output, mean, var, gamma_, beta_, eps_);
            return output;
        }
#endif

        auto centered = x - mean;
        auto inv_std = sqrt(var + eps_);
        auto x_hat = centered / inv_std;
        auto bn_out = x_hat * view_reshape(gamma_, {1, C, 1, 1}) + view_reshape(beta_, {1, C, 1, 1});
        if (apply_relu) {
            return relu(bn_out);
        }
        return bn_out;
    }
public:

    std::vector<Tensor<T>> parameters() const override {
        return {gamma_, beta_};
    }

    void to(Device device) override {
        gamma_ = gamma_.to(device);
        beta_ = beta_.to(device);
        running_mean_ = running_mean_.to(device);
        running_var_ = running_var_.to(device);
    }
};

template<typename T>
class Dropout : public Module<T> {
public:
    const char* name() const override { return "Dropout"; }

    T p_;
public:
    Dropout(T p = T(0.5)) : p_(p) {}

    Tensor<T> forward(const Tensor<T>& x) override {
        if (!this->is_training_ || p_ <= T(0)) {
            return x;
        }

        T p_keep = T(1) - p_;
        std::mt19937 gen(std::random_device{}());
        std::bernoulli_distribution dist(p_keep);

        Tensor<T> mask(x.shape(), x.device());
        T* m_data = mask.data();
        for (size_t i = 0; i < mask.total_elements(); ++i) {
            m_data[i] = dist(gen) ? T(1) : T(0);
        }

        if (x.device().type != DeviceType::CPU) {
            mask = mask.to(x.device());
        }

        Tensor<T> result = x * mask * (T(1) / p_keep);

        if (GradMode::is_enabled() && x.requires_grad()) {
            result.set_requires_grad(true);
            result.set_grad_fn(std::make_shared<DropoutBackward<T>>(x, mask, p_));
        }

        return result;
    }

    std::vector<Tensor<T>> parameters() const override { return {}; }
};

template<typename T>
class MaxPool2D : public Module<T> {
public:
    const char* name() const override { return "MaxPool2D"; }

    size_t kernel_size_, stride_, padding_;
    
public:
    MaxPool2D(size_t kernel_size, size_t stride = 0, size_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride), padding_(padding) {}

    Tensor<T> forward(const Tensor<T>& x) override {
        return max_pool2d(x, kernel_size_, stride_, padding_);
    }

    std::vector<Tensor<T>> parameters() const override { return {}; }
};

template<typename T>
class AdaptiveAvgPool2D : public Module<T> {
public:
    const char* name() const override { return "AdaptiveAvgPool2D"; }

    size_t output_size_;

    AdaptiveAvgPool2D(size_t output_size = 1) : output_size_(output_size) {}

    Tensor<T> forward(const Tensor<T>& x) override {
        auto shape = x.shape();
        if (shape.size() < 2)
            throw std::invalid_argument("AdaptiveAvgPool2D: input must have >= 2 dims");
        size_t H = shape[shape.size() - 2], W = shape[shape.size() - 1];
        if (output_size_ == 1) {
            auto pooled = sum(x, shape.size() - 2, true);
            pooled = sum(pooled, shape.size() - 1, true);
            return pooled * (T(1) / T(H * W));
        }
        throw std::runtime_error("AdaptiveAvgPool2D: only output_size=1 supported");
    }

    std::vector<Tensor<T>> parameters() const override { return {}; }
};

#include "Serialization.hpp"
#include "GraphExport.hpp"

template<typename T>
void Module<T>::save(const std::string& path) const {
    save_state_dict(path, parameters());
}

template<typename T>
void Module<T>::load(const std::string& path) {
    auto params = parameters();
    load_state_dict_into(path, params);
}

template<typename T>
void Module<T>::export_onnx(const std::string& path) {
    GraphExport<T>::save_graph(path, *this);
}