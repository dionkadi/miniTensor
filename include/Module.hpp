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
    virtual ~Module() = default;
    
    virtual Tensor<T> forward(const Tensor<T>& x) = 0;
    virtual std::vector<Tensor<T>> parameters() const = 0; 
    virtual void to(Device) {}

    Tensor<T> operator()(const Tensor<T>& x) {
        return forward(x);
    }
};

template<typename T>
class ReLU : public Module<T> {
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
    Tensor<T> forward(const Tensor<T>& x) override {
        return sigmoid(x);
    }

    std::vector<Tensor<T>> parameters() const override {
        return {};
    }
};

template<typename T>
class Linear : public Module<T> {
    Tensor<T> weight_;
    Tensor<T> bias_;

public:
    Linear(size_t in_features, size_t out_features, Device device = {}) {
        weight_ = Tensor<T>({in_features, out_features}, device);
        weight_.set_requires_grad(true);

        bias_ = Tensor<T>({1, out_features}, device);
        bias_.set_requires_grad(true);

        // Initialization
        std::mt19937 gen(42);
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
private:
    std::vector<std::shared_ptr<Module<T>>> modules_;

public:
    Sequential(std::initializer_list<std::shared_ptr<Module<T>>> modules) 
        : modules_(modules) 
    {}

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
};

template<typename T>
class Flatten : public Module<T> {
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
        std::mt19937 gen(42);
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
class MaxPool2D : public Module<T> {
    size_t kernel_size_, stride_, padding_;
    
public:
    MaxPool2D(size_t kernel_size, size_t stride = 0, size_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride), padding_(padding) {}

    Tensor<T> forward(const Tensor<T>& x) override {
        return max_pool2d(x, kernel_size_, stride_, padding_);
    }

    std::vector<Tensor<T>> parameters() const override { return {}; }
};