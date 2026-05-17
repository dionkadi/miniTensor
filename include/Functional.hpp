#pragma once

#include "TensorOps.hpp"

namespace F {

template<typename T>
Tensor<T> relu(const Tensor<T>& x) { return ::relu(x); }

template<typename T>
Tensor<T> sigmoid(const Tensor<T>& x) { return ::sigmoid(x); }

template<typename T>
Tensor<T> tanh(const Tensor<T>& x) { return ::tanh(x); }

template<typename T>
Tensor<T> softmax(const Tensor<T>& x) { return ::softmax(x); }

template<typename T>
Tensor<T> flatten(const Tensor<T>& x) { return ::flatten(x); }

template<typename T>
Tensor<T> dropout(const Tensor<T>& x, T p) {
    Dropout<T> d(p);
    return d.forward(x);
}

template<typename T>
Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& weight,
                 const Tensor<T>& bias, size_t stride, size_t padding) {
    return ::conv2d(input, weight, bias, stride, padding);
}

template<typename T>
Tensor<T> max_pool2d(const Tensor<T>& x, size_t k, size_t stride, size_t padding) {
    return ::max_pool2d(x, k, stride, padding);
}

template<typename T>
Tensor<T> batch_norm(const Tensor<T>& x, Tensor<T>& gamma, Tensor<T>& beta,
                     Tensor<T>& running_mean, Tensor<T>& running_var,
                     bool training, T eps, T momentum) {
    // Manual batch norm: x_hat = (x - mean) / sqrt(var + eps), y = gamma * x_hat + beta
    size_t C = x.shape()[1];
    T spatial_size = T(x.shape()[0] * x.shape()[2] * x.shape()[3]);

    auto mean = sum(x, 0, true);
    mean = sum(mean, 2, true);
    mean = sum(mean, 3, true);
    mean = mean / spatial_size;

    auto centered = x - mean;
    auto var = sum(pow2(centered), 0, true);
    var = sum(var, 2, true);
    var = sum(var, 3, true);
    var = var / spatial_size;

    if (training) {
        NoGradGuard guard;
        running_mean = running_mean * momentum + mean.reshape({C}) * (T(1) - momentum);
        running_var  = running_var  * momentum + var.reshape({C})  * (T(1) - momentum);
    }

    auto x_hat = centered / sqrt(var + eps);
    return x_hat * gamma.reshape({1, C, 1, 1}) + beta.reshape({1, C, 1, 1});
}

}