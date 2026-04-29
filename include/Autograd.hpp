#pragma once

#include "Tensor.hpp"

template<typename T>
struct AutogradNode {
    virtual ~AutogradNode() = default;
    virtual void apply(const Tensor<T>& grad_output) = 0;
    virtual std::vector<Tensor<T>> get_inputs() const = 0;
};

template<typename T>
struct AddBackward : public AutogradNode<T> {
    Tensor<T> tensor_a;
    Tensor<T> tensor_b;

    AddBackward(Tensor<T> a, Tensor<T> b) : tensor_a(a), tensor_b(b) {}

    void apply(const Tensor<T>& grad_output) override {
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
        return {tensor_a, tensor_b};
    }
};

template<typename T>
struct MatmulBackward : public AutogradNode<T> {
    Tensor<T> tensor_a;
    Tensor<T> tensor_b;

    MatmulBackward(Tensor<T> a, Tensor<T> b) : tensor_a(a), tensor_b(b) {}

    void apply(const Tensor<T>& grad_output) override {
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
        return {tensor_a, tensor_b};
    }
};