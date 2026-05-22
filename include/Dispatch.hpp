#pragma once

#include <functional>
#include <unordered_map>
#include <memory>
#include <string>

#include "Defines.hpp"

template<typename T> class Tensor;

enum class OpType : uint8_t {
    Add, Sub, Mul, Div, Matmul, Relu, Sigmoid, Softmax, Sum, Conv2D
};

using ErasedTensorPtr = void*;  // type-erased tensor pointer (platform-specific)

template<typename T>
using UnaryOpFn  = std::function<void(const Tensor<T>&, Tensor<T>&)>;
template<typename T>
using BinaryOpFn = std::function<void(const Tensor<T>&, const Tensor<T>&, Tensor<T>&)>;

template<typename T>
class OpRegistry {
public:
    static OpRegistry& instance() {
        static OpRegistry reg;
        return reg;
    }

    void register_unary(OpType op, UnaryOpFn<T> fn) { unary_ops_[op] = std::move(fn); }
    void register_binary(OpType op, BinaryOpFn<T> fn) { binary_ops_[op] = std::move(fn); }

    UnaryOpFn<T>* lookup_unary(OpType op) {
        auto it = unary_ops_.find(op);
        return it != unary_ops_.end() ? &it->second : nullptr;
    }

    BinaryOpFn<T>* lookup_binary(OpType op) {
        auto it = binary_ops_.find(op);
        return it != binary_ops_.end() ? &it->second : nullptr;
    }

private:
    OpRegistry() = default;
    std::unordered_map<OpType, UnaryOpFn<T>>  unary_ops_;
    std::unordered_map<OpType, BinaryOpFn<T>> binary_ops_;
};

template<typename T>
struct AutoRegistrar {
    AutoRegistrar(OpType op, UnaryOpFn<T> fn) {
        OpRegistry<T>::instance().register_unary(op, std::move(fn));
    }
    AutoRegistrar(OpType op, BinaryOpFn<T> fn) {
        OpRegistry<T>::instance().register_binary(op, std::move(fn));
    }
};

template<typename T>
void dispatch_unary_op(OpType op, const Tensor<T>& a, Tensor<T>& out) {
    auto* fn = OpRegistry<T>::instance().lookup_unary(op);
    if (fn) (*fn)(a, out);
}

template<typename T>
void dispatch_binary_op(OpType op, const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& out) {
    auto* fn = OpRegistry<T>::instance().lookup_binary(op);
    if (fn) (*fn)(a, b, out);
}
