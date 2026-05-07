#pragma once

#include <algorithm>
template<typename T>
struct Activation {
    virtual ~Activation() = default;
    virtual T forward(T x) = 0;
};

template<typename T>
struct ReLU : public Activation<T> {
    T forward(T x) override {
        return std::max(x, (T)0.0);
    }
};

template<typename T>
T relu(T x) { return std::max(x, (T)0.0); }