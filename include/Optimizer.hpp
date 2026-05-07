#pragma once

#include "Tensor.hpp"
#include "Autograd.hpp"

template<typename T>
class Optimizer {
protected:
    std::vector<Tensor<T>> parameters_;

public:
    Optimizer(const std::vector<Tensor<T>>& params) : parameters_(params) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;

    void zero_grad() {
        for (auto& p : parameters_) {
            p.zero_grad();
        }
    }
};

template<typename T>
class SGD : public Optimizer<T> {
private:
    T lr_;

public:
    SGD(const std::vector<Tensor<T>>& params, T lr) 
        : Optimizer<T>(params), lr_(lr) {}

    void step() override {
        NoGradGuard guard; 

        for (auto& p : this->parameters_) {
            if (p.grad().empty()) continue;
            
            Tensor<T> step = p.grad() * lr_;
            sub_(p, step); 
        }
    }
};