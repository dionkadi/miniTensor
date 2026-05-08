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
    T weight_decay_;

public:
    SGD(const std::vector<Tensor<T>>& params, T lr, T weight_decay = (T)0.0) 
        : Optimizer<T>(params), lr_(lr), weight_decay_(weight_decay) {}

    void step() override {
        NoGradGuard guard; 

        for (auto& p : this->parameters_) {
            if (p.grad().empty()) continue;
            
            Tensor<T> current_grad = p.grad();

            if (weight_decay_ > (T)0.0) {
                current_grad = current_grad + (p * weight_decay_);
            }

            Tensor<T> step = current_grad * lr_;
            sub_(p, step); 
        }
    }
};

template<typename T>
class Adam : public Optimizer<T> {
private:
    T lr_;
    T beta1_;
    T beta2_;
    T eps_;
    T weight_decay_;
    size_t t_;

    // State buffers for 1st and 2nd moments
    std::vector<Tensor<T>> m_;
    std::vector<Tensor<T>> v_;

public:
    Adam(const std::vector<Tensor<T>>& params, T lr = 0.001, T beta1 = 0.9, T beta2 = 0.999, T eps = 1e-8, T weight_decay = (T)0.0) 
        : Optimizer<T>(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0), weight_decay_(weight_decay)
    {
        
        // Initialize moment buffers to zero with the exact same shape and device as the parameters
        for (const auto& p : this->parameters_) {
            Tensor<T> m(p.shape(), p.device());
            m.fill((T)0.0);
            m_.push_back(m);

            Tensor<T> v(p.shape(), p.device());
            v.fill((T)0.0);
            v_.push_back(v);
        }
    }

    void step() override {
        NoGradGuard guard;
        t_++;

        for (size_t i = 0; i < this->parameters_.size(); ++i) {
            auto& p = this->parameters_[i];
            
            if (p.grad().empty()) continue;

            auto grad = p.grad();
            
            if (weight_decay_ > (T)0.0) {
                grad = grad + (p * weight_decay_);
            }

            // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            m_[i] = (m_[i] * beta1_) + (grad * ((T)1.0 - beta1_));

            // Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            v_[i] = (v_[i] * beta2_) + (pow2(grad) * ((T)1.0 - beta2_));

            // Compute bias-corrected first moment estimate
            T bias_correction1 = (T)1.0 - std::pow(beta1_, (T)t_);
            Tensor<T> m_hat = m_[i] * ((T)1.0 / bias_correction1);

            // Compute bias-corrected second raw moment estimate
            T bias_correction2 = (T)1.0 - std::pow(beta2_, (T)t_);
            Tensor<T> v_hat = v_[i] * ((T)1.0 / bias_correction2);

            // Update parameters: p = p - lr * m_hat / (sqrt(v_hat) + eps)
            Tensor<T> denom = sqrt(v_hat) + eps_;
            Tensor<T> step_size = (m_hat / denom) * lr_;
            
            sub_(p, step_size);
        }
    }
};