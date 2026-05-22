#pragma once

#include "Tensor.hpp"
#include "Autograd.hpp"
#include "TensorOps.hpp"

template<typename T>
class Optimizer {
protected:
    std::vector<Tensor<T>> parameters_;

public:
    Optimizer(const std::vector<Tensor<T>>& params) : parameters_(params) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;

    std::vector<Tensor<T>>& get_parameters() { return parameters_; }
    const std::vector<Tensor<T>>& get_parameters() const { return parameters_; }

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
        : Optimizer<T>(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay), t_(0)
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

        T bias_correction1 = (T)1.0 - std::pow(beta1_, (T)t_);
        T bias_correction2 = (T)1.0 - std::pow(beta2_, (T)t_);

        for (size_t i = 0; i < this->parameters_.size(); ++i) {
            auto& p = this->parameters_[i];
            if (p.grad().empty()) continue;

            auto& g = p.grad();

            if (p.device().type == DeviceType::CUDA) {
#if defined(USE_CUDA) || defined(USE_ROCM)
                adam_step_gpu(p, g, m_[i], v_[i],
                              lr_, beta1_, beta2_, eps_,
                              bias_correction1, bias_correction2, weight_decay_);
                continue;
#endif
            }

            // CPU fallback: decomposed operations
            auto grad = g;
            if (weight_decay_ > (T)0.0) {
                grad = grad + (p * weight_decay_);
            }
            m_[i] = (m_[i] * beta1_) + (grad * ((T)1.0 - beta1_));
            v_[i] = (v_[i] * beta2_) + (pow2(grad) * ((T)1.0 - beta2_));
            Tensor<T> m_hat = m_[i] * ((T)1.0 / bias_correction1);
            Tensor<T> v_hat = v_[i] * ((T)1.0 / bias_correction2);
            Tensor<T> denom = sqrt(v_hat) + eps_;
            Tensor<T> step_size = (m_hat / denom) * lr_;
            sub_(p, step_size);
        }
    }
};

template<typename T>
class GradientScaler {
public:
    GradientScaler(T init_scale = 65536.0, T growth_factor = 2.0,
                   T backoff_factor = 0.5, size_t growth_interval = 2000)
        : scale_(init_scale), growth_factor_(growth_factor),
          backoff_factor_(backoff_factor), growth_interval_(growth_interval),
          step_count_(0) {}

    T scale_loss(T loss) const { return loss * scale_; }

    bool update(bool overflow_occurred) {
        if (overflow_occurred) {
            scale_ *= backoff_factor_;
            step_count_ = 0;
            return false;
        }
        step_count_++;
        if (step_count_ >= growth_interval_) {
            scale_ *= growth_factor_;
            step_count_ = 0;
        }
        return true;
    }

    void unscale_gradients(Optimizer<T>& optimizer) {
        for (auto& p : optimizer.get_parameters()) {
            if (!p.grad().empty()) {
                p.grad() = p.grad() * (T(1) / scale_);
            }
        }
    }

    T scale() const { return scale_; }

private:
    T scale_;
    T growth_factor_;
    T backoff_factor_;
    size_t growth_interval_;
    size_t step_count_;
};