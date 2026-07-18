#pragma once

#include "Tensor.hpp"
#include "Autograd.hpp"
#include "TensorOps.hpp"

template<typename T>
class Optimizer {
protected:
    std::vector<Tensor<T>> parameters_;

    // 1-element device tensor holding the current learning rate.
    // Graph-capture-safe: the captured graph reads from lr_device_'s data
    // pointer. When the scheduler changes lr, we overwrite the contents
    // via sync_lr_device() — the pointer stays the same, so the captured
    // graph picks up the new value on the next replay.
    Tensor<T> lr_device_;
    bool lr_device_init_ = false;

    void init_lr_device(T lr) {
        if (!lr_device_init_ && !this->parameters_.empty()) {
            lr_device_ = Tensor<T>({1}, this->parameters_[0].device());
            lr_device_.fill(lr);
            lr_device_init_ = true;
        }
    }

    void sync_lr_device(T new_lr) {
        if (lr_device_init_) {
            lr_device_.fill(new_lr);
        }
    }

    // Graph-capture-safe gradient norm clipping.
    // Computes the global L2 norm across all parameter gradients and scales
    // them down if the norm exceeds max_norm. All operations are device-side
    // (no D2H sync), so the entire clip is captured inside the CUDA/HIP graph.
    void clip_grad_norm_impl(T max_norm) {
        if (max_norm <= T(0) || this->parameters_.empty()) return;

        // Accumulate sum of squared gradients into a 1-element tensor.
        Tensor<T> total_sq({1}, this->parameters_[0].device());
        total_sq.fill(T(0));

        for (auto& p : this->parameters_) {
            if (p.grad().empty()) continue;
            // Flatten to 1D, square, sum all elements → {1} tensor.
            auto flat = p.grad().reshape({p.grad().total_elements()});
            auto sq = pow2(flat);
            auto s = sum(sq, 0, true);
            add_(total_sq, s);
        }

        // total_norm = sqrt(total_sq)
        auto total_norm = sqrt(total_sq);

        // clip_coeff = max_norm / (total_norm + eps)
        auto denom = add_scalar(total_norm, T(1e-8));
        Tensor<T> max_norm_tensor({1}, total_sq.device());
        max_norm_tensor.fill(max_norm);
        auto clip_coeff = div(max_norm_tensor, denom);

        // scale = min(clip_coeff, 1.0) — no-op when norm <= max_norm
        auto scale = clamp_max_scalar(clip_coeff, T(1));

        // Apply scale to all gradients in-place
        for (auto& p : this->parameters_) {
            if (p.grad().empty()) continue;
            mul_(p.grad(), scale);
        }
    }

public:
    Optimizer(const std::vector<Tensor<T>>& params) : parameters_(params) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual T lr() const = 0;
    virtual void set_lr(T lr) = 0;

    std::vector<Tensor<T>>& get_parameters() { return parameters_; }
    const std::vector<Tensor<T>>& get_parameters() const { return parameters_; }

    // Access the device-resident LR tensor (for graph capture).
    Tensor<T>& lr_device() { return lr_device_; }

    // Checkpoint accessors for optimizer state serialization.
    // Subclasses override to expose their internal state buffers
    // (e.g. momentum velocity, Adam m/v) and scalar hyperparameters.
    virtual std::vector<Tensor<T>> state_buffers() const { return {}; }
    virtual void set_state_buffers(const std::vector<Tensor<T>>&) {}
    virtual std::vector<T> state_scalars() const { return {}; }
    virtual void set_state_scalars(const std::vector<T>&) {}
    virtual size_t current_step() const { return 0; }
    virtual void set_step(size_t) {}

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
    T momentum_;
    T clip_norm_;
    size_t t_ = 0;

    std::vector<Tensor<T>> velocity_;

public:
    SGD(const std::vector<Tensor<T>>& params, T lr, T weight_decay = (T)0.0,
        T momentum = (T)0.0, T clip_norm = (T)0.0) 
        : Optimizer<T>(params), lr_(lr), weight_decay_(weight_decay),
          clip_norm_(clip_norm)     
    {
        set_momentum(momentum);
        this->init_lr_device(lr);
    }

    T lr() const override { return lr_; }
    void set_lr(T lr) override { 
        lr_ = lr; 
        this->sync_lr_device(lr);
    }

    void set_momentum(T momentum) {
        momentum_ = momentum;
        if (momentum_ > T(0) && velocity_.empty()) {
            for (const auto& p : this->parameters_) {
                Tensor<T> v(p.shape(), p.device());
                v.fill(T(0));
                velocity_.push_back(v);
            }
        }
    }

    const std::vector<Tensor<T>>& velocity() const { return velocity_; }
    std::vector<Tensor<T>>& velocity() { return velocity_; }
    T weight_decay() const { return weight_decay_; }
    T momentum() const { return momentum_; }
    T clip_norm() const { return clip_norm_; }

    std::vector<Tensor<T>> state_buffers() const override { return velocity_; }
    void set_state_buffers(const std::vector<Tensor<T>>& bufs) override {
        for (size_t i = 0; i < velocity_.size() && i < bufs.size(); ++i)
            velocity_[i] = bufs[i];
    }
    std::vector<T> state_scalars() const override {
        return {lr_, weight_decay_, momentum_, clip_norm_};
    }
    void set_state_scalars(const std::vector<T>& s) override {
        lr_ = s[0];
        weight_decay_ = s[1];
        momentum_ = s[2];
        if (s.size() > 3) clip_norm_ = s[3];
        this->sync_lr_device(lr_);
    }
    size_t current_step() const override { return t_; }
    void set_step(size_t t) override { t_ = t; }


    void step() override {
        ++t_;
        NoGradGuard guard; 

        // Lazy init lr_device_ in case parameters weren't available at construction
        if (!this->lr_device_init_) {
            this->init_lr_device(lr_);
        }

        // Gradient clipping (graph-capture-safe, all device-side ops)
        if (clip_norm_ > T(0)) {
            this->clip_grad_norm_impl(clip_norm_);
        }

        for (size_t i = 0; i < this->parameters_.size(); ++i) {
            auto& p = this->parameters_[i];
            if (p.grad().empty()) continue;
            
            Tensor<T> current_grad = p.grad();

            if (weight_decay_ > (T)0.0) {
                current_grad = current_grad + (p * weight_decay_);
            }

            if (momentum_ > T(0)) {
                if (velocity_.size() <= i) {
                    Tensor<T> v(p.shape(), p.device());
                    v.fill(T(0));
                    velocity_.push_back(v);
                }
                // In-place velocity update: v = v * momentum + grad.
                // Keeps velocity_[i] at a fixed storage address, so graph capture
                // replays always read the current (not stale) velocity.
                {
                    Tensor<T> ms({1}, velocity_[i].device());
                    ms.fill(momentum_);
                    mul_(velocity_[i], ms);
                    add_(velocity_[i], current_grad);
                }
                // Use device-resident LR tensor so graph capture reads the
                // current lr from device memory instead of a baked scalar.
                Tensor<T> step = velocity_[i] * this->lr_device_;
                sub_(p, step);
            } else {
                Tensor<T> step = current_grad * this->lr_device_;
                sub_(p, step);
            }
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

    T lr() const override { return lr_; }
    void set_lr(T lr) override { lr_ = lr; }

    // Checkpoint state accessors for training resumption
    const std::vector<Tensor<T>>& m() const { return m_; }
    const std::vector<Tensor<T>>& v() const { return v_; }
    std::vector<Tensor<T>>& m() { return m_; }
    std::vector<Tensor<T>>& v() { return v_; }
    T beta1() const { return beta1_; }
    T beta2() const { return beta2_; }
    T eps() const { return eps_; }
    T weight_decay() const { return weight_decay_; }
    std::vector<Tensor<T>> state_buffers() const override {
        std::vector<Tensor<T>> r;
        r.reserve(m_.size() + v_.size());
        r.insert(r.end(), m_.begin(), m_.end());
        r.insert(r.end(), v_.begin(), v_.end());
        return r;
    }
    void set_state_buffers(const std::vector<Tensor<T>>& bufs) override {
        size_t n = m_.size();
        for (size_t i = 0; i < n && i < bufs.size(); ++i)
            m_[i] = bufs[i];
        for (size_t i = 0; i < n && n + i < bufs.size(); ++i)
            v_[i] = bufs[n + i];
    }
    std::vector<T> state_scalars() const override {
        return {lr_, beta1_, beta2_, eps_, weight_decay_};
    }
    void set_state_scalars(const std::vector<T>& s) override {
        lr_ = s[0];
        beta1_ = s[1];
        beta2_ = s[2];
        eps_ = s[3];
        weight_decay_ = s[4];
    }
    size_t current_step() const override { return t_; }
    void set_step(size_t t) override { t_ = t; }


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
class AdamW : public Optimizer<T> {
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
    AdamW(const std::vector<Tensor<T>>& params, T lr = 0.001, T beta1 = 0.9, T beta2 = 0.999, T eps = 1e-8, T weight_decay = (T)0.01) 
        : Optimizer<T>(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay), t_(0)
    {
        // Initialize moment buffers to zero
        for (const auto& p : this->parameters_) {
            Tensor<T> m(p.shape(), p.device());
            m.fill((T)0.0);
            m_.push_back(m);
            Tensor<T> v(p.shape(), p.device());
            v.fill((T)0.0);
            v_.push_back(v);
        }
    }

    T lr() const override { return lr_; }
    void set_lr(T lr) override { lr_ = lr; }

    // Checkpoint / state accessors
    const std::vector<Tensor<T>>& m() const { return m_; }
    const std::vector<Tensor<T>>& v() const { return v_; }
    std::vector<Tensor<T>>& m() { return m_; }
    std::vector<Tensor<T>>& v() { return v_; }
    T beta1() const { return beta1_; }
    T beta2() const { return beta2_; }
    T eps() const { return eps_; }

    std::vector<Tensor<T>> state_buffers() const override {
        std::vector<Tensor<T>> r;
        r.reserve(m_.size() + v_.size());
        r.insert(r.end(), m_.begin(), m_.end());
        r.insert(r.end(), v_.begin(), v_.end());
        return r;
    }
    void set_state_buffers(const std::vector<Tensor<T>>& bufs) override {
        size_t n = m_.size();
        for (size_t i = 0; i < n && i < bufs.size(); ++i)
            m_[i] = bufs[i];
        for (size_t i = 0; i < n && n + i < bufs.size(); ++i)
            v_[i] = bufs[n + i];
    }
    std::vector<T> state_scalars() const override {
        return {lr_, beta1_, beta2_, eps_, weight_decay_};
    }
    void set_state_scalars(const std::vector<T>& s) override {
        lr_ = s[0];
        beta1_ = s[1];
        beta2_ = s[2];
        eps_ = s[3];
        weight_decay_ = s[4];
    }
    size_t current_step() const override { return t_; }
    void set_step(size_t t) override { t_ = t; }

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
                adamw_step_gpu(p, g, m_[i], v_[i],
                               lr_, beta1_, beta2_, eps_,
                               bias_correction1, bias_correction2, weight_decay_);
                continue;
#endif
            }

            // CPU fallback: decoupled weight decay
            Tensor<T> grad = g;   // pure gradient
            m_[i] = (m_[i] * beta1_) + (grad * ((T)1.0 - beta1_));
            v_[i] = (v_[i] * beta2_) + (pow2(grad) * ((T)1.0 - beta2_));
            Tensor<T> m_hat = m_[i] * ((T)1.0 / bias_correction1);
            Tensor<T> v_hat = v_[i] * ((T)1.0 / bias_correction2);
            Tensor<T> denom = sqrt(v_hat) + eps_;
            // Adam update + decoupled weight decay as separate term
            Tensor<T> step_size = (m_hat / denom) * lr_ + p * lr_ * weight_decay_;
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