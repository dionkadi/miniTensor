#pragma once

#include "Tensor.hpp"
#include "Optimizer.hpp"

#include <cmath>
#include <fstream>
#include <vector>

// ============================================================================
// LRScheduler<T> — base class for learning rate schedulers
// ============================================================================
// Call pattern (matches PyTorch):
//   Most schedulers:  scheduler.step()        at end of each epoch
//   ReduceLROnPlateau: scheduler.step(metric) with validation loss
// ============================================================================

template<typename T>
class LRScheduler {
protected:
    Optimizer<T>& opt_;
    T base_lr_;
    int last_epoch_ = -1;   // -1 = no step taken, 0 = after 1st step()

public:
    LRScheduler(Optimizer<T>& opt)
        : opt_(opt), base_lr_(opt.lr()), last_epoch_(-1) {}
    virtual ~LRScheduler() = default;

    /// Called at epoch end. Subclass calls opt_.set_lr(new_lr).
    virtual void step() = 0;

    /// Current LR from the wrapped optimizer.
    T get_lr() const { return opt_.lr(); }

    /// Save internal state to an already-open binary stream.
    /// Called during checkpoint save (appended after optimizer data).
    virtual void save_state(std::ofstream& f) const {
        f.write(reinterpret_cast<const char*>(&last_epoch_), sizeof(last_epoch_));
        f.write(reinterpret_cast<const char*>(&base_lr_), sizeof(T));
    }

    /// Load internal state from an already-open binary stream.
    /// Called during checkpoint load.
    virtual void load_state(std::ifstream& f) {
        f.read(reinterpret_cast<char*>(&last_epoch_), sizeof(last_epoch_));
        f.read(reinterpret_cast<char*>(&base_lr_), sizeof(T));
        // Restore optimizer LR to base_lr_; step() will reapply schedule
        opt_.set_lr(base_lr_);
    }
};

// ============================================================================
// StepLR — decays LR by gamma every step_size epochs
// ============================================================================
//  lr = base_lr * gamma^{floor(epoch / step_size)}
// ============================================================================

template<typename T>
class StepLR : public LRScheduler<T> {
private:
    int step_size_;
    T gamma_;

public:
    StepLR(Optimizer<T>& opt, int step_size, T gamma = T(0.1))
        : LRScheduler<T>(opt), step_size_(step_size), gamma_(gamma) {}

    void step() override {
        this->last_epoch_++;
        if (this->last_epoch_ > 0 && this->last_epoch_ % step_size_ == 0) {
            int factor = this->last_epoch_ / step_size_;
            T new_lr = this->base_lr_ * std::pow(gamma_, factor);
            this->opt_.set_lr(new_lr);
        }
    }

    void save_state(std::ofstream& f) const override {
        LRScheduler<T>::save_state(f);
        f.write(reinterpret_cast<const char*>(&step_size_), sizeof(step_size_));
        f.write(reinterpret_cast<const char*>(&gamma_), sizeof(T));
    }

    void load_state(std::ifstream& f) override {
        LRScheduler<T>::load_state(f);
        f.read(reinterpret_cast<char*>(&step_size_), sizeof(step_size_));
        f.read(reinterpret_cast<char*>(&gamma_), sizeof(T));
        // Re-apply schedule based on restored last_epoch_
        int factor = this->last_epoch_ / step_size_;
        T new_lr = this->base_lr_ * std::pow(gamma_, factor);
        this->opt_.set_lr(new_lr);
    }
};

// ============================================================================
// MultiStepLR — decays LR by gamma at specified epoch milestones
// ============================================================================
//  lr = base_lr * gamma^{count of milestones reached}
// ============================================================================

template<typename T>
class MultiStepLR : public LRScheduler<T> {
private:
    std::vector<int> milestones_;
    T gamma_;

public:
    MultiStepLR(Optimizer<T>& opt, const std::vector<int>& milestones, T gamma = T(0.1))
        : LRScheduler<T>(opt), milestones_(milestones), gamma_(gamma)
    {
        // Milestones must be sorted
        for (size_t i = 1; i < milestones_.size(); ++i) {
            if (milestones_[i] <= milestones_[i - 1])
                throw std::invalid_argument("MultiStepLR milestones must be increasing");
        }
    }

    void step() override {
        this->last_epoch_++;
        int count = 0;
        for (auto m : milestones_) {
            if (this->last_epoch_ >= m) count++;
        }
        T new_lr = this->base_lr_ * std::pow(gamma_, count);
        this->opt_.set_lr(new_lr);
    }

    void save_state(std::ofstream& f) const override {
        LRScheduler<T>::save_state(f);
        uint32_t n = static_cast<uint32_t>(milestones_.size());
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (auto m : milestones_)
            f.write(reinterpret_cast<const char*>(&m), sizeof(m));
        f.write(reinterpret_cast<const char*>(&gamma_), sizeof(T));
    }

    void load_state(std::ifstream& f) override {
        LRScheduler<T>::load_state(f);
        uint32_t n;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        milestones_.resize(n);
        for (uint32_t i = 0; i < n; ++i)
            f.read(reinterpret_cast<char*>(&milestones_[i]), sizeof(milestones_[i]));
        f.read(reinterpret_cast<char*>(&gamma_), sizeof(T));
        // Re-apply schedule
        int count = 0;
        for (auto m : milestones_) {
            if (this->last_epoch_ >= m) count++;
        }
        T new_lr = this->base_lr_ * std::pow(gamma_, count);
        this->opt_.set_lr(new_lr);
    }
};

// ============================================================================
// ReduceLROnPlateau — decays LR when a metric stops improving
// ============================================================================
//  Call: scheduler.step(validation_loss) after each validation epoch.
//  The plain step() (no-arg) is a no-op for interface compatibility.
// ============================================================================

template<typename T>
class ReduceLROnPlateau : public LRScheduler<T> {
private:
    T factor_;
    int patience_;
    int cooldown_;
    T threshold_;
    T best_;
    int num_bad_;
    int cooldown_counter_;
    int epoch_;  // separate counter for metric-based steps

public:
    ReduceLROnPlateau(Optimizer<T>& opt, T factor = T(0.1), int patience = 10,
                      int cooldown = 0, T threshold = T(1e-4))
        : LRScheduler<T>(opt), factor_(factor), patience_(patience),
          cooldown_(cooldown), threshold_(threshold),
          best_(std::numeric_limits<T>::max()),
          num_bad_(0), cooldown_counter_(0), epoch_(0) {}

    /// No-op for interface compatibility; use step(T metric) instead.
    void step() override {}

    /// Main entry: call with validation loss after each epoch.
    void step(T metric) {
        this->last_epoch_++;
        epoch_++;

        // Check if metric improved
        if (metric < best_ - threshold_) {
            best_ = metric;
            num_bad_ = 0;
        } else {
            num_bad_++;
        }

        // Decrement cooldown
        if (cooldown_counter_ > 0)
            cooldown_counter_--;

        // If patience exhausted and not in cooldown, decay
        if (num_bad_ > patience_ && cooldown_counter_ == 0) {
            T new_lr = this->opt_.lr() * factor_;
            this->opt_.set_lr(new_lr);
            cooldown_counter_ = cooldown_;
            num_bad_ = 0;
        }
    }

    void save_state(std::ofstream& f) const override {
        LRScheduler<T>::save_state(f);
        f.write(reinterpret_cast<const char*>(&factor_), sizeof(T));
        f.write(reinterpret_cast<const char*>(&patience_), sizeof(patience_));
        f.write(reinterpret_cast<const char*>(&cooldown_), sizeof(cooldown_));
        f.write(reinterpret_cast<const char*>(&threshold_), sizeof(T));
        f.write(reinterpret_cast<const char*>(&best_), sizeof(T));
        f.write(reinterpret_cast<const char*>(&num_bad_), sizeof(num_bad_));
        f.write(reinterpret_cast<const char*>(&cooldown_counter_), sizeof(cooldown_counter_));
        f.write(reinterpret_cast<const char*>(&epoch_), sizeof(epoch_));
    }

    void load_state(std::ifstream& f) override {
        LRScheduler<T>::load_state(f);
        f.read(reinterpret_cast<char*>(&factor_), sizeof(T));
        f.read(reinterpret_cast<char*>(&patience_), sizeof(patience_));
        f.read(reinterpret_cast<char*>(&cooldown_), sizeof(cooldown_));
        f.read(reinterpret_cast<char*>(&threshold_), sizeof(T));
        f.read(reinterpret_cast<char*>(&best_), sizeof(T));
        f.read(reinterpret_cast<char*>(&num_bad_), sizeof(num_bad_));
        f.read(reinterpret_cast<char*>(&cooldown_counter_), sizeof(cooldown_counter_));
        f.read(reinterpret_cast<char*>(&epoch_), sizeof(epoch_));
    }
};

// ============================================================================
// CosineAnnealingLR — cosine decay from base_lr to eta_min over T_max epochs
// ============================================================================
//  lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max))
//
// After T_max, the value oscillates (clamped to eta_min for simplicity).
// ============================================================================

template<typename T>
class CosineAnnealingLR : public LRScheduler<T> {
private:
    int T_max_;
    T eta_min_;

public:
    CosineAnnealingLR(Optimizer<T>& opt, int T_max, T eta_min = T(0))
        : LRScheduler<T>(opt), T_max_(T_max), eta_min_(eta_min) {}

    void step() override {
        this->last_epoch_++;
        int epoch = this->last_epoch_;
        if (epoch <= 0) return;

        // Cosine formula with clamping after T_max
        T cos_val = std::cos(static_cast<T>(M_PI) * epoch / T_max_);
        T new_lr = eta_min_ + T(0.5) * (this->base_lr_ - eta_min_) * (T(1) + cos_val);

        // Clamp: LR should never go below eta_min_
        if (new_lr < eta_min_) new_lr = eta_min_;

        this->opt_.set_lr(new_lr);
    }

    void save_state(std::ofstream& f) const override {
        LRScheduler<T>::save_state(f);
        f.write(reinterpret_cast<const char*>(&T_max_), sizeof(T_max_));
        f.write(reinterpret_cast<const char*>(&eta_min_), sizeof(T));
    }

    void load_state(std::ifstream& f) override {
        LRScheduler<T>::load_state(f);
        f.read(reinterpret_cast<char*>(&T_max_), sizeof(T_max_));
        f.read(reinterpret_cast<char*>(&eta_min_), sizeof(T));
        // Re-apply schedule
        int epoch = this->last_epoch_;
        T cos_val = std::cos(static_cast<T>(M_PI) * epoch / T_max_);
        T new_lr = eta_min_ + T(0.5) * (this->base_lr_ - eta_min_) * (T(1) + cos_val);
        if (new_lr < eta_min_) new_lr = eta_min_;
        this->opt_.set_lr(new_lr);
    }
};
