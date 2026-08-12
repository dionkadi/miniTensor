#pragma once

#include "Tensor.hpp"
#include "TensorOps.hpp"

// ---------------------------------------------------------
// Mean Squared Error Loss
// MSE = (1/N) * \sum (preds - targets)^2
// ---------------------------------------------------------
template<typename T>
Tensor<T> mse_loss(const Tensor<T>& preds, const Tensor<T>& targets) {
    if (preds.shape() != targets.shape()) {
        throw std::invalid_argument("MSE loss requires preds and targets to have the same shape.");
    }

    Tensor<T> diff = preds - targets;
    Tensor<T> sq = pow2(diff);
    
    // Iteratively sum across all dimensions to reduce to a scalar
    Tensor<T> reduced = sq;
    for (int i = reduced.shape().size() - 1; i >= 0; --i) {
        reduced = sum(reduced, i, false);
    }
    
    T n = (T)preds.total_elements();
    return reduced * ((T)1.0 / n);
}

// ---------------------------------------------------------
// Cross Entropy Loss (with Softmax included)
// CE = (-1/N) * \sum (targets * \ln(\text{softmax}(logits)))
// Expected shapes: [batch_size, num_classes] for both.
// ---------------------------------------------------------
template<typename T>
Tensor<T> cross_entropy(const Tensor<T>& logits, const Tensor<T>& targets, T smoothing = T(0)) {
    if (logits.shape() != targets.shape()) {
        throw std::invalid_argument("Cross-entropy requires logits and targets to have the same shape.");
    }

    if (logits.device().type != DeviceType::CPU) {
        // GPU: fused forward + backward via CrossEntropyBackward node
        Tensor<T> loss = cross_entropy_fwd_gpu(logits, targets, smoothing);
        if (GradMode::is_enabled() && logits.requires_grad()) {
            loss.set_requires_grad(true);
            loss.set_grad_fn(std::make_shared<CrossEntropyBackward<T>>(logits, targets, smoothing));
        }
        return loss;
    }

    // Smoothed one-hot targets: (1-s)*y + s/C
    Tensor<T> sm_targets = targets * (T(1) - smoothing);
    if (smoothing > T(0)) {
        Tensor<T> uniform(targets.shape());
        uniform.fill(T(smoothing) / T(targets.shape()[1]));
        sm_targets = sm_targets + uniform;
    }

    Tensor<T> exp_l = exp(logits);
    Tensor<T> sum_exp = sum(exp_l, 1, true);
    Tensor<T> probs = exp_l / sum_exp;
    Tensor<T> log_probs = log(probs);
    Tensor<T> ce_terms = sm_targets * log_probs;
    ce_terms = ce_terms * (T)-1.0;
    Tensor<T> reduced = ce_terms;
    for (int i = reduced.shape().size() - 1; i >= 0; --i) {
        reduced = sum(reduced, i, false);
    }
    T batch_size = (T)logits.shape()[0];
    return reduced * ((T)1.0 / batch_size);
}