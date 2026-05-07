#pragma once

#include "Tensor.hpp"

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
Tensor<T> cross_entropy(const Tensor<T>& logits, const Tensor<T>& targets) {
    // Softmax: \exp(x_i) / \sum \exp(x_j)
    Tensor<T> exp_l = exp(logits);
    // Keepdims=true ensures broadcasting works division
    Tensor<T> sum_exp = sum(exp_l, 1, true); 
    Tensor<T> probs = exp_l / sum_exp;
    
    // Cross Entropy: targets * \ln(probs)
    Tensor<T> log_probs = log(probs);
    Tensor<T> ce_terms = targets * log_probs;
    
    ce_terms = ce_terms * (T)-1.0;
    
    // Reduce sum over all elements
    Tensor<T> reduced = ce_terms;
    for (int i = reduced.shape().size() - 1; i >= 0; --i) {
        reduced = sum(reduced, i, false);
    }
    
    // Average over the batch size
    T batch_size = (T)logits.shape()[0]; 
    return reduced * ((T)1.0 / batch_size);
}