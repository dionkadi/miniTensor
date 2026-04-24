#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

inline std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b) {
    size_t max_dims = std::max(shape_a.size(), shape_b.size());
    std::vector<size_t> out_shape(max_dims);

    int offset_a = max_dims - shape_a.size();
    int offset_b = max_dims - shape_b.size();
    
    for (int i = max_dims - 1; i >= 0; --i) {
        size_t a_dim = (i >= offset_a) ? shape_a[i - offset_a] : 1;
        size_t b_dim = (i >= offset_b) ? shape_b[i - offset_b] : 1;
        
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::invalid_argument("Tensors cannot be broadcast together.");
        }
        out_shape[i] = std::max(a_dim, b_dim);
    }
    return out_shape;
}