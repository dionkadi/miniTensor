#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

constexpr size_t MAX_DIMS = 8;

struct TensorInfo {
    size_t shape[MAX_DIMS];
    size_t strides[MAX_DIMS];
    size_t ndims;

    TensorInfo(const std::vector<size_t>& s, const std::vector<size_t>& st) {
        ndims = s.size();
        if (ndims > MAX_DIMS) throw std::runtime_error("Exceeded MAX_DIMS");
        for (size_t i = 0; i < ndims; ++i) {
            shape[i] = s[i];
            strides[i] = st[i];
        }
    }
};

inline void collapse_dims(TensorInfo& a, TensorInfo& b, TensorInfo& c) {
    if (a.ndims <= 1) return;

    TensorInfo na = a, nb = b, nc = c;
    size_t keep_idx = 0;

    for (size_t i = 1; i < a.ndims; ++i) {
        bool can_collapse = 
            (na.strides[keep_idx] == a.shape[i] * a.strides[i]) &&
            (nb.strides[keep_idx] == b.shape[i] * b.strides[i]) &&
            (nc.strides[keep_idx] == c.shape[i] * c.strides[i]);

        if (can_collapse) {
            na.shape[keep_idx] *= a.shape[i];
            na.strides[keep_idx] = a.strides[i]; // The stride becomes the innermost stride
            
            nb.shape[keep_idx] *= b.shape[i];
            nb.strides[keep_idx] = b.strides[i];
            
            nc.shape[keep_idx] *= c.shape[i];
            nc.strides[keep_idx] = c.strides[i];
        } else {
            // Cannot merge, so we must advance keep_idx and copy the dimension over
            keep_idx++;
            na.shape[keep_idx] = a.shape[i];
            na.strides[keep_idx] = a.strides[i];
            
            nb.shape[keep_idx] = b.shape[i];
            nb.strides[keep_idx] = b.strides[i];
            
            nc.shape[keep_idx] = c.shape[i];
            nc.strides[keep_idx] = c.strides[i];
        }
    }

    size_t final_ndims = keep_idx + 1;
    na.ndims = final_ndims;
    nb.ndims = final_ndims;
    nc.ndims = final_ndims;

    a = na; 
    b = nb; 
    c = nc;
}