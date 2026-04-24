#pragma once

#include "Tensor.hpp"
#include "TensorInfo.hpp"
#include <cstddef>

template<typename T>
void add_cpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    const T* a_ptr = A.data();
    const T* b_ptr = B.data();
    T* c_ptr = C.data();

    // Standard optimized CPU loop (could use OpenMP here later)
    for (size_t i = 0; i < A.total_elements(); ++i) {
        c_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

template<typename T>
void add_cpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    const T *A_ptr = A.data() + A.offset();
    const T *B_ptr = B.data() + B.offset();
    T *C_ptr = C.data() + C.offset();

    TensorInfo info_A(A.shape(), A.strides());
    TensorInfo info_B(B.shape(), B.strides());
    TensorInfo info_C(C.shape(), C.strides());

    collapse_dims(info_A, info_B, info_C);

    for (size_t i = 0; i < A.total_elements(); ++i) {
        size_t linear_idx = i;
        size_t phys_a = 0, phys_b = 0, phys_c = 0;

        for (int d = info_A.ndims - 1; d >= 0; --d) {
            size_t coord = linear_idx % info_A.shape[d];
            linear_idx /= info_A.shape[d];

            phys_a += coord * info_A.strides[d];
            phys_b += coord * info_B.strides[d];
            phys_c += coord * info_C.strides[d];
        }

        C_ptr[phys_c] = A_ptr[phys_a] + B_ptr[phys_b];
    }
}

template<typename T>
void matmul_cpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];

    const T* a_ptr = A.data() + A.offset();
    const T* b_ptr = B.data() + B.offset();
    T* c_ptr = C.data() + C.offset();

    // Naive O(N^3) matrix multiplication
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = 0;
            for (size_t k = 0; k < K; ++k) {
                // A[i, k] * B[k, j]
                sum += a_ptr[i * K + k] * b_ptr[k * N + j];
            }
            c_ptr[i * N + j] = sum;
        }
    }
}

template<typename T>
void matmul_cpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];

    const T* a_ptr = A.data() + A.offset();
    const T* b_ptr = B.data() + B.offset();
    T* c_ptr = C.data() + C.offset();

    // Cache strides locally for faster loop access
    size_t a_stride_m = A.strides()[0], a_stride_k = A.strides()[1];
    size_t b_stride_k = B.strides()[0], b_stride_n = B.strides()[1];
    size_t c_stride_m = C.strides()[0], c_stride_n = C.strides()[1];

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = 0;
            for (size_t k = 0; k < K; ++k) {
                // Read using exact strides
                T a_val = a_ptr[i * a_stride_m + k * a_stride_k];
                T b_val = b_ptr[k * b_stride_k + j * b_stride_n];
                sum += a_val * b_val;
            }
            c_ptr[i * c_stride_m + j * c_stride_n] = sum;
        }
    }
}