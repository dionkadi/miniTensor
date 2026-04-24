#pragma once
#include "Tensor.hpp"

template<typename T>
void add_gpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);

template<typename T>
void add_gpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);

template<typename T>
void matmul_gpu(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);

template<typename T>
void matmul_gpu_strided(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);