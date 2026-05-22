#include "../include/Tensor.hpp"
#include "../include/TensorOps.hpp"
#include "../include/Autograd.hpp"
#include <iostream>
#include <cassert>

int main() {
    using T = float;

    // ---- test 1: slice autograd ----
    {
        auto A = Tensor<T>({3, 4}, Device{DeviceType::CPU});
        A.set_requires_grad(true);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                A.at({i, j}) = static_cast<T>(i * 4 + j + 1);

        auto A_view = view_slice(A, 0, 0, 2);
        assert(A_view.requires_grad());
        assert(A_view.shape()[0] == 2 && A_view.shape()[1] == 4);

        auto loss = sum(A_view, 1, false);  // sum over last dim -> [2]
        loss = sum(loss, 0, false);          // sum over remaining -> scalar
        loss.backward();

        auto grad = A.grad().contiguous();
        for (size_t j = 0; j < 4; ++j) {
            assert(grad.at({0, j}) == T(1));
            assert(grad.at({1, j}) == T(1));
            assert(grad.at({2, j}) == T(0));
        }
        std::cout << "  [PASS] slice_autograd\n";
    }

    // ---- test 2: version counter shared by views ----
    {
        auto A = Tensor<T>({3, 4}, Device{DeviceType::CPU});
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                A.at({i, j}) = static_cast<T>(i * 4 + j + 1);

        uint32_t v0 = A.version();
        auto A_view = A.slice(0, 0, 2);
        assert(A_view.version() == v0);

        add_(A_view, A_view);  // in-place doubles values
        assert(A.version() == v0 + 1);
        assert(A_view.version() == v0 + 1);
        std::cout << "  [PASS] version_counter\n";
    }

    // ---- test 3: transpose autograd ----
    {
        auto A = Tensor<T>({2, 3}, Device{DeviceType::CPU});
        A.set_requires_grad(true);
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                A.at({i, j}) = static_cast<T>(i * 3 + j + 1);

        auto A_t = view_transpose(A, 0, 1);
        assert(A_t.shape()[0] == 3 && A_t.shape()[1] == 2);

        auto loss = sum(A_t, 1, false);  // sum over last -> [3]
        loss = sum(loss, 0, false);       // scalar
        loss.backward();

        auto grad = A.grad().contiguous();
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                assert(grad.at({i, j}) == T(1));
        std::cout << "  [PASS] transpose_autograd\n";
    }

    // ---- test 4: in-place mutation detection ----
    {
        auto A = Tensor<T>({3, 4}, Device{DeviceType::CPU});
        A.set_requires_grad(true);

        SavedTensor<T> saved(A);
        auto A_view = A.slice(0, 0, 2);
        A_view.set_requires_grad(false);  // bypass leaf-guard for test
        add_(A_view, A_view);  // bumps version

        bool threw = false;
        try { saved.unpack(); }
        catch (const std::runtime_error&) { threw = true; }
        assert(threw);
        std::cout << "  [PASS] mutation_detection\n";
    }

    // ---- test 5: reshape autograd ----
    {
        auto A = Tensor<T>({2, 3, 4}, Device{DeviceType::CPU});
        A.set_requires_grad(true);
        A.fill(T(1));

        auto B = view_reshape(A, {6, 4});
        assert(B.shape()[0] == 6 && B.shape()[1] == 4);

        auto loss = sum(B, 1, false);
        loss = sum(loss, 0, false);
        loss.backward();

        auto grad = A.grad().contiguous();
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                for (size_t k = 0; k < 4; ++k)
                    assert(grad.at({i, j, k}) == T(1));
        std::cout << "  [PASS] reshape_autograd\n";
    }

    std::cout << "\nAll view tests passed!\n";
    return 0;
}
