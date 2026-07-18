// Scheduler unit tests
// Tests: StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
// Compile: edit CMakeLists.txt to set test/scheduler_test.cpp as the source

#include "Module.hpp"
#include "Optimizer.hpp"
#include "Scheduler.hpp"
#include "Serialization.hpp"

#include <cmath>
#include <iostream>
#include <cassert>
#include <string>

// Minimal model for testing: Linear(4 -> 2)
template<typename T>
class TestModel : public Module<T> {
public:
    Linear<T> fc_;
    TestModel(Device device = {})
        : fc_(4, 2, device) {}
    Tensor<T> forward(const Tensor<T>& x) override { return fc_.forward(x); }
    std::vector<Tensor<T>> parameters() const override { return fc_.parameters(); }
};

template<typename T>
T approx(T a, T b, T eps = T(1e-5)) {
    return std::abs(a - b) < eps;
}

template<typename T>
void test_step_lr() {
    std::cout << "  StepLR...\n";
    TestModel<T> model;
    SGD<T> opt(model.parameters(), T(0.1));
    StepLR<T> sched(opt, /*step_size=*/3, /*gamma=*/T(0.5));

    // Initial LR
    assert(approx(opt.lr(), T(0.1)));
    assert(approx(sched.get_lr(), T(0.1)));

    // Epochs 0, 1, 2: no decay (step_size=3, only decays at epoch 2, 5, 8...)
    // After epoch 0 (last_epoch=0): no decay
    // After epoch 1 (last_epoch=1): no decay
    // After epoch 2 (last_epoch=2): 0.1 * 0.5^1 = 0.05
    for (int i = 0; i < 2; ++i) sched.step();
    assert(approx(opt.lr(), T(0.1)));

    sched.step();  // epoch 2 -> decay
    assert(approx(opt.lr(), T(0.05)));

    // Epochs 3, 4: no decay
    sched.step();
    assert(approx(opt.lr(), T(0.05)));
    sched.step();
    assert(approx(opt.lr(), T(0.05)));

    // Epoch 5: decay again -> 0.05 * 0.5 = 0.025
    sched.step();
    assert(approx(opt.lr(), T(0.025)));

    std::cout << "    PASS\n";
}

template<typename T>
void test_multi_step_lr() {
    std::cout << "  MultiStepLR...\n";
    TestModel<T> model;
    Adam<T> opt(model.parameters(), T(0.01));
    MultiStepLR<T> sched(opt, {10, 20, 30}, T(0.1));

    assert(approx(opt.lr(), T(0.01)));

    // Epochs 0-9: no decay
    for (int i = 0; i < 10; ++i) sched.step();
    assert(approx(opt.lr(), T(0.01)));

    // Epoch 10 -> first decay: 0.01 * 0.1 = 0.001
    sched.step();
    assert(approx(opt.lr(), T(0.001)));

    // Epochs 11-19: no additional decay
    for (int i = 0; i < 9; ++i) sched.step();
    assert(approx(opt.lr(), T(0.001)));

    // Epoch 20 -> second decay: 0.001 * 0.1 = 0.0001
    sched.step();
    assert(approx(opt.lr(), T(0.0001)));

    std::cout << "    PASS\n";
}

template<typename T>
void test_reduce_lr_on_plateau() {
    std::cout << "  ReduceLROnPlateau...\n";
    TestModel<T> model;
    SGD<T> opt(model.parameters(), T(0.1));
    ReduceLROnPlateau<T> sched(opt, /*factor=*/T(0.5), /*patience=*/2,
                                /*cooldown=*/1, /*threshold=*/T(1e-4));

    assert(approx(opt.lr(), T(0.1)));

    // Metric improves every call -> no decay
    sched.step(T(1.0));
    assert(approx(opt.lr(), T(0.1)));
    sched.step(T(0.5));
    assert(approx(opt.lr(), T(0.1)));
    sched.step(T(0.25));
    assert(approx(opt.lr(), T(0.1)));

    // Metric plateaus (doesn't improve beyond threshold)
    sched.step(T(0.25 - T(1e-6)));  // within threshold
    assert(approx(opt.lr(), T(0.1)));

    // One more bad epoch triggers patience=2
    sched.step(T(0.26));
    // num_bad = 2 > patience=2 -> decay to 0.05
    assert(approx(opt.lr(), T(0.05)));

    std::cout << "    PASS\n";
}

template<typename T>
void test_cosine_annealing_lr() {
    std::cout << "  CosineAnnealingLR...\n";
    TestModel<T> model;
    SGD<T> opt(model.parameters(), T(1.0));
    CosineAnnealingLR<T> sched(opt, /*T_max=*/10, /*eta_min=*/T(0));

    // Epoch 0: lr = 0 + 0.5 * (1-0) * (1 + cos(0)) = 0.5 * 2 = 1.0
    sched.step();
    assert(approx(opt.lr(), T(1.0)));

    // Epoch 5 (last_epoch=5): lr = 0 + 0.5 * (1) * (1 + cos(pi*5/10))
    // = 0.5 * (1 + cos(pi/2)) = 0.5 * (1 + 0) = 0.5
    for (int i = 0; i < 4; ++i) sched.step();
    T expected = T(0.5) * (T(1) + std::cos(T(M_PI) * 5 / 10));
    assert(approx(opt.lr(), expected, T(1e-6)));

    // Epoch 10 (last_epoch=10): lr = 0 + 0.5 * (1) * (1 + cos(pi*10/10))
    // = 0.5 * (1 - 1) = 0.0 (clamped to eta_min=0)
    for (int i = 0; i < 5; ++i) sched.step();
    assert(approx(opt.lr(), T(0.0), T(1e-6)));

    std::cout << "    PASS\n";
}

template<typename T>
void test_scheduler_checkpoint_roundtrip() {
    std::cout << "  Checkpoint roundtrip with scheduler...\n";

    // Create original model + optimizer + scheduler
    TestModel<T> model_a;
    Adam<T> opt_a(model_a.parameters(), T(0.1), T(0.9), T(0.999), T(1e-8), T(0));
    StepLR<T> sched_a(opt_a, /*step_size=*/2, /*gamma=*/T(0.5));

    // Run a few steps to get a non-trivial state
    for (int i = 0; i < 3; ++i) sched_a.step();
    T lr_before = opt_a.lr();
    assert(approx(lr_before, T(0.05)));  // epoch 2 triggered decay

    // Save checkpoint with scheduler
    {
        save_checkpoint<T>("/tmp/sched_ckpt.bin",
                           model_a.parameters(),
                           opt_a.state_buffers(),
                           opt_a.current_step(),
                           opt_a.state_scalars(),
                           &sched_a);
    }

    // Create new model + optimizer + scheduler
    TestModel<T> model_b;
    Adam<T> opt_b(model_b.parameters(), T(0.1), T(0.9), T(0.999), T(1e-8), T(0));
    StepLR<T> sched_b(opt_b, /*step_size=*/2, /*gamma=*/T(0.5));

    // Load checkpoint with scheduler
    {
        auto params_b = model_b.parameters();
        auto opt_buf = opt_b.state_buffers();
        auto opt_scalars = opt_b.state_scalars();
        size_t loaded_step;
        load_checkpoint_into<T>("/tmp/sched_ckpt.bin",
                                params_b, opt_buf,
                                loaded_step, opt_scalars,
                                &sched_b);
        opt_b.set_state_buffers(opt_buf);
        opt_b.set_state_scalars(opt_scalars);
        opt_b.set_step(loaded_step);
    }

    // Verify LR was restored
    assert(approx(opt_b.lr(), lr_before));
    assert(approx(sched_b.get_lr(), lr_before));

    // Verify scheduler resumes correctly: next step should see the same LR pattern
    sched_b.step();  // epoch 3 -> no decay (step_size=2, so epoch 4 triggers next)
    assert(approx(opt_b.lr(), lr_before));

    sched_b.step();  // epoch 4 -> decay again
    T expected_lr = T(0.05) * T(0.5);  // 0.025
    assert(approx(opt_b.lr(), expected_lr));

    std::cout << "    PASS\n";
}

int main() {
    using T = float;

    std::cout << "=== Scheduler Tests ===\n\n";

    std::cout << "[1] StepLR\n";
    test_step_lr<T>();

    std::cout << "\n[2] MultiStepLR\n";
    test_multi_step_lr<T>();

    std::cout << "\n[3] ReduceLROnPlateau\n";
    test_reduce_lr_on_plateau<T>();

    std::cout << "\n[4] CosineAnnealingLR\n";
    test_cosine_annealing_lr<T>();

    std::cout << "\n[5] Checkpoint roundtrip with scheduler\n";
    test_scheduler_checkpoint_roundtrip<T>();

    std::cout << "\n=== All tests PASSED ===\n";
    return 0;
}
