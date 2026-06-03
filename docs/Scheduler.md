# Learning Rate Schedulers

## Overview

Schedulers adjust the learning rate during training. They wrap an `Optimizer` and
mutate its internal `lr` between epochs. This library provides four strategies
matching PyTorch's `torch.optim.lr_scheduler`:

| Scheduler | Behavior |
|---|---|
| `StepLR` | `lr *= gamma` every `step_size` epochs |
| `MultiStepLR` | `lr *= gamma` at user-specified epoch milestones |
| `ReduceLROnPlateau` | `lr *= factor` when validation loss stagnates |
| `CosineAnnealingLR` | Cosine decay from `base_lr` to `eta_min` over `T_max` epochs |

## Usage

```cpp
#include "Optimizer.hpp"
#include "Scheduler.hpp"

// Create model and optimizer
auto model = std::make_shared<MyModel>();
Adam<float> opt(model->parameters(), 0.001);

// Create scheduler wrapping the optimizer
StepLR<float> scheduler(opt, /*step_size=*/10, /*gamma=*/0.5);

for (int epoch = 0; epoch < 100; ++epoch) {
    train_one_epoch(model, opt);
    validate(model);

    scheduler.step();  // ← decays LR after each epoch
    std::cout << "Current LR: " << scheduler.get_lr() << "\n";
}
```

### ReduceLROnPlateau (metric-driven)

```cpp
ReduceLROnPlateau<float> scheduler(opt, 0.1, 5);

for (int epoch = 0; epoch < 100; ++epoch) {
    float val_loss = validate(model);
    scheduler.step(val_loss);  // ← LR decays when val_loss plateaus
}
```

## API Reference

### `LRScheduler<T>` (base class)

| Method | Description |
|---|---|
| `LRScheduler(Optimizer<T>& opt)` | Wrap an optimizer |
| `void step() = 0` | Advance one epoch, adjust LR |
| `T get_lr()` | Current LR from the optimizer |
| `void save_state(ofstream&)` | Serialize internal state for checkpoint |
| `void load_state(ifstream&)` | Deserialize internal state from checkpoint |

### `StepLR<T>`

```cpp
StepLR(Optimizer<T>& opt, int step_size, T gamma = 0.1);
```

LR formula: `lr = base_lr * gamma^{floor(epoch / step_size)}`

### `MultiStepLR<T>`

```cpp
MultiStepLR(Optimizer<T>& opt, const std::vector<int>& milestones, T gamma = 0.1);
```

LR formula: `lr = base_lr * gamma^{count of milestones reached}`

Milestones must be sorted in increasing order (validated at construction).

### `ReduceLROnPlateau<T>`

```cpp
ReduceLROnPlateau(Optimizer<T>& opt, T factor = 0.1, int patience = 10,
                  int cooldown = 0, T threshold = 1e-4);
```

| Parameter | Meaning |
|---|---|
| `factor` | Multiplicative factor applied when decaying |
| `patience` | Number of epochs with no improvement before decay |
| `cooldown` | Epochs to wait after a decay before resuming normal patience |
| `threshold` | Minimum change to qualify as an improvement |

Call `scheduler.step(metric)` with the validation loss after each epoch.

### `CosineAnnealingLR<T>`

```cpp
CosineAnnealingLR(Optimizer<T>& opt, int T_max, T eta_min = 0);
```

LR formula: `lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max))`

The LR decays smoothly from `base_lr` to `eta_min` over `T_max` epochs. Values
below `eta_min` are clamped.

## Checkpoint Integration

Scheduler state can be saved alongside model params and optimizer state by
passing the scheduler pointer to `save_checkpoint` / `load_checkpoint_into`:

```cpp
// Save
save_checkpoint("model.ckpt", model->parameters(), opt.m(), opt.v(),
                opt.step(), opt.lr(), opt.beta1(), opt.beta2(), opt.eps(), opt.weight_decay(),
                &scheduler);

// Load
size_t step;
float lr, b1, b2, eps, wd;
load_checkpoint_into("model.ckpt", model->parameters(), opt.m(), opt.v(),
                     step, lr, b1, b2, eps, wd,
                     &scheduler);
opt.set_step(step);
```

The scheduler data is appended after the optimizer data in the same binary file,
preceded by a `uint8_t` marker (`1` = present). Old checkpoints without scheduler
data are handled gracefully — the scheduler is left in its initial state.

## Training with ResNet10 (example)

In `test/resnet.cpp`, a `CosineAnnealingLR` scheduler decays the LR from the
initial `0.001` down to `1e-6` over 30 epochs. The scheduler is saved in the
checkpoint and restored on resume, so a killed-and-restarted training run
continues with the correct LR schedule.
