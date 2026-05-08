#include "Tensor.hpp"
#include "Optimizer.hpp"
#include "Loss.hpp"
#include "Module.hpp"

int main() {
    // 1. Define the network
    Sequential<float> model({
        std::make_shared<Linear<float>>(2, 16),
        std::make_shared<Tanh<float>>(),
        std::make_shared<Linear<float>>(16, 1),
        std::make_shared<Sigmoid<float>>()
    });

    // 2. Initialize the Optimizer
    // model.parameters() traverses Sequential -> Linear -> returns {W, b}
    SGD<float> optim(model.parameters(), 0.1f);

    // 3. Dummy Training Loop
    Tensor<float> X({4, 2}); // Batch of 4, 2 features
    X.fill(1.0f);            // Replace with real data
    
    Tensor<float> Y({4, 1}); // Batch of 4 targets
    Y.fill(0.0f); 

    for (int epoch = 0; epoch < 100; ++epoch) {
        // Forward pass
        Tensor<float> preds = model(X);
        
        // Calculate loss (using MSE from your Loss.hpp)
        Tensor<float> loss = mse_loss(preds, Y);

        // Zero gradients, backward pass, step
        optim.zero_grad();
        loss.backward();
        optim.step();

        // Optional: Print loss to verify it's decreasing
        std::cout << "Epoch " << epoch << " Loss: " << loss.data()[0] << "\n";
    }

    return 0;
}