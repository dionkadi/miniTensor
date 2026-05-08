#include "Dataset.hpp"
#include "Tensor.hpp"
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include "Tensor.hpp"
#include "TensorOps.hpp"
#include "Module.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

template<typename T>
class MNISTDataset : public Dataset<T> {
private:
    Tensor<T> images_;   // [N, 1, 28, 28]  (1 channel)
    Tensor<T> labels_;   // [N, 10]         (one‑hot)

public:
    MNISTDataset(const std::string& images_path, const std::string& labels_path) {
        // Read images
        std::ifstream img(images_path, std::ios::binary);
        if (!img) throw std::runtime_error("Cannot open images file");

        uint32_t magic, num_images, rows, cols;
        img.read(reinterpret_cast<char*>(&magic), 4);
        img.read(reinterpret_cast<char*>(&num_images), 4);
        img.read(reinterpret_cast<char*>(&rows), 4);
        img.read(reinterpret_cast<char*>(&cols), 4);

        // MNIST is big‑endian; swap bytes if system is little‑endian
        auto swap = [](uint32_t x) {
            return ((x & 0xFF000000) >> 24) |
                   ((x & 0x00FF0000) >> 8)  |
                   ((x & 0x0000FF00) << 8)  |
                   ((x & 0x000000FF) << 24);
        };
        magic = swap(magic);
        num_images = swap(num_images);
        rows = swap(rows);
        cols = swap(cols);

        if (magic != 2051) throw std::runtime_error("Invalid MNIST image file");

        // Allocate tensor [N, 1, H, W]
        images_ = Tensor<T>({num_images, 1, rows, cols});

        // Read pixel data
        std::vector<unsigned char> buffer(num_images * rows * cols);
        img.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        img.close();

        T* out = images_.data();
        for (size_t i = 0; i < buffer.size(); ++i) {
            out[i] = static_cast<T>(buffer[i]) / T(255.0); // normalize
        }

        // Read labels
        std::ifstream lbl(labels_path, std::ios::binary);
        if (!lbl) throw std::runtime_error("Cannot open labels file");

        uint32_t label_magic, num_labels;
        lbl.read(reinterpret_cast<char*>(&label_magic), 4);
        lbl.read(reinterpret_cast<char*>(&num_labels), 4);
        label_magic = swap(label_magic);
        num_labels = swap(num_labels);

        if (label_magic != 2049) throw std::runtime_error("Invalid MNIST label file");
        if (num_labels != num_images) throw std::runtime_error("Label/image count mismatch");

        std::vector<unsigned char> lbl_data(num_labels);
        lbl.read(reinterpret_cast<char*>(lbl_data.data()), num_labels);
        lbl.close();

        // Convert to one‑hot tensor [N, 10]
        labels_ = Tensor<T>({num_labels, 10});
        labels_.fill(T(0.0));
        T* label_out = labels_.data();
        for (size_t i = 0; i < num_labels; ++i) {
            int digit = lbl_data[i];
            if (digit < 0 || digit > 9) throw std::runtime_error("Invalid label");
            label_out[i * 10 + digit] = T(1.0);
        }
    }

    size_t size() const override { return images_.shape()[0]; }

    std::pair<Tensor<T>, Tensor<T>> get(size_t index) const override {
        // slice returns a view – no copy
        return { images_.slice(0, index, index + 1),
                 labels_.slice(0, index, index + 1) };
    }
};

template<typename T>
T compute_accuracy(const Tensor<T>& logits, const Tensor<T>& labels_one_hot) {
    // logits [N, 10], labels_one_hot [N, 10]
    size_t N = logits.shape()[0];
    size_t correct = 0;
    const T* logits_ptr = logits.data();
    const T* labels_ptr = labels_one_hot.data();
    for (size_t i = 0; i < N; ++i) {
        int pred_class = 0, true_class = 0;
        T max_val = logits_ptr[i * 10];
        for (int j = 1; j < 10; ++j) {
            if (logits_ptr[i * 10 + j] > max_val) {
                max_val = logits_ptr[i * 10 + j];
                pred_class = j;
            }
        }
        for (int j = 0; j < 10; ++j) {
            if (labels_ptr[i * 10 + j] > T(0.5)) {
                true_class = j;
                break;
            }
        }
        if (pred_class == true_class) ++correct;
    }
    return static_cast<T>(correct) / N;
}

int main() {
    using T = float;

    MNISTDataset<T> train_ds("data/train-images-idx3-ubyte",
                             "data/train-labels-idx1-ubyte");
    MNISTDataset<T> test_ds("data/t10k-images-idx3-ubyte",
                            "data/t10k-labels-idx1-ubyte");

    const size_t batch_size = 64;
    DataLoader<T> train_loader(train_ds, batch_size, /*shuffle=*/true);

    auto model = std::make_shared<Sequential<T>>(
        std::initializer_list<std::shared_ptr<Module<T>>>{
            // Conv1: 1 -> 16, kernel 3, pad 1  -> [N, 16, 28, 28]
            std::make_shared<Conv2D<T>>(1, 16, 3, 1, 1),
            std::make_shared<ReLU<T>>(),
            // Conv2: 16 -> 32, kernel 3, pad 1 -> [N, 32, 28, 28]
            std::make_shared<Conv2D<T>>(16, 32, 3, 1, 1),
            std::make_shared<ReLU<T>>(),
            std::make_shared<MaxPool2D<T>>(2), // [N, 32, 14, 14]

            // Conv3: 32 -> 64, kernel 3, pad 1 -> [N, 64, 14, 14]
            std::make_shared<Conv2D<T>>(32, 64, 3, 1, 1),
            std::make_shared<ReLU<T>>(),
            std::make_shared<MaxPool2D<T>>(2), // [N, 64, 7, 7]

            std::make_shared<Flatten<T>>(),    // [N, 64*7*7] = [N, 3136]
            std::make_shared<Linear<T>>(3136, 128),
            std::make_shared<ReLU<T>>(),
            std::make_shared<Linear<T>>(128, 10)
        }
    );

    Device gpu(DeviceType::CUDA);
    model->to(gpu);

    Adam<T> optimizer(model->parameters(), T(0.001), T(0.9), T(0.999), T(1e-8), T(0.0));

    const int epochs = 5;
    std::cout << std::fixed << std::setprecision(4);

    std::vector<Tensor<T>> test_x_batches, test_y_batches;
    for (size_t i = 0; i < test_ds.size(); ++i) {
        auto [x, y] = test_ds.get(i);
        test_x_batches.push_back(x);
        test_y_batches.push_back(y);
    }
    Tensor<T> test_x = concat(test_x_batches, /*dim=*/0).to(gpu);
    Tensor<T> test_y = concat(test_y_batches, /*dim=*/0).to(gpu);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        T running_loss = 0;
        size_t batches = 0;
        auto start = std::chrono::high_resolution_clock::now();

        for (auto [batch_x, batch_y] : train_loader) {
            // Move batch to GPU
            batch_x = batch_x.to(gpu);
            batch_y = batch_y.to(gpu);

            optimizer.zero_grad();
            auto pred = model->forward(batch_x);
            auto loss = cross_entropy(pred, batch_y);
            loss.backward();
            optimizer.step();

            running_loss += loss.data()[0];
            ++batches;
        }

        // Evaluate on test set (no DataLoader – full batch)
        auto test_pred = model->forward(test_x);
        auto test_loss = cross_entropy(test_pred, test_y);
        auto accuracy = compute_accuracy(test_pred, test_y);

        auto end = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(end - start).count();

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << " | train loss: " << running_loss / batches
                  << " | test loss: " << test_loss.data()[0]
                  << " | test acc: " << std::setprecision(2) << accuracy * 100 << "%"
                  << " | time: " << sec << "s\n";
    }

    return 0;
}