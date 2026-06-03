// ResNet10 training on miniImageNet
// Tests: BatchNorm2d, Dropout, LayerNorm, Functional API, Skip connections,
//        Model serialization, ONNX-like graph export, Async DataLoader

#include "Dataset.hpp"
#include "Tensor.hpp"
#include "TensorOps.hpp"
#include "Module.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include "Serialization.hpp"
#include "GraphExport.hpp"
#include "Functional.hpp"
#include "GraphExecutor.hpp"
#include "Scheduler.hpp"

#include <format>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>
#include <map>
#include <dirent.h>
#include <sys/stat.h>
#include <csignal>

// STB image loader (single-header library)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ============================================================================
// MiniImageNet Dataset
// ============================================================================
// Structure: data/miniImagenet/<class_name>/<image_files>.JPEG
// 100 classes, 600 images per class = 60,000 total
// Variable image sizes (resize to target size during loading)

template<typename T>
class MiniImageNetDataset : public Dataset<T> {
private:
    std::vector<std::string> image_paths_;
    std::vector<int> labels_;
    std::map<std::string, int> class_to_idx_;
    std::vector<std::string> idx_to_class_;
    int target_height_;
    int target_width_;
    bool normalize_;
    
    // Random number generator for training augmentation
    std::mt19937 rng_;
    
public:
    MiniImageNetDataset(const std::string& root_dir, int target_size = 84,
                        bool normalize = true, unsigned int seed = 42)
        : target_height_(target_size), target_width_(target_size),
          normalize_(normalize), rng_(seed)
    {
        // Scan all class directories
        DIR* dir = opendir(root_dir.c_str());
        if (!dir) throw std::runtime_error("Cannot open directory: " + root_dir);
        
        struct dirent* entry;
        std::vector<std::string> class_names;
        
        while ((entry = readdir(dir)) != nullptr) {
            std::string name = entry->d_name;
            if (name == "." || name == "..") continue;
            
            std::string class_path = root_dir + "/" + name;
            struct stat st;
            if (stat(class_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
                class_names.push_back(name);
            }
        }
        closedir(dir);
        
        // Sort class names for consistent ordering
        std::sort(class_names.begin(), class_names.end());
        
        // Build class-to-index mapping
        for (size_t i = 0; i < class_names.size(); ++i) {
            class_to_idx_[class_names[i]] = static_cast<int>(i);
            idx_to_class_.push_back(class_names[i]);
        }
        
        std::cout << "Found " << class_names.size() << " classes" << std::endl;
        
        // Scan all images
        for (const auto& class_name : class_names) {
            std::string class_path = root_dir + "/" + class_name;
            int class_idx = class_to_idx_[class_name];
            
            DIR* class_dir = opendir(class_path.c_str());
            if (!class_dir) continue;
            
            while ((entry = readdir(class_dir)) != nullptr) {
                std::string filename = entry->d_name;
                if (filename.size() > 5 && 
                    (filename.substr(filename.size() - 5) == ".JPEG" ||
                     filename.substr(filename.size() - 4) == ".jpg")) {
                    image_paths_.push_back(class_path + "/" + filename);
                    labels_.push_back(class_idx);
                }
            }
            closedir(class_dir);
        }
        
        std::cout << "Total images: " << image_paths_.size() << std::endl;
    }
    
    size_t size() const override { return image_paths_.size(); }
    
    int num_classes() const { return static_cast<int>(class_to_idx_.size()); }
    
    std::pair<Tensor<T>, Tensor<T>> get(size_t index) const override {
        // Load image using stb_image
        int w, h, channels;
        unsigned char* data = stbi_load(image_paths_[index].c_str(), &w, &h, &channels, 3);
        
        if (!data) {
            throw std::runtime_error("Failed to load image: " + image_paths_[index]);
        }
        
        // Create output tensor [3, target_h, target_w]
        Tensor<T> image({size_t(1), size_t(3), 
                          static_cast<size_t>(target_height_), 
                          static_cast<size_t>(target_width_)});
        T* out = image.data();
        
        // Resize image to target size (simple bilinear interpolation)
        float x_ratio = static_cast<float>(w) / target_width_;
        float y_ratio = static_cast<float>(h) / target_height_;
        
        for (int y = 0; y < target_height_; ++y) {
            for (int x = 0; x < target_width_; ++x) {
                float src_x = x * x_ratio;
                float src_y = y * y_ratio;
                
                int x0 = static_cast<int>(src_x);
                int y0 = static_cast<int>(src_y);
                int x1 = std::min(x0 + 1, w - 1);
                int y1 = std::min(y0 + 1, h - 1);
                
                float x_frac = src_x - x0;
                float y_frac = src_y - y0;
                
                for (int c = 0; c < 3; ++c) {
                    float v00 = data[(y0 * w + x0) * 3 + c];
                    float v01 = data[(y0 * w + x1) * 3 + c];
                    float v10 = data[(y1 * w + x0) * 3 + c];
                    float v11 = data[(y1 * w + x1) * 3 + c];
                    
                    float v = v00 * (1 - x_frac) * (1 - y_frac) +
                              v01 * x_frac * (1 - y_frac) +
                              v10 * (1 - x_frac) * y_frac +
                              v11 * x_frac * y_frac;
                    
                    // Normalize to [0, 1] or use ImageNet normalization
                    if (normalize_) {
                        // Simple [0, 1] normalization
                        out[0 * target_height_ * target_width_ + y * target_width_ + x] = v / 255.0f;
                    } else {
                        out[0 * target_height_ * target_width_ + y * target_width_ + x] = v;
                    }
                }
            }
        }
        
        // Channels are already in order: R, G, B (stb_image loads as RGB)
        // No need to swap - the tensor layout is [C, H, W]
        
        stbi_image_free(data);
        
        // Create one-hot label tensor
        Tensor<T> label({1, static_cast<size_t>(class_to_idx_.size())});
        label.fill(T(0));
        label.data()[labels_[index]] = T(1);
        
        return {image, label};
    }
};

// ============================================================================
// BasicBlock for ResNet (with skip connection)
// ============================================================================
// BasicBlock: conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> (skip add) -> ReLU

template<typename T>
class BasicBlock : public Module<T> {
public:
    const char* name() const override { return "BasicBlock"; }
    
    Conv2D<T> conv1_;
    Conv2D<T> conv2_;
    BatchNorm2d<T> bn1_;
    BatchNorm2d<T> bn2_;
    bool downsample_;
    Conv2D<T> downsample_conv_;
    BatchNorm2d<T> downsample_bn_;
    
    BasicBlock(int in_channels, int out_channels, int stride = 1, Device device = {})
        : conv1_(in_channels, out_channels, 3, stride, 1, true, device),
          conv2_(out_channels, out_channels, 3, 1, 1, true, device),
          bn1_(out_channels, T(1e-5), T(0.1), device),
          bn2_(out_channels, T(1e-5), T(0.1), device),
          downsample_(stride != 1 || in_channels != out_channels),
          downsample_conv_(in_channels, out_channels, 1, stride, 0, true, device),
          downsample_bn_(out_channels, T(1e-5), T(0.1), device)
    {
    }
    
    Tensor<T> forward(const Tensor<T>& x) override {
        auto identity = x;
        
        auto out = conv1_.forward(x);
        out = bn1_.forward_relu(out);
        
        out = conv2_.forward(out);
        out = bn2_.forward(out);
        
        if (downsample_) {
            identity = downsample_conv_.forward(identity);
            identity = downsample_bn_.forward(identity);
        }
        
        out = out + identity;  // Skip connection
        out = relu(out);
        
        return out;
    }
    
    std::vector<Tensor<T>> parameters() const override {
        auto params = conv1_.parameters();
        auto p2 = conv2_.parameters();
        params.insert(params.end(), p2.begin(), p2.end());
        auto p3 = bn1_.parameters();
        params.insert(params.end(), p3.begin(), p3.end());
        auto p4 = bn2_.parameters();
        params.insert(params.end(), p4.begin(), p4.end());
        
        if (downsample_) {
            auto p5 = downsample_conv_.parameters();
            params.insert(params.end(), p5.begin(), p5.end());
            auto p6 = downsample_bn_.parameters();
            params.insert(params.end(), p6.begin(), p6.end());
        }
        
        return params;
    }
    
    void to(Device device) override {
        conv1_.to(device);
        conv2_.to(device);
        bn1_.to(device);
        bn2_.to(device);
        if (downsample_) {
            downsample_conv_.to(device);
            downsample_bn_.to(device);
        }
    }
    
    void train() {
        this->is_training_ = true;
        bn1_.train();
        bn2_.train();
        if (downsample_) downsample_bn_.train();
    }
    
    void eval() {
        this->is_training_ = false;
        bn1_.eval();
        bn2_.eval();
        if (downsample_) downsample_bn_.eval();
    }
};

// ============================================================================
// ResNet10: conv1 -> bn1 -> relu -> maxpool -> 
//           layer1 (2 blocks) -> layer2 (2 blocks) -> layer3 (2 blocks) ->
//           avgpool -> fc
// ============================================================================

template<typename T>
class ResNet10 : public Module<T> {
public:
    const char* name() const override { return "ResNet10"; }
    
    Conv2D<T> conv1_;
    BatchNorm2d<T> bn1_;
    MaxPool2D<T> maxpool_;
    
    BasicBlock<T> layer1_block1_;
    BasicBlock<T> layer1_block2_;
    BasicBlock<T> layer2_block1_;
    BasicBlock<T> layer2_block2_;
    BasicBlock<T> layer3_block1_;
    BasicBlock<T> layer3_block2_;
    
    size_t fc_in_features_;
    Linear<T> fc_;
    AdaptiveAvgPool2D<T> avg_pool_;
    
    ResNet10(int num_classes = 100, Device device = {})
        : conv1_(3, 64, 7, 2, 3, true, device),  // 84x84 -> 42x42
          bn1_(64, T(1e-5), T(0.1), device),
          maxpool_(3, 2, 1),                      // 42x42 -> 21x21
          layer1_block1_(64, 64, 1, device),
          layer1_block2_(64, 64, 1, device),
          layer2_block1_(64, 128, 2, device),     // 21x21 -> 11x11
          layer2_block2_(128, 128, 1, device),
          layer3_block1_(128, 256, 2, device),    // 11x11 -> 6x6
          layer3_block2_(256, 256, 1, device),
          fc_in_features_(256),
          fc_(fc_in_features_, num_classes, device),
          avg_pool_(1)
    {
    }
    
    Tensor<T> forward(const Tensor<T>& x) override {
        // Initial conv: 84x84 -> 42x42
        auto out = conv1_.forward(x);
        out = bn1_.forward_relu(out);
        
        // MaxPool: 42x42 -> 21x21
        out = maxpool_.forward(out);
        
        // Layer 1: 21x21 -> 21x21
        out = layer1_block1_.forward(out);
        out = layer1_block2_.forward(out);
        
        // Layer 2: 21x21 -> 11x11
        out = layer2_block1_.forward(out);
        out = layer2_block2_.forward(out);
        
        // Layer 3: 11x11 -> 6x6
        out = layer3_block1_.forward(out);
        out = layer3_block2_.forward(out);
        
        // Global average pooling: [N, 256, 6, 6] -> [N, 256, 1, 1]
        out = avg_pool_.forward(out);
        
        // Flatten: [N, 256, 1, 1] -> [N, 256]
        out = flatten(out);
        
        // FC: [N, 256] -> [N, num_classes]
        out = fc_.forward(out);
        
        return out;
    }
    
    std::vector<Tensor<T>> parameters() const override {
        auto params = conv1_.parameters();
        auto p = bn1_.parameters();
        params.insert(params.end(), p.begin(), p.end());
        
        auto p1 = layer1_block1_.parameters();
        params.insert(params.end(), p1.begin(), p1.end());
        auto p2 = layer1_block2_.parameters();
        params.insert(params.end(), p2.begin(), p2.end());
        auto p3 = layer2_block1_.parameters();
        params.insert(params.end(), p3.begin(), p3.end());
        auto p4 = layer2_block2_.parameters();
        params.insert(params.end(), p4.begin(), p4.end());
        auto p5 = layer3_block1_.parameters();
        params.insert(params.end(), p5.begin(), p5.end());
        auto p6 = layer3_block2_.parameters();
        params.insert(params.end(), p6.begin(), p6.end());
        
        auto pfc = fc_.parameters();
        params.insert(params.end(), pfc.begin(), pfc.end());
        
        return params;
    }
    
    void to(Device device) override {
        conv1_.to(device);
        bn1_.to(device);
        layer1_block1_.to(device);
        layer1_block2_.to(device);
        layer2_block1_.to(device);
        layer2_block2_.to(device);
        layer3_block1_.to(device);
        layer3_block2_.to(device);
        fc_.to(device);
    }
    
    void train() {
        this->is_training_ = true;
        bn1_.train();
        layer1_block1_.train();
        layer1_block2_.train();
        layer2_block1_.train();
        layer2_block2_.train();
        layer3_block1_.train();
        layer3_block2_.train();
    }
    
    void eval() {
        this->is_training_ = false;
        bn1_.eval();
        layer1_block1_.eval();
        layer1_block2_.eval();
        layer2_block1_.eval();
        layer2_block2_.eval();
        layer3_block1_.eval();
        layer3_block2_.eval();
    }
};

// ============================================================================
// Accuracy computation
// ============================================================================

template<typename T>
T compute_accuracy(const Tensor<T>& logits, const Tensor<T>& labels_one_hot) {
    size_t N = logits.shape()[0];
    size_t num_classes = logits.shape()[1];
    size_t correct = 0;
    
    const T* logits_ptr = logits.data();
    const T* labels_ptr = labels_one_hot.data();
    
    for (size_t i = 0; i < N; ++i) {
        int pred_class = 0;
        T max_val = logits_ptr[i * num_classes];
        
        for (size_t j = 1; j < num_classes; ++j) {
            if (logits_ptr[i * num_classes + j] > max_val) {
                max_val = logits_ptr[i * num_classes + j];
                pred_class = static_cast<int>(j);
            }
        }
        
        int true_class = 0;
        for (size_t j = 0; j < num_classes; ++j) {
            if (labels_ptr[i * num_classes + j] > T(0.5)) {
                true_class = static_cast<int>(j);
                break;
            }
        }
        
        if (pred_class == true_class) ++correct;
    }
    
    return static_cast<T>(correct) / static_cast<T>(N);
}

// ============================================================================
// Signal handling for graceful checkpoint save on SIGINT/SIGTERM
// ============================================================================

volatile sig_atomic_t g_interrupted = 0;
extern "C" void handle_signal(int) {
    g_interrupted = 1;
}

const std::string CHECKPOINT_PATH = "resnet10_checkpoint.bin";
const std::string CHECKPOINT_META_PATH = "resnet10_checkpoint_meta.bin";

template<typename T>
void save_checkpoint_meta(const std::string& path, int epoch, T best_val_acc) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    f.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));
    f.write(reinterpret_cast<const char*>(&best_val_acc), sizeof(T));
}

template<typename T>
bool load_checkpoint_meta(const std::string& path, int& epoch, T& best_val_acc) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(&epoch), sizeof(epoch));
    f.read(reinterpret_cast<char*>(&best_val_acc), sizeof(T));
    return f.good();
}

// ============================================================================
// ONNX Export (simplified text format)
// ============================================================================

template<typename T>
void export_onnx(const std::string& path, Module<T>& model, const std::vector<size_t>& input_shape) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path + " for writing");
    
    f << "ONNX Export (miniTensor format)\n";
    f << "================================\n\n";
    f << "ir_version: 7\n";
    f << "producer_name: \"miniTensor\"\n";
    f << "model_version: 1\n\n";
    
    f << "graph {\n";
    f << "  name: \"resnet10\"\n\n";
    
    // Input
    f << "  input {\n";
    f << "    name: \"input\"\n";
    f << "    type {\n";
    f << "      tensor_type {\n";
    f << "        elem_type: 1  // FLOAT\n";
    f << "        shape {\n";
    f << "          dim { dim_value: " << input_shape[0] << " }\n";
    f << "          dim { dim_value: " << input_shape[1] << " }\n";
    f << "          dim { dim_value: " << input_shape[2] << " }\n";
    f << "          dim { dim_value: " << input_shape[3] << " }\n";
    f << "        }\n";
    f << "      }\n";
    f << "    }\n";
    f << "  }\n\n";
    
    // Output
    f << "  output {\n";
    f << "    name: \"output\"\n";
    f << "    type {\n";
    f << "      tensor_type {\n";
    f << "        elem_type: 1  // FLOAT\n";
    f << "      }\n";
    f << "    }\n";
    f << "  }\n\n";
    
    // Extract graph structure
    auto info = GraphExport<T>::extract_graph(model);
    
    std::function<void(const typename GraphExport<T>::NodeInfo&, int)> print_node;
    print_node = [&](const typename GraphExport<T>::NodeInfo& node, int indent) {
        std::string sp(indent, ' ');
        f << sp << "node {\n";
        f << sp << "  name: \"" << node.type_name << "\"\n";
        f << sp << "  op_type: \"" << node.type_name << "\"\n";
        
        for (size_t i = 0; i < node.param_shapes.size(); ++i) {
            f << sp << "  attribute {\n";
            f << sp << "    name: \"weight_" << i << "\"\n";
            f << sp << "    type: TENSOR\n";
            f << sp << "    t {\n";
            f << sp << "      dims: [";
            for (size_t j = 0; j < node.param_shapes[i].size(); ++j) {
                if (j > 0) f << ", ";
                f << node.param_shapes[i][j];
            }
            f << "]\n";
            f << sp << "      data_type: 1  // FLOAT\n";
            f << sp << "    }\n";
            f << sp << "  }\n";
        }
        
        f << sp << "}\n";
        
        for (const auto& child : node.children) {
            print_node(child, indent);
        }
    };
    
    print_node(info, 2);
    
    f << "}\n";
    
    f.close();
    std::cout << "Exported ONNX model to " << path << std::endl;
}

// ============================================================================
// Train / Validation Split
// ============================================================================

template<typename T>
std::pair<std::unique_ptr<Dataset<T>>, std::unique_ptr<Dataset<T>>> 
train_val_split(Dataset<T>& dataset, double val_ratio = 0.1, unsigned int seed = 42) {
    size_t total = dataset.size();
    size_t val_size = static_cast<size_t>(total * val_ratio);
    size_t train_size = total - val_size;
    
    // Create indices
    std::vector<size_t> indices(total);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::mt19937 g(seed);
    std::shuffle(indices.begin(), indices.end(), g);
    
    std::vector<size_t> train_indices(indices.begin(), indices.begin() + train_size);
    std::vector<size_t> val_indices(indices.begin() + train_size, indices.end());
    
    // Create subset datasets
    class SubsetDataset : public Dataset<T> {
        const Dataset<T>& base_;
        std::vector<size_t> indices_;
    public:
        SubsetDataset(const Dataset<T>& base, std::vector<size_t> indices)
            : base_(base), indices_(std::move(indices)) {}
        size_t size() const override { return indices_.size(); }
        std::pair<Tensor<T>, Tensor<T>> get(size_t index) const override {
            return base_.get(indices_[index]);
        }
    };
    
    return {
        std::make_unique<SubsetDataset>(dataset, train_indices),
        std::make_unique<SubsetDataset>(dataset, val_indices)
    };
}

// ============================================================================
// Main
// ============================================================================

int main() {
    using T = float;
    
    std::cout << "=== ResNet10 Training on miniImageNet ===" << std::endl;
    
    // Load dataset
    std::cout << "\n[1] Loading miniImageNet dataset..." << std::endl;
    MiniImageNetDataset<T> full_dataset("data/miniImagenet", 84, true);
    
    int num_classes = full_dataset.num_classes();
    std::cout << "Number of classes: " << num_classes << std::endl;
    
    // Split into train/val
    auto [train_ds, val_ds] = train_val_split(full_dataset, 0.1, 42);
    std::cout << "Train samples: " << train_ds->size() << std::endl;
    std::cout << "Val samples: " << val_ds->size() << std::endl;
    
    // Create data loaders
    const size_t batch_size = 256;
    DataLoader<T> train_loader(*train_ds, batch_size, true);
    DataLoader<T> val_loader(*val_ds, batch_size, false);
    
    // Create model
    std::cout << "\n[2] Creating ResNet10 model..." << std::endl;
    auto model = std::make_shared<ResNet10<T>>(num_classes);
    
    Device gpu(DeviceType::CUDA);
    model->to(gpu);
    
    // Count parameters
    auto params = model->parameters();
    size_t num_params = 0;
    for (const auto& p : params) {
        num_params += p.total_elements();
    }
    std::cout << "Model parameters: " << num_params << std::endl;
    
    // Optimizer
    Adam<T> optimizer(model->parameters(), T(0.001), T(0.9), T(0.999), T(1e-8), T(0.0001));
    
    GraphExecutor<T> executor(model, optimizer,
        [](const Tensor<T>& pred, const Tensor<T>& target) {
            return cross_entropy(pred, target);
        });
    
    std::cout << "executor created\n";
    
    // Cosine annealing scheduler
    CosineAnnealingLR<T> scheduler(optimizer, 30, T(1e-6));
    
    const int epochs = 30;
    T best_val_acc = T(0);
    std::string best_weights_path = "best_resnet10.bin";
    int start_epoch = 0;
    
    // Resume from checkpoint if one exists
    {
        std::ifstream f(CHECKPOINT_PATH);
        if (f.good()) {
            f.close();
            std::cout << "\nCheckpoint found! Loading for resumption..." << std::endl;
            auto model_params = model->parameters();
            size_t loaded_step;
            T lr_tmp, b1, b2, eps_tmp, wd_tmp;
            load_checkpoint_into<T>(CHECKPOINT_PATH,
                                    model_params,
                                    optimizer.m(),
                                    optimizer.v(),
                                    loaded_step,
                                    lr_tmp, b1, b2, eps_tmp, wd_tmp,
                                    &scheduler);
            optimizer.set_step(loaded_step);
            
            int meta_epoch;
            T meta_best;
            if (load_checkpoint_meta(CHECKPOINT_META_PATH, meta_epoch, meta_best)) {
                start_epoch = meta_epoch + 1;
                if (meta_best > best_val_acc) best_val_acc = meta_best;
                std::cout << "Resuming from epoch " << start_epoch + 1 << "/" << epochs
                          << " (best val acc: " << (best_val_acc * 100) << "%)" << std::endl;
            }
        }
    }
    
    // Install signal handlers
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    
    // Lambda for saving checkpoint (used both at epoch end and on interrupt)
    auto save_checkpoint_now = [&](int current_epoch) {
        auto ckpt_params = model->parameters();
        const auto& const_opt = optimizer;
        save_checkpoint<T>(CHECKPOINT_PATH,
                          ckpt_params,
                          const_opt.m(), const_opt.v(),
                          const_opt.step(),
                          const_opt.lr(), const_opt.beta1(), const_opt.beta2(),
                          const_opt.eps(), const_opt.weight_decay(),
                          &scheduler);
        save_checkpoint_meta(CHECKPOINT_META_PATH, current_epoch, best_val_acc);
    };
    
    std::cout << "\n[3] Training for " << epochs << " epochs..." << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    for (int epoch = start_epoch; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        model->train();
        T train_loss = 0;
        size_t train_batches = 0;
        
        for (auto [bx, by] : train_loader) {
            auto bx_gpu = bx.to(gpu);
            auto by_gpu = by.to(gpu);
            
            auto loss = executor.step(bx_gpu, by_gpu);
            train_loss += loss.to({DeviceType::CPU}).data()[0];
            ++train_batches;

            if (g_interrupted) {
                save_checkpoint_now(epoch);
                std::cout << "\nInterrupted during epoch " << epoch + 1
                          << "! Checkpoint saved. Exiting." << std::endl;
                return 0;
            }
        }
        
        T avg_train_loss = train_loss / train_batches;
        
        // Validation phase
        model->eval();
        T val_loss = 0;
        T val_acc = 0;
        size_t val_batches = 0;
        
        {
            NoGradGuard guard;
            
            for (auto [bx, by] : val_loader) {
                bx = bx.to(gpu);
                by = by.to(gpu);
                
                auto pred = model->forward(bx);
                auto loss = cross_entropy(pred, by);
                
                val_loss += loss.to({DeviceType::CPU}).data()[0];
                val_acc += compute_accuracy(pred.to({DeviceType::CPU}), by.to({DeviceType::CPU}));
                ++val_batches;
            }
        }
        
        T avg_val_loss = val_loss / val_batches;
        T avg_val_acc = val_acc / val_batches;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        
        std::cout << "Epoch " << std::setw(2) << epoch + 1 << "/" << epochs
                  << " | train_loss: " << std::setprecision(4) << avg_train_loss
                  << " | val_loss: " << avg_val_loss
                  << " | val_acc: " << std::setprecision(2) << (avg_val_acc * 100) << "%"
                  << " | time: " << std::setprecision(1) << epoch_sec << "s";
        
        // Save best weights
        if (avg_val_acc > best_val_acc) {
            best_val_acc = avg_val_acc;
            model->save(best_weights_path);
            std::cout << " [BEST]";
        }
        std::cout << std::endl;
        
        // Save checkpoint for training resumption
        save_checkpoint_now(epoch);
        
        // Step scheduler (decays LR for next epoch)
        scheduler.step();
        
        // Graceful shutdown on SIGINT/SIGTERM (catch interrupts during validation)
        if (g_interrupted) {
            std::cout << "\nInterrupted! Checkpoint saved. Exiting." << std::endl;
            return 0;
        }
    }
    
    // Load best weights
    std::cout << "\n[4] Loading best weights..." << std::endl;
    model->load(best_weights_path);
    std::cout << "Best validation accuracy: " << (best_val_acc * 100) << "%" << std::endl;
    
    // Save final model
    std::cout << "\n[5] Saving model..." << std::endl;
    model->save("resnet10_final.bin");
    std::cout << "Saved final weights to resnet10_final.bin" << std::endl;
    
    // Export graph structure
    model->export_onnx("resnet10.graph");
    std::cout << "Saved graph structure to resnet10.graph" << std::endl;
    
    // Export ONNX (simplified format)
    export_onnx("resnet10.onnx.txt", *model, {1, 3, 84, 84});
    
    std::cout << "\n=== Training Complete ===" << std::endl;
    
    return 0;
}
