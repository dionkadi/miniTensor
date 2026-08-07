// ResNet10 training on miniImageNet
// Tests: BatchNorm2d, Dropout, LayerNorm, Functional API, Skip connections,
//        Model serialization, ONNX-like graph export, Async DataLoader

// NOTE: standard/library includes MUST come before the miniTensor headers.
// The HIP headers (via Tensor.hpp -> GpuUtils.hpp) define __noinline__ as a
// macro; GCC 16's <chrono> -> <format> uses [[__gnu__::__noinline__]] literally,
// which fails to parse when the macro is active. Ordering std includes first
// avoids the conflict (applies to any TU that mixes HIP headers + <chrono>).
#include <fmt/base.h>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <map>
#include <dirent.h>
#include <sys/stat.h>
#include <csignal>

#include "Dataset.hpp"
#include "Tensor.hpp"
#include "TensorOps.hpp"
#include "Module.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include "Serialization.hpp"
#include "GraphExport.hpp"
#include "Functional.hpp"
#include "Scheduler.hpp"

// STB image loader (single-header library)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ============================================================================
// MiniImageNet Dataset
// ============================================================================
// Structure: data/miniImagenet/<class_name>/<image_files>.JPEG
// 100 classes, 600 images per class = 60,000 total
// Variable image sizes (resize to target size during loading)
// Pipeline: one-time RAM cache at padded size (target + crop pad, uint8);
// train mode applies random crop + flip + color jitter + random erasing,
// eval mode uses a center crop. Normalized with ImageNet stats.

// Bilinear RGB resize (uint8). Maps dst pixel (dx,dy) to src coords with
// x_ratio = sw/dw, y_ratio = sh/dh, clamped 4-tap interpolation.
static void resize_rgb_bilinear(const unsigned char* src, int sw, int sh,
                                unsigned char* dst, int dw, int dh) {
    float x_ratio = static_cast<float>(sw) / dw;
    float y_ratio = static_cast<float>(sh) / dh;
    for (int dy = 0; dy < dh; ++dy) {
        float src_y = dy * y_ratio;
        int y0 = static_cast<int>(src_y);
        int y1 = std::min(y0 + 1, sh - 1);
        float y_frac = src_y - y0;
        for (int dx = 0; dx < dw; ++dx) {
            float src_x = dx * x_ratio;
            int x0 = static_cast<int>(src_x);
            int x1 = std::min(x0 + 1, sw - 1);
            float x_frac = src_x - x0;
            for (int c = 0; c < 3; ++c) {
                float v00 = src[(y0 * sw + x0) * 3 + c];
                float v01 = src[(y0 * sw + x1) * 3 + c];
                float v10 = src[(y1 * sw + x0) * 3 + c];
                float v11 = src[(y1 * sw + x1) * 3 + c];
                float v = v00 * (1 - x_frac) * (1 - y_frac) +
                          v01 * x_frac * (1 - y_frac) +
                          v10 * (1 - x_frac) * y_frac +
                          v11 * x_frac * y_frac;
                dst[(dy * dw + dx) * 3 + c] = static_cast<unsigned char>(v + 0.5f);
            }
        }
    }
}

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
    bool cache_images_;
    std::vector<unsigned char> cache_;
    bool has_cache_ = false;
    
    // Random number generator for training augmentation
    mutable std::mt19937 rng_;

    // Train/eval mode: controls augmentation (random crop + flip in train,
    // center crop in eval). Mutable so it can be toggled between phases.
    mutable bool train_mode_ = true;

    // ImageNet normalization constants
    static constexpr float imagenet_mean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float imagenet_std[3]  = {0.229f, 0.224f, 0.225f};

    // Padding (pixels) added around the target size for random cropping.
    static constexpr int crop_pad_ = 8;

public:
    MiniImageNetDataset(const std::string& root_dir, int target_size = 84,
                        bool normalize = true, unsigned int seed = 42,
                        bool cache_images = true)
        : target_height_(target_size), target_width_(target_size),
          normalize_(normalize), cache_images_(cache_images), rng_(seed)
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

        // One-time RAM cache: every image resized to padded (target + crop pad)
        // resolution as uint8 RGB. For target 84 + pad 8 this is 60K x 100 x 100
        // x 3 bytes ≈ 1.8 GB; the build takes a few minutes.
        if (cache_images_ && !image_paths_.empty()) {
            int padded_h = target_height_ + 2 * crop_pad_;
            int padded_w = target_width_  + 2 * crop_pad_;
            cache_.resize(image_paths_.size() * static_cast<size_t>(padded_h) * padded_w * 3);
            std::cout << "Caching " << image_paths_.size() << " images (" << (cache_.size() / (1024 * 1024)) << " MB)...\n" << std::flush;
            for (size_t i = 0; i < image_paths_.size(); ++i) {
                int w, h, channels;
                unsigned char* data = stbi_load(image_paths_[i].c_str(), &w, &h, &channels, 3);
                if (!data) throw std::runtime_error("Failed to load image: " + image_paths_[i]);
                resize_rgb_bilinear(data, w, h,
                                    cache_.data() + i * static_cast<size_t>(padded_h) * padded_w * 3,
                                    padded_w, padded_h);
                stbi_image_free(data);
            }
            has_cache_ = true;
        }
    }
    
    size_t size() const override { return image_paths_.size(); }
    
    int num_classes() const { return static_cast<int>(class_to_idx_.size()); }

    void set_train_mode(bool mode) const override { train_mode_ = mode; }

    std::pair<Tensor<T>, Tensor<T>> get(size_t index) const override {
        // Padded size: resize to (target + crop_pad) then crop to target.
        // This gives random crops in train mode and center crops in eval mode.
        int padded_h = target_height_ + 2 * crop_pad_;
        int padded_w = target_width_  + 2 * crop_pad_;

        // Source pixels: from the one-time RAM cache when available, otherwise
        // load the JPEG and resize into a local buffer on the fly.
        std::vector<unsigned char> local_padded;
        const unsigned char* padded = nullptr;
        if (has_cache_) {
            padded = cache_.data() + index * (size_t(padded_h) * padded_w * 3);
        } else {
            int w, h, channels;
            unsigned char* data = stbi_load(image_paths_[index].c_str(), &w, &h, &channels, 3);
            if (!data) {
                throw std::runtime_error("Failed to load image: " + image_paths_[index]);
            }
            local_padded.resize(static_cast<size_t>(padded_h) * padded_w * 3);
            resize_rgb_bilinear(data, w, h, local_padded.data(), padded_w, padded_h);
            stbi_image_free(data);
            padded = local_padded.data();
        }

        // Determine crop offset within the padded image
        int crop_x, crop_y;
        if (train_mode_) {
            std::uniform_int_distribution<int> crop_dist(0, 2 * crop_pad_);
            crop_x = crop_dist(rng_);
            crop_y = crop_dist(rng_);
        } else {
            crop_x = crop_pad_;
            crop_y = crop_pad_;
        }

        // Random horizontal flip (train only)
        bool do_flip = false;
        if (train_mode_) {
            std::bernoulli_distribution flip_dist(0.5);
            do_flip = flip_dist(rng_);
        }

        // Color jitter factors (train only): brightness, contrast, saturation
        float bf = 1.0f, cf = 1.0f, sf = 1.0f;
        if (train_mode_) {
            std::uniform_real_distribution<float> b_dist(0.8f, 1.2f);
            std::uniform_real_distribution<float> c_dist(0.8f, 1.2f);
            std::uniform_real_distribution<float> s_dist(0.8f, 1.2f);
            bf = b_dist(rng_); cf = c_dist(rng_); sf = s_dist(rng_);
        }

        // Create output tensor [1, 3, target_h, target_w]
        Tensor<T> image({size_t(1), size_t(3),
                          static_cast<size_t>(target_height_),
                          static_cast<size_t>(target_width_)});
        T* out = image.data();

        // Crop + flip + jitter + normalize in a single pass. The padded buffer
        // is already at target resolution, so pixels are read directly (no
        // interpolation needed here):
        for (int y = 0; y < target_height_; ++y) {
            for (int x = 0; x < target_width_; ++x) {
                // Map output pixel to padded-image coordinates
                int px = do_flip ? (target_width_ - 1 - x + crop_x) : (x + crop_x);
                int py = y + crop_y;

                float r = padded[(py * padded_w + px) * 3 + 0];
                float g = padded[(py * padded_w + px) * 3 + 1];
                float b = padded[(py * padded_w + px) * 3 + 2];

                // Color jitter (train only): brightness, then contrast, then
                // saturation using the luminance of the jittered pixel.
                if (train_mode_) {
                    r *= bf; g *= bf; b *= bf;
                    r = (r - 128.0f) * cf + 128.0f;
                    g = (g - 128.0f) * cf + 128.0f;
                    b = (b - 128.0f) * cf + 128.0f;
                    float lum = 0.299f * r + 0.587f * g + 0.114f * b;
                    r = lum + sf * (r - lum);
                    g = lum + sf * (g - lum);
                    b = lum + sf * (b - lum);
                    r = std::max(0.0f, std::min(255.0f, r));
                    g = std::max(0.0f, std::min(255.0f, g));
                    b = std::max(0.0f, std::min(255.0f, b));
                }

                for (int c = 0; c < 3; ++c) {
                    float v = (c == 0) ? r : ((c == 1) ? g : b);
                    T out_val;
                    if (normalize_) {
                        // ImageNet normalization: (v/255 - mean) / std
                        out_val = static_cast<T>((v / 255.0f - imagenet_mean[c]) / imagenet_std[c]);
                    } else {
                        out_val = static_cast<T>(v / 255.0f);
                    }
                    out[c * target_height_ * target_width_ + y * target_width_ + x] = out_val;
                }
            }
        }

        // Random erasing (train only): zero out a random rectangle
        if (train_mode_) {
            std::bernoulli_distribution erase_dist(0.5);
            if (erase_dist(rng_)) {
                // Random rectangle, area 2-15% of image, aspect 0.3-3.3, filled with 0.
                std::uniform_real_distribution<float> area_dist(0.02f, 0.15f);
                std::uniform_real_distribution<float> aspect_dist(0.3f, 3.3f);
                float area = area_dist(rng_) * (target_height_ * target_width_);
                float aspect = aspect_dist(rng_);
                int eh = std::min(target_height_, std::max(1, static_cast<int>(std::round(std::sqrt(area * aspect)))));
                int ew = std::min(target_width_,  std::max(1, static_cast<int>(std::round(std::sqrt(area / aspect)))));
                std::uniform_int_distribution<int> hy_dist(0, target_height_ - eh);
                std::uniform_int_distribution<int> wx_dist(0, target_width_ - ew);
                int hy = hy_dist(rng_);
                int wx = wx_dist(rng_);
                for (int y = hy; y < hy + eh; ++y)
                    for (int x = wx; x < wx + ew; ++x)
                        for (int c = 0; c < 3; ++c)
                            out[c * target_height_ * target_width_ + y * target_width_ + x] = T(0);
            }
        }
        
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
    Dropout<T> dropout_{T(0.5)};
    
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
        
        // Dropout on the 256-dim features before the FC head (p=0.5, train only)
        out = dropout_.forward(out);
        
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
        dropout_.train();
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
        dropout_.eval();
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
        Dataset<T>& base_;  // non-const: allows train/eval mode toggling
        std::vector<size_t> indices_;
    public:
        SubsetDataset(Dataset<T>& base, std::vector<size_t> indices)
            : base_(base), indices_(std::move(indices)) {}
        size_t size() const override { return indices_.size(); }
        std::pair<Tensor<T>, Tensor<T>> get(size_t index) const override {
            return base_.get(indices_[index]);
        }
        void set_train_mode(bool mode) const override { base_.set_train_mode(mode); }
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
    setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);
    std::cout << std::unitbuf;
    
    using T = float;
    
    std::cout << "=== ResNet10 Training on miniImageNet ===" << std::endl;
    
    // Load dataset
    fmt::print("\n[1] Loading miniImageNet dataset...\n");
    MiniImageNetDataset<T> full_dataset("data/miniImagenet", 84, true);
    
    int num_classes = full_dataset.num_classes();
    fmt::print("Number of classes: {}\n", num_classes);
    
    // Split into train/val
    fmt::print("\n[2] Splitting into train/val...\n");
    auto [train_ds, val_ds] = train_val_split(full_dataset, 0.1, 42);
    fmt::print("Train samples: {}\n", train_ds->size());
    fmt::print("Val samples: {}\n", val_ds->size());
    
    // Create data loaders
    const size_t batch_size = 128;
    DataLoader<T> train_loader(*train_ds, batch_size, true);
    DataLoader<T> val_loader(*val_ds, batch_size, false);
    
    fmt::print("Train batches: {}\n", (train_ds->size() + batch_size - 1) / batch_size);
    fmt::print("Val batches: {}\n", (val_ds->size() + batch_size - 1) / batch_size);

    // Create model
    fmt::print("\n[2] Creating ResNet10 model...\n");
    auto model = std::make_shared<ResNet10<T>>(num_classes);
    
    Device gpu(DeviceType::CUDA);
    model->to(gpu);
    
    // Count parameters
    auto params = model->parameters();
    size_t num_params = 0;
    for (const auto& p : params) {
        num_params += p.total_elements();
    }

    fmt::print("Model parameters: {}\n", num_params);
    // Optimizer — SGD with momentum and weight decay.
    // base_lr 0.01 (stable after warmup; diagnosed earlier that LR > 0.015 diverges).
    // Gradient clipping at 10.0 catches rare spikes (steady-state grad norm ~1-10).
    // weight_decay 1e-3 (up from 5e-4) + Dropout(0.5) on the 256-dim features before
    // the FC head, because the model (2.81M params, 54K train images) overfits hard:
    // previous run reached train_loss ~0.01 vs val_loss ~1.6 with val acc plateauing ~60%.
    //
    // NOTE: this run is deliberately EAGER (no graph capture) — see the loop comment
    // below. Dropout's CPU-side mask (std::mt19937 in Dropout::forward) would be baked
    // into a captured graph and replayed with a FIXED mask every batch, which is wrong;
    // a graph-safe Dropout needs a GPU-side RNG kernel (library change).
    //
    // 400 epochs total (down from 1000): the previous run gained almost nothing from
    // epochs 30-500 at lr ~0.01 (val loss stuck ~1.9-2.1 while train loss fell to 0.1)
    // and only improved once cosine decayed LR below ~0.005. 20-epoch warmup, then
    // cosine from 0.01 to 1e-5 over the remaining 380 epochs.
    const T base_lr = T(0.01);
    const int epochs = 400;
    const int warmup_epochs = 20;
    SGD<T> optimizer(model->parameters(), base_lr, T(0.001), 0.9, T(10.0));
    // AdamW<T> optimizer(model->parameters(), T(0.001), T(0.9), T(0.999), T(1e-8), T(0.01));
    
    // Cosine annealing scheduler — T_max accounts for warmup epochs.
    // Warmup epochs use a linear LR ramp; cosine takes over afterward.
    CosineAnnealingLR<T> scheduler(optimizer, epochs - warmup_epochs, T(1e-5));
    
    T best_val_acc = T(0);
    std::string best_weights_path = "best_resnet10.bin";
    int start_epoch = 0;
    
    // Install signal handlers
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    
    // Lambda for saving checkpoint (epoch end or interrupt)
    auto save_checkpoint_now = [&](int current_epoch) {
        save_checkpoint<T>(CHECKPOINT_PATH,
                          model->parameters(),
                          optimizer.state_buffers(),
                          optimizer.current_step(),
                          optimizer.state_scalars(),
                          &scheduler);
        save_checkpoint_meta(CHECKPOINT_META_PATH, current_epoch, best_val_acc);
    };
    
    // Resume from checkpoint if one exists
    {
        std::ifstream f(CHECKPOINT_PATH);
        if (f.good()) {
            f.close();
            fmt::print("\nCheckpoint found! Loading for resumption...\n");
            auto model_params = model->parameters();
            auto opt_buf = optimizer.state_buffers();
            auto opt_scalars = optimizer.state_scalars();
            size_t loaded_step;
            load_checkpoint_into<T>(CHECKPOINT_PATH,
                                    model_params, opt_buf,
                                    loaded_step, opt_scalars,
                                    &scheduler);
            optimizer.set_state_buffers(opt_buf);
            optimizer.set_state_scalars(opt_scalars);
            optimizer.set_step(loaded_step);
            
            int meta_epoch;
            T meta_best;
            if (load_checkpoint_meta(CHECKPOINT_META_PATH, meta_epoch, meta_best)) {
                start_epoch = meta_epoch + 1;
                if (meta_best > best_val_acc) best_val_acc = meta_best;
                fmt::print("Resuming from epoch {} of {} (best val acc: {:.2f}%)\n", start_epoch + 1, epochs, (best_val_acc * 100));
            }
        }
    }
    
    fmt::print("\n[3] Training for {} epochs...\n", epochs);
    
    // GPU-resident loss accumulator (avoids per-batch D2H sync)
    Tensor<T> loss_accum({1}, gpu);
    
    // Number of full training batches (drop partial last batch — graph
    // capture requires a fixed batch shape)
    // size_t num_full_batches = train_ds->size() / batch_size;
    
    for (int epoch = start_epoch; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        model->train();
        full_dataset.set_train_mode(true);

        // Linear warmup: ramp LR from base_lr/warmup to base_lr over warmup_epochs.
        // After warmup, the cosine scheduler manages LR.
        if (epoch < warmup_epochs) {
            T warmup_lr = base_lr * T(epoch + 1) / T(warmup_epochs);
            optimizer.set_lr(warmup_lr);
        }
        
        loss_accum.fill(T(0));
        size_t train_batches = 0;
        
        for (auto [bx, by] : train_loader) {
            // Skip partial last batch — training loop assumes fixed batch size
            if (bx.shape()[0] != batch_size) break;
            
            auto bx_gpu = bx.to(gpu);
            auto by_gpu = by.to(gpu);
            
            // Deliberately EAGER: no graph capture for this run.
            // Dropout's CPU-side mask (std::mt19937 in Dropout::forward) would be baked
            // into a captured graph and replayed with a FIXED mask every batch — wrong;
            // a graph-safe Dropout needs a GPU-side RNG kernel (library change).
            // BatchNorm running stats ARE graph-safe now (device kernels folded outside
            // the captured graph, see BatchNorm2d::fold_running_stats in Module.hpp),
            // so a dropout-free graph-mode run would work.
            model->train();
            optimizer.zero_grad();
            auto preds = model->forward(bx_gpu);
            auto loss = cross_entropy(preds, by_gpu);
            loss.backward();
            optimizer.step();

            // Accumulate loss on GPU (no D2H sync per batch)
            add_(loss_accum, loss);
            ++train_batches;

            if (g_interrupted) {
                save_checkpoint_now(std::max(0, epoch - 1));
                std::cout << "Interrupted by user\n";
                return 0;
            }
        }
        
        // Single D2H sync at epoch end
        T avg_train_loss = loss_accum.to({DeviceType::CPU}).data()[0] / T(train_batches);
        
        // Validation phase
        model->eval();
        full_dataset.set_train_mode(false);
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
        fmt::print("Epoch {} of {} | lr: {:.6f} | train_loss: {:.4f} | val_loss: {:.4f} | val_acc: {:.2f}% | time: {:.1f}s", epoch + 1, epochs, optimizer.lr(), avg_train_loss, avg_val_loss, (avg_val_acc * 100), epoch_sec);
        // std::cout << "Epoch " << std::setw(2) << epoch + 1 << "/" << epochs
        //           << " | lr: " << std::setprecision(6) << optimizer.lr()
        //           << " | train_loss: " << std::setprecision(4) << avg_train_loss
        //           << " | val_loss: " << avg_val_loss
        //           << " | val_acc: " << std::setprecision(2) << (avg_val_acc * 100) << "%"
        //           << " | time: " << std::setprecision(1) << epoch_sec << "s";
        
        // Save best weights
        if (avg_val_acc > best_val_acc) {
            best_val_acc = avg_val_acc;
            model->save(best_weights_path);
            fmt::print(" [BEST]");
        }
        fmt::print("\n");
        
        // Save checkpoint for training resumption
        save_checkpoint_now(epoch);
        
        // Step scheduler only after warmup completes.
        // During warmup, LR is managed by the linear ramp above.
        if (epoch >= warmup_epochs) {
            scheduler.step();
        }
        
        // Graceful shutdown on SIGINT/SIGTERM (catch interrupts during validation)
        if (g_interrupted) {
            fmt::print("\nInterrupted! Checkpoint saved. Exiting.\n");
            save_checkpoint_now(epoch);
            return 0;
        }
    }
    
    // Load best weights
    fmt::print("\n[4] Loading best weights...\n");
    model->load(best_weights_path);
    fmt::print("Best validation accuracy: {:.2f}%\n", (best_val_acc * 100));
    
    // Save final model
    fmt::print("\n[5] Saving model...\n");
    model->save("resnet10_final.bin");
    fmt::print("Saved final weights to resnet10_final.bin\n");
    
    // Export graph structure
    model->export_onnx("resnet10.graph");
    fmt::print("Saved graph structure to resnet10.graph\n");

    // Export ONNX (simplified format)
    export_onnx("resnet10.onnx.txt", *model, {1, 3, 84, 84});
    
    fmt::print("\n=== Training Complete ===\n");
    
    return 0;
}
