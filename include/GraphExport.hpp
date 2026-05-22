#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <memory>

#include "Module.hpp"

template<typename T>
class GraphExport {
public:
    struct NodeInfo {
        std::string type_name;
        std::vector<std::vector<size_t>> param_shapes;
        std::vector<NodeInfo> children;
    };

    static NodeInfo extract_graph(Module<T>& module) {
        NodeInfo info;
        info.type_name = module.name();

        for (auto& p : module.parameters()) {
            info.param_shapes.push_back(p.shape());
        }

        if (auto* seq = dynamic_cast<Sequential<T>*>(&module)) {
            info.type_name = "Sequential";
            for (auto& child : seq->modules()) {
                info.children.push_back(extract_graph(*child));
            }
        }

        return info;
    }

    static void save_graph(const std::string& path, Module<T>& module) {
        auto info = extract_graph(module);
        std::ofstream ofs(path, std::ios::binary);
        write_node(ofs, info);
        ofs.close();
    }

    static NodeInfo load_graph(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        return load_graph(ifs);
    }

    static NodeInfo load_graph(std::ifstream& ifs) {
        NodeInfo info;
        uint32_t name_len = 0, n_params = 0, n_children = 0;

        ifs.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        info.type_name.resize(name_len);
        ifs.read(info.type_name.data(), name_len);

        ifs.read(reinterpret_cast<char*>(&n_params), sizeof(n_params));
        info.param_shapes.resize(n_params);
        for (uint32_t i = 0; i < n_params; ++i) {
            uint32_t ndim = 0;
            ifs.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
            info.param_shapes[i].resize(ndim);
            for (uint32_t d = 0; d < ndim; ++d) {
                uint64_t dim = 0;
                ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                info.param_shapes[i][d] = static_cast<size_t>(dim);
            }
        }

        ifs.read(reinterpret_cast<char*>(&n_children), sizeof(n_children));
        for (uint32_t i = 0; i < n_children; ++i) {
            info.children.push_back(load_graph(ifs));
        }

        return info;
    }

private:
    static void write_node(std::ofstream& ofs, const NodeInfo& info) {
        uint32_t name_len = static_cast<uint32_t>(info.type_name.size());
        ofs.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        ofs.write(info.type_name.data(), name_len);

        uint32_t n_params = static_cast<uint32_t>(info.param_shapes.size());
        ofs.write(reinterpret_cast<const char*>(&n_params), sizeof(n_params));
        for (auto& shape : info.param_shapes) {
            uint32_t ndim = static_cast<uint32_t>(shape.size());
            ofs.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
            for (auto d : shape) {
                uint64_t dim = static_cast<uint64_t>(d);
                ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            }
        }

        uint32_t n_children = static_cast<uint32_t>(info.children.size());
        ofs.write(reinterpret_cast<const char*>(&n_children), sizeof(n_children));
        for (auto& child : info.children) {
            write_node(ofs, child);
        }
    }
};

template<typename T>
std::unique_ptr<Module<T>> build_from_graph(const typename GraphExport<T>::NodeInfo& info,
                                            Device device = {}) {
    const auto& name = info.type_name;

    if (name == "Sequential") {
        auto seq = std::make_unique<Sequential<T>>();
        for (auto& child : info.children) {
            if (auto mod = build_from_graph<T>(child, device)) {
                seq->add(std::move(mod));
            }
        }
        return seq;
    }

    if (name == "Linear") {
        return std::make_unique<Linear<T>>(
            info.param_shapes.size() >= 1 ? info.param_shapes[0][0] : 0,
            info.param_shapes.size() >= 1 ? info.param_shapes[0][1] : 0,
            device);
    }

    if (name == "Conv2D") {
        return std::make_unique<Conv2D<T>>(
            info.param_shapes.size() >= 1 ? static_cast<int>(info.param_shapes[0][0]) : 1,
            info.param_shapes.size() >= 1 ? static_cast<int>(info.param_shapes[0][1]) : 1,
            info.param_shapes.size() >= 1 ? static_cast<int>(info.param_shapes[0][2]) : 3,
            1, device);
    }

    if (name == "ReLU")     return std::make_unique<ReLU<T>>();
    if (name == "Tanh")     return std::make_unique<Tanh<T>>();
    if (name == "Sigmoid")  return std::make_unique<Sigmoid<T>>();
    if (name == "Flatten")  return std::make_unique<Flatten<T>>();
    if (name == "MaxPool2D")return std::make_unique<MaxPool2D<T>>(2);

    if (name == "LayerNorm") {
        return std::make_unique<LayerNorm<T>>(
            info.param_shapes.size() >= 1 ? static_cast<int>(info.param_shapes[0][0]) : 128,
            device);
    }

    if (name == "BatchNorm2d") {
        return std::make_unique<BatchNorm2d<T>>(
            info.param_shapes.size() >= 1 ? static_cast<int>(info.param_shapes[0][0]) : 32,
            device);
    }

    if (name == "Dropout")  return std::make_unique<Dropout<T>>(0.5f);

    return nullptr;
}
