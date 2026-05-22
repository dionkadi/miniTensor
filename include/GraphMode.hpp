#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <iostream>

#include "Tensor.hpp"

template<typename T>
struct GraphNode {
    enum class Kind { Input, Op };

    Kind kind;
    std::string op_name;
    std::vector<size_t> input_ids;
    size_t id = 0;
    std::vector<size_t> shape;
    Device device;
};

template<typename T>
class ComputationGraph {
public:
    std::vector<GraphNode<T>> nodes;
    std::vector<Tensor<T>> tensors;

    size_t add_input(Tensor<T> t) {
        GraphNode<T> node;
        node.kind = GraphNode<T>::Kind::Input;
        node.id = nodes.size();
        node.shape = t.shape();
        node.device = t.device();
        nodes.push_back(node);
        tensors.push_back(t);
        return node.id;
    }

    size_t add_op(const std::string& name, std::vector<size_t> inputs, Tensor<T> output) {
        GraphNode<T> node;
        node.kind = GraphNode<T>::Kind::Op;
        node.op_name = name;
        node.input_ids = std::move(inputs);
        node.id = nodes.size();
        node.shape = output.shape();
        node.device = output.device();
        nodes.push_back(std::move(node));
        tensors.push_back(output);
        return nodes.size() - 1;
    }

    void print(std::ostream& os = std::cout) const {
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& n = nodes[i];
            if (n.kind == GraphNode<T>::Kind::Input) {
                os << "%" << i << " = input(";
                for (size_t j = 0; j < n.shape.size(); ++j) {
                    os << n.shape[j] << (j + 1 < n.shape.size() ? "," : "");
                }
                os << ")\n";
            } else {
                os << "%" << i << " = " << n.op_name << "(";
                for (size_t j = 0; j < n.input_ids.size(); ++j) {
                    os << "%" << n.input_ids[j] << (j + 1 < n.input_ids.size() ? ", " : "");
                }
                os << ")\n";
            }
        }
    }

    Tensor<T> get_output(size_t id) const { return tensors[id]; }
};

template<typename T>
class GraphTracer {
public:
    ComputationGraph<T> trace(Module<T>& model, const Tensor<T>& input) {
        ComputationGraph<T> graph;
        std::unordered_map<const void*, size_t> memo;  // impl_ptr -> graph node id

        auto current_id = graph.add_input(input);
        Tensor<T> x = input;

        if (auto* seq = dynamic_cast<Sequential<T>*>(&model)) {
            for (auto& mod : seq->modules()) {
                auto out = mod->forward(x);
                current_id = graph.add_op(mod->name(), {current_id}, out);
                x = out;
            }
        } else {
            auto out = model.forward(x);
            current_id = graph.add_op(model.name(), {current_id}, out);
            x = out;
        }

        return graph;
    }

    static void replay_graph(ComputationGraph<T>& graph,
                             std::function<Tensor<T>(const std::string&, const std::vector<Tensor<T>>&)> executor) {
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            const auto& n = graph.nodes[i];
            if (n.kind == GraphNode<T>::Kind::Input) continue;

            std::vector<Tensor<T>> inputs;
            for (auto id : n.input_ids) {
                inputs.push_back(graph.tensors[id]);
            }

            graph.tensors[i] = executor(n.op_name, inputs);
        }
    }
};
