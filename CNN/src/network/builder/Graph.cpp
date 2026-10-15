#include <cnn/network/builder/Graph.hpp>

namespace cnn{
Builder::Builder() {
    edges[START].push_back(INVALID);
}

void Builder::add_node(std::string name=utils::random_str(), cnn::Layer* layer) {
    if (name == START || name == END || name == INVALID)
        throw std::invalid_argument("This node's name is reserved");

    if (nodes_name_count[name] > 0)
        throw std::invalid_argument("This name has already taken");

    if (nodes_layer_count[layer] > 0)
        throw std::invalid_argument("This layer has already been assigned to another node");

    nodes[name] = layer;
    nodes_name_count[name]++;
    nodes_layer_count[layer]++;
}

void Builder::add_edge(const std::string& from, const std::string& to) {
    std::ostringstream edge_name_oss;

    edge_name_oss << "[" << from << "] --> [" << to << "]";
    if (edges_count[edge_name_oss.str()] > 0)
        throw std::invalid_argument("Duplicate edge is not allowed");

    if (!nodes_name_count[from])
        throw std::invalid_argument("The source node has not been added");

    if (!nodes_name_count[to])
        throw std::invalid_argument("The destination node has not been added");

    if (from == START) {
        auto it = std::find(edges[from].begin(), edges[from].end(), INVALID);
        if (it != edges[from].end())
            edges[from].erase(it);
    }

    edges_count[edge_name_oss.str()]++;
    edges[from].push_back(to);
}

cnn::Network Builder::compile() {
    auto it = std::find(edges[START].begin(), edges[START].end(), INVALID);
    if (edges[START].empty() || it != edges[START].end())
        throw std::runtime_error("There must be an input layer");

    // Traverse down the graph
    has_end = false;
    current_level = 0;
    forward_validate(START);

    if (!has_end)
        throw std::runtime_error("There must be a layer that connect to the dense layer");

    
}

void Builder::forward_validate(const std::string& node) {
    for (std::size_t l = 0; l < nodes_stack.size(); ++l)
        for (const auto& l_node : nodes_stack[l])
            if (l_node == node && l <= current_level)
                throw std::runtime_error("Deadlock found! Only forward residues are allowed");

    if (edges[node].empty())
        throw std::runtime_error("Dangling node found!");

    nodes_stack[current_level++].push_back(node);
    for (const auto& next : edges[node]) {
        if (next == END) {
            if (has_end)
                throw std::runtime_error("There can only be 1 connection to the dense layer for this version");
            has_end = true;
        }

        forward_validate(node);
    }

    --current_level;
}

}
