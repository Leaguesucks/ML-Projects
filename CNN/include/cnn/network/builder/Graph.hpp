#pragma once

#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <cstddef>
#include <algorithm>

#include <cnn/utils/String.hpp>

#include <cnn/layer/Layer.hpp>
#include <cnn/layer/Convolution.hpp>
#include <cnn/layer/Pool.hpp>

#include <cnn/network/Network.hpp>

#define START "START_NODE"
#define END "END_NODE"
#define INVALID "INVALID_NODE"

namespace cnn {

/**
 * @brief Responsible for building a network
 * @author Dang Nguyen
 * @date 14/10/2026
 * @note This was somewhat inspired by langgraph
 */
class Builder {
    private:
        std::vector<std::vector<std::string>> nodes_stack; // Layer oredered in level from START (inputs) -> END (dense layer)
        std::unordered_map<std::string, cnn::Layer*> nodes;
        std::unordered_map<std::string, std::vector<std::string>> edges;

        std::unordered_map<std::string, std::size_t> nodes_name_count; // Avoid duplicate node names.
        std::unordered_map<cnn::Layer*, std::size_t> nodes_layer_count; // Avoid duplicate layer
        std::unordered_map<std::string, std::size_t> edges_count; // Avoid duplicate edges. The name is in the format 

        bool has_end = false;
        std::size_t current_level = 0;

    public:
        /**
         * @brief Default constructor
         */
        Builder();

        /**
         * @brief Add a node (layer)
         * @param name The name of the node. Will be randomized if not given
         * @param layer The layer to add for this node
         */
        void add_node(std::string name=utils::random_str(), cnn::Layer* layer);

        /**
         * @brief Add an edge
         * @param from The source edge
         * @param to The destination edge
         */
        void add_edge(const std::string& from, const std::string& to);

        /**
         * @brief Compile the graph
         * @return A fully connected functional CNN
         */
        cnn::Network compile();

    private:
        /**
         * @brief Traverse down the graph and validate
         * @param node The current node to validate
         */
        void forward_validate(const std::string& node);
};

}  