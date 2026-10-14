#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <cstddef>
#include <algorithm>
#include <memory>

#include <cnn/utils/String.hpp>

#include <cnn/layer/Layer.hpp>
#include <cnn/layer/Convolution.hpp>
#include <cnn/layer/Pool.hpp>

#define START "START_NODE"
#define END "END_NODE"

namespace cnn {

/**
 * @brief Responsible for building a network
 * @author Dang Nguyen
 * @date 14/10/2026
 * @note This was somewhat inspired by langgraph
 */
class Builder {
    private:
        std::vector<cnn::Layer*> layers_stack; // Layer oredered from START (inputs) -> END (dense layer)
        std::unordered_map<std::string, cnn::Layer*> nodes;
        std::unordered_map<std::string, std::string> edges;

    public:
        /**
         * @brief Add a node (layer)
         * @param name The name of the node. Will be randomized if not given
         * @param layer The layer to add for this node
         */
        void add_node(std::string name=utils::random_str(), cnn::Layer& layer);
};

}  