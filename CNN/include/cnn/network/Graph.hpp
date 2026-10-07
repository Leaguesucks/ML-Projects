#pragma once

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
#include <cnn/layer/Activation.hpp>

#define START "START_NODE"
#define END "END_NODE"

namespace cnn {
/**
 * @brief Support non-linear CNN architecture, such as resnet
 * @date 10/07/2026
 * @author Dang Nguyen
 * @note This was inspired from langgraph.
 * @note The network starts at the START node (input layer) and ends at the END node (where it is connected to a DNN)
 */
class LayerGraph {
    private:
        std::unordered_map<std::string, cnn::Layer> nodes;
        std::unordered_map<std::string, std::vector<std::string>> edges;

    public:
        /**
         * @brief Add (a) layer(s) (node(s)) to the graph
         * @param names The name of each layer. If blank or empty then it will be randomized.
         *              MUST BE UNIQUE.
         * @param layers The layers to add to the graph
         */
        void add_nodes(const std::vector<std::string>& names, const std::vector<cnn::Layer>& layers);

        /**
         * @brief Connect the layers together. One layer can be connected to multiple others
         * @param from The source layers
         * @param to The destination layers
         */
        void add_edges(const std::vector<std::string>& from, const std::vector<std::string>& to);

        /**
         * @brief Delete one or multiple nodes
         * @param names The names of the nodes to delete
         */
        void delete_nodes(const std::vector<std::string>& names);

        /**
         * @brief Delete one or multiple edges
         * @param from The source layers
         * @param to The destination layers
         */
        void delete_edges(const std::vector<std::string>& from, const std::vector<std::string>& to);

        /**
         * @brief Compile the graph
         */
        void compile();

        std::unordered_map<std::string, cnn::Layer>& get_nodes() {return nodes;}
        std::unordered_map<std::string, std::vector<std::string>>& get_edges() {return edges;}
};
}