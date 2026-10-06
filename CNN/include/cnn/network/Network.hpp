#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <omp.h>

#include <cnn/layer/Layer.hpp>
#include <dnn/Layer.h>
#include <dnn/Network.h>

namespace cnn {

/**
 * @brief CNN Network
 * @date 10/6/2026
 * @author Dang Nguyen
 */
class Network {
    private:
        std::vector<cnn::Layer> conv_layers;
        std::vector<double> inputs, outputs;
        dnn::Network densed_layer;

    public:
        /**
         * @brief Default constructor
         * @param conv_layers Array of convolutions layers to connect e.g., inputs -> conv_layers[0] -> con_layers[1] -> ... -> densed_layer -> output
         * @param densed_layer Dense neural network for this CNN, usually is the one produces the final output
         */
        Network(const std::vector<cnn::Layer>& conv_layers, const dnn::Network& densed_layer);

};

} // namespace cnn
