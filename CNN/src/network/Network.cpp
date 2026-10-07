#include <cnn/network/Network.hpp>

namespace cnn {

Network::Network(const std::vector<cnn::OperationLayer>& conv_layers, const dnn::Network& densed_layer)
: conv_layers(conv_layers), densed_layer(densed_layer) {
    if (conv_layers.empty())
        throw std::invalid_argument("The convolution layers cannot be empty");

    if (this->densed_layer.get_layers().empty())
        throw std::invalid_argument("The densed layer cannot be empty");

    for (auto iter = this->conv_layers.begin() + 1; iter < this->conv_layers.end() - 1; ++iter)
        if (iter->get_out_channels() != (iter + 1)->get_in_channels() ||
            iter->get_out_data_side() != (iter + 1)->get_in_data_side())
            throw std::invalid_argument("Mismatch inputs/outputs between convolution layers");

    if (this->conv_layers.back().get_out_data_side() * this->conv_layers.back().get_out_channels() !=
        this->densed_layer.get_layers()[0].get_n_inputs())
        throw std::invalid_argument("Mismatch inputs/outputs between convolution layers and densed layer");

}

} // namespace cnn
