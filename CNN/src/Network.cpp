#include <cnn/network/Network.hpp>

namespace cnn {

Network::Network(const std::vector<std::vector<ConvolutionLayer>>& conv_layers,
                const std::vector<dnn::Network>& densed_layers,
                std::size_t max_threads)
                : max_threads(max_threads), side(0), conv_layers(conv_layers), densed_layers(densed_layers) {
    if (conv_layers.empty())
        throw std::invalid_argument("The conv_layers must not be empty");
    if (densed_layers.empty())
        throw std::invalid_argument("The densed layers must not be empty");
    if (max_threads <= 0)
        throw std::invalid_argument("There must be at least 1 thread");
}

void Network::set_max_threads(std::size_t max_threads) {
    if (max_threads <= 0)
        throw std::invalid_argument("There must be at least 1 thread");
    this->max_threads = max_threads;
}

} // namespace cnn
