#include <cnn/layer/Layer.hpp>

namespace cnn {
Layer::Layer(std::size_t in_channels=1, std::size_t stride=1, 
             std::size_t padding=1)
: in_channels(in_channels), stride(stride),
  padding(padding) {
    if (in_channels <= 0)
        throw std::invalid_argument("There must be at least one input channel");
    if (stride <= 0)
        throw std::invalid_argument("The stride must be greater than zero");
}
}