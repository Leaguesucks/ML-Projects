#include <cnn/layer/Layer.hpp>

namespace cnn {
Layer::Layer(std::size_t in_channels=1, std::size_t out_channels=1,
             std::size_t in_data_side, std::size_t out_data_side) 
: in_channels(in_channels), out_channels(out_channels),
  in_data_side(in_data_side), out_data_side(out_data_side) {
    out_data.assign(out_data_side * out_data_side * out_channels, 0.0);
}

OperationLayer::OperationLayer(std::size_t in_channels=1, std::size_t out_channels=1,
              std::size_t stride=1, std::size_t padding=0,
              std::size_t in_data_side, std::size_t out_data_side)
: Layer(in_channels, out_channels, in_data_side, out_data_side),
  stride(stride), padding(padding) {
    if (stride <= 0)
        throw std::invalid_argument("The stride must be greater than zero");
}
}