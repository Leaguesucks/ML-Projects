#include <cnn/layer/Layer.hpp>

namespace cnn {
Layer::Layer(std::size_t in_channels=1, std::size_t stride=1, 
             std::size_t padding=1, PaddingType padd_type=ZERO_PADD)
: in_channels(in_channels), stride(stride),
  padding(padding), padd_type(padd_type) {}
}