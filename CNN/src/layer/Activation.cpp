#include <cnn/layer/Activation.hpp>

namespace cnn {
ActivationLayer::ActivationLayer(std::size_t in_out_channels=1,
              const std::string& name=utils::random_str(),
              dnn::Activation_Type activation_type=dnn::RELU,
              std::size_t in_out_data_side)
: cnn::Layer(in_out_channels, in_out_channels, name, in_out_data_side, in_out_data_side),
  activation_type(activation_type) {
    layer_type = cnn::ACTV;
}

void ActivationLayer::forward(const std::vector<double>& data) {
    if (out_data_side * out_data_side * out_channels != data.size())
        throw std::invalid_argument("Data size mistmatch");

    in_data = data;

    switch (activation_type) {
        case dnn::RELU:
            for (std::size_t i; i < data.size(); ++i)
                out_data
    }
}
}