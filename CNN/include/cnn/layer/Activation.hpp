#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>

#include <dnn/Layer.h>
#include <cnn/layer/Layer.hpp>

namespace cnn {
class ActivationLayer : Layer {
    private:
        dnn::Activation_Type activation_type;

    public:
        ActivationLayer(std::size_t in_out_channels=1,
              dnn::Activation_Type activation_type=dnn::RELU,
              std::size_t in_out_data_side);

        void forward(const std::vector<double>& data) override;
};
}