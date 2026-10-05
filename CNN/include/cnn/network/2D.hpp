#pragma once

#include <cnn/network/Network.hpp>

namespace cnn {

/**
 * @brief Square Matrix to store data for black and white image
 * @author Dang Nguyen
 * @date 10/4/2026
 */
class Network2D : public Network {
    public:
        Network2D(std::vector<std::vector<ConvolutionLayer>> conv_layers,
                  std::vector<dnn::Network> densed_layers,
                  std::size_t max_threads=std::thread::hardware_concurrency());

        void forward(std::size_t side, const std::vector<double>& inputs) override;
        void back_propagation(const std::vector<std::vector<double>>& Ys) override;

    protected:
        void conv(const std::vector<std::vector<double>>& kernel, std::size_t stride=1) override;
        void activate(dnn::Activation_Type type=dnn::RELU) override;
        void pool(std::size_t p_side, std::size_t stride=2, PoolingType type=MAX_POOLING) override;
};

} // namespace cnn
