#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <limits>

#include <cnn/layer/Layer.hpp>

namespace cnn {
class PoolLayer : cnn::Layer {
    private:
        cnn::PoolingType pooling_type;
        std::size_t window_side;

    public:
        PoolLayer(std::size_t in_channels=1, std::size_t stride=2, 
                  std::size_t padding=0,
                  std::size_t window_side=2, PoolingType pooling_type=cnn::MAX_POOLING);

        void forward(std::size_t side, const std::vector<double>& data) override;

        cnn::PoolingType get_pooling_type() {return pooling_type;}
        std::size_t get_window_side() {return window_side;}

    private:
        /**
         * @brief Perform the pooling operation
         */
        void pool();
}; 

}