#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <limits>

#include <cnn/layer/Layer.hpp>

namespace cnn {
class PoolLayer : cnn::OperationLayer {
    private:
        cnn::PoolingType pooling_type;
        std::size_t window_side;

    public:
        PoolLayer(std::size_t in_out_channels=1,
            const std::string& name=utils::random_str(),
            std::size_t stride=1, std::size_t padding=0,
            std::size_t window_side=2, cnn::PoolingType pooling_type=cnn::MAX_POOLING,
            std::size_t in_data_side);

        void forward(const std::vector<double>& data) override;

        cnn::PoolingType get_pooling_type() {return pooling_type;}
        std::size_t get_window_side() {return window_side;}

    private:
        /**
         * @brief Perform the pooling operation
         */
        void pool();
}; 

}