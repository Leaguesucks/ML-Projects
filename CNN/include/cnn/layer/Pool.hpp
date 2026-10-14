#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <limits>

#include <cnn/layer/Layer.hpp>

namespace cnn {
class PoolLayer : public cnn::OperationLayer {
    private:
        cnn::PoolingType pooling_type;
        std::size_t window_side;

        // Output index -> max input index
        std::vector<std::size_t> max_indices; 

        // Input index -> pooling output indices whose averaging windows contain this input
        std::unordered_map<std::size_t, std::vector<std::size_t>> contributors;

    public:
        PoolLayer(std::size_t in_out_channels=1,
            std::size_t stride=1, std::size_t padding=0,
            std::size_t window_side=2, cnn::PoolingType pooling_type=cnn::MAX_POOLING,
            std::size_t in_data_side);

        const std::vector<double>& forward(const std::vector<double>& data) override;
        void backward() override;

        const std::vector<double>& get_final_out_data() override {return out_data;}

        cnn::PoolingType get_pooling_type() {return pooling_type;}
        std::size_t get_window_side() {return window_side;}

    protected:
        void calculate_prev_dldas() override;

    private:
        /**
         * @brief Perform the pooling operation
         */
        void pool();
}; 

}