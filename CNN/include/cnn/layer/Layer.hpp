#pragma once

#include <stdexcept>
#include <cstddef>
#include <vector>
#include <string>

#include <cnn/utils/String.hpp>

namespace cnn {
enum PoolingType {
    MAX_POOLING,
    AVERAGE_POOLING
};

enum ConvolutionLayerType {
    CONV,
    POOL,
    ACTV,
};

/**
 * @brief Represent a layer in the convolution layer
 * @date 10/5/2026
 * @author Dang Nguyen
 */
class Layer {
    protected:
        ConvolutionLayerType layer_type;

        std::size_t in_channels, out_channels;
        std::size_t in_data_side, out_data_side;
        std::vector<double> in_data; // in_data_side x in_data_side x in_channels
        std::vector<double> out_data; // out_data_side x out_data_side x out_channels

    public:
        /**
         * @brief Default constructor
         * @param in_channels The number of input channels
         * @param out_channels The number of output channels
         * @param in_data_side The size = side x side of the input data
         * @param out_data_side The size = side x side of the output data
         */
        Layer(std::size_t in_channels=1, std::size_t out_channels=1,
              std::size_t in_data_side, std::size_t out_data_side);

        /**
         * @brief Forward the data through this layer
         * @param data The data to feed through this layer as a flat vector
         */
        virtual void forward(const std::vector<double>& data) = 0;

        ConvolutionLayerType get_layer_type() {return layer_type;}
        std::size_t get_in_channels() {return in_channels;}
        std::size_t get_in_data_side() {return in_data_side;}
        std::size_t get_out_channels() {return out_channels;}
        std::size_t get_out_data_side() {return out_data_side;}

        std::vector<double>& get_in_data() {return in_data;}
        std::vector<double>& get_out_data() {return out_data;}
};

/**
 * @brief Represent a feature extraction layer
 * @date 10/5/2026
 * @author Dang Nguyen
 * @note The operations in the layer will be parallelized using #pragma omp ..., assuming the 
 *       threads has already been initialized using #pragma omp parallel num_threads.
 *       So for example:
 *       
 *       #pragma omp parallel num_threads(...) {
 *          OperationLayer.conv(...)
 *          ...
 *        }
 *       
 *      Then, int OperationLayer.conv(...)
 *      
 *      #pragma omp for...
 */
class OperationLayer : public Layer {
    protected:
        std::size_t stride, padding;

    public:
        /**
         * @brief Default constructor
         * @param stride How fast is the operation window moving
         * @param padding The number of padding row/columns -> produce a (side + padding)^2 matrix
         * @param padd_type The padding type
         */
        OperationLayer(std::size_t in_channels=1, std::size_t out_channels=1,
              std::size_t stride=1, std::size_t padding=0,
              std::size_t in_data_side, std::size_t out_data_side);

};
}
