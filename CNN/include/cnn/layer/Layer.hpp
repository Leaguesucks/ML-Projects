#pragma once

#include <cstddef>
#include <vector>

namespace cnn {
enum PoolingType {
    MAX_POOLING,
    AVERAGE_POOLING
};

enum ConvolutionLayerType {
    CONV,
    ACTIVATE,
    POOL,
    DENSE
};

enum PaddingType {
    ZERO_PADD,
    SAME_PADD
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
 *          Layer.conv(...)
 *          ...
 *        }
 *       
 *      Then, int Layer.conv(...)
 *      
 *      #pragma omp for...
 */
class Layer {
    protected:
        ConvolutionLayerType layer_type;
        PaddingType padd_type;
        std::size_t stride, padding;
        std::size_t in_channels, out_channels;
        std::size_t in_data_side, out_data_side;
        std::vector<double> *in_data, out_data; 

    public:
        /**
         * @brief Default constructor
         * @param in_channels Number of input channels (1 for greyscale, 3 for RGB)
         * @param stride How fast is the operation window moving
         * @param padding The number of padding row/columns -> produce a (side + padding)^2 matrix
         * @param padd_type The padding type
         */
        Layer(std::size_t in_channels=1, std::size_t stride=1, 
              std::size_t padding=1, PaddingType padd_type=ZERO_PADD);

        /**
         * @brief Forward the data through this layer
         */

        ConvolutionLayerType get_layer() {return layer_type;}
        std::size_t get_out_channels_num() {return out_channels;}
        std::size_t get_out_data_side() { return out_data_side;}
};
}
