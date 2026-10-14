#pragma once

#include <stdexcept>
#include <cstddef>
#include <vector>
#include <string>
#include <omp.h>

#include <cnn/utils/String.hpp>
#include <cnn/layer/Convolution.hpp>
#include <cnn/layer/Pool.hpp>

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
        
        Layer* next_layer = nullptr; // The right next layer of this layer. Null if this is the last layer
        std::vector<Layer*> from_residues; // The residue layers that connect to this layer i.e., res layers -> this layer

        bool is_first_layer = false; // True if this is the first layer. Used to save some calculation effort in back propagation.

        std::size_t in_channels, out_channels;
        std::size_t in_data_side, out_data_side;
        std::vector<double> in_data; // in_data_side x in_data_side x in_channels
        std::vector<double> out_data; // out_data_side x out_data_side x out_channels. The output produced by this layer ONLY (no residue)

        // For back propagation
        std::vector<double> deltas; // dL/dz.
        std::vector<double> prev_dldas; // dL/da from the previous layer

    public:
        /**
         * @brief Default constructor
         * @param in_channels The number of input channels
         * @param out_channels The number of output channels
         * @param in_data_side The size = side x side of the input data
         * @param out_data_side The size = side x side of the output data
         */
        Layer(std::size_t in_channels, std::size_t out_channels,
              std::size_t in_data_side, std::size_t out_data_side);

        /**
         * @brief Forward the data through this layer
         * @param data The data to feed through this layer as a flat vector
         * @return The output + residues of this layer
         */
        virtual const std::vector<double>& forward(const std::vector<double>& data) = 0;

        /**
         * @brief Perform back-propagation
         * @note The reason why we do not return anything is because back propagation returns lots of different fields.
         */
        virtual void backward() = 0;

        /**
         * @return The final output data (e.g., after activation) of a layer
         */
        virtual const std::vector<double>& get_final_out_data() = 0;

        ConvolutionLayerType get_layer_type() {return layer_type;}
        std::size_t get_in_channels() {return in_channels;}
        std::size_t get_in_data_side() {return in_data_side;}
        std::size_t get_out_channels() {return out_channels;}
        std::size_t get_out_data_side() {return out_data_side;}

        const std::vector<double>& get_in_data() {return in_data;}
        const std::vector<double>& get_out_data() {return out_data;}

        std::vector<double>& get_deltas() {return deltas;}
        std::vector<double>& get_prev_dldas() {return prev_dldas;}

        void set_is_first_layer(bool is_first_layer) {this->is_first_layer = is_first_layer;}
        bool get_is_first_layer() {return is_first_layer;} 

        void set_next_layer(Layer* next_layer) {this->next_layer = next_layer;}
        const Layer* get_next_layer() {return next_layer;}

        std::vector<Layer*>& get_from_residues() {return from_residues;}

    protected:
        /**
         * @brief Calculate the dL/da from the previous layer i.e., dL/dInput of this layer
         */
        virtual void calculate_prev_dldas() = 0;

        /**
         * @brief Propagate the complete deltas to the residue sources
         */
        void backward_residue_deltas();
};

/**
 * @brief Represent a feature extraction layer
 * @date 10/5/2026
 * @author Dang Nguyen
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
        OperationLayer(std::size_t in_channels, std::size_t out_channels,
              std::size_t stride, std::size_t padding,
              std::size_t in_data_side, std::size_t out_data_side);

        std::size_t get_stride() {return stride;}
        std::size_t get_padding() {return padding;}

};
}
