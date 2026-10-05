#pragma once

#include <algorithm>
#include <thread>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <cmath>

#include <cnn/layer/Layer.hpp>
#include <dnn/Layer.h>
#include <dnn/Network.h>

namespace cnn {

enum Channel {
    R,
    G,
    B
};

enum Dimension {
    D2,
    D3
};

/**
 * @brief Represents a convolution layer. Applies multiple layers to form a convnet
 */
struct ConvolutionLayer {
    ConvolutionLayerType type;
    PoolingType pooling_type;
    dnn::Activation_Type activation_type;

    std::vector<std::vector<double>> kernel;
    std::size_t stride;
    std::size_t p_side;

    std::size_t in_channels;
    std::size_t out_channels;
};

/**
 * @brief Template class for a convolution network
 * @author Dang Nguyen
 * @date 9/30/2026
 */
class Network {
    protected:
        Dimension dimension;

        std::size_t max_threads;
        std::size_t side;

        std::vector<std::vector<ConvolutionLayer>> conv_layers;
        std::vector<dnn::Network> densed_layers;
        std::vector<double> data;

    public:
        /**
         * @brief Construct a new convolution neural network
         * @param conv_layers The layers of the convolution network, usually size 1 for 2D images and 3 for 3D images
         * @param densed_layers The dense layers of the network, usually size 1 for 2D images and 3 for 3D images
         * @param max_threads The maximum number of threads to use for parallelization
         */
        Network(const std::vector<std::vector<ConvolutionLayer>>& conv_layers,
                const std::vector<dnn::Network>& densed_layers,
                std::size_t max_threads=std::thread::hardware_concurrency());

        /**
         * @brief Forward the input data into the network
         * @param side Construct a side x side matrix
         * @param inputs The input data to forward into the network
         */
        void virtual forward(std::size_t side, const std::vector<double>& inputs) = 0;

        /**
         * @brief Perform back propagation on each densed layer
         * @param Ys The expected output for each densed layer
         */
        void virtual back_propagation(const std::vector<std::vector<double>>& Ys) = 0;
        
        Dimension get_dimension() {return dimension;}
        
        void set_max_threads(std::size_t max_threads);
        std::size_t get_max_threads() {return max_threads;}
        
        std::size_t get_side() {return side;}
        std::vector<double>& get_data() {return data;}

        std::vector<std::vector<ConvolutionLayer>>& get_conv_layers() {return conv_layers;}
        std::vector<dnn::Network>& get_densed_layers() {return densed_layers;}

    protected:
            /**
         * @brief Convolute this matrix using a kernel
         * @param kernel The kernel to applied convolution to the current matrix, Must be squared
         * @param stride How fast should we slide the kernel accross the matrix
         */
        virtual void conv(const std::vector<std::vector<double>>& kernel, std::size_t stride=1) = 0;

        /**
         * @brief Filter the matrix through an activation fucntion
         * @param type The type of activation function to applied
         */
        virtual void activate(dnn::Activation_Type type=dnn::RELU) = 0;

        /**
         * @brief Applied pooling operation on the matrix
         * @param p_side The dimensions of the p_side x p_side pooling window
         * @param stride How fast should we slide the pooling window accross the matrix
         * @param type The type of pooling: MAX_POOLING | AVERAGE_POOLING
         */
        virtual void pool(std::size_t p_side, std::size_t stride=2, PoolingType type=MAX_POOLING) = 0;

};

} // namespace cnn
