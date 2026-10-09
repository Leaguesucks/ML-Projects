#pragma once

#include <stdexcept>
#include <cstddef>
#include <vector>
#include <cmath>
#include <omp.h>

#include <dnn/Layer.h>
#include <cnn/layer/Layer.hpp>

namespace cnn {
/**
 * @brief The convolution layer. Produce n x n x f output where n is the input side and f is 
 *        the number of filter
 * @date 10/6/2026
 * @author Dang Nguyen
 */
class ConvolutionLayer : public cnn::OperationLayer {
    private:
        std::size_t num_filters; // = Number of out channels
        std::size_t kernel_side; // side x side filter
        std::vector<double> kernels; // side x side x num_filters x in_channels
        std::vector<double> biases; // out_data_side x out_data_side x out_channels biases

        std::size_t n_residue; // The number of residue of this layer i.e., how many layers is this layer connected to
        std::vector<double> residues; // out_data_side x out_data_side x out_channels x n_residue

        std::size_t n_back_residue; // For back propagation i.e., how many layers does this layer connect to
        
        std::vector<double> activated_out_data;
        dnn::Activation_Type activation_type;

        // Only convolution layer needs these parameters for back propagation and Adam optimizer
        std::vector<double> kernel_gradients;
        std::vector<double> bias_gradients;

        // Adam optimizer
        std::vector<double> kernel_mts;
        std::vector<double> kernel_vts;

        std::vector<double> bias_mts;
        std::vector<double> bias_vts;

    public:
        ConvolutionLayer(std::size_t in_channels=1, std::size_t out_channels=1,
              std::size_t stride=1, std::size_t padding=0, 
              std::size_t n_residue=0, std::size_t n_back_residue=0,
              dnn::Activation_Type activation_type=dnn::RELU,
              std::size_t in_data_side, 
              const std::vector<double>& kernels, const std::vector<double>& biases);

        const std::vector<double>& forward(const std::vector<double>& data) override;

        /**
         * @brief Unique overload to the convolution layer to accept residues
         * @param data The data, usually from the previous layer
         * @param residues THe residues from the previous layers
         */
        const std::vector<double>& forward(const std::vector<double>& data, const std::vector<double>& residues);

        void backward(cnn::Layer& next_layer, std::vector<cnn::Layer*>& residue_layers) override;

        void set_n_residue(std::size_t n_residue) {this->n_residue = n_residue;}
        std::size_t get_n_residue() {return n_residue;}

        void set_n_back_residue(std::size_t n_back_residue) {this->n_back_residue = n_back_residue;}
        std::size_t get_n_back_residue() {return n_back_residue;}

        std::size_t get_num_filter() {return num_filters;}
        std::size_t get_kernel_side() {return kernel_side;}

        std::vector<double>& get_kernels() {return kernels;}
        std::vector<double>& get_biases() {return biases;}
        std::vector<double>& get_kernel_gradients() {return kernel_gradients;}
        std::vector<double>& get_bias_gradients() {return bias_gradients;}

        std::vector<double>& get_kernel_mts() {return kernel_mts;}
        std::vector<double>& get_kernel_vts() {return kernel_vts;}

        std::vector<double>& get_bias_mts() {return bias_mts;}
        std::vector<double>& get_bias_vts() {return bias_vts;}
};
}