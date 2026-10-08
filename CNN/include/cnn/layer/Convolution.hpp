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

        bool activate; // Used to set up for residue operation
        dnn::Activation_Type activation_type;

        std::vector<double> kernel_gradients;
        std::vector<double> biases_gradients;

        std::vector<double> kernel_mts;
        std::vector<double> kernel_vts;

        std::vector<double> bias_mts;
        std::vector<double> bias_vts;

        std::vector<double> zs;

        std::vector<double> deltas;
        std::vector<double> dldaas;

    public:
        ConvolutionLayer(std::size_t in_channels=1, std::size_t out_channels=1,
              std::size_t stride=1, std::size_t padding=0, 
              bool activate=true, dnn::Activation_Type activation_type=dnn::RELU,
              std::size_t in_data_side, 
              const std::vector<double>& kernels, const std::vector<double>& biases);

        void forward(const std::vector<double>& data) override;

        std::size_t get_num_filter() {return num_filters;}
        std::size_t get_kernel_side() {return kernel_side;}
        std::vector<double>& get_kernels() {return kernels;}

    private:
        /**
         * @brief Perform convolution operation on the input data
         */
        void convolution();
};
}