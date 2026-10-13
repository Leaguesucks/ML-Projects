#pragma once

#include <stdexcept>
#include <cstddef>
#include <vector>
#include <cmath>
#include <omp.h>

#include <dnn/Layer.h>
#include <cnn/layer/Layer.hpp>
#include <cnn/layer/Pool.hpp>

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
        std::vector<double> biases; // out_channels biases
        
        std::vector<double> activated_out_data;
        std::vector<double> pre_activated_out_data;
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
              dnn::Activation_Type activation_type=dnn::RELU,
              std::size_t in_data_side, 
              const std::vector<double>& kernels, const std::vector<double>& biases);

        const std::vector<double>& forward(const std::vector<double>& data) override;

        void backward() override;

        const std::vector<double>& get_final_out_data() override {return activated_out_data;}

        dnn::Activation_Type get_activation_type() {return activation_type;}

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

        const std::vector<double>& get_pre_activated_out_data() {return pre_activated_out_data;}

    protected:
        void calculate_prev_dldas() override;

    private:
        /**
         * @brief Calculate the gradients for this layer
         */
        void calculate_gradients();

        /**
         * @brief Calculate the bias gradients
         */
        void calculate_bias_gradients();
};
}