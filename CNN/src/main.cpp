#include <cnn/network/2D.hpp>

#include <iostream>
#include <vector>

int main() {
    dnn::Network densed_network(
        std::vector<dnn::Layer>{dnn::Layer(1, std::vector<double>(4, 1.0), dnn::RELU)},
        dnn::SSE
    );

    cnn::ConvolutionLayer convolution{};
    convolution.type = cnn::CONV;
    convolution.kernel = std::vector<std::vector<double>>(3, std::vector<double>(3, 1.0));
    convolution.stride = 1;

    cnn::ConvolutionLayer activation{};
    activation.type = cnn::ACTIVATE;
    activation.activation_type = dnn::RELU;

    cnn::ConvolutionLayer pooling{};
    pooling.type = cnn::POOL;
    pooling.pooling_type = cnn::MAX_POOLING;
    pooling.p_side = 2;
    pooling.stride = 1;

    cnn::ConvolutionLayer dense{};
    dense.type = cnn::DENSE;

    cnn::Network2D network({{convolution, activation, pooling, dense}}, {densed_network}, 1);
    network.forward(3, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    std::cout << "CNN output: " << network.get_data().front() << '\n';
}
