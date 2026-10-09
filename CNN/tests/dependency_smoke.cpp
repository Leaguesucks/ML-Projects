#include <cnn/network/Network.hpp>
#include <cnn/network/2D.hpp>
#include <cnn/layer/Layer.hpp>
#include <cnn/layer/Layer.hpp>
#include <dnn/Adam.h>
#include <dnn/MNIST.h>
#include <dnn/Network.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

static_assert(!std::is_same_v<dnn::Network, cnn::Network>);
static_assert(std::is_convertible_v<cnn::Network2D*, cnn::Network*>);
static_assert(!std::is_abstract_v<cnn::Network2D>);

cnn::ConvolutionLayer dense_layer() {
    cnn::ConvolutionLayer layer{};
    layer.type = cnn::DENSE;
    return layer;
}

dnn::Network dense_network(const std::vector<double>& weights) {
    return dnn::Network({dnn::OperationLayer(1, weights, dnn::RELU)}, dnn::SSE);
}

// Expose individual operations for the original matrix regression values.
class TestMatrix : public cnn::Network2D {
public:
    TestMatrix(std::size_t input_side, const std::vector<double>& inputs,
               std::size_t threads)
        : Network2D({{dense_layer()}},
                    {dense_network(std::vector<double>(inputs.size(), 1.0))}, threads) {
        side = input_side;
        data = inputs;
    }

    using cnn::Network2D::conv;
    using cnn::Network2D::n_residue;
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void require_values(const std::vector<double>& actual,
                    const std::vector<double>& expected, const char* message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t i = 0; i < actual.size(); ++i)
        require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <= 1e-12,
                message);
}

void check_matrix_operations() {
    const std::vector<double> inputs{1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<std::vector<double>> kernel(3, std::vector<double>(3, 1.0));
    TestMatrix parallel(3, inputs, 2);
    parallel.conv(kernel);
    require(parallel.get_dimension() == cnn::D2 && parallel.get_side() == 3,
            "Convolution dimensions changed");
    require_values(parallel.get_data(), {12, 21, 16, 27, 45, 33, 24, 39, 28},
                   "Convolution output changed");

    TestMatrix serial(3, inputs, 1);
    serial.conv(kernel);
    require_values(serial.get_data(), parallel.get_data(),
                   "Serial and parallel convolution outputs differ");

    TestMatrix strided(3, inputs, 2);
    strided.conv(kernel, 2);
    require(strided.get_side() == 2, "Strided convolution dimensions changed");
    require_values(strided.get_data(), {12, 16, 24, 28},
                   "Strided convolution output changed");

    TestMatrix activated(3, {-1, 2, -3, 4, 5, -6, 7, -8, 9}, 2);
    activated.n_residue(dnn::RELU);
    require_values(activated.get_data(), {0, 2, 0, 4, 5, 0, 7, 0, 9},
                   "ReLU output changed");
}

cnn::Network2D pipeline(cnn::PoolingType pooling_type, std::size_t threads) {
    cnn::ConvolutionLayer convolution{};
    convolution.type = cnn::CONV;
    convolution.kernel = {{0, 0, 0}, {0, -1, 0}, {0, 0, 0}};
    convolution.stride = 1;

    cnn::ConvolutionLayer activation{};
    activation.type = cnn::ACTIVATE;
    activation.activation_type = dnn::RELU;

    cnn::ConvolutionLayer pooling{};
    pooling.type = cnn::POOL;
    pooling.pooling_type = pooling_type;
    pooling.p_side = 2;
    pooling.stride = 2;

    return cnn::Network2D({{convolution, activation, pooling, dense_layer()}},
                          {dense_network({1, 2, 3, 4})}, threads);
}

void check_forward_pipeline(cnn::PoolingType pooling_type, double expected_output) {
    const std::vector<double> inputs{
        -1, 2, -3, 4, 5, -6, 7, -8, 9, 10, -11, 12, -13, 14, 15, -16};
    auto serial = pipeline(pooling_type, 1);
    auto parallel = pipeline(pooling_type, 2);
    serial.forward(4, inputs);
    parallel.forward(4, inputs);
    require(parallel.get_dimension() == cnn::D2 && parallel.get_side() == 2,
            "Forward propagation dimensions changed");
    require_values(parallel.get_data(), {expected_output},
                   "Convolution, activation, pooling and dense pipeline output changed");
    require_values(serial.get_data(), parallel.get_data(),
                   "Serial and parallel forward outputs differ");

    auto& dense = parallel.get_densed_layers()[0];
    require_values(dense.get_Y_HAT(), parallel.get_data(),
                   "Forward propagation did not preserve the shared DNN output");

    dnn::Adam optimizer(dense);
    optimizer.zero_gradients();
    parallel.back_propagation({{expected_output - 5.0}});
    const std::vector<double> pooled = pooling_type == cnn::MAX_POOLING
        ? std::vector<double>{6, 8, 13, 16}
        : std::vector<double>{1.75, 2.75, 3.25, 6.75};
    std::vector<double> expected_gradients;
    for (double value : pooled)
        expected_gradients.push_back(10.0 * value);
    require_values(dense.get_layers()[0].get_gradients(), expected_gradients,
                   "CNN backpropagation did not reach the shared DNN weights");
    require_values(dense.get_layers()[0].get_biases_gradients(), {10},
                   "CNN backpropagation did not reach the shared DNN biases");

    parallel.forward(4, inputs);
    require_values(parallel.get_data(), {expected_output},
                   "Repeated forward propagation did not restore the input matrix");
}

void check_empty_chain_rejected() {
    try {
        cnn::Network2D invalid({{}}, {dense_network({1})}, 1);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("An empty convolution chain was accepted");
}

int main() {
    try {
        check_matrix_operations();
        check_forward_pipeline(cnn::MAX_POOLING, 125.0);
        check_forward_pipeline(cnn::AVERAGE_POOLING, 44.0);
        check_empty_chain_rejected();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    std::cout << "CNN operations, forward pipeline and shared DNN dependency passed\n";
}
