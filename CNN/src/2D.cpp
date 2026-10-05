#include <cnn/network/2D.hpp>

#include <algorithm>
#include <utility>

namespace cnn {

Network2D::Network2D(std::vector<std::vector<ConvolutionLayer>> conv_layers,
                     std::vector<dnn::Network> densed_layers,
                     std::size_t max_threads)
                     : Network(conv_layers, densed_layers, max_threads) {

    if (conv_layers.size() != 1)
        throw std::invalid_argument("The conv_layers size must be 1 for 2D images");
    if (conv_layers[0].empty())
        throw std::invalid_argument("The convolution layer chain must not be empty");
    if (conv_layers[0].back().type != DENSE)
        throw std::invalid_argument("The last layer of conv_layers must be a DENSE layer for 2D images");
    if (densed_layers.size() != 1)
        throw std::invalid_argument("The densed layers size must be 1 for 2D images");

    dimension = D2;
}

void Network2D::forward(std::size_t side, const std::vector<double>& inputs) {
    if (inputs.empty() || inputs.size() != side * side)
        throw std::invalid_argument("The input size must be equal to side * side");

    this->side = side;

    if (this->data.empty())
        this->data = std::vector(inputs);
    else {
        this->data.clear();
        this->data.resize(inputs.size());
        std::copy(inputs.begin(), inputs.end(), this->data.begin());
    }

    for (const auto& conv_layer : conv_layers[0])
        switch (conv_layer.type) {
            case cnn::CONV:
                conv(conv_layer.kernel, conv_layer.stride);
                break;

            case cnn::ACTIVATE:
                activate(conv_layer.activation_type);
                break;

            case cnn::POOL:
                pool(conv_layer.p_side, conv_layer.stride, conv_layer.pooling_type);
                break;

            case cnn::DENSE:
                densed_layers[0].forward_propagation(data);
                data = densed_layers[0].get_Y_HAT();
                break;

            default:
                throw std::invalid_argument("Unsupported layer type");
                break;
        }
}

void Network2D::back_propagation(const std::vector<std::vector<double>>& Ys) {
    if (Ys.size() != densed_layers.size())
        throw std::invalid_argument("The size of the expected outputs must equal the size of the densed layers");

    densed_layers[0].back_propagation(Ys[0]);
}

void Network2D::conv(const std::vector<std::vector<double>>& kernel, std::size_t stride) {
    if (kernel.empty() || kernel.size() > side || kernel.size() % 2 == 0)
        throw std::invalid_argument("The kernel dimension must be odd, not empty "
            "and smaller than the matrix side");

    if (stride <= 0 || stride >= side)
        throw std::invalid_argument("The stride must be greater than zero and smaller than the matrix side");

    std::size_t k_middle = kernel.size() / 2;

    for (const auto& row : kernel)
            if (row.size() != kernel.size())
                throw std::invalid_argument("The kernel must be a square matrix");

    const std::size_t output_side = (side - kernel.size() + 2 * k_middle) / stride + 1;
    const std::size_t num_pixels = output_side * output_side;
    std::vector<double> new_data(num_pixels, 0.0);

    #pragma omp parallel for num_threads(max_threads) schedule(static)
    for (std::ptrdiff_t pixel = 0; pixel < static_cast<std::ptrdiff_t>(num_pixels); ++pixel) {
        const std::size_t out_row = static_cast<std::size_t>(pixel) / output_side;
        const std::size_t out_col = static_cast<std::size_t>(pixel) % output_side;
        const int cur_row = static_cast<int>(out_row * stride);
        const int cur_col = static_cast<int>(out_col * stride);

        double sum = 0.0;
        for (int row = cur_row - static_cast<int>(k_middle); row <= cur_row + static_cast<int>(k_middle); ++row)
            for (int col = cur_col - static_cast<int>(k_middle); col <= cur_col + static_cast<int>(k_middle); ++col) {
                if (row < 0 || row >= static_cast<int>(side) || col < 0 || col >= static_cast<int>(side))
                    continue;

                const std::size_t kernel_row = static_cast<std::size_t>(row - cur_row + static_cast<int>(k_middle));
                const std::size_t kernel_col = static_cast<std::size_t>(col - cur_col + static_cast<int>(k_middle));

                sum += data[static_cast<std::size_t>(row) * side + static_cast<std::size_t>(col)] * kernel[kernel_row][kernel_col];
            }

        new_data[static_cast<std::size_t>(pixel)] = sum;
    }

    data = std::move(new_data);
    side = output_side;
}

void Network2D::activate(dnn::Activation_Type type) {
    switch (type) {
        case dnn::RELU:
            #pragma omp parallel for num_threads(max_threads) schedule(static)
            for (std::size_t i = 0; i < data.size(); ++i)
                data[i] = std::max(0.0, data[i]);
            break;

        default:
            throw std::invalid_argument("Unsupported activation function");
            break;
    }
}

void Network2D::pool(std::size_t p_side, std::size_t stride, PoolingType type) {
    if (stride <= 0 || stride >= side)
        throw std::invalid_argument("The stride must be greater than zero and smaller than the matrix side");

    if (p_side <= 0 || p_side > side)
        throw std::invalid_argument("Pooling window must be greater than zero and no larger than the matrix side");

    std::size_t output_side = (side - p_side) / stride + 1;
    std::vector<double> new_data(output_side * output_side, 0.0);

    switch (type) {
        case cnn::AVERAGE_POOLING:

            #pragma omp parallel for num_threads(max_threads) schedule(static)
            for (std::size_t pixel = 0; pixel < output_side * output_side; ++pixel) {
                const std::size_t out_row = pixel / output_side;
                const std::size_t out_col = pixel % output_side;
                const std::size_t cur_row = out_row * stride;
                const std::size_t cur_col = out_col * stride;

                for (std::size_t i = 0; i < p_side; ++i)
                    for (std::size_t j = 0; j < p_side; ++j)
                        new_data[pixel] += data[(cur_row + i) * side + (cur_col + j)];

                new_data[pixel] /= static_cast<double>(p_side * p_side);
            }

            break;

        case cnn::MAX_POOLING:

            #pragma omp parallel for num_threads(max_threads) schedule(static)
            for (std::size_t pixel = 0; pixel < output_side * output_side; ++pixel) {
                const std::size_t out_row = pixel / output_side;
                const std::size_t out_col = pixel % output_side;
                const std::size_t cur_row = out_row * stride;
                const std::size_t cur_col = out_col * stride;

                new_data[pixel] = data[cur_row * side + cur_col];
                for (std::size_t i = 0; i < p_side; ++i)
                    for (std::size_t j = 0; j < p_side; ++j)
                        if (data[(cur_row + i) * side + (cur_col + j)] > new_data[pixel])
                            new_data[pixel] = data[(cur_row + i) * side + (cur_col + j)];
            }

            break;

        default:
            throw std::invalid_argument("Unsupported pooling type");
    }

    side = output_side;
    data = std::move(new_data);
}

} // namespace cnn
