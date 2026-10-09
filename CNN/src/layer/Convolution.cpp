#include <cnn/layer/Convolution.hpp>

namespace cnn {
ConvolutionLayer::ConvolutionLayer(std::size_t in_channels=1, std::size_t out_channels=1,
              std::size_t stride=1, std::size_t padding=0, 
              std::size_t n_residue=0, std::size_t n_back_residue=0,
              dnn::Activation_Type activation_type=dnn::RELU,
              std::size_t in_data_side,
              const std::vector<double>& kernels, const std::vector<double>& biases)
: cnn::OperationLayer(in_channels, out_channels, 
    stride, padding, in_data_side, (in_data_side - kernel_side + 2 * padding) / stride + 1),
  num_filters(out_channels), n_residue(n_residue), n_back_residue(n_back_residue),
  activation_type(activation_type),
  kernel_side(kernel_side), kernels(kernels), biases(biases) {
    if (num_filters <= 0)
        throw std::invalid_argument("There must be at least one filter");
    if (kernel_side <= 0)
        throw std::invalid_argument("The kernel side cannot be zero");
    if (kernel_side * kernel_side * num_filters * in_channels != kernels.size())
        throw std::invalid_argument("Mismatch kernels size");
    if (biases.size() != out_data_side * out_data_side * out_channels)
        throw std::invalid_argument("Mismatch biases size");

    activated_out_data.assign(out_data_side * out_data_side * out_channels, 0.0);

    layer_type = cnn::CONV;
}

const std::vector<double>& ConvolutionLayer::forward(const std::vector<double>& data) {
    if (in_data_side * in_data_side * in_channels != data.size())
        throw std::invalid_argument("Mismatch data size");

    in_data = data;
    std::fill(out_data.begin(), out_data.end(), 0.0);
    std::fill(activated_out_data.begin(), activated_out_data.end(), 0.0);

    const std::size_t kernel_middle = kernel_side / 2;
    const std::size_t out_pixels = out_data_side * out_data_side;
    const int shifting = static_cast<int>(kernel_middle) - static_cast<int>(padding);

    #pragma omp parallel for schedule(static)
    for (std::size_t f = 0; f < num_filters; ++f) {
        for (std::size_t pixel = 0; pixel < out_pixels; ++pixel) {
            out_data[pixel * out_channels + f] = biases[pixel * out_channels + f];

            const std::size_t out_row = pixel / out_data_side;
            const std::size_t out_col = pixel % out_data_side;

            int center_row = static_cast<int>(out_row * stride) + shifting;
            int center_col = static_cast<int>(out_col * stride) + shifting;

            for (int row = center_row - static_cast<int>(kernel_middle); row <= center_row + static_cast<int>(kernel_middle); ++row)
                for (int col = center_col - static_cast<int>(kernel_middle); col <= center_col + static_cast<int>(kernel_middle); ++col) {
                    std::size_t kernel_row = static_cast<std::size_t>(
                        row - center_row + static_cast<int>(kernel_middle)
                    );

                    std::size_t kernel_col = static_cast<std::size_t>(
                        col - center_col + static_cast<int>(kernel_middle)
                    );

                    // Zero padding
                    if (row < 0 || row >= static_cast<int>(in_data_side) || col < 0 || col >= static_cast<int>(in_data_side))
                        continue;

                    for (std::size_t in_ch = 0; in_ch < in_channels; ++in_ch) {
                        out_data[pixel * out_channels + f] +=
                        in_data[
                            (
                                static_cast<std::size_t>(row) * in_data_side + static_cast<std::size_t>(col)
                            ) 
                            * in_channels
                            + in_ch
                        ] *
                        kernels[
                            (
                                (kernel_row * kernel_side + kernel_col) * num_filters + f
                            )
                            * in_channels
                            + in_ch
                        ];
                    }
                }

            activated_out_data[pixel * out_channels + f] = out_data[pixel * out_channels + f];
            for (std::size_t r = 0; r < n_residue; ++r)
                activated_out_data[pixel * out_channels + f] += residues[(pixel * out_channels + f) * n_residue + r];

            switch (activation_type) {
                case dnn::RELU:
                    activated_out_data[pixel * out_channels + f] = std::max(0.0, activated_out_data[pixel * out_channels + f]);
                    break;

                default:
                    throw std::runtime_error("Unsupported activation type");
            }
        }
    }

    return activated_out_data;
}

const std::vector<double>& ConvolutionLayer::forward(const std::vector<double>& data, const std::vector<double>& residues) {
    if (!n_residue)
        throw std::invalid_argument("Wrong function overload: This layer accepts no residue");
    if (n_residue * out_data_side * out_data_side * out_channels != residues.size())
        throw std::invalid_argument("Residues size mistmatch");

    this->residues = residues;
    return forward(data);
}

void ConvolutionLayer::backward(cnn::Layer& next_layer, std::vector<cnn::Layer*>& residue_layers) {
    if (n_back_residue != residue_layers.size())
        throw std::invalid_argument("Mismatch residue layers size");

    kernel_gradients.assign(kernel_side * kernel_side * num_filters * in_channels, 0.0);
    bias_gradients.assign(out_data_side * out_data_side * out_channels, 0.0);
    deltas.assign(out_data_side * out_data_side * out_channels, 0.0);

    // Calculate deltas for this layer
    for (std::size_t next = 0; next < n_back_residue; ++next) {
        switch (residue_layers[next]->get_layer_type()) {
            case cnn::CONV:
                ConvolutionLayer* next_layer = static_cast<ConvolutionLayer*>(residue_layers[next]);
                const std::size_t nK = next_layer->get_kernel_side();
                const std::size_t nS = next_layer->get_kernel_side();
                const std::size_t nP = next_layer->get_padding();
                const std::size_t nG = next_layer->get_num_filter();
                const std::size_t nO = next_layer->get_out_data_side();

                for (std::size_t A = 0; A < out_data_side * out_data_side * out_channels; ++A) {
                    std::size_t f = A % out_channels;
                    std::size_t x = (A / out_channels) / out_data_side;
                    std::size_t y = (A / out_channels) % out_data_side;

                    for (std::size_t each_next_delta = 0; each_next_delta < nO * nO * nG; ++each_next_delta) {
                        std::size_t g = each_next_delta % nP;
                        std::size_t u = (each_next_delta / nP) / nO;
                        std::size_t v = (each_next_delta / nP) % nO;

                        
                    }
                }

                break;
        }
    }
}

}