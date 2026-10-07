#include <cnn/layer/Convolution.hpp>

namespace cnn {
ConvolutionLayer::ConvolutionLayer(std::size_t in_channels=1, std::size_t out_channels=1,
              const std::string& name=utils::random_str(),
              std::size_t stride=1, std::size_t padding=0, std::size_t in_data_side,
              const std::vector<double>& kernels, const std::vector<double>& biases)
: cnn::OperationLayer(in_channels, out_channels, name, 
    stride, padding, in_data_side, (in_data_side - kernel_side + 2 * padding) / stride + 1),
  num_filters(out_channels),
  kernel_side(kernel_side), kernels(kernels), biases(biases) {
    if (num_filters <= 0)
        throw std::invalid_argument("There must be at least one filter");
    if (kernel_side <= 0)
        throw std::invalid_argument("The kernel side cannot be zero");
    if (kernel_side * kernel_side * num_filters * in_channels != kernels.size())
        throw std::invalid_argument("Mismatch kernels size");
    if (biases.size() != out_data_side * out_data_side * out_channels)
        throw std::invalid_argument("Mismatch biases size");

    layer_type = cnn::CONV;
}

void ConvolutionLayer::forward(const std::vector<double>& data) {
    if (in_data_side * in_data_side * in_channels != data.size())
        throw std::invalid_argument("Mismatch data size");

    in_data = data;
    std::fill(out_data.begin(), out_data.end(), 0.0);

    convolution();
}

void ConvolutionLayer::convolution() {
    const std::size_t kernel_middle = kernel_side / 2;
    const std::size_t out_pixels = out_data_side * out_data_side;
    const int shifting = static_cast<int>(kernel_middle) - static_cast<int>(padding);

    #pragma omp for schedule(static)
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

                    for (std::size_t in_ch = 0; in_ch < in_channels; ++in_ch)
                        out_data[
                            (out_row * out_data_side + out_col) * out_channels + f
                        ] +=
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

    }    
}


}