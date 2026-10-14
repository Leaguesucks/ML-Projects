#include <cnn/layer/Convolution.hpp>

namespace cnn {
ConvolutionLayer::ConvolutionLayer(std::size_t in_channels=1, std::size_t out_channels=1,
              std::size_t stride=1, std::size_t padding=0, 
              dnn::Activation_Type activation_type=dnn::RELU,
              std::size_t in_data_side,
              const std::vector<double>& kernels, const std::vector<double>& biases)
: cnn::OperationLayer(in_channels, out_channels, 
    stride, padding, in_data_side, (in_data_side - kernel_side + 2 * padding) / stride + 1),
  num_filters(out_channels), activation_type(activation_type),
  kernel_side(kernel_side), kernels(kernels), biases(biases) {
    if (kernel_side <= 0)
        throw std::invalid_argument("The kernel side cannot be zero");
    if (kernel_side * kernel_side * num_filters * in_channels != kernels.size())
        throw std::invalid_argument("Mismatch kernels size");
    if (biases.size() != out_channels)
        throw std::invalid_argument("Mismatch biases size");

    activated_out_data.assign(out_data_side * out_data_side * out_channels, 0.0);
    pre_activated_out_data.assign(out_data_side * out_data_side * out_channels, 0.0);

    kernel_gradients.assign(kernel_side * kernel_side * num_filters * in_channels, 0.0);
    bias_gradients.assign(out_channels, 0.0);
    deltas.assign(out_data_side * out_data_side * out_channels, 0.0);

    layer_type = cnn::CONV;
}

const std::vector<double>& ConvolutionLayer::forward(const std::vector<double>& data) {
    if (in_data_side * in_data_side * in_channels != data.size())
        throw std::invalid_argument("Mismatch data size");

    // Pre-cleanup to prepare for back propagation
    std::fill(kernel_gradients.begin(), kernel_gradients.end(), 0.0);
    std::fill(bias_gradients.begin(), bias_gradients.end(), 0.0);
    std::fill(deltas.begin(), deltas.end(), 0.0);

    in_data = data;
    std::fill(out_data.begin(), out_data.end(), 0.0);
    std::fill(activated_out_data.begin(), activated_out_data.end(), 0.0);
    std::fill(pre_activated_out_data.begin(), pre_activated_out_data.end(), 0.0);

    const std::size_t kernel_middle = kernel_side / 2;
    const std::size_t out_pixels = out_data_side * out_data_side;
    const int shifting = static_cast<int>(kernel_middle) - static_cast<int>(padding);

    #pragma omp parallel for schedule(static)
    for (std::size_t f = 0; f < num_filters; ++f) {
        for (std::size_t pixel = 0; pixel < out_pixels; ++pixel) {
            out_data[pixel * out_channels + f] = biases[f];

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

            pre_activated_out_data[pixel * out_channels + f] = out_data[pixel * out_channels + f];
            for (std::size_t r = 0; r < from_residues.size(); ++r) {
                if (from_residues[r]->get_final_out_data().size() != out_data_side * out_data_side * out_channels)
                    throw std::runtime_error("Mismatch residue outputs size");

                pre_activated_out_data[pixel * out_channels + f] += from_residues[r]->get_final_out_data()[pixel * out_channels + f];
            }

            switch (activation_type) {
                case dnn::RELU:
                    activated_out_data[pixel * out_channels + f] = std::max(0.0, pre_activated_out_data[pixel * out_channels + f]);
                    break;

                default:
                    throw std::runtime_error("Unsupported activation type");
            }
        }
    }

    return activated_out_data;
}

void ConvolutionLayer::backward() {
    if (next_layer == nullptr)
        throw std::runtime_error("Final convolution layer must be handle outside of backward()");
    
    auto& dldas = next_layer->get_prev_dldas();

    // Add the main-path contribution to this layer's delta
    switch (activation_type) {
        case dnn::RELU:

            #pragma omp parallel for schedule(static)
            for (std::size_t d = 0; d < out_data_side * out_data_side * out_channels; ++d)
                deltas[d] += (pre_activated_out_data[d] > 0.0) ? dldas[d] : 0.0;
            
            break;

        default:
            throw std::runtime_error("Unsupported activation function");
    }

    // Propagate the complete delta to the residue sources
    backward_residue_deltas();

    calculate_prev_dldas();
    calculate_gradients();
    calculate_bias_gradients();
}

void ConvolutionLayer::calculate_prev_dldas() {
    if (is_first_layer) // Save calculation effort
        return;

    std::fill(prev_dldas.begin(), prev_dldas.end(), 0.0);
    
    const auto S = static_cast<std::ptrdiff_t>(stride);
    const auto P = static_cast<std::ptrdiff_t>(padding);
    const auto H = static_cast<std::ptrdiff_t>(out_data_side);

    #pragma omp parallel for schedule(static)
    for (std::size_t A = 0; A < in_data_side * in_data_side * in_channels; ++A) {
        const std::size_t f = A % in_channels;
        const auto x = static_cast<std::ptrdiff_t>((A / in_channels) / in_data_side);
        const auto y = static_cast<std::ptrdiff_t>((A / in_channels) % in_data_side);

        double sum = 0.0;
        for (std::size_t i = 0; i < kernel_side; ++i) {
            const auto numerator_u = x + P - static_cast<std::ptrdiff_t>(i);
            
            if (numerator_u < 0 || numerator_u % S != 0)
                continue;

            const auto u = numerator_u / S;

            if (u >= H)
                continue;

            for (std::size_t j = 0; j < kernel_side; ++j) {
                const auto numerator_v = y + P - static_cast<std::ptrdiff_t>(j);

                if (numerator_v < 0 || numerator_v % S != 0)
                    continue;

                const auto v = numerator_v / S;

                if (v >= H)
                    continue;

                const std::size_t pixel = static_cast<std::size_t>(u) * out_data_side + static_cast<std::size_t>(v);

                for (std::size_t g = 0; g < out_channels; ++g) {
                    const std::size_t D = pixel * out_channels + g;
                    const std::size_t W = ((i * kernel_side + j) * out_channels + g) * in_channels + f;
                    sum += deltas[D] * kernels[W];
                }
            }
        }

        prev_dldas[A] = sum;
    }
}

void ConvolutionLayer::calculate_gradients() {
    const std::size_t S = stride;
    const std::size_t P = padding;

    #pragma omp parallel for schedule(static)
    for (std::size_t W = 0; W < kernel_side * kernel_side * num_filters * in_channels; ++W) {
        const std::size_t c = W % in_channels;
        const std::size_t f = (W / in_channels) % num_filters;
        const std::size_t i = ((W / in_channels) / num_filters) / kernel_side;
        const std::size_t j = ((W / in_channels) / num_filters) % kernel_side;

        for (std::size_t r = 0; r < out_data_side; ++r)
            for (std::size_t s = 0; s < out_data_side; ++s) {
                const int in_row = static_cast<int>(r * S) + static_cast<int>(i) - static_cast<int>(P);
                const int in_col = static_cast<int>(s * S) + static_cast<int>(j) - static_cast<int>(P);

                if (in_row < 0 || in_row >= in_data_side || in_col < 0 || in_col >= in_data_side)
                    continue;

                kernel_gradients[W] += deltas[(r * out_data_side + s) * out_channels + f]
                * in_data[((r * S + i - P) * in_data_side + (s * S + j - P)) * in_channels + c];
            }             
    }
}

void ConvolutionLayer::calculate_bias_gradients() {
    const std::size_t out_pixels = out_data_side * out_data_side;

    #pragma omp parallel for schedule(static)
    for (std::size_t f = 0; f < out_channels; ++f) {
        double sum = 0.0;

        for (std::size_t pixel = 0; pixel < out_pixels; ++pixel) {
            const std::size_t d = pixel * out_channels + f;
            sum += deltas[d];
        }

        bias_gradients[f] += sum;
    }
}

}