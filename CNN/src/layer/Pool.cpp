#include <cnn/layer/Pool.hpp>

namespace cnn {
PoolLayer::PoolLayer(std::size_t in_out_channels=1,
            std::size_t stride=1, std::size_t padding=0,
            std::size_t window_side=2, cnn::PoolingType pooling_type=cnn::MAX_POOLING,
            std::size_t in_data_side)
: cnn::OperationLayer(in_out_channels, in_out_channels, 
    stride, padding, in_data_side, (in_data_side - window_side + 2 * padding) / stride + 1), 
window_side(window_side), pooling_type(pooling_type) {
    if (window_side <= 0)
        throw std::invalid_argument("The window slide cannot be zero");
    layer_type = cnn::POOL;

    max_indices.assign(out_data_side * out_data_side * out_channels, 0.0);
}

const std::vector<double>& PoolLayer::forward(const std::vector<double>& data) {
    if (in_data_side * in_data_side * in_channels != data.size())
        throw std::invalid_argument("Mismatch data size");

    in_data = data;
    std::fill(out_data.begin(), out_data.end(), 0.0);
    std::fill(deltas.begin(), deltas.end(), 0.0);
    std::fill(max_indices.begin(), max_indices.end(), 0.0);

    contributors.clear();

    pool();
    return out_data;
}

void PoolLayer::backward() {
    if (next_layer == nullptr)
        throw std::runtime_error("Last convolutional layer must be handled explicitly");

    auto& dldas = next_layer->get_prev_dldas(); // == main deltas of the pooling layer
    if (out_data_side * out_data_side * out_channels != dldas.size())
        throw std::runtime_error("Mismatch dL/da size");
    
    // Add the main-path contribution to this layer deltas
    #pragma omp parallel for schedule(static)
    for (std::size_t d = 0; d < dldas.size(); ++d)
        deltas[d] += dldas[d];

    // Propagate back to each source residue layers
    backward_residue_deltas();
    calculate_prev_dldas();
}

void PoolLayer::calculate_prev_dldas() {
    std::fill(prev_dldas.begin(), prev_dldas.end(), 0.0);

    switch (pooling_type) {
        case cnn::MAX_POOLING:
            
            #pragma omp parallel for schedule(static)
            for (std::size_t o = 0; o < deltas.size(); ++o) {
                const std::size_t i = max_indices[o];

                if (i >= in_data_side * in_data_side * in_channels)
                    throw std::runtime_error("Invalid max index");

                #pragma omp atomic update
                prev_dldas[i] += deltas[o];
            }

            break;

        case cnn::AVERAGE_POOLING:
            const std::size_t K = window_side;
            const std::size_t N_INPUT = in_data_side * in_data_side * in_channels;
            const std::size_t N_OUTPUT = out_data_side * out_data_side * out_channels;

            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < N_INPUT; ++i) {
                const auto it = contributors.find(i);
                
                double cum_grad = 0.0;
                if (it != contributors.end())
                    for (const std::size_t o : it->second)
                        cum_grad += deltas[o];

                prev_dldas[i] = cum_grad / (K * K);
            }

            break;

        default:
            throw std::runtime_error("Unsupported pooling type");
    }
}

void PoolLayer::pool() {
    const std::size_t out_pixels = out_data_side * out_data_side * out_channels;

    switch (pooling_type) {
        case cnn::MAX_POOLING:
            // Flatten for maximum concurrency
            #pragma omp parallel for schedule(static)
            for (std::size_t out_pixel = 0; out_pixel < out_pixels; ++out_pixel) {
                const std::size_t cur_out_mat = out_pixel / out_channels;
                const std::size_t cur_out_row = cur_out_mat / out_data_side;
                const std::size_t cur_out_col = cur_out_mat % out_data_side;
                const std::size_t cur_ch = out_pixel % out_channels;

                const int cur_row = static_cast<int>(cur_out_row * stride) - static_cast<int>(padding);
                const int cur_col = static_cast<int>(cur_out_col * stride) - static_cast<int>(padding);

                double max_val = std::numeric_limits<double>::lowest();
                bool found_value = false;
                std::size_t max_in_coords = 0.0;
                for (int row = cur_row; row < cur_row + static_cast<int>(window_side); ++row)
                    for (int col = cur_col; col < cur_col + static_cast<int>(window_side); ++col) {
                        if (row < 0 || row >= static_cast<int>(in_data_side) ||
                            col < 0 || col >= static_cast<int>(in_data_side)) // Behave like -inf for max pooling
                                continue;

                        std::size_t in_coords = (static_cast<std::size_t>(row) * in_data_side + static_cast<std::size_t>(col))
                                                * in_channels + cur_ch;

                        if (in_data[in_coords] > max_val) {
                            found_value = true;

                            max_val = in_data[in_coords];
                            max_in_coords = in_coords;
                        }
                    }
                            
                if (found_value) {
                    out_data[out_pixel] = max_val;
                    max_indices[out_pixel] = max_in_coords;
                } else
                    out_data[out_pixel] = 0.0;
            }

            break;

        case cnn::AVERAGE_POOLING:
            // Flatten for maximum concurrency
            #pragma omp parallel for schedule (static)
            for (std::size_t out_pixel = 0; out_pixel < out_pixels; ++out_pixel) {
                const std::size_t cur_out_mat = out_pixel / out_channels;
                const std::size_t cur_out_row = cur_out_mat / out_data_side;
                const std::size_t cur_out_col = cur_out_mat % out_data_side;
                const std::size_t cur_ch = out_pixel % out_channels;

                const int cur_row = static_cast<int>(cur_out_row * stride) - static_cast<int>(padding);
                const int cur_col = static_cast<int>(cur_out_col * stride) - static_cast<int>(padding);

                double local_sum = 0.0;
                for (int row = cur_row; row < cur_row + static_cast<int>(window_side); ++row)
                    for (int col = cur_col; col < cur_col + static_cast<int>(window_side); ++col) {
                        if (row < 0 || row >= static_cast<int>(in_data_side) ||
                            col < 0 || col >= static_cast<int>(in_data_side))
                                continue;

                        std::size_t in_pixel = (
                            static_cast<std::size_t>(row) * in_data_side + static_cast<std::size_t>(col)
                        ) * in_channels + cur_ch;

                        contributors[in_pixel].push_back(out_pixel);
                        local_sum += in_data[in_pixel];
                    }
                            
                out_data[out_pixel] = local_sum / (window_side * window_side);
            }

            break;

        default:
            throw std::runtime_error("Unsupported pooling type");
    }
}

}