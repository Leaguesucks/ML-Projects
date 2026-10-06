#include <cnn/layer/Pool.hpp>

namespace cnn {
PoolLayer::PoolLayer(std::size_t in_channels=1, std::size_t stride=2, 
                     std::size_t padding=0,
                     std::size_t window_side=2, PoolingType pooling_type=cnn::MAX_POOLING)
: cnn::Layer(in_channels, stride, padding), window_side(window_side),
  pooling_type(pooling_type) {
    if (window_side <= 0)
        throw std::invalid_argument("The window slide cannot be zero");

    out_channels = in_channels;
}

void PoolLayer::forward(std::size_t side, const std::vector<double>& data) {
    if (side <= 0)
        throw std::invalid_argument("The data side cannot be zero");
    if (side * side * in_channels != data.size())
        throw std::invalid_argument("Data size mismatch");
    
    in_data_side = side;
    in_data = data;

    out_data_side = (side - window_side + 2 * padding) / stride + 1;
    out_data.assign(out_data_side * out_data_side * out_channels, 0.0);

    pool();
}

void PoolLayer::pool() {
    const std::size_t out_pixels = out_data_side * out_data_side * out_channels;

    switch (pooling_type) {
        case cnn::MAX_POOLING:
            // Flatten for maximum concurrency
            #pragma omp for schedule (static)
            for (std::size_t out_pixel = 0; out_pixel < out_pixels; ++out_pixel) {
                const std::size_t cur_out_mat = out_pixel / out_channels;
                const std::size_t cur_out_row = cur_out_mat / out_data_side;
                const std::size_t cur_out_col = cur_out_mat % out_data_side;
                const std::size_t cur_ch = out_pixel % out_channels;

                const int cur_row = static_cast<int>(cur_out_row * stride) - static_cast<int>(padding);
                const int cur_col = static_cast<int>(cur_out_col * stride) - static_cast<int>(padding);

                double max_val = std::numeric_limits<double>::lowest();
                bool found_value = false;
                for (int row = cur_row; row < cur_row + static_cast<int>(window_side); ++row)
                    for (int col = cur_col; col < cur_col + static_cast<int>(window_side); ++col) {
                        if (row < 0 || row >= static_cast<int>(in_data_side) ||
                            col < 0 || col >= static_cast<int>(in_data_side)) // Behave like -inf for max pooling
                                continue;

                        if (in_data[
                            (row * static_cast<int>(in_data_side) + col) 
                            * static_cast<int>(in_channels) 
                            + static_cast<int>(cur_ch)
                        ] > max_val) {
                            found_value = true;

                            max_val = in_data[
                                (row * static_cast<int>(in_data_side) + col) 
                                * static_cast<int>(in_channels) 
                                + static_cast<int>(cur_ch)
                            ];
                        }
                    }
                            
                out_data[out_pixel] = (found_value) ? max_val : 0.0;
            }

            break;

        case cnn::AVERAGE_POOLING:
            // Flatten for maximum concurrency
            #pragma omp for schedule (static)
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

                        local_sum += in_data[
                            (
                                static_cast<std::size_t>(row) * in_data_side +
                                static_cast<std::size_t>(col)
                            )
                            * in_channels
                            + cur_ch
                        ];
                    }
                            
                out_data[out_pixel] = local_sum / (window_side * window_side);
            }

            break;

        default:
            throw std::runtime_error("Unsupported pooling type");
    }
}

}