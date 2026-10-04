#pragma once

#include <Matrix.hpp>

/**
 * @brief Square Matrix to store data for black and white image
 * @author Dang Nguyen
 * @date 10/4/2026
 */
class SquareMatrix2D : SquareMatrix {
    public:
        SquareMatrix2D(std::size_t side, double value=0.0, std::size_t max_threads=MAX_THREADS);
        SquareMatrix2D(std::size_t side, const std::vector<double>& data, std::size_t max_threads=MAX_THREADS);

        void conv(const std::vector<std::vector<double>>& kernel, std::size_t stride=1) override;
        void activate(ActivationType type=RELU) override;
};