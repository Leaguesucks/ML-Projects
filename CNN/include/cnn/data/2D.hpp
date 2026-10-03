#pragma once

#include <Matrix.hpp>

class SquareMatrix2D : SquareMatrix {
    public:
        /**
         * @brief Construct a 2D square matrix filled with the default value
         * @param side Construct a side x side matrix
         * @param value The default value of the matrix
         */
        SquareMatrix2D(std::size_t dim, double value=0.0);

        /**
         * @brief Construct a 2D square matrix using a flat array
         * @param side Construct a side x side matrix
         * @param data The data of the square matrix flattened
         */
        SquareMatrix2D(std::size_t side, const std::vector<double>& data);

        void conv(const std::vector<std::vector<double>>& kernel, std::size_t stride=1) override;
};