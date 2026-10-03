#pragma once

#include <thread>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <cmath>

#include <omp.h>

#include <dnn/Layer.hpp>

#define MAX_THREADS std::thread::hardware_concurrency()

enum Channel {
    R,
    G,
    B
};

enum Dimension {
    D2,
    D3
};

enum PoolingType {
    MAX_POOLING,
    AVERAGE_POOLING
};

/**
 * @brief Template class for a square matrix
 * @author Dang Nguyen
 * @date 9/30/2026
 */
class SquareMatrix {
    protected:
        Dimension dimension;

        std::size_t side;
        std::vector<double> data;

    public:
        /**
         * @brief Construct a square matrix filled with the default value
         * @param size Construct a size x size matrix
         * @param value The default value of the matrix
         */
        SquareMatrix(std::size_t size, double value=0.0);

        /**
         * @brief Construct a square matrix from an existed data
         * @param size Construct a size x size matrix
         * @param data The data to initialize the matrix
         */
        SquareMatrix(std::size_t size, const std::vector<double>& data);

        /**
         * @brief Convolute this matrix using a kernel
         * @param kernel The kernel to applied convolution to the current matrix, Must be squared
         * @param stride How fast should we slide the kernel accross the matrix
         */
        virtual void conv(const std::vector<std::vector<double>>& kernel, std::size_t stride=1) = 0;

        /**
         * @brief Filter the matrix through an activation fucntion
         * @param type The type of activation function to applied
         */
        virtual void activate(ActivationType type=RELU) = 0;

        /**
         * @brief Applied pooling operation on the matrix
         * @param dim_p The dimensions of the pooling window
         * @param stride How fast should we slide the pooling window accross the matrix
         * @param type The type of pooling: MAX_POOLING | AVERAGE_POOLING
         */
        virtual void pool(const std::vector<std::size_t>& dim_p, int stride=2, PoolingType type=MAX_POOLING) = 0;
        
        Dimension get_dimension() {return dimension;}
        std::size_t get_size() {return side;}
        std::vector<double>& get_data() {return data;}
};