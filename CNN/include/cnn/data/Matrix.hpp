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

        std::size_t max_threads;
        std::size_t side;
        std::vector<double> data;

    public:
        /**
         * @brief Dummy default constructor
         */
        SquareMatrix() {max_threads = MAX_THREADS;}

        /**
         * @brief Construct a square matrix filled with the default value
         * @param side Construct a side x side matrix
         * @param value The default value of the matrix
         * @param max_threads Max number of threads to handle operations
         */
        SquareMatrix(std::size_t side, double value=0.0, std::size_t max_threads=MAX_THREADS);

        /**
         * @brief Construct a square matrix from an existed data
         * @param side Construct a side x side matrix
         * @param data The data to initialize the matrix
         * @param max_threads Max number of threads to handle operations
         */
        SquareMatrix(std::size_t side, const std::vector<double>& data, std::size_t max_threads=MAX_THREADS);

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

        /**
         * @brief Applied another set of data to this network
         * @param side Construct a side x side matrix
         * @param data Another set of data to construct
         */
        virtual void set_data(std::size_t side, const std::vector<double>& data) = 0;
        
        Dimension get_dimension() {return dimension;}
        
        void set_max_threads(std::size_t max_threads) {
            if (max_threads <= 0)
                throw std::invalid_argument("There must be at least 1 thread");
            this->max_threads = max_threads;
        }

        std::size_t get_max_threads() {return max_threads;}
        
        std::size_t get_side() {return side;}
        std::vector<double>& get_data() {return data;}
};