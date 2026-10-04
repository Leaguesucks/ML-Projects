#include <cnn/data/Matrix.hpp>

SquareMatrix::SquareMatrix(std::size_t side, double value=0.0, std::size_t max_threads=MAX_THREADS)
: side(side), data(side * side, value), max_threads(max_threads) {
    if (max_threads <= 0)
        throw std::invalid_argument("There must be at least 1 thread");
}

SquareMatrix::SquareMatrix(std::size_t side, const std::vector<double>& data, std::size_t max_threads=MAX_THREADS)
: side(side), data(data), max_threads(max_threads) {
    if (max_threads <= 0)
        throw std::invalid_argument("There must be at least 1 thread");
}