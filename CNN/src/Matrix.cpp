#include <cnn/data/Matrix.hpp>

SquareMatrix::SquareMatrix(std::size_t size, double value=0.0)
: side(size), data(std::pow(size, 2), value) {}

SquareMatrix::SquareMatrix(std::size_t size, const std::vector<double>& data)
: side(size), data(data) {}