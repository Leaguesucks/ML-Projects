#pragma once

#include <cstddef>
#include <vector>

namespace dnn {

class Network;

double calculate_loss(
    Network& network,
    std::vector<double>& X,
    std::vector<double>& Y
);

void test_gradient_check(
    size_t input_size,
    const std::vector<size_t>& hidden_sizes,
    size_t output_size
);

} // namespace dnn
