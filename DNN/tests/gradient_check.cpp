#include "gradient_check.h"

#include <dnn/Network.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace dnn {

double calculate_loss(dnn::Network& network, std::vector<double>& X,std::vector<double>& Y) {
    network.forward_propagation(X);
    std::vector<double> prediction = network.get_Y_HAT();

    double total_loss = 0.0;
    for (size_t i = 0; i < Y.size(); ++i)
        total_loss += network.loss(Y[i], prediction[i], dnn::CATEGORICAL_CROSS_ENTROPY);

    return total_loss;
}


void test_gradient_check(size_t input_size, 
    const std::vector<size_t>& hidden_sizes, size_t output_size) {
    // Gradient-check parameters
    constexpr double epsilon = 1e-5;

    // A gradient is considered correct when:
    // |analytical - numerical| <= absolute_tolerance + relative_tolerance * scale
    constexpr double absolute_tolerance = 1e-8;
    constexpr double relative_tolerance = 1e-5;

    std::cout << std::fixed << std::setprecision(10);

    std::cout << "========================================\n";
    std::cout << "Backpropagation Gradient Check\n";
    std::cout << "========================================\n\n";

    // ============================================================
    // Validate architecture
    // ============================================================

    if (input_size == 0)
        throw std::invalid_argument("Input size must be greater than zero");
    if (output_size == 0)
        throw std::invalid_argument("Output size must be greater than zero");

    for (size_t size : hidden_sizes)
        if (size == 0)
            throw std::invalid_argument("Hidden layer size must be greater than zero");

    // ============================================================
    // Create hidden layers
    // ============================================================

    std::vector<dnn::Layer> layers;
    size_t previous_size = input_size;
    for (size_t layer_index = 0; layer_index < hidden_sizes.size(); ++layer_index) {
        size_t current_size = hidden_sizes[layer_index];

        std::vector<std::vector<double>> weights(current_size, std::vector<double>(previous_size));

        // --------------------------------------------------------
        // Deterministic weights
        // --------------------------------------------------------

        for (size_t neuron = 0; neuron < current_size; ++neuron) {
            for (size_t weight = 0; weight < previous_size; ++weight) {
                double value = static_cast<double>((
                            (layer_index + 1) * 17
                            + (neuron + 1) * 31
                            + (weight + 1) * 13
                        ) % 200
                    ) / 1000.0;
                value -= 0.1;

                weights[neuron][weight] = value;
            }
        }

        layers.emplace_back(weights, dnn::RELU);
        previous_size = current_size;
    }

    // ============================================================
    // Create output layer
    // ============================================================

    std::vector<std::vector<double>> weights(output_size, std::vector<double>(previous_size));

    for (size_t neuron = 0; neuron < output_size; ++neuron) {
        for (size_t weight = 0; weight < previous_size; ++weight) {
            double value = static_cast<double>(
                    (
                        101
                        + (neuron + 1) * 23
                        + (weight + 1) * 37
                    ) % 160
                ) / 1000.0;
            value -= 0.08;
            weights[neuron][weight] = value;
        }
    }

    layers.emplace_back(weights, dnn::SOFTMAX);

    // ============================================================
    // Create network
    // ============================================================

    dnn::Network network(layers, dnn::CATEGORICAL_CROSS_ENTROPY);

    // ============================================================
    // Deterministic input
    // ============================================================

    std::vector<double> X(input_size);

    for (size_t i = 0; i < input_size; ++i)
        X[i] = std::sin(static_cast<double>(i) * 0.37);

    // ============================================================
    // One-hot target
    // ============================================================

    std::vector<double> Y(output_size, 0.0);
    size_t target_class = output_size / 2;

    Y[target_class] = 1.0;

    // ============================================================
    // Forward propagation
    // ============================================================

    network.forward_propagation(X);
    std::vector<double> prediction = network.get_Y_HAT();
    std::cout << "Network output:\n";

    for (size_t i = 0; i < prediction.size(); ++i)
        std::cout << "  [" << i << "] " << prediction[i] << '\n';

    // ============================================================
    // Verify Softmax
    // ============================================================

    double probability_sum = 0.0;
    for (double p : prediction) {
        if (!std::isfinite(p))
            throw std::runtime_error("Softmax produced a non-finite probability");
        if (p <= 0.0)
            throw std::runtime_error("Softmax produced a non-negative probability");

        probability_sum += p;
    }

    std::cout << "\nProbability sum: " << probability_sum << '\n';
    if (std::abs(probability_sum - 1.0) > 1e-12)
        throw std::runtime_error("Softmax probabilities do not sum to 1");

    // ============================================================
    // Original loss
    // ============================================================

    double original_loss = dnn::calculate_loss(network, X, Y);
    if (!std::isfinite(original_loss))
        throw std::runtime_error("Initial loss is not finite");

    std::cout << "Original loss:   " << original_loss << "\n\n";

    // ============================================================
    // Backpropagation
    // ============================================================

    network.forward_back_propagation(X, Y);

    // ============================================================
    // Gradient-check statistics
    // ============================================================

    size_t total_tests = 0;
    size_t failed_tests = 0;

    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;

    size_t worst_L = 0;
    size_t worst_J = 0;
    size_t worst_I = 0;

    double worst_analytical = 0.0;
    double worst_numerical = 0.0;

    // ============================================================
    // Iterate over every layer
    // ============================================================

    for (size_t L = 0; L < network.get_layers().size(); ++L) {
        dnn::Layer& current_layer = network.get_layers()[L];
        auto& current_weights = current_layer.get_weights();
        auto& current_grads = current_layer.get_gradients();
        int n_neurons = current_layer.get_n_neurons();
        int n_inputs = current_layer.get_n_inputs();

        if (current_weights.size() % n_inputs != 0 || current_weights.size() / n_neurons != static_cast<size_t>(n_inputs))
            throw std::runtime_error("Weight size in each neuron does not match");
        if (current_grads.size() != static_cast<size_t>(n_neurons * n_inputs))
            throw std::runtime_error("Gradient size of this layer does not match the number of neurons in it");

        // ========================================================
        // Every neuron
        // ========================================================

        for (size_t J = 0; J < static_cast<size_t>(n_neurons); ++J) {

            // ====================================================
            // Every weight
            // ====================================================

            for (size_t I = 0; I < static_cast<size_t>(n_inputs); ++I) {
                ++total_tests;

                // Store original weight
                double original_weight = current_weights[J * n_inputs + I];

                // L(w + epsilon)
                current_weights[J * n_inputs + I] = original_weight + epsilon;

                double loss_plus = dnn::calculate_loss(network, X, Y);

                // L(w - epsilon)
                current_weights[J * n_inputs + I] = original_weight - epsilon;

                double loss_minus = dnn::calculate_loss(network, X, Y);

                // Restore original weight
                current_weights[J * n_inputs + I] = original_weight;

                // Validate losses
                if (!std::isfinite(loss_plus) || !std::isfinite(loss_minus))
                    throw std::runtime_error("Numerical gradient produced a non-finite loss");

                // ------------------------------------------------
                // Numerical gradient
                //
                // Central difference:
                //
                // dL/dw ≈
                //     (L(w + ε) - L(w - ε))
                //     / (2ε)
                // ------------------------------------------------

                double numerical_gradient = (loss_plus - loss_minus) / (2.0 * epsilon);

                // Analytical gradient
                double analytical_gradient = current_grads[J * n_inputs + I];

                if (!std::isfinite(analytical_gradient))
                    throw std::runtime_error("Backpropagation produced a non-finite gradient");

                // Calculate errors
                double absolute_error = std::abs(analytical_gradient - numerical_gradient);

                // Use the larger magnitude as the scale.
                //
                // This avoids artificially inflating the
                // relative error when one gradient is tiny.

                double scale = std::max(std::abs(analytical_gradient), std::abs(numerical_gradient));
                double relative_error = (scale > 0.0) ? absolute_error / scale : 0.0;

                // Combined tolerance
                double allowed_error = absolute_tolerance + relative_tolerance * scale;

                // =================================================
                // Track worst gradient
                //
                // Use absolute error as the primary metric.
                // This prevents tiny gradients from appearing
                // "worst" merely because of relative error.
                // =================================================

                if (absolute_error > max_absolute_error) {
                    max_absolute_error = absolute_error;
                    max_relative_error = relative_error;

                    worst_L = L;
                    worst_J = J;
                    worst_I = I;

                    worst_analytical = analytical_gradient;
                    worst_numerical = numerical_gradient;
                }

                // Check gradient
                if (absolute_error > allowed_error) {
                    ++failed_tests;
                    std::cout
                        << "FAIL,"
                        << "  Layer = " << L << ","
                        << "  Neuron = " << J << ","
                        << "  Weight = " << I << ","
                        << "  Analytical = "
                        << analytical_gradient << ","
                        << "  Numerical = "
                        << numerical_gradient << ","
                        << "  Absolute error = "
                        << absolute_error << ","
                        << "  Relative error = "
                        << relative_error << ","
                        << "  Allowed error = "
                        << allowed_error << ","
                        << '\n';
                }
            }
        }
    }

    // ============================================================
    // Results
    // ============================================================

    std::cout << "\n";
    std::cout << "Gradient Check\n";
    std::cout << "---------------\n";
    std::cout << "Total weights tested: " << total_tests << '\n';
    std::cout << "Failed tests:         " << failed_tests << '\n';
    
    std::cout << "\nWorst gradient:\n";
    
    std::cout << "  Layer:              " << worst_L << '\n';
    std::cout << "  Neuron:             " << worst_J << '\n';
    std::cout << "  Weight:             " << worst_I << '\n';
    std::cout << "  Analytical:         " << worst_analytical << '\n';
    std::cout << "  Numerical:          " << worst_numerical << '\n';
    std::cout << "  Absolute error:     " << max_absolute_error << '\n';
    std::cout << "  Relative error:     " << max_relative_error << '\n';
    std::cout << "  Absolute tolerance: " << absolute_tolerance << '\n';
    std::cout << "  Relative tolerance: " << relative_tolerance << '\n';

    // Final result
    if (failed_tests == 0)
        std::cout << "\nPASS: " << "All weight gradients match.\n";
    else
        std::cout << "\nFAIL: " << "Some weight gradients do not match.\n";
}

} // namespace dnn
