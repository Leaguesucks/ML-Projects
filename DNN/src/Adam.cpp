#include <dnn/Adam.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

Adam::Adam(Network& network,
           double a,
           double B1, double B2,
           double epsilon) :
           a(a), B1(B1), B2(B2),
           epsilon(epsilon), network(network) {
    if (a <= 0.0)
        throw std::invalid_argument("Learning rate must be greater than zero");

    if (B1 < 0.0 || B1 >= 1.0)
        throw std::invalid_argument("B1 must be in range [0, 1)");

    if (B2 < 0.0 || B2 >= 1.0)
        throw std::invalid_argument("B2 must be in range [0, 1)");

    if (epsilon <= 0.0)
        throw std::invalid_argument("Epsilon must be greater than zero");
}

void Adam::zero_gradients() {
    for (Layer& layer : network.get_layers()) {
        std::fill(layer.get_gradients().begin(), layer.get_gradients().end(), 0.0);
        std::fill(layer.get_biases_gradients().begin(), layer.get_biases_gradients().end(), 0.0);
    }
}

void Adam::accumulate_gradients(const std::vector<double>& X, const std::vector<double>& Y) {
    network.forward_propagation(X);
    network.back_propagation(Y);
}

void Adam::update_weights(size_t batch_size) {
    if (batch_size == 0)
        throw std::invalid_argument("Batch size must be greater than zero");

    const double scale = 1.0 / static_cast<double>(batch_size);
    size_t& t = network.get_time_step();
    ++t; // Advance Adam time step

    const double bias_correction_1 = 1.0 - std::pow(B1, static_cast<double>(t));
    const double bias_correction_2 = 1.0 - std::pow(B2, static_cast<double>(t));

    for (Layer& layer : network.get_layers()) {
        auto& weights = layer.get_weights();
        auto& grads = layer.get_gradients();
        auto& biases = layer.get_biases();
        auto& mts = layer.get_mts();
        auto& vts = layer.get_vts();
        auto& bias_mts = layer.get_bias_mts();
        auto& bias_vts = layer.get_bias_vts();
        auto& biases_grads = layer.get_biases_gradients();
        int n_weights = layer.get_n_inputs();
        
        if (weights.size() != grads.size())
            throw std::runtime_error("Weights and gradient sizes do not match");
        if (mts.size() != weights.size() || vts.size() != weights.size())
            throw std::runtime_error("Adam moment vectors must match weight vector size");
        if (biases.size() != biases_grads.size())
            throw std::runtime_error("Biases and biases gradients sizes do not match");

        for (size_t neuron_index = 0; neuron_index < static_cast<size_t>(layer.get_n_neurons()); ++neuron_index) {

            // Update weights
            for (size_t weight_index = 0; weight_index < static_cast<size_t>(n_weights); weight_index++) {
                double mt_hat, vt_hat;
                double g = grads[neuron_index * n_weights + weight_index] * scale;

                mts[neuron_index * n_weights + weight_index] = B1 * mts[neuron_index * n_weights + weight_index] + (1.0 - B1) * g;
                vts[neuron_index * n_weights + weight_index] = B2 * vts[neuron_index * n_weights + weight_index] + (1.0 - B2) * g * g;

                mt_hat = mts[neuron_index * n_weights + weight_index] / bias_correction_1;
                vt_hat = vts[neuron_index * n_weights + weight_index] / bias_correction_2;

                weights[neuron_index * n_weights + weight_index] -= a * mt_hat / (std::sqrt(vt_hat) + epsilon);
            }
            
            // Update bias
            double bias_mt_hat, bias_vt_hat;
            double bias_mt = bias_mts[neuron_index];
            double bias_vt = bias_vts[neuron_index];
            double bias = biases[neuron_index];
            double bias_grad = biases_grads[neuron_index] * scale;

            bias_mt = B1 * bias_mt + (1.0 - B1) * bias_grad;
            bias_vt = B2 * bias_vt + (1.0 - B2) * bias_grad * bias_grad;

            bias_mt_hat = bias_mt / bias_correction_1;
            bias_vt_hat = bias_vt / bias_correction_2;

            bias -= a * bias_mt_hat / (std::sqrt(bias_vt_hat) + epsilon);

            biases[neuron_index] = bias;
            bias_mts[neuron_index] = bias_mt;
            bias_vts[neuron_index] = bias_vt;
        }
    }
}
