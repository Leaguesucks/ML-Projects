#include <dnn/Layer.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

Layer::Layer(int n_neurons, int n_inputs, ActivationType activation_type, bool random) : 
activation_type(activation_type), gradients(n_neurons * n_inputs, 0.0),
mts(n_neurons * n_inputs, 0.0), vts(n_neurons * n_inputs, 0.0),
bias_mts(n_neurons, 0.0), bias_vts(n_neurons, 0.0), 
biases(n_neurons, 0.0), biases_gradients(n_neurons, 0.0),
as(n_neurons, 0.0), zs(n_neurons, 0.0),
deltas(n_neurons, 0.0), dldas(n_neurons, 0.0),
n_neurons(n_neurons), n_inputs(n_inputs) {
    if (!random)
        return;

    if (n_neurons <= 0)
        throw std::invalid_argument("Each layer must have at least one neuron");
    if (n_inputs <= 0)
        throw std::invalid_argument("Each neuron must take at least one input");

    double deviation = normal_Xavier_Deviation(n_inputs, n_neurons);

    std::mt19937_64 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, deviation);

    weights.reserve(n_neurons * n_inputs);
    for (size_t i = 0; i < static_cast<size_t>(n_neurons); i++)
        for (size_t j = 0; j < static_cast<size_t>(n_inputs); j++)
            weights.push_back(dist(rng));

}

Layer::Layer(const std::vector<std::vector<double>>& weights_layer, ActivationType activation_type) :
Layer(weights_layer.size(), weights_layer.empty() ? 0 : weights_layer[0].size(), activation_type, false) {
    if (weights_layer.empty())
        throw std::invalid_argument("Each layer must at least have one neuron");
    if (weights_layer[0].empty())
        throw std::invalid_argument("Each neuron must have at least one weight");

    weights.reserve(n_neurons * n_inputs);
    for (const auto& weight_layer : weights_layer) {
        if (weight_layer.size() != static_cast<size_t>(n_inputs))
            throw std::invalid_argument(
                "All neurons in a layer must have the same number of weights"
            );

        for (double w : weight_layer)
            weights.push_back(w);
    }
}

Layer::Layer(int n_neurons, const std::vector<double>& weights, ActivationType activation_type) :
Layer(n_neurons, weights.size() / n_neurons, activation_type, false) {
    if (n_neurons <= 0)
        throw std::invalid_argument("Each layer must have at least one neuron");
    if (weights.empty())
        throw std::invalid_argument("Each layer must at least have one neuron with at least one weight");
    if (weights.size() % n_neurons != 0)
        throw std::invalid_argument("Each neuron in a layer must have the same number of weights");

    this->weights = weights;
}

double Layer::normal_Xavier_Deviation(int n_inputs, int n_neurons) {
    return std::sqrt(2.0 / (n_inputs + n_neurons));
}

double Layer::relu(double x) {
    return std::max(0.0, x);
}

double Layer::softmax(double x, double eX, double mX) {
    return std::exp(x - mX) / eX;
}

double Layer::d_relu(double x) {
    return (x > 0.0) ? 1.0 : 0.0;
}

double Layer::d_softmax(int i, int j) {
    if (i == j) {
        double s = as[i];
        return s * (1 - s);
    } else {
        return - as[i] * as[j];
    }
}

double Layer::activate(const std::vector<double>& args, ActivationType activation_type) {
    switch (activation_type) {
        case SOFTMAX:
            return softmax(args[0], args[1], args[2]);
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

double Layer::activate(double x, ActivationType activation_type) {
    switch (activation_type) {
        case RELU:
            return relu(x);
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

double Layer::d_activate(int i, int j, ActivationType activation_type) {
    switch (activation_type) {
        case SOFTMAX:
            return d_softmax(i, j);
        case RELU:
            return (i == j) ? d_relu(zs[i]) : 0.0;
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

void Layer::weighted_sums(const std::vector<double>& inputs) {
    for (size_t i = 0; i < static_cast<size_t>(n_neurons); i++) {
        double sum = biases[i];
        
        for (size_t j = 0; j < static_cast<size_t>(n_inputs); j++) {
            sum += weights[i * n_inputs + j] * inputs[j];
        }
        zs[i] = sum;
    }
}

void Layer::forward(const std::vector<double>& inputs) {
    if (inputs.size() != static_cast<size_t>(n_inputs))
        throw std::invalid_argument("Input size does not match the number of inputs for this layer");

    weighted_sums(inputs);
    switch (activation_type) {
        case SOFTMAX: {
            double max_sum = *std::max_element(zs.begin(), zs.end());
            double eX = 0.0;

            for (double z : zs)
                eX += std::exp(z - max_sum);

            for (size_t i = 0; i < static_cast<size_t>(n_neurons); ++i)
                as[i] = activate({zs[i], eX, max_sum}, SOFTMAX);
            break;
        }

        case RELU: {
            for (size_t i = 0; i < static_cast<size_t>(n_neurons); ++i)
                as[i] = activate(zs[i], RELU);
            break;
        }

        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

ActivationType Layer::get_activation_type() {
    return activation_type;
}

bool Layer::is_single_activation() {
    return (activation_type <= RELU) ? true : false;
}

std::vector<double>& Layer::get_weights() {
    return weights;
}

std::vector<double>& Layer::get_gradients() {
    return gradients;
}

std::vector<double>& Layer::get_mts() {
    return mts;
}

std::vector<double>& Layer::get_vts() {
    return vts;
}

std::vector<double>& Layer::get_bias_mts() {
    return bias_mts;
}

std::vector<double>& Layer::get_bias_vts() {
    return bias_vts;
}

std::vector<double>& Layer::get_biases() {
    return biases;
}

std::vector<double>& Layer::get_biases_gradients() {
    return biases_gradients;
}

std::vector<double>& Layer::get_as() {
    return as;
}

std::vector<double>& Layer::get_zs() {
    return zs;
}

std::vector<double>& Layer::get_deltas() {
    return deltas;
}

std::vector<double>& Layer::get_dldas() {
    return dldas;
}

int Layer::get_n_neurons() {
    return n_neurons;
}

int Layer::get_n_inputs() {
    return n_inputs;
}
