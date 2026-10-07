#include <dnn/Network.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace dnn {

Network::Network(int n_inputs, const std::vector<Layer_Architecture>& architectures, Loss_Type loss_type) 
: loss_type(loss_type) {
    if (architectures.empty())
        throw std::invalid_argument("Network must have at least one layer");

    layers.reserve(architectures.size() );
    layers.emplace_back(architectures[0].n_neurons, n_inputs, architectures[0].activation_type, true);
    
    for (size_t i = 1; i < architectures.size(); i++)
        layers.emplace_back(architectures[i].n_neurons, 
            architectures[i-1].n_neurons, architectures[i].activation_type, true);
}

Network::Network(const std::vector<Layer>& layers, Loss_Type loss_type) 
: layers(layers), loss_type(loss_type) {}

Network::Network(const std::string& filename) {
    load(filename);
}

void Network::forward_propagation(const std::vector<double>& inputs) {
    auto y_hats = inputs;
    Xs = inputs;

    for (Layer& layer : layers) {
        layer.forward(y_hats);
        y_hats = layer.get_as();
    }
    Y_HAT = y_hats;
}

double Network::loss(double y, double y_hat, Loss_Type type) {
    switch (type) {
        case BINARY_CROSS_ENTROPY:
            return binary_cross_entropy(y, y_hat);
        case CATEGORICAL_CROSS_ENTROPY:
            return categorical_cross_entropy(y, y_hat);
        case SSE:
            return sse(y, y_hat);
        default:
            throw std::invalid_argument("Invalid type");
    }
}

double Network::d_loss(double y, double y_hat, Loss_Type type) {
    switch (type) {
        case BINARY_CROSS_ENTROPY:
            return d_binary_cross_entropy(y, y_hat);
        case CATEGORICAL_CROSS_ENTROPY:
            return d_categorical_cross_entropy(y, y_hat);
        case SSE:
            return d_sse(y, y_hat);
        default:
            throw std::invalid_argument("Invalid type");
    }
}

double Network::binary_cross_entropy(double y, double y_hat) {
    return -(y * std::log(y_hat) + (1 - y) * std::log(1 - y_hat));
}

double Network::d_binary_cross_entropy(double y, double y_hat) {
    return -((y / y_hat) - (1 - y) / (1 - y_hat));
}

double Network::sse(double y, double y_hat) {
    return (y - y_hat) * (y - y_hat);
}

double Network::d_sse(double y, double y_hat) {
    return -2 * (y - y_hat);
}

std::vector<Layer>& Network::get_layers() {
    return layers;
}

double Network::categorical_cross_entropy(double y, double y_hat) {
    return -y * std::log(std::max(y_hat, 1e-12));
}

double Network::d_categorical_cross_entropy(double y, double y_hat) {
    return -y / (std::max(y_hat, 1e-12));
}

double Network::total_loss(const std::vector<double>& X, const std::vector<double>& Y) {
    double losses = 0.0;

    forward_propagation(X);
    for (size_t i = 0; i < Y.size(); i++)
        losses += loss(Y[i], Y_HAT[i], loss_type);
    
    back_propagation(Y);
    return losses;
}

double Network::total_loss(const std::vector<double>& Y) {
    double losses = 0.0;

    for (size_t i = 0; i < Y.size(); i++)
        losses += loss(Y[i], Y_HAT[i], loss_type);
    return losses;
}

void Network::back_propagation(const std::vector<double>& Y) {
    if (layers.empty())
        throw std::runtime_error("Network has no layers");
    if (Y.size() != static_cast<size_t>(layers.back().get_n_neurons()))
        throw std::invalid_argument("Size of Y does not match the number of neurons in the output layer");

    for (size_t l = layers.size(); l-- > 0; ) {
        Layer& current_layer = layers[l];
        Activation_Type atype = current_layer.get_activation_type();

        auto& grads = current_layer.get_gradients();
        auto& as = current_layer.get_as();
        auto& deltas = current_layer.get_deltas();
        auto& dldas = current_layer.get_dldas();
        auto& biases_grads = current_layer.get_biases_gradients();
        int n_neurons = current_layer.get_n_neurons();
        int n_weights = current_layer.get_n_inputs();

        auto& prev_as = (l == 0) ? Xs : layers[l - 1].get_as();

        // Calculate dLda for each neuron in this layer
        if (l == layers.size() - 1)
            for (size_t k = 0; k < static_cast<size_t>(n_neurons); k++)
                dldas[k] = d_loss(Y[k], as[k], loss_type);
        else {
            Layer& next_layer = layers[l + 1];
            auto& next_weights = next_layer.get_weights();
            auto& next_deltas = next_layer.get_deltas();
            int next_n_neurons = next_layer.get_n_neurons();
            int next_n_weights = next_layer.get_n_inputs();

            std::fill(dldas.begin(), dldas.end(), 0.0);
            for (size_t j = 0; j < static_cast<size_t>(next_n_neurons); j++)
                for (size_t k = 0; k < static_cast<size_t>(n_neurons); k++)
                    dldas[k] += next_weights[j * next_n_weights + k] * next_deltas[j];
        }

        // Calculate delta for each neuron in this layer
        for (size_t i = 0; i < static_cast<size_t>(n_neurons); i++) {
            double delta = 0.0;

            if (current_layer.is_single_activation())
                delta = dldas[i] * current_layer.d_activate(i, i, atype);
            else
                for (size_t j = 0; j < static_cast<size_t>(n_neurons); j++)
                    delta += dldas[j] * current_layer.d_activate(j, i, atype);

            deltas[i] = delta;

            if (accumulate_gradients)
                biases_grads[i] += delta;
            else
                biases_grads[i] = delta;
        }

        // Calculate the gradient for each neuron in this layer
        for (size_t i = 0; i < static_cast<size_t>(n_neurons); i++)
            for (size_t j = 0; j < static_cast<size_t>(n_weights); j++)
                if (accumulate_gradients)
                    grads[i * n_weights + j] += deltas[i] * prev_as[j];
                else
                    grads[i * n_weights + j] = deltas[i] * prev_as[j];
    }
}

void Network::forward_back_propagation(const std::vector<double>& X, const std::vector<double>& Y) {
    forward_propagation(X);
    back_propagation(Y);
}

std::vector<double>& Network::get_Y_HAT() {
    return Y_HAT;
}

void Network::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);

    if (!file)
        throw std::runtime_error("Could not open file for writing: " + filename);

    constexpr char MAGIC[] = "FNN";

    file.write(MAGIC, sizeof(MAGIC) - 1);
    file.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));

    uint64_t input_size = static_cast<uint64_t>(layers[0].get_n_inputs());
    file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));

    uint64_t n_layers = layers.size();
    file.write(reinterpret_cast<const char*>(&n_layers), sizeof(n_layers));

    uint32_t loss_val = static_cast<uint32_t>(loss_type);
    file.write(reinterpret_cast<const char*>(&loss_val), sizeof(loss_val));

    uint64_t adam_t = static_cast<uint64_t>(t);
    file.write(reinterpret_cast<const char*>(&adam_t), sizeof(adam_t));

    for (Layer& layer : layers) {
        auto& weights = layer.get_weights();
        auto& biases = layer.get_biases();
        auto& mts = layer.get_mts();
        auto& vts = layer.get_vts();
        auto& bias_mts = layer.get_bias_mts();
        auto& bias_vts = layer.get_bias_vts();
        uint32_t n_weights = layer.get_n_inputs();

        uint64_t n_neurons = layer.get_n_neurons();
        file.write(reinterpret_cast<const char*>(&n_neurons), sizeof(n_neurons));

        uint32_t activation = static_cast<uint32_t>(layer.get_activation_type());
        file.write(reinterpret_cast<const char*>(&activation), sizeof(activation));

        file.write(reinterpret_cast<const char*>(&n_weights), sizeof(n_weights));
        file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(mts.data()), mts.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(vts.data()), vts.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(bias_mts.data()), bias_mts.size() * sizeof(double));
        file.write(reinterpret_cast<const char*>(bias_vts.data()), bias_vts.size() * sizeof(double));
    }

    if (!file)
        throw std::runtime_error("Error occured while saving the network");
}

void Network::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file)
        throw std::runtime_error("Could not open file: " + filename);

    char magic[3];
    file.read(magic, sizeof(magic));

    if (std::string(magic, 3) != "FNN")
        throw std::runtime_error("Invalid saved file");

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (version != VERSION)
        throw std::runtime_error("Unsupproted version");

    uint64_t input_size;
    file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));

    uint64_t n_layers;
    file.read(reinterpret_cast<char*>(&n_layers), sizeof(n_layers));

    uint32_t loss_val;
    file.read(reinterpret_cast<char*>(&loss_val), sizeof(loss_val));
    loss_type = static_cast<Loss_Type>(loss_val);

    uint64_t adam_t;
    file.read(reinterpret_cast<char*>(&adam_t), sizeof(adam_t));
    t = static_cast<size_t>(adam_t);

    layers.clear();
    layers.reserve(n_layers);

    for (uint64_t L = 0; L < n_layers; L++) {
        uint64_t n_neurons;
        file.read(reinterpret_cast<char*>(&n_neurons), sizeof(n_neurons));

        uint32_t activation_value;
        file.read(reinterpret_cast<char*>(&activation_value), sizeof(activation_value));

        Activation_Type activation = static_cast<Activation_Type>(activation_value);

        uint32_t n_weights;
        file.read(reinterpret_cast<char*>(&n_weights), sizeof(n_weights));

        std::vector<double> weights(static_cast<size_t>(n_neurons * n_weights));
        file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(double));

        std::vector<double> biases(static_cast<size_t>(n_neurons));
        file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(double));

        std::vector<double> mts(static_cast<size_t>(n_neurons * n_weights));
        file.read(reinterpret_cast<char*>(mts.data()), mts.size() * sizeof(double));

        std::vector<double> vts(static_cast<size_t>(n_neurons * n_weights));
        file.read(reinterpret_cast<char*>(vts.data()), vts.size() * sizeof(double));    

        std::vector<double> bias_mts(static_cast<size_t>(n_neurons));
        file.read(reinterpret_cast<char*>(bias_mts.data()), bias_mts.size() * sizeof(double));

        std::vector<double> bias_vts(static_cast<size_t>(n_neurons));
        file.read(reinterpret_cast<char*>(bias_vts.data()), bias_vts.size() * sizeof(double));

        Layer layer(n_neurons, weights, activation);
        layer.get_biases() = std::move(biases);
        layer.get_mts() = std::move(mts);
        layer.get_vts() = std::move(vts);
        layer.get_bias_mts() = std::move(bias_mts); 
        layer.get_bias_vts() = std::move(bias_vts);
        layers.push_back(std::move(layer));
    }

    if (!file)
        throw std::runtime_error("Error occured while reading from saved file");
    Xs.resize(input_size);
}

bool Network::get_accumulate_gradients() {
    return accumulate_gradients;
}

void Network::set_accumulate_gradients(bool accumulate) {
    accumulate_gradients = accumulate;
}

size_t& Network::get_time_step() {
    return t;
}

} // namespace dnn
