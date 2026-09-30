#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

#include <dnn/Layer.h>

/** @brief Loss functions supported by Network. */
enum Loss_Type {
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY,
    SSE
};

/** @brief Size and activation of one fully connected layer. */
struct Layer_Architecture {
    int n_neurons; // Number of outputs produced by this layer.
    Activation_Type activation_type;
};

/**
 * @brief A sequence of fully connected layers with loss and optimizer state.
 *
 * Forward propagation stores the latest inputs and outputs for subsequent
 * loss calculation and backpropagation.
 */
class Network {
    private:
        std::vector<Layer> layers;
        std::vector<double> Xs; // The inputs to feed into this network
        std::vector<double> Y_HAT; // The outputs of this network
        Loss_Type loss_type;
        size_t t; // The time step for the Adam optimizer
        const uint32_t VERSION = 4;
        bool accumulate_gradients = false; // Sum sample gradients when enabled.

    public:
        /**
         * @brief Create layers with randomly initialized weights.
         * @param n_inputs Number of values accepted by the first layer.
         * @param architectures Size and activation of each layer, in order.
         * @param loss_type Loss function used by this network.
         */
        Network(int n_inputs, const std::vector<Layer_Architecture>& architectures, Loss_Type loss_type);

        /**
         * @brief Create a network from existing layers.
         * @param layers Layers and their current parameters, in order.
         * @param loss_type Loss function used by this network.
         */
        Network(const std::vector<Layer>& layers, Loss_Type loss_type);

        /**
         * @brief Restore a network from its serialized binary form.
         * @param filename Path to a file produced by save().
         */
        Network(const std::string& filename);

        /**
         * @brief Run the network and store the latest output values.
         * @param inputs Input vector for the first layer.
         * @see get_Y_HAT()
         */
        void forward_propagation(const std::vector<double>& inputs);

        /**
         * @brief Evaluate one scalar loss contribution.
         * @param y Expected output value.
         * @param y_hat Actual output value.
         * @param type Loss function to evaluate.
         * @return Loss contribution for this output.
         */
        double loss(double y, double y_hat, Loss_Type type);

        /**
         * @brief Differentiate one scalar loss contribution.
         * @param y Expected output value.
         * @param y_hat Actual output value.
         * @param type Loss function to differentiate.
         * @return Derivative of the loss with respect to y_hat.
         */
        double d_loss(double y, double y_hat, Loss_Type type);

        /**
         * @brief Run a sample, calculate its loss, and backpropagate it.
         * @param X Input values for the sample.
         * @param Y Expected output values.
         * @return Sum of the output loss contributions.
         * @note This overload updates gradients; the Y-only overload does not.
         */
        double total_loss(const std::vector<double>& X, const std::vector<double>& Y);

        /**
         * @brief Calculate loss from the most recent forward output.
         * @param Y Expected output values.
         * @return Sum of the output loss contributions.
         * @pre Call forward_propagation() first.
         */
        double total_loss(const std::vector<double>& Y);

        /** @brief Access the layers and their mutable training state. */
        std::vector<Layer>& get_layers();

        /**
         * @brief Calculate weight and bias gradients for the latest input.
         * @param Y Expected output values.
         * @pre Call forward_propagation() first.
         * @note Gradients replace previous values unless accumulation is enabled.
         */
        void back_propagation(const std::vector<double>& Y);

        /**
         * @brief Run forward propagation followed by backpropagation.
         * @param X Input values for the sample.
         * @param Y Expected output values.
         */
        void forward_back_propagation(const std::vector<double>& X, const std::vector<double>& Y);

        /** @brief Access the output vector from the latest forward pass. */
        std::vector<double>& get_Y_HAT();

        /** @brief Check whether backpropagation sums into existing gradients. */
        bool get_accumulate_gradients();

        /**
         * @brief Select gradient accumulation or replacement during backpropagation.
         * @param accumulate True to add sample gradients to existing values.
         */
        void set_accumulate_gradients(bool accumulate);

        /** @brief Access the mutable Adam update counter. */
        size_t& get_time_step();

        /**
         * @brief Save network parameters and Adam state in binary format.
         * @param filename Destination path.
         */
        void save(const std::string& filename);

        /**
         * @brief Replace network state from a previously saved binary file.
         * @param filename Path to a file produced by save().
         */
        void load(const std::string& filename);

    private:
        /**
         * @param y Expected output value.
         * @param y_hat Actual output value.
         * @return Binary cross-entropy contribution.
         */
        double binary_cross_entropy(double y, double y_hat);

        /**
         * @param y Expected output value.
         * @param y_hat Actual output value.
         * @return Categorical cross-entropy contribution.
         */
        double categorical_cross_entropy(double y, double y_hat);

        /**
         * @param y Expected output value.
         * @param y_hat Actual output value.
         * @return Squared error contribution.
         */
        double sse(double y, double y_hat);

        /**
         * @param y Expected output value.
         * @param y_hat Actual output value.
         * @return Binary cross-entropy derivative with respect to y_hat.
         */
        double d_binary_cross_entropy(double y, double y_hat);

        /**
         * @param y Expected output value.
         * @param y_hat Actual output value.
         * @return Categorical cross-entropy derivative with respect to y_hat.
         */
        double d_categorical_cross_entropy(double y, double y_hat);

        /**
         * @param y Expected output value.
         * @param y_hat Actual output value.
         * @return Squared error derivative with respect to y_hat.
         */
        double d_sse(double y, double y_hat);
};
