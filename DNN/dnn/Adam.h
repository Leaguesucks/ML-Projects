#pragma once

#include <vector>
#include <cstddef>

#include <dnn/Network.h>

/**
 * @brief Applies Adam updates to the weights and biases of a Network.
 *
 * The optimizer uses gradient and moment arrays stored in the network's layers.
 */
class Adam {
    private:
        double a; // The learning rate
        double B1, B2; // Decay rates for the moving averages of the gradient
        double epsilon; // To avoid division by zero
        
        Network& network; // The network to train

    public:
        /**
         * @brief Attach an Adam optimizer to a network.
         * @param network Network whose parameters and optimizer state are updated.
         * @param a Learning rate; must be positive.
         * @param B1 First-moment decay rate in [0, 1).
         * @param B2 Second-moment decay rate in [0, 1).
         * @param epsilon Positive denominator offset.
         */
        Adam(Network& network, double a = 0.001, double B1 = 0.9, double B2 = 0.999, double epsilon = 1e-8);

        /**
         * @brief Clear all weight and bias gradients before a new batch.
         */
        void zero_gradients();

        /**
         * @brief Run forward and backward propagation for one sample.
         * @param X Input values for the sample.
         * @param Y Expected output values.
         * @note To sum gradients across samples, enable accumulation on the
         *       network before calling this method. Weights are updated by
         *       update_weights(), not by this method.
         */
        void accumulate_gradients(const std::vector<double>& X, const std::vector<double>& Y);

        /**
         * @brief Apply one Adam update using the accumulated gradients.
         * @param batch_size Number of samples represented by the gradients;
         *                   must be greater than zero.
         * @note Gradients are averaged by batch_size, and the network's Adam
         *       time step advances once per call.
         */
        void update_weights(size_t batch_size);
};
