#pragma once

#include <vector>

namespace dnn {

/**
 * @brief Activation functions supported by a layer.
 */
enum Activation_Type {
    RELU,
    SOFTMAX
};

/**
 * @brief A fully connected layer with row-major weights and per-neuron biases.
 *
 * Weight `(i, j)` is stored at `i * n_inputs + j`. The layer stores its
 * activations, pre-activation sums, gradients, and Adam moments.
 */
class Layer {
    private:
        Activation_Type activation_type; // Activation used by this layer.

        std::vector<double> weights, gradients;
        std::vector<double> mts, vts; // Adam moments for weights.
        std::vector<double> bias_mts, bias_vts; // Adam moments for biases.
        std::vector<double> biases;
        std::vector<double> biases_gradients;
        std::vector<double> as, zs; // Activations and pre-activation sums.
        std::vector<double> deltas, dldas; // Backpropagation values.
        int n_neurons;
        int n_inputs;

    public:
        /**
         * @brief Create a layer with the given dimensions.
         * @param n_neurons Number of neurons in this layer.
         * @param n_inputs Number of inputs to each neuron.
         * @param activation_type Activation applied after the weighted sums.
         * @param random If true, initialize weights randomly; if false, leave
         *               weights empty for another constructor to supply.
         */
        Layer(int n_neurons, int n_inputs, Activation_Type activation_type, bool random);

        /**
         * @brief Create a layer from one weight vector per neuron.
         * @param weights_layer Weight vectors, all with the same nonzero length.
         * @param activation_type Activation applied after the weighted sums.
         */
        Layer(const std::vector<std::vector<double>>& weights_layer, Activation_Type activation_type);

        /**
         * @brief Create a layer from flattened, row-major weights.
         * @param n_neurons Number of neurons represented by weights.
         * @param weights Concatenated weight vector for each neuron.
         * @param activation_type Activation applied after the weighted sums.
         */
        Layer(int n_neurons, const std::vector<double>& weights, Activation_Type activation_type);

        /**
         * @brief Evaluate Softmax for one neuron.
         * @param arg Values `[z, sum_exp, max_z]` for the current layer.
         * @param activate_type Activation to evaluate; only SOFTMAX is supported.
         * @return The neuron's Softmax activation.
         */
        double activate(const std::vector<double>& arg, Activation_Type activate_type);

        /**
         * @brief Evaluate ReLU for one neuron.
         * @param x Pre-activation sum for the neuron.
         * @param activation_type Activation to evaluate; only RELU is supported.
         * @return The neuron's ReLU activation.
         */
        double activate(double x, Activation_Type activation_type);

        /**
         * @brief Evaluate one entry of the activation Jacobian.
         * @param i Index of the activation output.
         * @param j Index of the pre-activation input.
         * @param activation_type Activation whose derivative is required.
         * @return The derivative of activation i with respect to input j.
         */
        double d_activate(int i, int j, Activation_Type activation_type);

        /**
         * @brief Check whether each output depends only on its matching input.
         * @return True for RELU and false for SOFTMAX.
         */
        bool is_single_activation();

        /**
         * @brief Compute and store weighted sums and activations for one input.
         * @param inputs One value per input connection.
         */
        void forward(const std::vector<double>& inputs);

        Activation_Type get_activation_type();

        std::vector<double>& get_weights();
        std::vector<double>& get_gradients();
        std::vector<double>& get_mts();
        std::vector<double>& get_vts();

        std::vector<double>& get_bias_mts();
        std::vector<double>& get_bias_vts();
        std::vector<double>& get_biases();
        std::vector<double>& get_biases_gradients();
        std::vector<double>& get_as();
        std::vector<double>& get_zs();
        std::vector<double>& get_deltas();
        std::vector<double>& get_dldas();

        int get_n_neurons();
        int get_n_inputs();

    private:
        /**
         * @brief Store each neuron's weighted sum in zs.
         * @param inputs One value per input connection.
         */
        void weighted_sums(const std::vector<double>& inputs);

        /**
         * @brief Apply rectified linear activation.
         * @param x Pre-activation sum.
         * @return The greater of zero and x.
         */
        double relu(double x);

        /**
         * @brief Apply numerically stabilized Softmax for one neuron.
         * @param x This neuron's pre-activation sum.
         * @param eX Sum of exp(z - mX) across the layer.
         * @param mX Maximum pre-activation sum in the layer.
         * @return This neuron's Softmax activation.
         */
        double softmax(double x, double eX, double mX);

        /**
         * @param x Pre-activation sum for one neuron.
         * @return One if x is positive, otherwise zero.
         */
        double d_relu(double x);

        /**
         * @param i Index of the Softmax activation output.
         * @param j Index of the pre-activation input.
         * @return The Softmax Jacobian entry for i and j.
         */
        double d_softmax(int i, int j);

        /**
         * @param n_inputs Number of inputs to each neuron.
         * @param n_neurons Number of neurons in the layer.
         * @return Standard deviation used by the random initializer.
         */
        double normal_Xavier_Deviation(int n_inputs, int n_neurons);
};

} // namespace dnn
