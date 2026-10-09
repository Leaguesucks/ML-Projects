#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <iterator>

#include <dnn/Network.h>
#include <dnn/MNIST.h>
#include <dnn/Adam.h>

namespace {
    constexpr std::size_t N_OUTPUTS = 10;
    constexpr std::size_t EPOCH_SIZE = 20;
    constexpr std::size_t BATCH_SIZE = 256;
    constexpr char SAVED_FILE[] = "training/mnist_train.bin";
}

namespace dnn {

void train(std::vector<dnn::MNIST_Image>& training_data, const std::vector<dnn::MNIST_Image>& test_data,
    dnn::Network& network, dnn::Adam& optimizer, std::mt19937& rng) {
    std::vector<double> Y(N_OUTPUTS, 0.0);

    std::cout << "Start training..." << "\n\n";
    double best_accuracy = 98.12; // The best accuracy so far
    double minimum_loss = 0.129846; // The minimum loss so far

    for (size_t epoch = 0; epoch < EPOCH_SIZE; epoch++) {
        std::shuffle(training_data.begin(), training_data.end(), rng);

        for (size_t start = 0; start < training_data.size(); start += BATCH_SIZE) {
            size_t end = std::min(start + BATCH_SIZE, training_data.size());
            size_t batch_size = end - start;
            optimizer.zero_gradients();

            for (size_t i = start; i < end; ++i) {
                const auto& image = training_data[i];

                std::fill(Y.begin(), Y.end(), 0.0);
                Y[image.label] = 1.0;
                optimizer.accumulate_gradients(image.pixels, Y);
            }
            optimizer.update_weights(batch_size);
        }
        
        // Evaluate the model on the test dataset
        double total_loss = 0.0;
        size_t n_correct = 0;
        for (size_t i = 0; i < test_data.size(); i++) {
            const auto& image = test_data[i];
            std::fill(Y.begin(), Y.end(), 0.0);
            Y[image.label] = 1.0;

            total_loss += network.total_loss(image.pixels, Y); // Call forward_propagation
            const auto& Y_HAT = network.get_Y_HAT();
            
            size_t predicted_label = std::distance(Y_HAT.begin(), std::max_element(Y_HAT.begin(), Y_HAT.end()));
            if (predicted_label == image.label)
                n_correct++;
        }

        double accuracy = static_cast<double>(n_correct) / test_data.size() * 100.0;
        double average_loss = total_loss / test_data.size();

        if (accuracy > best_accuracy) {
            std::cout << "CURRENT BEST\n";
            best_accuracy = accuracy;
            network.save(SAVED_FILE);
        } else if (accuracy == best_accuracy && average_loss < minimum_loss) {
            std::cout << "CURRENT BEST\n";
            minimum_loss = average_loss;
            network.save(SAVED_FILE);
        }

        std::cout << "Epoch:        " << epoch + 1 <<"\n";
        std::cout << "Average loss: " << average_loss << "\n";
        std::cout << "Accuracy:     " << accuracy << "%\n\n";
    }
}

} // namespace dnn

int main() {
    std::random_device rd;
    std::mt19937 g(rd());

    dnn::MNIST mnist, mnist_test;

    mnist.load(
        "mnist/train-images.idx3-ubyte",
        "mnist/train-labels.idx1-ubyte"
    );

    mnist_test.load(
        "mnist/t10k-images.idx3-ubyte",
        "mnist/t10k-labels.idx1-ubyte"
    );

    dnn::Network network(SAVED_FILE);
    network.set_accumulate_gradients(true);

    dnn::Adam optimizer(network);

    dnn::train(mnist.get_data(), mnist_test.get_data(), network, optimizer, g);
    
    return 0;
}
