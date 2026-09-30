#include <dnn/MNIST.h>
#include <dnn/Network.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>

int main() {
    MNIST mnist;
    mnist.load("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte");
    if (mnist.get_data().size() != 10000) {
        std::cerr << "MNIST test image count differs from the pre-refactor baseline\n";
        return 1;
    }

    Network network("training/mnist_train.bin");
    network.forward_propagation(mnist.get_data().front().pixels);
    const auto& prediction = network.get_Y_HAT();
    const auto predicted_label = std::distance(
        prediction.begin(), std::max_element(prediction.begin(), prediction.end()));

    if (mnist.get_data().front().label != 7 || prediction.size() != 10 ||
        predicted_label != 7 ||
        std::abs(prediction[7] - 0.99999999480757928) > 1e-12) {
        std::cerr << "Saved-model prediction differs from the pre-refactor baseline\n";
        return 1;
    }

    std::cout << "Saved-model prediction matches the pre-refactor baseline\n";
}
