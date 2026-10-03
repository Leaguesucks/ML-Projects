#include <dnn/MNIST.hpp>

#include <stdexcept>
#include <utility>

uint32_t MNIST::read_uint32(std::ifstream& file) {
    uint8_t bytes[4];

    file.read(reinterpret_cast<char*>(bytes), 4);

    return
        (static_cast<uint32_t>(bytes[0]) << 24) |
        (static_cast<uint32_t>(bytes[1]) << 16) |
        (static_cast<uint32_t>(bytes[2]) << 8)  |
         static_cast<uint32_t>(bytes[3]);
}

void MNIST::load(const std::string& image_file, const std::string& label_file) {
    std::ifstream images(image_file, std::ios::binary);
    std::ifstream labels(label_file, std::ios::binary);

    if (!images || !labels)
        throw std::runtime_error("Unable to open MNIST files");

    uint32_t image_magic = read_uint32(images);
    uint32_t n_images    = read_uint32(images);
    uint32_t rows        = read_uint32(images);
    uint32_t cols        = read_uint32(images);

    uint32_t label_magic = read_uint32(labels);
    uint32_t n_labels    = read_uint32(labels);

    if (image_magic != 2051)
        throw std::runtime_error("Invalid image file");

    if (label_magic != 2049)
        throw std::runtime_error("Invalid label file");

    if (n_images != n_labels)
        throw std::runtime_error("Image/label count mismatch");

    if (rows != 28 || cols != 28)
        throw std::runtime_error("Unexpected image dimensions");

    data.reserve(n_images);

    for (uint32_t i = 0; i < n_images; ++i) {
        MNIST_Image image;
        image.pixels.resize(rows * cols);

        for (double& pixel : image.pixels) {
            uint8_t value;

            images.read(reinterpret_cast<char*>(&value), 1);
            pixel = static_cast<double>(value) / 255.0;
        }

        uint8_t label;

        labels.read(reinterpret_cast<char*>(&label), 1);
        image.label = label;
        data.push_back(std::move(image));
    }
}

std::vector<MNIST_Image>& MNIST::get_data() {
    return data;
}

void MNIST::save_image(const MNIST_Image& image, const std::string& filename) {
        std::ofstream file(filename);

    if (!file)
        throw std::runtime_error("Unable to open output file");

    // Label
    file << static_cast<int>(image.label) << '\n';

    // 28x28 pixels
    for (int row = 0; row < 28; ++row) {
        for (int col = 0; col < 28; ++col) {
            file << image.pixels[row * 28 + col];

            if (col != 27)
                file << ' ';
        }

        file << '\n';
    }
}
