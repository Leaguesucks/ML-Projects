#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

namespace dnn {

/**
 * @brief A 28 by 28 MNIST image and its digit label.
 *
 * Pixel values are stored in row-major order, normalized to [0, 1].
 */
struct MNIST_Image {
    std::vector<double> pixels;
    uint8_t label;
};

/**
 * @brief Read MNIST image and label files into memory.
 */
class MNIST {
    private:
        std::vector<MNIST_Image> data;

    public:
        /**
         * @brief Append matching images and labels from MNIST IDX files.
         * @param image_file Path to the image IDX file.
         * @param label_file Path to the label IDX file.
         * @note The existing data vector is not cleared before loading.
         */
        void load(const std::string& image_file, const std::string& label_file);
        
        /**
         * @brief Access the images loaded so far.
         * @return A mutable reference to the dataset.
         */
        std::vector<MNIST_Image>& get_data();

        /**
         * @brief Write an image as a plain-text label and 28 rows of pixels.
         * @param image Image to write.
         * @param filename Destination path.
         */
        void save_image(const MNIST_Image& image, const std::string& filename);

    private:
        /**
         * @param file Binary stream to read.
         * @return The next four bytes interpreted as a big-endian integer.
         */
        uint32_t read_uint32(std::ifstream& file);
};

} // namespace dnn
