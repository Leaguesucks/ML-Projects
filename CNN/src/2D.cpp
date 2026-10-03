#include <cnn/data/2D.hpp>

SquareMatrix2D::SquareMatrix2D(std::size_t size, double value) 
: SquareMatrix(size, value) {
    dimension = D2;
}

SquareMatrix2D::SquareMatrix2D(std::size_t size, const std::vector<double>& data)
: SquareMatrix(size, data) {
    dimension = D2;

    if (data.size() % size != 0 || data.size() / size != size)
        throw std::invalid_argument("The data must be a size x size square matrix");
}

void SquareMatrix2D::conv(const std::vector<std::vector<double>>& kernel, std::size_t stride=1) {
    if (kernel.empty() || kernel.size() > side || kernel.size() % 2 == 0)
        throw std::invalid_argument("The kernel dimension must be odd, not empty "
            "and smaller than the matrix size");

    if (stride <= 0 || stride >= side)
        throw std::invalid_argument("The stride must be greater than zero and smaller than the matrix size");

    std::size_t k_middle = kernel.size() / 2;

    for (const auto& row : kernel)
            if (row.size() != kernel.size())
                throw std::invalid_argument("The kernel must be a square matrix");

    const int num_pixels = static_cast<int>(std::pow(side / MAX_THREADS, 2));
    const int num_threads = std::min(num_pixels, static_cast<int>(MAX_THREADS));
    const int pixels_per_thread = num_pixels / num_threads;

    std::vector<double> new_data(std::pow(side, 2), 0.0);
    auto thread_conv = [this, k_middle, &kernel, &new_data](std::vector<int> pixels) {
        for (int pixel : pixels) {
        
            int curRow = pixel / side;
            int curCol = pixel % side;

            for (int row = curRow - k_middle; row <= curRow + k_middle; row++)
                for (int col = curCol - k_middle; col <= curCol + k_middle; col++)
                    if (row >= 0 && row < side && col >= 0 && col < side)
                        new_data[pixel] += data[row * side + col] * kernel[row - (curRow - k_middle)][col - (curCol - k_middle)];
        }
    };

    std::vector<std::thread> threads;
    std::vector<int> pixels;

    pixels.reserve(pixels_per_thread);
    for (int row = 0; row < side; row += static_cast<int>(stride))
        for (int col = 0; col < side; col += static_cast<int>(stride)) {
            if (pixels.size() >= pixels_per_thread) {
                threads.emplace_back(thread_conv, pixels);
                pixels.clear();
            }

            pixels.push_back(row * side + col);
        }

    for (auto& thread : threads)
        thread.join();
    
    data = std::move(new_data);
}
