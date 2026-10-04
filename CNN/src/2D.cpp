#include <cnn/data/2D.hpp>

SquareMatrix2D::SquareMatrix2D(std::size_t side, double value=0.0, std::size_t max_threads=MAX_THREADS) 
: SquareMatrix(side, value, max_threads) {
    dimension = D2;
}

SquareMatrix2D::SquareMatrix2D(std::size_t side, const std::vector<double>& data, std::size_t max_threads=MAX_THREADS)
: SquareMatrix(side, data, max_threads) {
    dimension = D2;

    if (data.size() % side != 0 || data.size() / side != side)
        throw std::invalid_argument("The data must be a side x side square matrix");
}

void SquareMatrix2D::conv(const std::vector<std::vector<double>>& kernel, std::size_t stride=1) {
    if (kernel.empty() || kernel.size() > side || kernel.size() % 2 == 0)
        throw std::invalid_argument("The kernel dimension must be odd, not empty "
            "and smaller than the matrix side");

    if (stride <= 0 || stride >= side)
        throw std::invalid_argument("The stride must be greater than zero and smaller than the matrix side");

    std::size_t k_middle = kernel.size() / 2;

    for (const auto& row : kernel)
            if (row.size() != kernel.size())
                throw std::invalid_argument("The kernel must be a square matrix");

    const std::size_t output_side = (side - kernel.size() + 2 * k_middle) / stride + 1;
    const std::size_t num_pixels = output_side * output_side;
    const std::size_t num_threads = std::min<std::size_t>(num_pixels, max_threads);
    const std::size_t pixels_per_thread = num_pixels / num_threads;
    const std::size_t extras = num_pixels % num_threads;

    std::vector<double> new_data(output_side * output_side, 0.0);
    auto thread_conv = [this, k_middle, stride, output_side, &kernel, &new_data](std::size_t start_pixel, std::size_t end_pixel) {
        for (std::size_t pixel = start_pixel; pixel < end_pixel; ++pixel) {
            const std::size_t out_row = pixel / output_side;
            const std::size_t out_col = pixel % output_side;
            const int cur_row = static_cast<int>(out_row * stride);
            const int cur_col = static_cast<int>(out_col * stride);

            double sum = 0.0;
            for (int row = cur_row - static_cast<int>(k_middle);
                 row <= cur_row + static_cast<int>(k_middle);
                 row++)
                for (int col = cur_col - static_cast<int>(k_middle);
                     col <= cur_col + static_cast<int>(k_middle);
                     col++) {
                        
                    if (row < 0 || row >= static_cast<int>(side) || col < 0 || col >= static_cast<int>(side))
                        continue;

                    const std::size_t kernel_row = static_cast<std::size_t>(row - cur_row + static_cast<int>(k_middle));
                    const std::size_t kernel_col = static_cast<std::size_t>(col - cur_col + static_cast<int>(k_middle));

                    sum += data[static_cast<std::size_t>(row) * side + static_cast<std::size_t>(col)] * kernel[kernel_row][kernel_col];
                }
            
            new_data[pixel] = sum;
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::size_t start = 0;
    for (std::size_t i = 0; i < num_threads; i++) {
        const std::size_t count = pixels_per_thread + (i < extras ? 1 : 0); // Apply more pixels to the first thread
        const std::size_t end = start + count;

        threads.emplace_back(thread_conv, start, end);
        start = end;
    }

    for (auto& thread : threads)
        thread.join();

    data = std::move(new_data);
    side = output_side;
}

void SquareMatrix2D::activate(ActivationType type=RELU) {
    switch (type) {
        case RELU:
            #pragma omp parallel for num_threads(max_threads)
            for (std::size_t i = 0; i < data.size(); ++i)
                data[i] = std::max(0.0, data[i]);
            break;
        
        default:
            throw std::invalid_argument("Unsupported activation function");
            break;
    }
}
