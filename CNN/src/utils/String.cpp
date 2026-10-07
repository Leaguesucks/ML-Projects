#include <cnn/utils/String.hpp>

namespace utils {
std::string random_str(std::size_t length=0, const std::string& char_set="") {
    std::size_t len;
    std::string ch_set;

    len = (length == 0) ? 25 : length;
    if (char_set.empty())
        ch_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_+=";
    else 
        ch_set = char_set;

    std::string result;
    result.resize(len);
    srand(static_cast<unsigned int>(time(0)));

    for (std::size_t i = 0; i < len; ++i)
        result[i] = ch_set[rand() % ch_set.length()];

    return result;
}
}