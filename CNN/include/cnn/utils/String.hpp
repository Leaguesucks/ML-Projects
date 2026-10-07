#pragma once

#include <string>
#include <cstdlib>
#include <ctime>

namespace utils {
/**
 * @brief Generate a random string
 * @param length The length of the string. If zero then use the default length of 25
 * @param char_set The character sets that form the random string. If empty then use the default character set
 */
std::string random_str(std::size_t length=0, const std::string& char_set="");
}