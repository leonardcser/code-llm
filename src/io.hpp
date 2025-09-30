#ifndef IO_HPP
#define IO_HPP

#include <string>

namespace io {

std::string read_file(const std::string &relative_path);

bool matches(const std::string &str, const std::string &pat, size_t s = 0, size_t p = 0);

bool matches_glob(const std::string &str, const std::string &pattern);

std::string concatenate_files(const std::vector<std::string>& paths);

}

#endif