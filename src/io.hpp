#ifndef IO_HPP
#define IO_HPP

#include "tokenizer.hpp"
#include <functional>
#include <string>
#include <vector>

namespace io {

std::string read_file(const std::string &relative_path);

bool matches(const std::string &str, const std::string &pat, size_t s = 0,
             size_t p = 0);

bool matches_glob(const std::string &str, const std::string &pattern);

std::string concatenate_files(
    const std::vector<std::string> &paths,
    std::function<std::string(const std::string &)> transform =
        [](const std::string &s) { return s; });

void save_txt(const std::vector<std::string> &data,
              const std::string &filename);
std::vector<std::string> load_txt(const std::string &filename);

void save_tokens(const std::vector<tokenizer::TokenId> &tokens,
                 const std::string &filename);
std::vector<tokenizer::TokenId> load_tokens(const std::string &filename);

} // namespace io

#endif
