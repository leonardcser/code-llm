#include "tokenizer.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

std::string read_file(const std::string &relative_path) {
    std::filesystem::path source_dir = std::filesystem::path(__FILE__)
                                           .parent_path()
                                           .parent_path(); // go up from src/
    std::filesystem::path full_path = source_dir / relative_path;

    std::ifstream file(full_path, std::ios::in | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + full_path.string());
    }

    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

bool matches(const std::string &str, const std::string &pat, size_t s = 0,
             size_t p = 0) {
    if (p == pat.length()) {
        return s == str.length();
    }
    if (s == str.length()) {
        for (size_t i = p; i < pat.length(); ++i) {
            if (pat[i] != '*') return false;
        }
        return true;
    }
    char pc = pat[p];
    if (pc == '*') {
        if (matches(str, pat, s, p + 1)) return true;
        if (s < str.length()) {
            return matches(str, pat, s + 1, p);
        }
        return false;
    } else if (pc == '?' || pc == str[s]) {
        return matches(str, pat, s + 1, p + 1);
    }
    return false;
}

bool matches_glob(const std::string &str, const std::string &pattern) {
    return matches(str, pattern);
}

std::string read_dir(const std::string &relative_dir,
                     const std::string &glob_pattern, size_t max_files = 0) {
    std::filesystem::path source_dir =
        std::filesystem::path(__FILE__).parent_path().parent_path();
    std::filesystem::path dir_path = source_dir / relative_dir;
    if (!std::filesystem::exists(dir_path) ||
        !std::filesystem::is_directory(dir_path)) {
        throw std::runtime_error("Failed to open directory: " +
                                 dir_path.string());
    }
    std::string result;
    size_t total_size = 0;
    size_t file_count = 0;
    try {
        for (const auto &entry :
             std::filesystem::recursive_directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (matches_glob(filename, glob_pattern)) {
                    if (max_files > 0 && file_count >= max_files) break;
                    total_size += std::filesystem::file_size(entry.path());
                    ++file_count;
                }
            }
        }
    } catch (const std::exception &e) {
    }
    result.reserve(total_size);
    file_count = 0;
    try {
        for (const auto &entry :
             std::filesystem::recursive_directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (matches_glob(filename, glob_pattern)) {
                    if (max_files > 0 && file_count >= max_files) break;
                    std::ifstream file(entry.path(),
                                       std::ios::in | std::ios::binary);
                    if (file.is_open()) {
                        std::string content(
                            (std::istreambuf_iterator<char>(file.rdbuf())),
                            std::istreambuf_iterator<char>());
                        result += std::move(content);
                    }
                    ++file_count;
                }
            }
        }
    } catch (const std::exception &e) {
    }
    return result;
}

int main() {
    std::string txt = read_dir("data/py150/data", "*.py", 10000);
    // std::string txt = read_file("data/shakespeare/shakespeare.txt");
    const std::string pattern =
        R"( ?[A-Za-z_][A-Za-z_.]*|[0-9]{1,3}| ?[^ _A-Za-z0-9]+[\r\n]*|\s+$|\s+(?!\S)|\s)";
    // std::string pattern =
    //     R"([sdmt]|ll|ve|re|[^\r\na-zA-Z0-9]?[a-zA-Z]+|[0-9]{1,3}|
    //     ?[^\sa-zA-Z0-9]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s)";

    tokenizer::Ranks ranks;
    tokenizer::bpe_train(txt, 50000, pattern, ranks);

    tokenizer::save(ranks, "out/tok.bin");

    std::string example = read_file("data/py150/data/00/wikihouse/asset.py");
    auto tokens = tokenizer::encode(example, ranks, pattern);

    // std::cout << tokenizer::visualize(tokens, ranks) << std::endl;
    std::cout << "Used a total of " << tokens.size() << " tokens" << std::endl;

    return 0;
}
