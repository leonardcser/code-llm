#include "io.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

namespace io {

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

bool matches(const std::string &str, const std::string &pat, size_t s, size_t p) {
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

std::string concatenate_files(const std::vector<std::string>& paths) {
    std::string result;
    size_t total_size = 0;
    // First pass: calculate total size
    for (const auto& path : paths) {
        if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) {
            total_size += std::filesystem::file_size(path);
        }
    }
    result.reserve(total_size);
    // Second pass: read and append contents
    for (const auto& path : paths) {
        if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) {
            std::ifstream file(path, std::ios::in | std::ios::binary);
            if (file.is_open()) {
                std::string content((std::istreambuf_iterator<char>(file.rdbuf())),
                                    std::istreambuf_iterator<char>());
                result += std::move(content);
            }
        }
    }
    return result;
}

} // namespace io
