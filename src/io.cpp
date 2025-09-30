#include "io.hpp"
#include "threading.hpp"
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>
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

bool matches(const std::string &str, const std::string &pat, size_t s,
             size_t p) {
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

std::string
concatenate_files(const std::vector<std::string> &paths,
                  std::function<std::string(const std::string &)> transform) {
    std::string result;
    size_t total_size = 0;
    // First pass: calculate total size
    for (const auto &path : paths) {
        if (std::filesystem::exists(path) &&
            std::filesystem::is_regular_file(path)) {
            total_size += std::filesystem::file_size(path);
        }
    }
    result.reserve(total_size);

    // Parallel reading using thread pool
    const size_t num_threads = std::thread::hardware_concurrency();
    threading::ThreadPool thread_pool(num_threads);
    const size_t chunk_size = (paths.size() + num_threads - 1) / num_threads;

    std::vector<std::string> thread_results(num_threads);
    std::vector<size_t> chunk_sizes(num_threads, 0);

    // Calculate chunk sizes
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, paths.size());
        for (size_t i = start; i < end; ++i) {
            if (std::filesystem::exists(paths[i]) &&
                std::filesystem::is_regular_file(paths[i])) {
                chunk_sizes[t] += std::filesystem::file_size(paths[i]);
            }
        }
        thread_results[t].reserve(chunk_sizes[t]);
    }

    // Enqueue tasks
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, paths.size());
        thread_pool.enqueue([&, t, start, end]() {
            for (size_t i = start; i < end; ++i) {
                const auto &path = paths[i];
                if (std::filesystem::exists(path) &&
                    std::filesystem::is_regular_file(path)) {
                    std::ifstream file(path, std::ios::in | std::ios::binary);
                    if (file.is_open()) {
                        std::string content(
                            (std::istreambuf_iterator<char>(file.rdbuf())),
                            std::istreambuf_iterator<char>());
                        thread_results[t] += transform(content);
                    }
                }
            }
        });
    }

    thread_pool.wait();

    // Append thread results to final result
    for (auto &chunk : thread_results) {
        result += std::move(chunk);
    }

    return result;
}

void save_txt(const std::vector<std::string> &data,
              const std::string &filename) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open " + filename + " for writing");
    }
    for (const auto &s : data) {
        file << s << '\n';
    }
    if (!file) {
        throw std::runtime_error("Failed to write to " + filename);
    }
}

std::vector<std::string> load_txt(const std::string &filename) {
    std::vector<std::string> data;
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open " + filename + " for reading");
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            data.push_back(std::move(line));
        }
    }
    return data;
}

} // namespace io
