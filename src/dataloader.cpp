#include "dataloader.hpp"
#include "io.hpp"
#include <algorithm>
#include <filesystem>
#include <random>
#include <stdexcept>

namespace dataloader {

std::mt19937 rng(42);

std::vector<std::string> load_file_paths(const std::string &relative_folder,
                                         const std::string &glob_pattern,
                                         size_t max_files) {
    std::filesystem::path source_dir =
        std::filesystem::path(__FILE__).parent_path().parent_path();
    std::filesystem::path dir_path = source_dir / relative_folder;
    if (!std::filesystem::exists(dir_path) ||
        !std::filesystem::is_directory(dir_path)) {
        throw std::runtime_error("Failed to open directory: " +
                                 dir_path.string());
    }
    std::vector<std::string> paths;
    size_t count = 0;
    try {
        for (const auto &entry :
             std::filesystem::recursive_directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (io::matches_glob(filename, glob_pattern)) {
                    paths.push_back(entry.path().string());
                    ++count;
                    if (max_files > 0 && count >= max_files) {
                        break;
                    }
                }
            }
        }
    } catch (const std::exception &e) {
        // Ignore errors
    }
    return paths;
}

void shuffle(std::vector<std::string> &paths) {
    if (paths.empty()) return;
    std::shuffle(paths.begin(), paths.end(), rng);
}

std::tuple<std::vector<std::string>, std::vector<std::string>>
split(const std::vector<std::string> &paths, double train_ratio) {
    size_t total = paths.size();
    if (train_ratio > 1.0 || train_ratio < 0.0) {
        throw std::invalid_argument("train_ratio must be between 0.0 and 1.0");
    }
    size_t train_size = static_cast<size_t>(total * train_ratio);

    std::vector<std::string> train(paths.begin(), paths.begin() + train_size);
    auto test_start = paths.begin() + train_size;
    std::vector<std::string> test(test_start, paths.end());

    return {std::move(train), std::move(test)};
}

void set_seed(unsigned int seed) {
    rng.seed(seed);
}

} // namespace dataloader
