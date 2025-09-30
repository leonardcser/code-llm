#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <string>
#include <vector>
#include <tuple>

namespace dataloader {

std::vector<std::string> load_file_paths(const std::string& relative_folder, const std::string& glob_pattern = "*", size_t max_files = 0);

void shuffle(std::vector<std::string>& paths);

std::tuple<std::vector<std::string>, std::vector<std::string>>
split(const std::vector<std::string>& paths, double train_ratio = 0.7);

void set_seed(unsigned int seed);

}

#endif