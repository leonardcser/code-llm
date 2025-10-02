#include "lib/io.hpp"
#include "lib/tokenizer.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

int main() {
    YAML::Node config = YAML::LoadFile("params.yaml");

    auto tok = tokenizer::load(config["data"]["tok_file"].as<std::string>());

    std::vector<std::string> val_paths =
        io::load_txt(config["data"]["val_paths_file"].as<std::string>());
    for (const auto &path : val_paths) {
        std::string content = io::read_file(path);
        auto tokens = tokenizer::encode(content, tok);

        std::cout << "File: " << path << std::endl << std::endl;
        std::cout << tokenizer::visualize(tokens, tok) << std::endl;
        std::cout << "Used a total of " << tokens.size() << " tokens"
                  << std::endl
                  << std::endl;
        std::cout << "Press Enter to continue to next file..." << std::endl;
        std::string input;
        std::getline(std::cin, input);
    }
    std::cout << "Processed all validation files." << std::endl;
    return 0;
}
