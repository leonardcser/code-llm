#include "lib/tokenizer.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <text to tokenize...>"
                  << std::endl;
        std::cerr << "Example: " << argv[0] << " \"Hello world!\"" << std::endl;
        return 1;
    }

    // Load tokenizer from params.yaml
    YAML::Node config = YAML::LoadFile("params.yaml");
    auto tok = tokenizer::load(config["data"]["tok_file"].as<std::string>());

    // Concatenate all arguments (skip program name at argv[0])
    std::string input;
    for (int i = 1; i < argc; ++i) {
        if (i > 1) input += " ";
        input += argv[i];
    }

    // Tokenize the input
    auto tokens = tokenizer::encode(input, tok);

    // Print visualization
    std::cout << tokenizer::visualize(tokens, tok) << std::endl;
    std::cout << "\nTokens: " << tokens.size() << std::endl;

    return 0;
}
