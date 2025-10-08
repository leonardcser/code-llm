#include "lib/io.hpp"
#include "lib/tokenizer.hpp"
#include <algorithm>
#include <iostream>
#include <yaml-cpp/yaml.h>

int main(int argc, char *argv[]) {
    YAML::Node config = YAML::LoadFile("params.yaml");

    // Load tokenizer
    // Parse command line arguments
    size_t max_tokens = 100; // default
    if (argc > 1) {
        max_tokens = std::stoul(argv[1]);
    }

    const std::string tok_file =
        config["tokenize"]["tok_file"].as<std::string>();
    const std::string val_file = config["data"]["val_file"].as<std::string>();

    // Load tokenizer
    std::cout << "Loading tokenizer from " << tok_file << "..." << std::endl;
    auto tok = tokenizer::load(tok_file);
    std::cout << "Tokenizer loaded (vocab size: " << tok.ranks.size() << ")"
              << std::endl;

    // Load validation tokens
    std::cout << "Loading validation tokens from " << val_file << "..."
              << std::endl;
    auto val_tokens = io::load_tokens(val_file);
    std::cout << "Loaded " << val_tokens.size() << " tokens" << std::endl;

    // Determine how many tokens to decode
    size_t tokens_to_decode = std::min(max_tokens, val_tokens.size());
    std::cout << "\nDecoding " << tokens_to_decode << " tokens..." << std::endl;

    // Extract subset of tokens
    std::vector<tokenizer::TokenId> subset(
        val_tokens.begin(), val_tokens.begin() + tokens_to_decode);

    // Decode tokens
    std::string decoded = tokenizer::decode(subset, tok);

    // Output decoded text
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "DECODED TEXT:" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << decoded << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}
