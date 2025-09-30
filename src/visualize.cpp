#include "io.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <string>
#include <vector>

int main() {
    auto tok = tokenizer::load("out/tok.bin");

    std::vector<std::string> val_paths = io::load_txt("val_paths.txt");
    for (const auto &path : val_paths) {
        std::string content = io::read_file(path);
        auto tokens = tokenizer::encode(content, tok.ranks, tok.pattern);

        std::cout << "File: " << path << std::endl << std::endl;
        std::cout << tokenizer::visualize(tokens, tok.ranks) << std::endl;
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
