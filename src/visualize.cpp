#include "io.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <string>

int main() {
    auto tok = tokenizer::load("out/tok.bin");

    std::string example_path =
        std::string("data/py150/data/00/wikihouse/asset.py");
    std::string example = io::read_file(example_path);
    auto tokens = tokenizer::encode(example, tok.ranks, tok.pattern);

    std::cout << tokenizer::visualize(tokens, tok.ranks) << std::endl;
    std::cout << "Used a total of " << tokens.size() << " tokens" << std::endl;
    return 0;
}
