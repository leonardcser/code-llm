#include "dataloader.hpp"
#include "io.hpp"
#include "tokenizer.hpp"
#include <filesystem>
#include <iostream>
#include <utility>

int main() {
    std::string train_txt = "train_paths.txt";
    std::string val_txt = "val_paths.txt";

    std::vector<std::string> train_paths, val_paths;
    size_t total_size;
    if (std::filesystem::exists(train_txt) &&
        std::filesystem::exists(val_txt)) {
        train_paths = io::load_txt(train_txt);
        val_paths = io::load_txt(val_txt);
        total_size = train_paths.size() + val_paths.size();
    } else {
        auto paths = dataloader::load_file_paths("data/py150/data", "*.py");
        dataloader::set_seed(42);
        dataloader::shuffle(paths);
        auto [t, v] = dataloader::split(paths, 0.7);
        train_paths = std::move(t);
        val_paths = std::move(v);
        total_size = paths.size();
        io::save_txt(train_paths, train_txt);
        io::save_txt(val_paths, val_txt);
    }

    std::cout << "Training data: " << train_paths.size() << std::endl;
    std::cout << "Validation data: " << val_paths.size() << std::endl;
    std::cout << "Total data: " << total_size << std::endl << std::endl;
    std::cout << "Reading files..." << std::endl;
    std::string txt = io::concatenate_files(train_paths);
    const std::string pattern =
        R"( ?[A-Za-z_][A-Za-z_.]*|%(?:\.\d+)?[sdifFeEgGxXoc%]|[0-9]{1,3}| ?[^ %_A-Za-z0-9]+[\r\n]*|%|\s+$|\s+(?=\s)|\s)";
    // std::string pattern =
    //     R"([sdmt]|ll|ve|re|[^\r\na-zA-Z0-9]?[a-zA-Z]+|[0-9]{1,3}|
    //     ?[^\sa-zA-Z0-9]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s)";

    std::cout << "Training tokenizer..." << std::endl;
    tokenizer::Ranks ranks;

    size_t vocab_size = 50000;
    // size_t max_unique_words = 100000;
    size_t max_unique_words = 0;
    tokenizer::bpe_train(txt, vocab_size, pattern, ranks, max_unique_words,
                         1000);

    std::filesystem::create_directories("out");
    tokenizer::Tokenizer tok{ranks, pattern};
    tokenizer::save(tok, "out/tok.bin");

    // Use first train path as example if available, else fallback
    std::string example_path =
        train_paths.empty()
            ? std::string("data/py150/data/00/wikihouse/asset.py")
            : train_paths[0];
    std::string example = io::read_file(example_path);
    auto tokens = tokenizer::encode(example, ranks, pattern);

    // std::cout << tokenizer::visualize(tokens, ranks) << std::endl;
    std::cout << "Used a total of " << tokens.size() << " tokens" << std::endl;

    return 0;
}
