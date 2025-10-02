#include "lib/dataloader.hpp"
#include "lib/io.hpp"
#include "lib/text.hpp"
#include "lib/tokenizer.hpp"
#include <filesystem>
#include <functional>
#include <iostream>
#include <sstream>
#include <utility>

int main() {
    const unsigned int SEED = 42;
    const double SPLIT = 0.7;
    const std::string DATASET_PATH = "data/py150/data";
    const std::string TRAIN_TXT = "out/tokenize/train_paths.txt";
    const std::string VAL_TXT = "out/tokenize/val_paths.txt";
    const std::string TOK_BIN = "out/tokenize/tok.bin";
    const std::string TRAIN_BIN = "out/tokenize/train.bin";
    const std::string VAL_BIN = "out/tokenize/val.bin";
    const size_t VOCAB_SIZE = 20000;
    const size_t MAX_UNIQUE_WORDS = 0;

    // Define special tokens (UNK is automatically added)
    const tokenizer::SpecialTokensInput special_tokens(
        "",              // BOS token (empty = unused)
        "<|endoftext|>", // EOS token
        "<|pad|>"        // PAD token
    );

    std::filesystem::create_directories("out/tokenize");

    auto paths = dataloader::load_file_paths(DATASET_PATH, "*.py");
    dataloader::set_seed(SEED);
    dataloader::shuffle(paths);
    auto [t, v] = dataloader::split(paths, SPLIT);
    io::save_txt(t, TRAIN_TXT);
    io::save_txt(v, VAL_TXT);
    std::vector<std::string> train_paths = std::move(t);
    std::vector<std::string> val_paths = std::move(v);

    size_t total_size = paths.size();
    std::cout << "Training data: " << train_paths.size() << std::endl;
    std::cout << "Validation data: " << val_paths.size() << std::endl;
    std::cout << "Total data: " << total_size << std::endl << std::endl;
    std::cout << "Reading files..." << std::endl;

    auto trim_and_ascii = [](const std::string &input) -> std::string {
        std::istringstream iss(input);
        std::ostringstream oss;
        std::string line;
        bool skipping = true;
        while (std::getline(iss, line)) {
            // Trim leading whitespace
            size_t start = line.find_first_not_of(" \t");
            if (start == std::string::npos) {
                // Empty line, skip if in skipping mode
                if (skipping) continue;
                oss << line << '\n';
                continue;
            }
            std::string trimmed = line.substr(start);
            if (skipping && trimmed.empty()) continue;
            if (skipping && trimmed[0] == '#') {
                // Skip full comment line at start
                continue;
            }
            skipping = false;
            // For non-skipped lines, remove inline comments if any
            size_t pos = line.find('#');
            if (pos != std::string::npos &&
                line.find_first_not_of(" \t", 0) < pos) {
                // Only remove if # is after non-whitespace
                line.erase(pos);
            }
            oss << line << '\n';
        }
        return to_ascii(oss.str());
    };

    std::string txt = io::concatenate_files(train_paths, trim_and_ascii);
    const std::string pattern =
        R"( ?[A-Za-z_(][A-Za-z_.]*|%(?:\.\d+)?[sdifFeEgGxXoc%]|[0-9]{1,3}| ?[^ %_A-Za-z0-9]+(?: ")?[\r\n]*|%|\s+$|\s+(?=\s)|\s)";
    // std::string pattern =
    //     R"([sdmt]|ll|ve|re|[^\r\na-zA-Z0-9]?[a-zA-Z]+|[0-9]{1,3}|
    //     ?[^\sa-zA-Z0-9]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s)";

    std::cout << "Training tokenizer..." << std::endl;

    auto tok = tokenizer::bpe_train(txt, VOCAB_SIZE, pattern, special_tokens,
                                    MAX_UNIQUE_WORDS, 5000);

    tokenizer::save(tok, TOK_BIN);

    // Encode train data
    std::cout << "\nEncoding training data..." << std::endl;
    std::string train_txt_concat =
        io::concatenate_files(train_paths, trim_and_ascii, "<|endoftext|>");
    auto train_tokens = tokenizer::encode(train_txt_concat, tok);
    std::cout << "Train tokens: " << train_tokens.size() << std::endl;

    // Encode val data
    std::cout << "Encoding validation data..." << std::endl;
    std::string val_txt_concat =
        io::concatenate_files(val_paths, trim_and_ascii, "<|endoftext|>");
    auto val_tokens = tokenizer::encode(val_txt_concat, tok);
    std::cout << "Val tokens: " << val_tokens.size() << std::endl;

    // Save train tokens
    std::cout << "\nSaving train tokens..." << std::endl;
    io::save_tokens(train_tokens, TRAIN_BIN);
    std::cout << "Saved to out/train.bin" << std::endl;

    // Save val tokens
    std::cout << "Saving val tokens..." << std::endl;
    io::save_tokens(val_tokens, VAL_BIN);
    std::cout << "Saved to out/val.bin" << std::endl;

    std::cout << "\nTraining complete!" << std::endl;
    return 0;
}
