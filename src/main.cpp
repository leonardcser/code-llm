#include "dataloader.hpp"
#include "io.hpp"
#include "text.hpp"
#include "tokenizer.hpp"
#include <filesystem>
#include <functional>
#include <iostream>
#include <sstream>
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

    // vocab_size = number of BPE merges (excludes 256 byte tokens + special)
    size_t vocab_size = 50000;
    // size_t max_unique_words = 100000;
    size_t max_unique_words = 0;

    // Define special tokens (UNK is automatically added)
    tokenizer::SpecialTokensInput special_tokens(
        "",                 // No BOS token
        "<|endoftext|>",    // EOS token
        "<|pad|>"           // PAD token
    );

    auto tok = tokenizer::bpe_train(txt, vocab_size, pattern, special_tokens,
                                    max_unique_words, 5000);

    std::cout << "\nTokenizer created with special tokens:" << std::endl;
    std::cout << "  Vocab size: " << tok.ranks.size() << std::endl;
    std::cout << "  Special tokens: " << tok.special_tokens.size() << std::endl;
    std::cout << "  UNK: " << tok.unk_token_id << std::endl;
    std::cout << "  BOS: " << tok.bos_token_id << std::endl;
    std::cout << "  EOS: " << tok.eos_token_id << std::endl;
    std::cout << "  PAD: " << tok.pad_token_id << std::endl;

    std::filesystem::create_directories("out");
    tokenizer::save(tok, "out/tok.bin");

    // Encode train data
    std::cout << "\nEncoding training data..." << std::endl;
    std::string train_txt_concat =
        io::concatenate_files(train_paths, trim_and_ascii);
    auto train_tokens = tokenizer::encode(train_txt_concat, tok);
    std::cout << "Train tokens: " << train_tokens.size() << std::endl;

    // Encode val data
    std::cout << "Encoding validation data..." << std::endl;
    std::string val_txt_concat =
        io::concatenate_files(val_paths, trim_and_ascii);
    auto val_tokens = tokenizer::encode(val_txt_concat, tok);
    std::cout << "Val tokens: " << val_tokens.size() << std::endl;

    // Save train tokens
    std::cout << "\nSaving train tokens..." << std::endl;
    io::save_tokens(train_tokens, "out/train.bin");
    std::cout << "Saved to out/train.bin" << std::endl;

    // Save val tokens
    std::cout << "Saving val tokens..." << std::endl;
    io::save_tokens(val_tokens, "out/val.bin");
    std::cout << "Saved to out/val.bin" << std::endl;

    std::cout << "\nTraining complete!" << std::endl;
    return 0;
}
