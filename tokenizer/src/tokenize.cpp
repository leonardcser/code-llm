#include "lib/dataloader.hpp"
#include "lib/io.hpp"
#include "lib/text.hpp"
#include "lib/tokenizer.hpp"
#include <filesystem>
#include <functional>
#include <iostream>
#include <sstream>
#include <yaml-cpp/yaml.h>

int main() {
    YAML::Node config = YAML::LoadFile("params.yaml");

    // Load configuration from params.yaml
    const unsigned int seed = config["data"]["seed"].as<unsigned int>();
    const std::string dataset_path =
        config["data"]["dataset_path"].as<std::string>();
    const std::string glob_pattern =
        config["data"]["glob_pattern"].as<std::string>();
    const std::string tok_file = config["data"]["tok_file"].as<std::string>();
    const std::string dataset_file =
        config["data"]["dataset_file"].as<std::string>();
    const size_t vocab_size = config["data"]["vocab_size"].as<size_t>();
    const size_t max_unique_words =
        config["data"]["max_unique_words"].as<size_t>();
    const std::string bos_token = config["data"]["bos_token"].as<std::string>();
    const std::string eos_token = config["data"]["eos_token"].as<std::string>();
    const std::string pad_token = config["data"]["pad_token"].as<std::string>();
    const std::string pattern = config["data"]["pattern"].as<std::string>();

    // Define special tokens (UNK is automatically added)
    const tokenizer::SpecialTokensInput special_tokens(
        bos_token, // BOS token (empty = unused)
        eos_token, // EOS token
        pad_token  // PAD token
    );

    std::filesystem::create_directories("out/tokenize");

    auto paths = dataloader::load_file_paths(dataset_path, glob_pattern);
    dataloader::set_seed(seed);
    dataloader::shuffle(paths);

    std::cout << "Total files: " << paths.size() << std::endl;
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

    std::string txt = io::concatenate_files(paths, trim_and_ascii);

    std::cout << "Training tokenizer..." << std::endl;

    auto tok = tokenizer::bpe_train(txt, vocab_size, pattern, special_tokens,
                                    max_unique_words, 5000);

    tokenizer::save(tok, tok_file);

    // Encode all data
    std::cout << "\nEncoding all data..." << std::endl;
    std::string all_txt_concat =
        io::concatenate_files(paths, trim_and_ascii, eos_token);
    auto all_tokens = tokenizer::encode(all_txt_concat, tok);
    std::cout << "Total tokens: " << all_tokens.size() << std::endl;

    // Save all tokens to single file
    std::cout << "\nSaving all tokens..." << std::endl;
    io::save_tokens(all_tokens, dataset_file);
    std::cout << "Saved to " << dataset_file << std::endl;

    std::cout << "\nTokenization complete!" << std::endl;
    return 0;
}
