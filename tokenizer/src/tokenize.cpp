#include "lib/dataloader.hpp"
#include "lib/io.hpp"
#include "lib/text.hpp"
#include "lib/tokenizer.hpp"
#include "lib/threading.hpp"
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <yaml-cpp/yaml.h>

// Parse human-readable size string (e.g., "100MB", "1GB", "256M")
size_t parse_size_string(const std::string &size_str) {
    if (size_str == "0") return 0;

    std::string num_part;
    std::string unit_part;

    for (char c : size_str) {
        if (std::isdigit(c) || c == '.') {
            num_part += c;
        } else if (std::isalpha(c)) {
            unit_part += std::toupper(c);
        }
    }

    if (num_part.empty()) {
        throw std::runtime_error("Invalid size string: " + size_str);
    }

    double value = std::stod(num_part);
    size_t multiplier = 1;

    if (unit_part == "K" || unit_part == "KB") {
        multiplier = 1024;
    } else if (unit_part == "M" || unit_part == "MB") {
        multiplier = 1024 * 1024;
    } else if (unit_part == "G" || unit_part == "GB") {
        multiplier = 1024 * 1024 * 1024;
    } else if (unit_part == "T" || unit_part == "TB") {
        multiplier = 1024ULL * 1024 * 1024 * 1024;
    } else if (!unit_part.empty()) {
        throw std::runtime_error("Unknown unit: " + unit_part);
    }

    return static_cast<size_t>(value * multiplier);
}

// Concatenate files in parallel up to a maximum size limit
std::string
concatenate_files_limited(const std::vector<std::string> &paths,
                          std::function<std::string(const std::string &)> transform,
                          size_t max_size) {
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t batch_size = std::max(size_t(1), std::min(size_t(10000), paths.size() / (num_threads * 2)));

    std::string result;
    result.reserve(std::min(max_size, size_t(100 * 1024 * 1024))); // Reserve 100MB initially
    size_t current_size = 0;

    std::cout << "Processing files in batches of " << batch_size << "..." << std::endl;

    for (size_t batch_start = 0; batch_start < paths.size() && (max_size == 0 || current_size < max_size); batch_start += batch_size) {
        size_t batch_end = std::min(batch_start + batch_size, paths.size());

        // Process batch in parallel
        threading::ThreadPool thread_pool(num_threads);
        std::vector<std::string> batch_results(batch_end - batch_start);

        for (size_t i = batch_start; i < batch_end; ++i) {
            size_t batch_idx = i - batch_start;
            thread_pool.enqueue([&, i, batch_idx]() {
                const auto &path = paths[i];
                if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) {
                    std::ifstream file(path, std::ios::in | std::ios::binary);
                    if (file.is_open()) {
                        std::string content(
                            (std::istreambuf_iterator<char>(file.rdbuf())),
                            std::istreambuf_iterator<char>());
                        batch_results[batch_idx] = transform(content);
                    }
                }
            });
        }

        thread_pool.wait();

        // Append batch results to result
        for (const auto &content : batch_results) {
            if (max_size > 0 && current_size >= max_size) {
                break;
            }

            if (max_size > 0 && current_size + content.size() > max_size) {
                size_t remaining = max_size - current_size;
                result += content.substr(0, remaining);
                current_size = max_size;
                break;
            }

            result += content;
            current_size += content.size();
        }

        std::cout << "Processed " << batch_end << "/" << paths.size()
                  << " files (" << current_size << "/" << max_size << " bytes)" << std::endl;

        if (max_size > 0 && current_size >= max_size) {
            break;
        }
    }

    return result;
}

// Encode files in parallel and save to chunked output files
void encode_files_chunked(
    const std::vector<std::string> &paths,
    const tokenizer::Tokenizer &tok,
    const std::string &dataset_dir,
    size_t chunk_size,
    const std::string &eos_token,
    std::function<std::string(const std::string &)> transform) {

    std::filesystem::create_directories(dataset_dir);

    // Process files in batches
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t batch_size = std::max(size_t(1), paths.size() / (num_threads * 4));

    std::vector<tokenizer::TokenId> current_chunk;
    current_chunk.reserve(chunk_size);
    size_t chunk_index = 0;
    size_t total_tokens = 0;

    std::cout << "Processing " << paths.size() << " files in batches of " << batch_size << "..." << std::endl;

    for (size_t batch_start = 0; batch_start < paths.size(); batch_start += batch_size) {
        size_t batch_end = std::min(batch_start + batch_size, paths.size());

        // Process batch in parallel
        threading::ThreadPool thread_pool(num_threads);
        std::vector<std::vector<tokenizer::TokenId>> batch_results(batch_end - batch_start);

        for (size_t i = batch_start; i < batch_end; ++i) {
            size_t batch_idx = i - batch_start;
            thread_pool.enqueue([&, i, batch_idx]() {
                const auto &path = paths[i];
                if (std::filesystem::exists(path) &&
                    std::filesystem::is_regular_file(path)) {
                    std::ifstream file(path, std::ios::in | std::ios::binary);
                    if (file.is_open()) {
                        std::string content(
                            (std::istreambuf_iterator<char>(file.rdbuf())),
                            std::istreambuf_iterator<char>());
                        std::string transformed = transform(content);
                        if (!eos_token.empty()) {
                            transformed += eos_token;
                        }
                        batch_results[batch_idx] = tokenizer::encode(transformed, tok);
                    }
                }
            });
        }

        thread_pool.wait();

        // Append batch results to current chunk
        for (const auto &tokens : batch_results) {
            for (const auto &token : tokens) {
                current_chunk.push_back(token);
                total_tokens++;

                // Save chunk if it reaches the limit
                if (current_chunk.size() >= chunk_size) {
                    std::ostringstream filename;
                    filename << dataset_dir << "/chunk_"
                             << std::setfill('0') << std::setw(6) << chunk_index << ".bin";

                    io::save_tokens(current_chunk, filename.str());
                    std::cout << "Saved chunk " << chunk_index << " ("
                              << current_chunk.size() << " tokens) to "
                              << filename.str() << std::endl;

                    current_chunk.clear();
                    current_chunk.reserve(chunk_size);
                    chunk_index++;
                }
            }
        }

        // Progress update
        std::cout << "Processed " << batch_end << "/" << paths.size()
                  << " files (" << total_tokens << " tokens)" << std::endl;
    }

    // Save remaining tokens in final chunk
    if (!current_chunk.empty()) {
        std::ostringstream filename;
        filename << dataset_dir << "/chunk_"
                 << std::setfill('0') << std::setw(6) << chunk_index << ".bin";

        io::save_tokens(current_chunk, filename.str());
        std::cout << "Saved final chunk " << chunk_index << " ("
                  << current_chunk.size() << " tokens) to "
                  << filename.str() << std::endl;
    }

    std::cout << "Total tokens: " << total_tokens << std::endl;
    std::cout << "Total chunks: " << (chunk_index + 1) << std::endl;
}

int main() {
    YAML::Node config = YAML::LoadFile("params.yaml");

    // Load configuration from params.yaml
    const unsigned int seed = config["tokenize"]["seed"].as<unsigned int>();
    const std::string dataset_path =
        config["tokenize"]["dataset_path"].as<std::string>();
    const std::string glob_pattern =
        config["tokenize"]["glob_pattern"].as<std::string>();
    const std::string tok_file =
        config["tokenize"]["tok_file"].as<std::string>();
    const std::string dataset_dir =
        config["tokenize"]["dataset_dir"].as<std::string>();
    const std::string max_train_size_str =
        config["tokenize"]["max_train_size"].as<std::string>();
    const std::string chunk_size_str =
        config["tokenize"]["chunk_size"].as<std::string>();
    const size_t vocab_size = config["tokenize"]["vocab_size"].as<size_t>();
    const size_t max_unique_words =
        config["tokenize"]["max_unique_words"].as<size_t>();
    const std::string bos_token =
        config["tokenize"]["bos_token"].as<std::string>();
    const std::string eos_token =
        config["tokenize"]["eos_token"].as<std::string>();
    const std::string pad_token =
        config["tokenize"]["pad_token"].as<std::string>();
    const std::string cursor_token =
        config["tokenize"]["cursor_token"].as<std::string>();
    const std::string edit_start_token =
        config["tokenize"]["edit_start_token"].as<std::string>();
    const std::string edit_end_token =
        config["tokenize"]["edit_end_token"].as<std::string>();
    const std::string pattern = config["tokenize"]["pattern"].as<std::string>();

    // Parse size parameters
    const size_t max_train_size = parse_size_string(max_train_size_str);
    const size_t chunk_size = parse_size_string(chunk_size_str);

    // Define special tokens (UNK is automatically added)
    const tokenizer::SpecialTokensInput special_tokens(
        bos_token,          // BOS token (empty = unused)
        eos_token,          // EOS token
        pad_token,          // PAD token
        cursor_token,       // CURSOR token
        edit_start_token,   // EDIT_START token
        edit_end_token      // EDIT_END token
    );

    std::filesystem::create_directories("out/tokenize");

    auto paths = dataloader::load_file_paths(dataset_path, glob_pattern);
    dataloader::set_seed(seed);
    dataloader::shuffle(paths);

    std::cout << "Total files: " << paths.size() << std::endl;

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

    // Load data for training tokenizer (with size limit)
    std::cout << "Loading training data";
    if (max_train_size > 0) {
        std::cout << " (limited to " << max_train_size_str << ")";
    }
    std::cout << "..." << std::endl;

    std::string txt = concatenate_files_limited(paths, trim_and_ascii, max_train_size);
    std::cout << "Loaded " << txt.size() << " bytes for training" << std::endl;

    std::cout << "\nTraining tokenizer..." << std::endl;

    auto tok = tokenizer::bpe_train(txt, vocab_size, pattern, special_tokens,
                                    max_unique_words, 5000);

    tokenizer::save(tok, tok_file);
    std::cout << "Tokenizer saved to " << tok_file << std::endl;

    // Encode all data in chunks
    std::cout << "\nEncoding all data to chunks..." << std::endl;
    encode_files_chunked(paths, tok, dataset_dir, chunk_size, eos_token, trim_and_ascii);

    std::cout << "\nTokenization complete!" << std::endl;
    return 0;
}
