#include "tokenizer.hpp"
#include "text.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <regex>
#include <unordered_map>

using TokenId = uint32_t;
using Word = std::vector<TokenId>;
using WordList = std::vector<Word>;

using PairCount = std::unordered_map<uint64_t, size_t>;
using Clock = std::chrono::steady_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

// Helper function to split text into words and convert to byte tokens
WordList tokenize_to_words(const std::string &text,
                           const std::string &pattern_str) {
    WordList words;

    auto push_char = [](unsigned char c) -> TokenId {
        return static_cast<TokenId>(c);
    };

    if (!pattern_str.empty()) {
        const std::regex pattern(pattern_str, std::regex_constants::optimize);
        words.reserve(text.length() / 5); // Rough estimate for word count

        std::sregex_iterator iter(text.begin(), text.end(), pattern);
        std::sregex_iterator end;

        for (; iter != end; ++iter) {
            const std::string &word = iter->str();
            Word token_word;
            token_word.reserve(word.length());

            for (unsigned char c : word) {
                token_word.push_back(push_char(c));
            }
            words.push_back(std::move(token_word));
        }

        const bool has_multi_char =
            std::any_of(words.begin(), words.end(),
                        [](const Word &word) { return word.size() > 1; });

        if (has_multi_char) return words;
    }

    WordList fallback_words;
    fallback_words.reserve(text.length() / 6 + 1);

    Word current_word;
    current_word.reserve(16);

    auto flush_word = [&]() {
        if (!current_word.empty()) {
            fallback_words.emplace_back();
            fallback_words.back().swap(current_word);
        }
    };

    for (unsigned char raw_c : text) {
        if (std::isspace(static_cast<unsigned char>(raw_c))) {
            flush_word();
            Word whitespace_token(1, push_char(raw_c));
            fallback_words.push_back(std::move(whitespace_token));
        } else {
            current_word.push_back(push_char(raw_c));
        }
    }
    flush_word();

    if (!fallback_words.empty()) return fallback_words;

    WordList single_word(1);
    Word &all_tokens = single_word.front();
    all_tokens.reserve(text.size());
    for (unsigned char raw_c : text) {
        all_tokens.push_back(push_char(raw_c));
    }
    return single_word;
}

// Helper function to merge pairs in word list
void merge_pairs(WordList &words, TokenId first_token, TokenId second_token,
                 TokenId new_token) {
    for (auto &word : words) {
        const size_t size = word.size();
        if (size < 2) continue;

        size_t write = 0;
        size_t read = 0;

        while (read < size) {
            if (read + 1 < size && word[read] == first_token &&
                word[read + 1] == second_token) {
                word[write++] = new_token;
                read += 2;
            } else {
                word[write++] = word[read++];
            }
        }

        if (write < size) word.resize(write);
    }
}

constexpr uint64_t encode_pair(TokenId first, TokenId second) {
    return (static_cast<uint64_t>(first) << 32) | static_cast<uint64_t>(second);
}

constexpr TokenId first_from_pair(uint64_t key) {
    return static_cast<TokenId>(key >> 32);
}

constexpr TokenId second_from_pair(uint64_t key) {
    return static_cast<TokenId>(key & 0xffffffffu);
}

void tokenizer::bpe_train(std::string &text, size_t vocab_size,
                          const std::string &pattern, Ranks &ranks) {
    if (vocab_size < 256) {
        throw std::invalid_argument("vocab_size must be at least 256");
    }

    const auto total_start = Clock::now();

    const auto normalize_start = Clock::now();
    text = to_ascii(text);
    const DurationMs normalize_time =
        std::chrono::duration_cast<DurationMs>(Clock::now() - normalize_start);

    ranks.reserve(vocab_size);

    std::vector<std::string> vocab;
    vocab.reserve(vocab_size);

    for (int i = 0; i < 256; ++i) {
        std::string byteToken = std::string(1, static_cast<char>(i));
        ranks[byteToken] = i;
        vocab.push_back(byteToken);
    }

    DurationMs tokenize_time = DurationMs::zero();
    const auto tokenize_start = Clock::now();
    // Split text into words using provided regex pattern
    WordList words = tokenize_to_words(text, pattern);
    tokenize_time += std::chrono::duration_cast<DurationMs>(Clock::now() -
                                                            tokenize_start);

    // Main BPE loop
    DurationMs pair_count_time = DurationMs::zero();
    DurationMs find_best_time = DurationMs::zero();
    DurationMs merge_time = DurationMs::zero();
    size_t iteration_count = 0;
    while (ranks.size() < vocab_size) {
        ++iteration_count;
        // Count pair frequencies
        const auto stats_start = Clock::now();
        PairCount stats;
        stats.reserve(10000); // Reasonable initial capacity

        for (const auto &word : words) {
            for (size_t i = 0; i < word.size() - 1; ++i) {
                ++stats[encode_pair(word[i], word[i + 1])];
            }
        }
        pair_count_time += std::chrono::duration_cast<DurationMs>(
            Clock::now() - stats_start);

        if (stats.empty()) break;

        // Find most frequent pair
        const auto find_start = Clock::now();
        auto max_iter = std::max_element(
            stats.begin(), stats.end(), [](const auto &a, const auto &b) {
                if (a.second != b.second) return a.second < b.second;
                return a.first < b.first; // Tie-breaking for determinism
            });
        find_best_time += std::chrono::duration_cast<DurationMs>(Clock::now() -
                                                                 find_start);

        if (max_iter->second <= 1) break; // No more meaningful merges
        // Extract the most common pair
        const uint64_t pair_key = max_iter->first;
        const TokenId first_token = first_from_pair(pair_key);
        const TokenId second_token = second_from_pair(pair_key);
        std::string new_token = vocab[first_token] + vocab[second_token];

        // Add new token to ranks
        TokenId token_id = static_cast<TokenId>(vocab.size());
        ranks[new_token] = static_cast<int>(token_id);
        vocab.push_back(new_token);

        // Merge pairs in all words
        const auto merge_start = Clock::now();
        merge_pairs(words, first_token, second_token, token_id);
        merge_time += std::chrono::duration_cast<DurationMs>(Clock::now() -
                                                             merge_start);

        // Optional: Progress reporting
        if (ranks.size() % 100 == 0) {
            std::cout << "Vocab size: " << ranks.size() << "/" << vocab_size
                      << ", merged: " << vocab[first_token] << " + "
                      << vocab[second_token] << " (freq: " << max_iter->second
                      << ")" << std::endl;
        }
    }

    const DurationMs total_time =
        std::chrono::duration_cast<DurationMs>(Clock::now() - total_start);

    std::cout << "BPE training completed. Final vocabulary size: "
              << ranks.size() << std::endl;
    std::cout << "[bpe_train] normalize: " << normalize_time.count()
              << " ms" << std::endl;
    std::cout << "[bpe_train] tokenize_to_words: " << tokenize_time.count()
              << " ms" << std::endl;
    std::cout << "[bpe_train] pair counting: " << pair_count_time.count()
              << " ms" << std::endl;
    std::cout << "[bpe_train] find best pair: " << find_best_time.count()
              << " ms" << std::endl;
    std::cout << "[bpe_train] merge pairs: " << merge_time.count() << " ms"
              << std::endl;
    std::cout << "[bpe_train] iterations: " << iteration_count << std::endl;
    std::cout << "[bpe_train] total: " << total_time.count() << " ms"
              << std::endl;
}

std::string base64_encode(const std::string &input) {
    static const char chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    int val = 0, valb = -6;
    for (unsigned char c : input) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            result.push_back(chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) result.push_back(chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (result.size() % 4)
        result.push_back('=');
    return result;
}

// Check if string is valid UTF-8
bool is_valid_utf8(const std::string &str) {
    for (size_t i = 0; i < str.length();) {
        unsigned char c = str[i];
        int len = 1;

        if (c >= 0x80) {
            if ((c & 0xE0) == 0xC0)
                len = 2;
            else if ((c & 0xF0) == 0xE0)
                len = 3;
            else if ((c & 0xF8) == 0xF0)
                len = 4;
            else
                return false; // Invalid UTF-8 start byte

            // Check continuation bytes
            for (int j = 1; j < len; j++) {
                if (i + j >= str.length() || (str[i + j] & 0xC0) != 0x80) {
                    return false;
                }
            }
        }
        i += len;
    }
    return true;
}

void tokenizer::save_to_json(const Ranks &ranks, const std::string &path) {
    std::map<std::string, int> sorted;
    for (const auto &[token, rank] : ranks) {
        std::string key;
        if (is_valid_utf8(token)) {
            key = token;
        } else {
            key = "b64:" + base64_encode(token);
        }
        sorted[key] = rank;
    }

    nlohmann::json j(sorted);

    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    file << j.dump(4);
}
