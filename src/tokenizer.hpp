#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace tokenizer {
using Ranks = std::unordered_map<std::string, int>;
using TokenId = uint32_t;

// Training function
// max_unique_words: limit number of unique words to process (0 = no limit)
// Useful for training on very large datasets - samples high-frequency words
void bpe_train(std::string &text, size_t vocab_size, const std::string &pattern,
               Ranks &ranks, size_t max_unique_words = 0,
               size_t logging_interval = 1000);

// Save/load functions
void save(const Ranks &ranks, const std::string &filename);
Ranks load(const std::string &filename);

// Optimized encoding/decoding functions with direct lookup and thread-local
// regex
std::vector<TokenId> encode(const std::string &text, const Ranks &ranks,
                            const std::string &pattern);
std::string decode(const std::vector<TokenId> &tokens, const Ranks &ranks);

std::string visualize(const std::vector<TokenId> &tokens, const Ranks &ranks);
} // namespace tokenizer
