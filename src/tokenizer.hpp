#pragma once

#include <stddef.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace tokenizer {
using Ranks = std::unordered_map<std::string, int>;
using TokenId = uint32_t;

// Training function
void bpe_train(std::string &text, size_t vocab_size, const std::string &pattern,
               Ranks &ranks);

// Save/load functions

// Optimized encoding/decoding functions with direct lookup and thread-local
// regex
std::vector<TokenId> encode(const std::string &text, const Ranks &ranks,
                            const std::string &pattern);
std::string decode(const std::vector<TokenId> &tokens, const Ranks &ranks);

std::string visualize(const std::vector<TokenId> &tokens, const Ranks &ranks);
} // namespace tokenizer
