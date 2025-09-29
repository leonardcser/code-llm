#pragma once

#include <stddef.h>
#include <string>
#include <unordered_map>

namespace tokenizer {
using Ranks = std::unordered_map<std::string, int>;

// IDEAS:
// - normalize to ASCII 256
//
// - trim comments in header
// - format files??? (for training tokenizer subset)
void bpe_train(std::string &text, size_t vocab_size, const std::string &pattern,
               Ranks &ranks);

void save_to_json(const Ranks &ranks, const std::string &path);

} // namespace tokenizer
