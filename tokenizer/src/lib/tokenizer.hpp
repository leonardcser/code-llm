#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tokenizer {
using Ranks = std::unordered_map<std::string, int>;
using TokenId = uint32_t;

using OffsetPair = std::pair<size_t, size_t>;
using OffsetList = std::vector<OffsetPair>;

// Special token metadata
struct SpecialToken {
    std::string content;
    TokenId id;
    bool special; // if true, can be skipped during decoding

    SpecialToken() : content(""), id(0), special(true) {}
    SpecialToken(std::string c, TokenId i, bool s = true)
        : content(c), id(i), special(s) {}
};

// Map from token string to its metadata
using SpecialTokensMap = std::unordered_map<std::string, SpecialToken>;

// Input struct for specifying special tokens during training
struct SpecialTokensInput {
    std::string bos_token;  // Beginning of sequence (e.g., "<|startoftext|>")
    std::string eos_token;  // End of sequence (e.g., "<|endoftext|>")
    std::string pad_token;  // Padding token (e.g., "<|pad|>")
    // UNK token is always added automatically - not user-specified

    SpecialTokensInput() = default;
    SpecialTokensInput(std::string bos, std::string eos, std::string pad)
        : bos_token(bos), eos_token(eos), pad_token(pad) {}
};

struct Tokenizer {
    Ranks ranks;
    std::string pattern;
    SpecialTokensMap special_tokens;

    // Common special token IDs (0 = not set)
    TokenId unk_token_id = 0;
    TokenId bos_token_id = 0;
    TokenId eos_token_id = 0;
    TokenId pad_token_id = 0;
};

// Training function
// vocab_size: number of BPE merges (excludes 256 byte tokens and special tokens)
// max_unique_words: limit number of unique words to process (0 = no limit)
//   Useful for training on very large datasets - samples high-frequency words
// special_tokens_input: optional special tokens (BOS, EOS, PAD)
//   UNK token is always added automatically
// Returns a fully configured Tokenizer with:
//   - IDs 0-255: byte tokens
//   - IDs 256 to (256+vocab_size-1): BPE merge tokens
//   - IDs (256+vocab_size) onwards: special tokens (UNK always last)
Tokenizer bpe_train(std::string &text, size_t vocab_size,
                    const std::string &pattern,
                    const SpecialTokensInput &special_tokens_input = {},
                    size_t max_unique_words = 0,
                    size_t logging_interval = 1000);

// Save/load functions
void save(const Tokenizer &tokenizer, const std::string &filename);
Tokenizer load(const std::string &filename);

// Encoding/decoding functions
std::vector<TokenId> encode(const std::string &text,
                            const Tokenizer &tokenizer);
std::string decode(const std::vector<TokenId> &tokens,
                   const Tokenizer &tokenizer,
                   bool skip_special_tokens = false);

std::string visualize(const std::vector<TokenId> &tokens,
                      const Tokenizer &tokenizer);
} // namespace tokenizer
