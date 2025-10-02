#include "tokenizer.hpp"
#include "threading.hpp"
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <algorithm>
#include <cctype>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <reflex/matcher.h>
#include <reflex/pattern.h>
#include <thread>
#include <utility>
#include <vector>

using tokenizer::OffsetList;
using tokenizer::OffsetPair;
using tokenizer::TokenId;
using Word = std::vector<TokenId>;
using WordList = std::vector<Word>;
using PairCount = absl::flat_hash_map<uint64_t, size_t>;
using Clock = std::chrono::steady_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

// Symbol for vector-based representation
struct Symbol {
    TokenId token;
    ssize_t prev; // Index of previous symbol (-1 if none)
    ssize_t next; // Index of next symbol (-1 if none)
    bool deleted; // Mark for deletion instead of actually deleting

    Symbol(TokenId t, ssize_t p, ssize_t n)
        : token(t), prev(p), next(n), deleted(false) {}
};

// Priority queue entry for pair frequencies
struct PairEntry {
    uint64_t pair_key;
    size_t frequency;

    bool operator<(const PairEntry &other) const {
        // Max heap: higher frequency = higher priority
        if (frequency != other.frequency) {
            return frequency < other.frequency;
        }
        // Tie-break by pair key (lower is better for consistency)
        return pair_key > other.pair_key;
    }
};

// Thread-local RE/flex matcher cache (compiles pattern once)
thread_local std::unordered_map<std::string, std::unique_ptr<reflex::Pattern>>
    pattern_cache;

// Helper to get/create pattern for regex (compiled once)
reflex::Pattern *get_pattern(const std::string &pattern_str) {
    auto it = pattern_cache.find(pattern_str);
    if (it == pattern_cache.end()) {
        auto pat = std::make_unique<reflex::Pattern>(pattern_str.c_str());
        it = pattern_cache.emplace(pattern_str, std::move(pat)).first;
        return it->second.get();
    }
    return it->second.get();
}

// Helper to create matcher with input and pattern (use unique_ptr to manage)
std::unique_ptr<reflex::Matcher> create_matcher(const std::string &input,
                                                reflex::Pattern *pat) {
    return std::make_unique<reflex::Matcher>(pat, reflex::Input(input.c_str()));
}

// Helper function to split text into word offsets (start, end indices in text)
// - efficient, no early vectors
OffsetList tokenize_to_offsets(const std::string &text,
                               const std::string &pattern_str) {
    OffsetList offsets;

    if (!pattern_str.empty()) {
        reflex::Pattern *pat = get_pattern(pattern_str);
        auto matcher = create_matcher(text, pat);
        offsets.reserve(text.length() / 3);

        while (matcher->find()) {
            size_t start = matcher->first();
            size_t len = matcher->size();
            offsets.emplace_back(start, start + len);
        }

        // Early return if multi-char "words" exist (check lengths >1 byte)
        bool has_multi_char = false;
        for (const auto &off : offsets) {
            if (off.second - off.first > 1) {
                has_multi_char = true;
                break;
            }
        }
        if (has_multi_char) return offsets;
    }

    // Fallback: Manual split on whitespace transitions, grouping consecutive
    // spaces
    offsets.reserve(text.length() / 4 + 1);
    size_t start = 0;
    bool prev_space = true; // Assume leading space to handle initial non-space
    for (size_t i = 0; i <= text.length(); ++i) {
        bool is_space = (i < text.length())
                            ? std::isspace(static_cast<unsigned char>(text[i]))
                            : false;
        if (i == text.length() || is_space != prev_space) {
            if (i > start) {
                offsets.emplace_back(start, i);
            }
            start = i;
            prev_space = is_space;
        }
    }

    if (offsets.empty() && !text.empty()) {
        offsets.emplace_back(0, text.length());
    }

    return offsets;
}

// Lazy conversion: offsets to byte-token words (only for unique offsets)
WordList offsets_to_words(const std::string &text, const OffsetList &offsets) {
    WordList words;
    words.reserve(offsets.size());
    auto push_char = [](unsigned char c) -> TokenId {
        return static_cast<TokenId>(c);
    };

    for (const auto &off : offsets) {
        size_t len = off.second - off.first;
        if (len == 0) continue;
        Word token_word;
        token_word.reserve(len);
        for (size_t j = 0; j < len; ++j) {
            token_word.push_back(
                push_char(static_cast<unsigned char>(text[off.first + j])));
        }
        words.push_back(std::move(token_word));
    }
    return words;
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

tokenizer::Tokenizer
tokenizer::bpe_train(std::string &text, size_t vocab_size,
                     const std::string &pattern,
                     const SpecialTokensInput &special_tokens_input,
                     size_t max_unique_words, size_t logging_interval) {
    // vocab_size now means number of BPE merges (excludes 256 byte tokens)
    if (vocab_size == 0) {
        throw std::invalid_argument("vocab_size must be at least 1");
    }

    const auto total_start = Clock::now();

    // Total vocab will be: 256 bytes + vocab_size BPE merges
    const size_t total_bpe_vocab = 256 + vocab_size;

    Ranks ranks;
    ranks.reserve(total_bpe_vocab);

    std::vector<std::string> vocab;
    vocab.reserve(total_bpe_vocab);

    for (int i = 0; i < 256; ++i) {
        std::string byteToken = std::string(1, static_cast<char>(i));
        ranks[byteToken] = i;
        vocab.push_back(byteToken);
    }

    DurationMs tokenize_time = DurationMs::zero();
    const auto tokenize_start = Clock::now();
    // Split into offsets (fast, allocation-light)
    OffsetList offsets = tokenize_to_offsets(text, pattern);

    // Dedup offsets (hash text slices, no copies/ conversions yet)
    std::vector<size_t> word_counts_vec;
    word_counts_vec.reserve(offsets.size());

    // Define hasher and eq as structs
    struct OffsetHash {
        const std::string &text_ref;
        OffsetHash(const std::string &t) : text_ref(t) {}
        size_t operator()(const OffsetPair &off) const {
            size_t h = 0;
            for (size_t i = off.first; i < off.second; ++i) {
                unsigned char c = static_cast<unsigned char>(text_ref[i]);
                h ^= std::hash<unsigned char>{}(c) + 0x9e3779b9 + (h << 6) +
                     (h >> 2);
            }
            return h;
        }
    };
    struct OffsetEq {
        const std::string &text_ref;
        OffsetEq(const std::string &t) : text_ref(t) {}
        bool operator()(const OffsetPair &a, const OffsetPair &b) const {
            if (a.second - a.first != b.second - b.first) return false;
            return std::equal(text_ref.begin() + a.first,
                              text_ref.begin() + a.second,
                              text_ref.begin() + b.first);
        }
    };

    // Construct hash map with custom hash/eq and reserve
    absl::flat_hash_map<OffsetPair, size_t, OffsetHash, OffsetEq> offset_index(
        offsets.size() / 2, OffsetHash(text), OffsetEq(text));
    OffsetList unique_offsets;
    unique_offsets.reserve(offsets.size() / 2);
    for (const auto &off : offsets) {
        if (off.second <= off.first) continue;
        auto [it, inserted] =
            offset_index.try_emplace(off, unique_offsets.size());
        if (inserted) {
            unique_offsets.push_back(off);
            word_counts_vec.push_back(1);
        } else {
            ++word_counts_vec[it->second];
        }
    }
    offsets = std::move(unique_offsets);

    // Convert only unique to words (major saving)
    WordList words = offsets_to_words(text, offsets);

    tokenize_time +=
        std::chrono::duration_cast<DurationMs>(Clock::now() - tokenize_start);

    // Frequency-based sampling if max_unique_words is set
    DurationMs sampling_time = DurationMs::zero();
    if (max_unique_words > 0 && words.size() > max_unique_words) {
        const auto sampling_start = Clock::now();

        // Create indices sorted by frequency (descending)
        std::vector<size_t> indices(words.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        // Sort by count descending (keep most frequent words)
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return word_counts_vec[a] > word_counts_vec[b];
        });

        // Keep only top max_unique_words
        indices.resize(max_unique_words);

        // Build sampled word list
        WordList sampled_words;
        std::vector<size_t> sampled_counts;
        sampled_words.reserve(max_unique_words);
        sampled_counts.reserve(max_unique_words);

        for (size_t idx : indices) {
            sampled_words.push_back(std::move(words[idx]));
            sampled_counts.push_back(word_counts_vec[idx]);
        }

        words = std::move(sampled_words);
        word_counts_vec = std::move(sampled_counts);

        sampling_time = std::chrono::duration_cast<DurationMs>(Clock::now() -
                                                               sampling_start);
        std::cout << "[bpe_train] Sampled " << words.size()
                  << " most frequent unique words from larger dataset"
                  << std::endl;
    }

    // Build vector-based symbol representation for each word
    std::vector<std::vector<Symbol>> word_symbols(words.size());

    for (size_t wi = 0; wi < words.size(); ++wi) {
        if (words[wi].empty()) continue;

        word_symbols[wi].reserve(words[wi].size());
        for (size_t i = 0; i < words[wi].size(); ++i) {
            ssize_t prev_idx = (i == 0) ? -1 : static_cast<ssize_t>(i - 1);
            ssize_t next_idx =
                (i == words[wi].size() - 1) ? -1 : static_cast<ssize_t>(i + 1);
            word_symbols[wi].emplace_back(words[wi][i], prev_idx, next_idx);
        }
    }

    // Main BPE loop with priority queue
    DurationMs pair_count_time = DurationMs::zero();
    DurationMs find_best_time = DurationMs::zero();
    DurationMs merge_time = DurationMs::zero();
    size_t iteration_count = 0;

    // Create thread pool (reused throughout training)
    const size_t num_threads = std::thread::hardware_concurrency();
    threading::ThreadPool thread_pool(num_threads);
    const size_t chunk_size = (words.size() + num_threads - 1) / num_threads;

    // Initial pair counting with position tracking (parallelized)
    const auto initial_count_start = Clock::now();

    struct ThreadLocalData {
        PairCount pair_counts;
        absl::flat_hash_map<uint64_t, absl::flat_hash_set<size_t>>
            where_to_update;
    };

    std::vector<ThreadLocalData> thread_data(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        thread_pool.enqueue([&, t]() {
            const size_t start = t * chunk_size;
            const size_t end = std::min(start + chunk_size, words.size());

            auto &local_counts = thread_data[t].pair_counts;
            auto &local_positions = thread_data[t].where_to_update;

            for (size_t wi = start; wi < end; ++wi) {
                const size_t mult = word_counts_vec[wi];
                const auto &symbols = word_symbols[wi];

                for (size_t i = 0; i < symbols.size(); ++i) {
                    if (symbols[i].deleted || symbols[i].next == -1) continue;

                    const size_t next_idx =
                        static_cast<size_t>(symbols[i].next);
                    if (symbols[next_idx].deleted) continue;

                    uint64_t pair_key =
                        encode_pair(symbols[i].token, symbols[next_idx].token);
                    local_counts[pair_key] += mult;
                    local_positions[pair_key].insert(wi);
                }
            }
        });
    }

    thread_pool.wait();

    // Merge thread-local counts and positions
    PairCount pair_counts;
    absl::flat_hash_map<uint64_t, absl::flat_hash_set<size_t>> where_to_update;
    pair_counts.reserve(words.size() * 4);

    for (const auto &td : thread_data) {
        for (const auto &[pair_key, freq] : td.pair_counts) {
            pair_counts[pair_key] += freq;
        }
        for (const auto &[pair_key, positions] : td.where_to_update) {
            where_to_update[pair_key].insert(positions.begin(),
                                             positions.end());
        }
    }

    // Build priority queue from initial counts
    std::priority_queue<PairEntry> pq;
    for (const auto &[pair_key, freq] : pair_counts) {
        if (freq > 1) {
            pq.push({pair_key, freq});
        }
    }
    pair_count_time += std::chrono::duration_cast<DurationMs>(
        Clock::now() - initial_count_start);

    while (ranks.size() < total_bpe_vocab && !pq.empty()) {
        ++iteration_count;

        // Find best pair
        const auto find_start = Clock::now();
        PairEntry best_entry = pq.top();
        pq.pop();

        // Verify the frequency is still accurate (may be stale)
        auto freq_it = pair_counts.find(best_entry.pair_key);
        if (freq_it == pair_counts.end() ||
            freq_it->second != best_entry.frequency) {
            // Stale entry, skip it
            continue;
        }

        uint64_t pair_key = best_entry.pair_key;
        size_t pair_frequency = best_entry.frequency;
        find_best_time +=
            std::chrono::duration_cast<DurationMs>(Clock::now() - find_start);

        if (pair_frequency <= 1) break;

        const TokenId first_token = first_from_pair(pair_key);
        const TokenId second_token = second_from_pair(pair_key);
        std::string new_token = vocab[first_token] + vocab[second_token];

        // Add new token to ranks
        TokenId token_id = static_cast<TokenId>(vocab.size());
        ranks[new_token] = static_cast<int>(token_id);
        vocab.push_back(new_token);

        // Get positions where this pair occurs
        auto pos_it = where_to_update.find(pair_key);
        if (pos_it == where_to_update.end()) continue;

        // Copy positions before erasing
        absl::flat_hash_set<size_t> positions = std::move(pos_it->second);
        where_to_update.erase(pos_it);

        // Merge pairs in symbol vectors and update pair counts (serial)
        const auto merge_start = Clock::now();

        absl::flat_hash_map<uint64_t, int64_t> pair_deltas;
        absl::flat_hash_map<uint64_t, absl::flat_hash_set<size_t>>
            new_positions;

        for (size_t wi : positions) {
            if (wi >= word_symbols.size()) continue;

            const size_t mult = word_counts_vec[wi];
            auto &symbols = word_symbols[wi];

            for (size_t i = 0; i < symbols.size(); ++i) {
                if (symbols[i].deleted || symbols[i].next == -1) continue;

                const size_t next_idx = static_cast<size_t>(symbols[i].next);
                if (next_idx >= symbols.size() || symbols[next_idx].deleted)
                    continue;

                if (symbols[i].token == first_token &&
                    symbols[next_idx].token == second_token) {
                    // Record pairs that will be removed
                    if (symbols[i].prev >= 0) {
                        const size_t prev_idx =
                            static_cast<size_t>(symbols[i].prev);
                        if (prev_idx < symbols.size() &&
                            !symbols[prev_idx].deleted) {
                            uint64_t left_pair = encode_pair(
                                symbols[prev_idx].token, first_token);
                            pair_deltas[left_pair] -= mult;
                        }
                    }
                    if (symbols[next_idx].next >= 0) {
                        const size_t next_next_idx =
                            static_cast<size_t>(symbols[next_idx].next);
                        if (next_next_idx < symbols.size() &&
                            !symbols[next_next_idx].deleted) {
                            uint64_t right_pair = encode_pair(
                                second_token, symbols[next_next_idx].token);
                            pair_deltas[right_pair] -= mult;
                        }
                    }
                    pair_deltas[pair_key] -= mult;

                    // Merge the pair: update current symbol
                    symbols[i].token = token_id;
                    symbols[i].next = symbols[next_idx].next;

                    // Mark next symbol as deleted
                    symbols[next_idx].deleted = true;

                    // Update the next symbol's prev pointer
                    if (symbols[i].next >= 0) {
                        const size_t new_next_idx =
                            static_cast<size_t>(symbols[i].next);
                        if (new_next_idx < symbols.size()) {
                            symbols[new_next_idx].prev =
                                static_cast<ssize_t>(i);
                        }
                    }

                    // Record new pairs
                    if (symbols[i].prev >= 0) {
                        const size_t prev_idx =
                            static_cast<size_t>(symbols[i].prev);
                        if (prev_idx < symbols.size() &&
                            !symbols[prev_idx].deleted) {
                            uint64_t new_left_pair =
                                encode_pair(symbols[prev_idx].token, token_id);
                            pair_deltas[new_left_pair] += mult;
                            new_positions[new_left_pair].insert(wi);
                        }
                    }
                    if (symbols[i].next >= 0) {
                        const size_t new_next_idx =
                            static_cast<size_t>(symbols[i].next);
                        if (new_next_idx < symbols.size() &&
                            !symbols[new_next_idx].deleted) {
                            uint64_t new_right_pair = encode_pair(
                                token_id, symbols[new_next_idx].token);
                            pair_deltas[new_right_pair] += mult;
                            new_positions[new_right_pair].insert(wi);
                        }
                    }
                }
            }
        }

        // Apply deltas and update priority queue
        for (const auto &[pk, delta] : pair_deltas) {
            auto it = pair_counts.find(pk);
            if (it != pair_counts.end()) {
                int64_t new_count = static_cast<int64_t>(it->second) + delta;
                if (new_count > 1) {
                    it->second = static_cast<size_t>(new_count);
                    pq.push({pk, it->second});
                } else {
                    pair_counts.erase(it);
                }
            } else if (delta > 0) {
                pair_counts[pk] = static_cast<size_t>(delta);
                if (delta > 1) {
                    pq.push({pk, static_cast<size_t>(delta)});
                }
            }
        }

        // Update positions map
        for (auto &[pk, pos_set] : new_positions) {
            where_to_update[pk].insert(pos_set.begin(), pos_set.end());
        }

        merge_time +=
            std::chrono::duration_cast<DurationMs>(Clock::now() - merge_start);

        // Optional: Progress reporting
        if (ranks.size() % logging_interval == 0) {
            std::cout << "[bpe_train] Vocab size: " << ranks.size() << "/"
                      << total_bpe_vocab << ", merged: " << vocab[first_token]
                      << " + " << vocab[second_token]
                      << " (freq: " << pair_frequency << ")" << std::endl;
        }
    }

    const DurationMs total_time =
        std::chrono::duration_cast<DurationMs>(Clock::now() - total_start);

    std::cout << "[bpe_train] BPE training completed. Final vocabulary size: "
              << ranks.size() << std::endl;
    std::cout << "[bpe_train] tokenize_to_words: " << tokenize_time.count()
              << " ms" << std::endl;
    if (sampling_time.count() > 0) {
        std::cout << "[bpe_train] sampling: " << sampling_time.count() << " ms"
                  << std::endl;
    }
    std::cout << "[bpe_train] pair counting: " << pair_count_time.count()
              << " ms" << std::endl;
    std::cout << "[bpe_train] find best pair: " << find_best_time.count()
              << " ms" << std::endl;
    std::cout << "[bpe_train] merge pairs: " << merge_time.count() << " ms"
              << std::endl;
    std::cout << "[bpe_train] iterations: " << iteration_count << std::endl;
    std::cout << "[bpe_train] total: " << total_time.count() << " ms"
              << std::endl;

    // Create tokenizer with trained vocabulary
    Tokenizer tokenizer;
    tokenizer.ranks = std::move(ranks);
    tokenizer.pattern = pattern;

    // Add special tokens at the end (after all BPE tokens)
    // Special token IDs start at 256 + vocab_size
    TokenId special_id = total_bpe_vocab;

    std::cout << "[bpe_train] Adding special tokens starting at ID "
              << special_id << "..." << std::endl;

    // Add user-specified special tokens
    if (!special_tokens_input.bos_token.empty()) {
        tokenizer.special_tokens[special_tokens_input.bos_token] =
            SpecialToken(special_tokens_input.bos_token, special_id, true);
        tokenizer.bos_token_id = special_id;
        std::cout << "  BOS: " << special_tokens_input.bos_token << " (ID "
                  << special_id << ")" << std::endl;
        special_id++;
    }

    if (!special_tokens_input.eos_token.empty()) {
        tokenizer.special_tokens[special_tokens_input.eos_token] =
            SpecialToken(special_tokens_input.eos_token, special_id, true);
        tokenizer.eos_token_id = special_id;
        std::cout << "  EOS: " << special_tokens_input.eos_token << " (ID "
                  << special_id << ")" << std::endl;
        special_id++;
    }

    if (!special_tokens_input.pad_token.empty()) {
        tokenizer.special_tokens[special_tokens_input.pad_token] =
            SpecialToken(special_tokens_input.pad_token, special_id, true);
        tokenizer.pad_token_id = special_id;
        std::cout << "  PAD: " << special_tokens_input.pad_token << " (ID "
                  << special_id << ")" << std::endl;
        special_id++;
    }

    // Always add UNK token (not user-specified)
    const std::string unk_token = "<|unk|>";
    tokenizer.special_tokens[unk_token] =
        SpecialToken(unk_token, special_id, true);
    tokenizer.unk_token_id = special_id;
    std::cout << "  UNK: " << unk_token << " (ID " << special_id
              << ") [auto-added]" << std::endl;

    return tokenizer;
}

// Helper struct for merge priority queue
struct Merge {
    size_t pos;
    int rank;
    TokenId new_id;

    bool operator<(const Merge &other) const {
        // Min heap: lower rank = higher priority
        if (rank != other.rank) {
            return rank > other.rank; // Reverse for min heap
        }
        return pos > other.pos;
    }
};

// Symbol for linked list approach
struct EncodeSymbol {
    TokenId c;
    ssize_t prev;
    ssize_t next;

    EncodeSymbol(TokenId token, ssize_t p, ssize_t n)
        : c(token), prev(p), next(n) {}
};

// BPE encoding with priority queue (optimized like Rust implementation)
std::vector<TokenId> byte_pair_encode(
    const std::string &piece, const tokenizer::Ranks &ranks,
    const absl::flat_hash_map<uint64_t, std::pair<int, TokenId>> &merge_map) {
    if (piece.empty()) {
        return {};
    }

    // Direct lookup first - if the whole piece is a token, return it
    auto it = ranks.find(piece);
    if (it != ranks.end()) {
        return {static_cast<TokenId>(it->second)};
    }

    // Fall back to character-level tokens for single characters
    if (piece.length() == 1) {
        unsigned char c = static_cast<unsigned char>(piece[0]);
        return {static_cast<TokenId>(c)};
    }

    // Convert to symbols with prev/next links
    std::vector<EncodeSymbol> symbols;
    symbols.reserve(piece.length());
    for (size_t i = 0; i < piece.length(); ++i) {
        unsigned char c = static_cast<unsigned char>(piece[i]);
        ssize_t prev_idx = (i == 0) ? -1 : static_cast<ssize_t>(i - 1);
        ssize_t next_idx =
            (i == piece.length() - 1) ? -1 : static_cast<ssize_t>(i + 1);
        symbols.emplace_back(static_cast<TokenId>(c), prev_idx, next_idx);
    }

    // Build priority queue of merges
    std::priority_queue<Merge> queue;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
        uint64_t pair_key = encode_pair(symbols[i].c, symbols[i + 1].c);
        auto merge_it = merge_map.find(pair_key);
        if (merge_it != merge_map.end()) {
            queue.push({i, merge_it->second.first, merge_it->second.second});
        }
    }

    // Process merges
    while (!queue.empty()) {
        Merge top = queue.top();
        queue.pop();

        // Skip if position is invalid or symbol was already merged
        if (top.pos >= symbols.size() || symbols[top.pos].next == -1) {
            continue;
        }

        size_t next_pos = static_cast<size_t>(symbols[top.pos].next);
        if (next_pos >= symbols.size()) {
            continue;
        }

        // Verify this is still the right pair (not stale queue entry)
        uint64_t current_pair =
            encode_pair(symbols[top.pos].c, symbols[next_pos].c);
        auto verify_it = merge_map.find(current_pair);
        if (verify_it == merge_map.end() ||
            verify_it->second.second != top.new_id) {
            continue;
        }

        // Merge: update current symbol
        symbols[top.pos].c = top.new_id;
        symbols[top.pos].next = symbols[next_pos].next;

        // Update next symbol's prev if it exists
        if (symbols[next_pos].next >= 0) {
            size_t next_next_pos = static_cast<size_t>(symbols[next_pos].next);
            if (next_next_pos < symbols.size()) {
                symbols[next_next_pos].prev = static_cast<ssize_t>(top.pos);
            }
        }

        // Mark next symbol as removed
        symbols[next_pos].next = -2; // Use -2 to mark as deleted

        // Add new merge with previous symbol
        if (symbols[top.pos].prev >= 0) {
            size_t prev_pos = static_cast<size_t>(symbols[top.pos].prev);
            uint64_t new_pair = encode_pair(symbols[prev_pos].c, top.new_id);
            auto new_merge = merge_map.find(new_pair);
            if (new_merge != merge_map.end()) {
                queue.push({prev_pos, new_merge->second.first,
                            new_merge->second.second});
            }
        }

        // Add new merge with next symbol
        if (symbols[top.pos].next >= 0) {
            size_t new_next_pos = static_cast<size_t>(symbols[top.pos].next);
            if (new_next_pos < symbols.size()) {
                uint64_t new_pair =
                    encode_pair(top.new_id, symbols[new_next_pos].c);
                auto new_merge = merge_map.find(new_pair);
                if (new_merge != merge_map.end()) {
                    queue.push({top.pos, new_merge->second.first,
                                new_merge->second.second});
                }
            }
        }
    }

    // Collect non-deleted symbols
    std::vector<TokenId> result;
    result.reserve(symbols.size());
    for (const auto &sym : symbols) {
        if (sym.next != -2) { // Not deleted
            result.push_back(sym.c);
        }
    }

    return result;
}

// Build decoder map helper function
std::unordered_map<TokenId, std::string>
build_decoder_map(const tokenizer::Ranks &ranks) {
    std::unordered_map<TokenId, std::string> decoder;
    decoder.reserve(ranks.size());

    for (const auto &[token, id] : ranks) {
        decoder[static_cast<TokenId>(id)] = token;
    }

    return decoder;
}

// Build merge map: maps (token1, token2) -> (rank, merged_token_id)
// Reconstructs merges from vocabulary by finding the correct split for each
// token
absl::flat_hash_map<uint64_t, std::pair<int, TokenId>>
build_merge_map(const tokenizer::Ranks &ranks) {
    absl::flat_hash_map<uint64_t, std::pair<int, TokenId>> merge_map;
    merge_map.reserve(ranks.size());

    // For each token, find the split where both parts were created earliest
    // (i.e., max(left_rank, right_rank) is minimized)
    for (const auto &[token_str, rank] : ranks) {
        if (token_str.length() <= 1 || rank < 256) {
            continue; // Skip base byte tokens
        }

        // Find the best split: the one where both parts existed earliest
        int best_max_rank = INT_MAX;
        TokenId best_left_id = 0;
        TokenId best_right_id = 0;
        bool found = false;

        for (size_t split = 1; split < token_str.length(); ++split) {
            std::string left = token_str.substr(0, split);
            std::string right = token_str.substr(split);

            auto left_it = ranks.find(left);
            if (left_it == ranks.end()) continue;

            auto right_it = ranks.find(right);
            if (right_it == ranks.end()) continue;

            int left_rank = left_it->second;
            int right_rank = right_it->second;

            // Both parts must have been created before this token
            if (left_rank >= rank || right_rank >= rank) continue;

            int max_rank = std::max(left_rank, right_rank);

            // Pick the split where both parts were created earliest
            if (max_rank < best_max_rank) {
                best_max_rank = max_rank;
                best_left_id = static_cast<TokenId>(left_rank);
                best_right_id = static_cast<TokenId>(right_rank);
                found = true;
            }
        }

        if (found) {
            uint64_t pair_key = encode_pair(best_left_id, best_right_id);
            // Use the token's rank as the merge rank
            merge_map[pair_key] = {rank, static_cast<TokenId>(rank)};
        }
    }

    return merge_map;
}

// Internal helper: encode a piece without special tokens
static std::vector<tokenizer::TokenId>
encode_piece(const std::string &text, const tokenizer::Ranks &ranks,
             const std::string &pattern) {
    std::vector<tokenizer::TokenId> result;
    result.reserve(text.length() / 2);

    // Build and cache merge map (thread-local, built once per tokenizer)
    static thread_local absl::flat_hash_map<uint64_t,
                                            std::pair<int, tokenizer::TokenId>>
        cached_merge_map;
    static thread_local const tokenizer::Ranks *cached_ranks = nullptr;

    if (cached_ranks != &ranks) {
        cached_merge_map = build_merge_map(ranks);
        cached_ranks = &ranks;
    }

    if (!pattern.empty()) {
        reflex::Pattern *pat = get_pattern(pattern);
        auto matcher = create_matcher(text, pat);

        while (matcher->find()) {
            size_t start = matcher->first();
            size_t len = matcher->size();
            std::string piece(text.data() + start, len);

            // Try direct lookup first
            auto it = ranks.find(piece);
            if (it != ranks.end()) {
                result.push_back(static_cast<tokenizer::TokenId>(it->second));
            } else {
                // Fall back to BPE encoding
                std::vector<tokenizer::TokenId> encoded =
                    byte_pair_encode(piece, ranks, cached_merge_map);
                result.insert(result.end(), encoded.begin(), encoded.end());
            }
        }
    } else {
        // Fallback: whole text as one piece
        std::string piece = text;
        auto it = ranks.find(piece);
        if (it != ranks.end()) {
            result.push_back(static_cast<tokenizer::TokenId>(it->second));
        } else {
            std::vector<tokenizer::TokenId> encoded =
                byte_pair_encode(piece, ranks, cached_merge_map);
            result.insert(result.end(), encoded.begin(), encoded.end());
        }
    }

    return result;
}

// Encode with special token support
std::vector<TokenId> tokenizer::encode(const std::string &text,
                                       const Tokenizer &tokenizer) {
    // First, check if entire text is a special token
    auto special_it = tokenizer.special_tokens.find(text);
    if (special_it != tokenizer.special_tokens.end()) {
        return {special_it->second.id};
    }

    // Fast path: if no special tokens, encode directly
    if (tokenizer.special_tokens.empty()) {
        return encode_piece(text, tokenizer.ranks, tokenizer.pattern);
    }

    // Pre-scan text once to find all special token positions
    struct SpecialMatch {
        size_t pos;
        size_t len;
        TokenId id;
    };
    std::vector<SpecialMatch> matches;
    matches.reserve(text.length() / 100); // Estimate

    for (const auto &[token_str, st] : tokenizer.special_tokens) {
        size_t search_pos = 0;
        while ((search_pos = text.find(token_str, search_pos)) !=
               std::string::npos) {
            matches.push_back({search_pos, token_str.length(), st.id});
            search_pos += token_str.length();
        }
    }

    // Sort matches by position
    std::sort(matches.begin(), matches.end(),
              [](const SpecialMatch &a, const SpecialMatch &b) {
                  return a.pos < b.pos;
              });

    // Remove overlapping matches (keep first occurrence)
    if (!matches.empty()) {
        std::vector<SpecialMatch> filtered;
        filtered.reserve(matches.size());
        filtered.push_back(matches[0]);
        for (size_t i = 1; i < matches.size(); ++i) {
            if (matches[i].pos >= filtered.back().pos + filtered.back().len) {
                filtered.push_back(matches[i]);
            }
        }
        matches = std::move(filtered);
    }

    // Now encode text segments between special tokens
    std::vector<TokenId> result;
    result.reserve(text.length() / 2);
    size_t pos = 0;

    for (const auto &match : matches) {
        // Encode text before special token
        if (match.pos > pos) {
            std::string piece = text.substr(pos, match.pos - pos);
            std::vector<TokenId> encoded =
                encode_piece(piece, tokenizer.ranks, tokenizer.pattern);
            result.insert(result.end(), encoded.begin(), encoded.end());
        }
        // Add special token
        result.push_back(match.id);
        pos = match.pos + match.len;
    }

    // Encode remaining text
    if (pos < text.length()) {
        std::string piece = text.substr(pos);
        std::vector<TokenId> encoded =
            encode_piece(piece, tokenizer.ranks, tokenizer.pattern);
        result.insert(result.end(), encoded.begin(), encoded.end());
    }

    return result;
}

// Decode with special token support
std::string tokenizer::decode(const std::vector<TokenId> &tokens,
                              const Tokenizer &tokenizer,
                              bool skip_special_tokens) {
    // Build decoder map once
    auto decoder = build_decoder_map(tokenizer.ranks);

    // Build reverse lookup for special tokens (ID -> token)
    std::unordered_map<TokenId, std::string> special_decoder;
    for (const auto &[token_str, st] : tokenizer.special_tokens) {
        special_decoder[st.id] = token_str;
    }

    std::string result;
    result.reserve(tokens.size() * 2);

    for (TokenId token : tokens) {
        // Check if it's a special token
        auto special_it = special_decoder.find(token);
        if (special_it != special_decoder.end()) {
            if (!skip_special_tokens) {
                result += special_it->second;
            }
            continue;
        }

        // Regular token decode
        auto it = decoder.find(token);
        if (it != decoder.end()) {
            result += it->second;
        } else {
            // Fall back to single character for unknown tokens
            if (token < 256) {
                result += static_cast<char>(token);
            }
        }
    }

    return result;
}

std::string tokenizer::visualize(const std::vector<TokenId> &tokens,
                                 const Tokenizer &tokenizer) {
    auto decoder = build_decoder_map(tokenizer.ranks);

    // Build reverse lookup for special tokens (ID -> token)
    std::unordered_map<TokenId, std::string> special_decoder;
    for (const auto &[token_str, st] : tokenizer.special_tokens) {
        special_decoder[st.id] = token_str;
    }
    std::string result;
    for (TokenId token : tokens) {
        std::string tok_str;

        // Check if it's a special token first
        auto special_it = special_decoder.find(token);
        if (special_it != special_decoder.end()) {
            tok_str = special_it->second;
        } else {
            auto it = decoder.find(token);
            if (it != decoder.end()) {
                tok_str = it->second;
            } else if (token < 256) {
                tok_str = std::string(1, static_cast<char>(token));
            } else {
                continue;
            }
        }
        // Generate color
        unsigned int hash_val = static_cast<unsigned int>(token);
        hash_val ^= hash_val >> 16;
        hash_val *= 0x85ebca6bU;
        hash_val ^= hash_val >> 13;
        hash_val *= 0xc2b2ae35U;
        hash_val ^= hash_val >> 16;
        int r = (hash_val >> 16) & 0xFF;
        int g = (hash_val >> 8) & 0xFF;
        int b = hash_val & 0xFF;
        // Adjust to pastel: blend with white
        double factor = 0.6;
        r = static_cast<int>(r * factor + 255 * (1 - factor));
        g = static_cast<int>(g * factor + 255 * (1 - factor));
        b = static_cast<int>(b * factor + 255 * (1 - factor));
        int fr = 0, fg_col = 0, fb = 0;
        result += "\x1b[48;2;" + std::to_string(r) + ";" + std::to_string(g) +
                  ";" + std::to_string(b) + "m";
        result += "\x1b[38;2;" + std::to_string(fr) + ";" +
                  std::to_string(fg_col) + ";" + std::to_string(fb) + "m";

        result += tok_str;
        result += "\x1b[0m";
    }
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    return result;
}

void tokenizer::save(const Tokenizer &tokenizer, const std::string &filename) {
    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " +
                                 filename);
    }

    cereal::BinaryOutputArchive archive(os);
    archive(tokenizer.ranks);
    archive(tokenizer.pattern);

    // Save special tokens
    size_t num_special = tokenizer.special_tokens.size();
    archive(num_special);
    for (const auto &[token, st] : tokenizer.special_tokens) {
        archive(st.content, st.id, st.special);
    }

    // Save special token IDs
    archive(tokenizer.unk_token_id);
    archive(tokenizer.bos_token_id);
    archive(tokenizer.eos_token_id);
    archive(tokenizer.pad_token_id);
}

tokenizer::Tokenizer tokenizer::load(const std::string &filename) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " +
                                 filename);
    }

    Tokenizer tok;
    cereal::BinaryInputArchive archive(is);
    archive(tok.ranks);
    archive(tok.pattern);

    // Load special tokens
    size_t num_special = 0;
    archive(num_special);
    for (size_t i = 0; i < num_special; ++i) {
        std::string content;
        TokenId id;
        bool special;
        archive(content, id, special);
        tok.special_tokens[content] = SpecialToken(content, id, special);
    }

    // Load special token IDs
    archive(tok.unk_token_id);
    archive(tok.bos_token_id);
    archive(tok.eos_token_id);
    archive(tok.pad_token_id);

    return tok;
}
