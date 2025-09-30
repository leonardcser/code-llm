#include "tokenizer.hpp"
#include "text.hpp"
#include "threading.hpp"
#include <absl/container/flat_hash_map.h>
#include <algorithm>
#include <cctype>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <queue>
#include <regex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using tokenizer::TokenId;
using Word = std::vector<TokenId>;
using WordList = std::vector<Word>;
using PairCount = absl::flat_hash_map<uint64_t, size_t>;
using Clock = std::chrono::steady_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

// Symbol node for linked list representation
struct SymbolNode {
    TokenId token;
    SymbolNode *prev;
    SymbolNode *next;

    SymbolNode(TokenId t) : token(t), prev(nullptr), next(nullptr) {}
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

// Thread-local regex cache to avoid recompilation
thread_local std::unordered_map<std::string, std::regex> regex_cache;

// Helper function to get thread-local regex
const std::regex &get_thread_local_regex(const std::string &pattern) {
    auto it = regex_cache.find(pattern);
    if (it == regex_cache.end()) {
        it = regex_cache
                 .emplace(pattern,
                          std::regex(pattern, std::regex_constants::optimize))
                 .first;
    }
    return it->second;
}

// Helper function to split text into words and convert to byte tokens
WordList tokenize_to_words(const std::string &text,
                           const std::string &pattern_str) {
    WordList words;

    auto push_char = [](unsigned char c) -> TokenId {
        return static_cast<TokenId>(c);
    };

    if (!pattern_str.empty()) {
        const std::regex &pattern = get_thread_local_regex(pattern_str);
        words.reserve(text.length() / 3);

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
    fallback_words.reserve(text.length() / 4 + 1);

    Word current_word;
    current_word.reserve(32); // Larger initial capacity

    auto flush_word = [&]() {
        if (!current_word.empty()) {
            fallback_words.emplace_back();
            fallback_words.back().swap(current_word);
            current_word.clear();
            current_word.reserve(32);
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
                          const std::string &pattern, Ranks &ranks,
                          size_t max_unique_words, size_t logging_interval) {
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
    // Deduplicate identical words to reduce repeated work; keep multiplicities
    std::vector<size_t> word_counts_vec;
    word_counts_vec.reserve(words.size());
    absl::flat_hash_map<std::string, size_t> word_index;
    word_index.reserve(words.size() * 2);
    {
        WordList unique_words;
        unique_words.reserve(words.size());
        for (const auto &w : words) {
            if (w.empty()) continue;
            std::string key;
            key.resize(w.size());
            for (size_t i = 0; i < w.size(); ++i) {
                key[i] = static_cast<char>(static_cast<unsigned char>(w[i]));
            }
            auto [it, inserted] =
                word_index.try_emplace(key, unique_words.size());
            if (inserted) {
                unique_words.push_back(w);
                word_counts_vec.push_back(1);
            } else {
                ++word_counts_vec[it->second];
            }
        }
        if (!unique_words.empty()) {
            words.swap(unique_words);
        }
    }
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

    // Build linked list representation for each word
    std::vector<SymbolNode *> word_heads(words.size(), nullptr);

    for (size_t wi = 0; wi < words.size(); ++wi) {
        if (words[wi].empty()) continue;

        SymbolNode *head = new SymbolNode(words[wi][0]);
        word_heads[wi] = head;
        SymbolNode *current = head;

        for (size_t i = 1; i < words[wi].size(); ++i) {
            SymbolNode *node = new SymbolNode(words[wi][i]);
            current->next = node;
            node->prev = current;
            current = node;
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

    // Initial pair counting (parallelized with thread pool)
    const auto initial_count_start = Clock::now();

    std::vector<PairCount> thread_pair_counts(num_threads);

    // Reserve space for each thread's hash map
    for (auto &tpc : thread_pair_counts) {
        tpc.reserve(words.size() / num_threads + 1024);
    }

    for (size_t t = 0; t < num_threads; ++t) {
        thread_pool.enqueue([&, t]() {
            const size_t start = t * chunk_size;
            const size_t end = std::min(start + chunk_size, words.size());

            for (size_t wi = start; wi < end; ++wi) {
                const size_t mult = word_counts_vec[wi];
                SymbolNode *node = word_heads[wi];

                while (node && node->next) {
                    uint64_t pair_key =
                        encode_pair(node->token, node->next->token);
                    thread_pair_counts[t][pair_key] += mult;
                    node = node->next;
                }
            }
        });
    }

    thread_pool.wait();

    // Merge thread-local counts
    PairCount pair_counts;
    pair_counts.reserve(words.size() * 4);
    for (const auto &tpc : thread_pair_counts) {
        for (const auto &[pair_key, freq] : tpc) {
            pair_counts[pair_key] += freq;
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

    while (ranks.size() < vocab_size && !pq.empty()) {
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

        // Merge pairs in linked lists and update pair counts (parallelized)
        const auto merge_start = Clock::now();

        // Thread-local delta maps
        std::vector<absl::flat_hash_map<uint64_t, int64_t>> thread_deltas(
            num_threads);

        for (size_t t = 0; t < num_threads; ++t) {
            thread_pool.enqueue([&, t]() {
                const size_t start = t * chunk_size;
                const size_t end = std::min(start + chunk_size, words.size());
                auto &local_deltas = thread_deltas[t];

                for (size_t wi = start; wi < end; ++wi) {
                    const size_t mult = word_counts_vec[wi];
                    SymbolNode *node = word_heads[wi];

                    while (node && node->next) {
                        if (node->token == first_token &&
                            node->next->token == second_token) {
                            // Record pairs that will be removed
                            if (node->prev) {
                                uint64_t left_pair =
                                    encode_pair(node->prev->token, first_token);
                                local_deltas[left_pair] -= mult;
                            }
                            if (node->next->next) {
                                uint64_t right_pair = encode_pair(
                                    second_token, node->next->next->token);
                                local_deltas[right_pair] -= mult;
                            }
                            local_deltas[pair_key] -= mult;

                            // Merge the pair
                            SymbolNode *to_delete = node->next;
                            node->token = token_id;
                            node->next = to_delete->next;
                            if (to_delete->next) {
                                to_delete->next->prev = node;
                            }
                            // Memory pool handles cleanup, no delete needed

                            // Record new pairs
                            if (node->prev) {
                                uint64_t new_left_pair =
                                    encode_pair(node->prev->token, token_id);
                                local_deltas[new_left_pair] += mult;
                            }
                            if (node->next) {
                                uint64_t new_right_pair =
                                    encode_pair(token_id, node->next->token);
                                local_deltas[new_right_pair] += mult;
                            }

                            // Continue from merged node
                            if (node->next) {
                                node = node->next;
                            } else {
                                break;
                            }
                        } else {
                            node = node->next;
                        }
                    }
                }
            });
        }

        thread_pool.wait();

        // Merge thread-local deltas
        absl::flat_hash_map<uint64_t, int64_t> pair_deltas;
        for (const auto &td : thread_deltas) {
            for (const auto &[pk, delta] : td) {
                pair_deltas[pk] += delta;
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

        merge_time +=
            std::chrono::duration_cast<DurationMs>(Clock::now() - merge_start);

        // Optional: Progress reporting
        if (ranks.size() % logging_interval == 0) {
            std::cout << "[bpe_train] Vocab size: " << ranks.size() << "/"
                      << vocab_size << ", merged: " << vocab[first_token]
                      << " + " << vocab[second_token]
                      << " (freq: " << pair_frequency << ")" << std::endl;
        }
    }

    // Memory pool automatically cleans up all nodes

    const DurationMs total_time =
        std::chrono::duration_cast<DurationMs>(Clock::now() - total_start);

    std::cout << "[bpe_train] BPE training completed. Final vocabulary size: "
              << ranks.size() << std::endl;
    std::cout << "[bpe_train] normalize: " << normalize_time.count() << " ms"
              << std::endl;
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
}

// BPE encoding with direct token lookup
std::vector<TokenId>
byte_pair_encode(const std::string &piece, const tokenizer::Ranks &ranks,
                 const std::unordered_map<TokenId, std::string> &decoder) {
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

    // Convert to vector of character tokens initially
    std::vector<TokenId> tokens;
    tokens.reserve(piece.length());
    for (unsigned char c : piece) {
        tokens.push_back(static_cast<TokenId>(c));
    }

    // Apply BPE merges
    bool merged = true;
    while (merged && tokens.size() > 1) {
        merged = false;
        std::string best_pair;
        int best_rank = -1;
        size_t best_pos = 0;

        // Find the highest-ranking pair to merge
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            const std::string *left_piece = nullptr;
            const std::string *right_piece = nullptr;
            std::string left_fallback;
            std::string right_fallback;

            auto left_it = decoder.find(tokens[i]);
            if (left_it != decoder.end()) {
                left_piece = &left_it->second;
            } else if (tokens[i] < 256) {
                left_fallback.assign(1, static_cast<char>(tokens[i]));
                left_piece = &left_fallback;
            } else {
                continue;
            }

            auto right_it = decoder.find(tokens[i + 1]);
            if (right_it != decoder.end()) {
                right_piece = &right_it->second;
            } else if (tokens[i + 1] < 256) {
                right_fallback.assign(1, static_cast<char>(tokens[i + 1]));
                right_piece = &right_fallback;
            } else {
                continue;
            }

            std::string pair;
            pair.reserve(left_piece->size() + right_piece->size());
            pair.append(*left_piece);
            pair.append(*right_piece);

            auto pair_it = ranks.find(pair);
            if (pair_it != ranks.end()) {
                if (best_rank == -1 || pair_it->second < best_rank) {
                    best_pair = pair;
                    best_rank = pair_it->second;
                    best_pos = i;
                    merged = true;
                }
            }
        }

        // Apply the best merge
        if (merged) {
            TokenId new_token = static_cast<TokenId>(best_rank);
            std::vector<TokenId> new_tokens;
            new_tokens.reserve(tokens.size());

            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i == best_pos) {
                    new_tokens.push_back(new_token);
                    ++i; // Skip the next token as it's part of the merge
                } else {
                    new_tokens.push_back(tokens[i]);
                }
            }
            tokens = std::move(new_tokens);
        }
    }

    return tokens;
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

// encode function with direct lookup
std::vector<TokenId> tokenizer::encode(const std::string &text,
                                       const Ranks &ranks,
                                       const std::string &pattern) {
    std::vector<TokenId> result;
    result.reserve(text.length() / 2);

    const std::regex &regex_pattern = get_thread_local_regex(pattern);
    auto decoder = build_decoder_map(ranks);

    // Tokenize using regex
    std::sregex_iterator iter(text.begin(), text.end(), regex_pattern);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
        const std::string piece = iter->str();

        // Try direct lookup first
        auto it = ranks.find(piece);
        if (it != ranks.end()) {
            result.push_back(static_cast<TokenId>(it->second));
        } else {
            // Fall back to BPE encoding
            std::vector<TokenId> encoded =
                byte_pair_encode(piece, ranks, decoder);
            result.reserve(result.size() + encoded.size());
            result.insert(result.end(), encoded.begin(), encoded.end());
        }
    }

    return result;
}

std::string tokenizer::decode(const std::vector<TokenId> &tokens,
                              const Ranks &ranks) {
    // Build decoder map once
    auto decoder = build_decoder_map(ranks);

    std::string result;
    result.reserve(tokens.size() * 2);

    for (TokenId token : tokens) {
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
                                 const Ranks &ranks) {
    auto decoder = build_decoder_map(ranks);
    std::string result;
    for (TokenId token : tokens) {
        auto it = decoder.find(token);
        std::string tok_str;
        if (it != decoder.end()) {
            tok_str = it->second;
        } else if (token < 256) {
            tok_str = std::string(1, static_cast<char>(token));
        } else {
            continue;
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
        double lum = 0.299 * r + 0.587 * g + 0.114 * b;
        if (lum < 50) {
            r = static_cast<int>(r * 1.5);
            if (r > 255) r = 255;
            g = static_cast<int>(g * 1.5);
            if (g > 255) g = 255;
            b = static_cast<int>(b * 1.5);
            if (b > 255) b = 255;
            lum = 0.299 * r + 0.587 * g + 0.114 * b;
        }
        int fr, fg_col, fb;
        if (lum > 128) {
            fr = fg_col = fb = 0;
        } else {
            fr = fg_col = fb = 255;
        }
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

void tokenizer::save(const Ranks &ranks, const std::string &filename) {
    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " +
                                 filename);
    }

    cereal::BinaryOutputArchive archive(os);
    archive(ranks);
}

tokenizer::Ranks tokenizer::load(const std::string &filename) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " +
                                 filename);
    }

    Ranks ranks;
    cereal::BinaryInputArchive archive(is);
    archive(ranks);

    return ranks;
}
