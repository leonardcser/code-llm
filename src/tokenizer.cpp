#include "tokenizer.hpp"
#include "text.hpp"

void tok_bpe_train(std::string &text, size_t vocab_size) {
    if (vocab_size < 256) {
        throw std::invalid_argument("vocab_size must be at least 256");
    }
    text = to_ascii(text);
}
