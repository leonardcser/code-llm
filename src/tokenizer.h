#pragma once

#include <cstddef>

typedef struct {

} Tok;

// IDEAS:
// - normalize to ASCII 256
//
// - trim comments in header
// - format files??? (for training tokenizer subset)
void tok_bpe_train(char *text, size_t vocab_size);
