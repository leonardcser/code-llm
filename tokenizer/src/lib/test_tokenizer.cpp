#include "tokenizer.hpp"
#include <cassert>

int main() {
    // Create a simple test tokenizer with small vocabulary
    std::string test_text = "hello world hello";
    std::string pattern = R"(\w+|\s+)";

    // Encode-decode roundtrip tests without special tokens
    {
        tokenizer::SpecialTokensInput special_tokens("", "", "");
        auto tokenizer = tokenizer::bpe_train(test_text, 300, pattern, special_tokens, 0, 1000);

        // Test case 1: Simple sentence
        std::string text1 = "hello world";
        auto tokens1 = tokenizer::encode(text1, tokenizer);
        auto decoded1 = tokenizer::decode(tokens1, tokenizer, false);
        assert(decoded1 == text1);

        // Test case 2: Sentence with punctuation
        std::string text2 = "This is a test. Isn't it?";
        auto tokens2 = tokenizer::encode(text2, tokenizer);
        auto decoded2 = tokenizer::decode(tokens2, tokenizer, false);
        assert(decoded2 == text2);

        // Test case 3: Numbers and special chars
        std::string text3 = "123 45! @#$%";
        auto tokens3 = tokenizer::encode(text3, tokenizer);
        auto decoded3 = tokenizer::decode(tokens3, tokenizer, false);
        assert(decoded3 == text3);

        // Test case 4: Empty string
        std::string text4 = "";
        auto tokens4 = tokenizer::encode(text4, tokenizer);
        auto decoded4 = tokenizer::decode(tokens4, tokenizer, false);
        assert(decoded4 == text4);

        // Test case 5: Single word
        std::string text5 = "tokenizer";
        auto tokens5 = tokenizer::encode(text5, tokenizer);
        auto decoded5 = tokenizer::decode(tokens5, tokenizer, false);
        assert(decoded5 == text5);
    }

    // Encode-decode roundtrip tests with special tokens
    {
        tokenizer::SpecialTokensInput special_tokens("", "<|endoftext|>", "<|pad|>");
        auto tokenizer = tokenizer::bpe_train(test_text, 300, pattern, special_tokens, 0, 1000);

        // Test case 1: Simple sentence with EOS
        std::string text1 = "hello world<|endoftext|>";
        auto tokens1 = tokenizer::encode(text1, tokenizer);
        auto decoded1 = tokenizer::decode(tokens1, tokenizer, false);
        assert(decoded1 == text1);

        // Test case 2: Sentence with PAD
        std::string text2 = "This is a test<|pad|>";
        auto tokens2 = tokenizer::encode(text2, tokenizer);
        auto decoded2 = tokenizer::decode(tokens2, tokenizer, false);
        assert(decoded2 == text2);

        // Test case 3: Multiple special tokens
        std::string text3 = "<|endoftext|><|pad|>hello<|endoftext|>";
        auto tokens3 = tokenizer::encode(text3, tokenizer);
        auto decoded3 = tokenizer::decode(tokens3, tokenizer, false);
        assert(decoded3 == text3);

        // Test case 4: Text with unknown token
        std::string text4 = "unknown_token<|endoftext|>";
        auto tokens4 = tokenizer::encode(text4, tokenizer);
        auto decoded4 = tokenizer::decode(tokens4, tokenizer, false);
        assert(decoded4.find("<unk>") != std::string::npos); // Should replace unknown with <unk>

        // Test case 5: Empty string with special
        std::string text5 = "<|endoftext|>";
        auto tokens5 = tokenizer::encode(text5, tokenizer);
        auto decoded5 = tokenizer::decode(tokens5, tokenizer, false);
        assert(decoded5 == text5);
    }

    // Test 1: BPE train with special tokens
    {
        tokenizer::SpecialTokensInput special_tokens(
            "", "<|endoftext|>", "<|pad|>");
        auto tokenizer = tokenizer::bpe_train(test_text, 300, pattern,
                                              special_tokens, 0, 1000);

        // Should have 3 special tokens: EOS, PAD, and auto-added UNK
        assert(tokenizer.special_tokens.size() == 3);
        assert(tokenizer.unk_token_id != 0);
        assert(tokenizer.bos_token_id == 0);
        assert(tokenizer.eos_token_id != 0);
        assert(tokenizer.pad_token_id != 0);

        // Verify token IDs are at the end: 256 + 300 = 556 onwards
        assert(tokenizer.eos_token_id >= 556);
        assert(tokenizer.pad_token_id >= 556);
        assert(tokenizer.unk_token_id >= 556);
    }

    // Test 2: Encode with special tokens
    {
        tokenizer::SpecialTokensInput special_tokens("", "<|endoftext|>", "");
        auto tokenizer = tokenizer::bpe_train(test_text, 300, pattern,
                                              special_tokens, 0, 1000);

        std::string text = "<|endoftext|>hello<|endoftext|>";
        auto tokens = tokenizer::encode(text, tokenizer);

        // Should have special tokens
        assert(tokens.size() > 0);
        assert(tokens[0] == tokenizer.eos_token_id); // First is endoftext
        assert(tokens[tokens.size() - 1] ==
               tokenizer.eos_token_id); // Last is endoftext
    }

    // Test 3: Decode with special tokens
    {
        tokenizer::SpecialTokensInput special_tokens("", "<|endoftext|>", "");
        auto tokenizer = tokenizer::bpe_train(test_text, 300, pattern,
                                              special_tokens, 0, 1000);

        std::string text = "<|endoftext|>hello<|endoftext|>";
        auto tokens = tokenizer::encode(text, tokenizer);
        auto decoded = tokenizer::decode(tokens, tokenizer, false);

        // Should contain the special tokens
        assert(decoded.find("<|endoftext|>") != std::string::npos);
        assert(decoded.find("hello") != std::string::npos);
    }

    // Test 4: Decode with special tokens (skip)
    {
        tokenizer::SpecialTokensInput special_tokens("", "<|endoftext|>", "");
        auto tokenizer = tokenizer::bpe_train(test_text, 300, pattern,
                                              special_tokens, 0, 1000);

        std::string text = "<|endoftext|>hello<|endoftext|>";
        auto tokens = tokenizer::encode(text, tokenizer);
        auto decoded = tokenizer::decode(tokens, tokenizer, true);

        // Should NOT contain the special tokens
        assert(decoded.find("<|endoftext|>") == std::string::npos);
        assert(decoded.find("hello") != std::string::npos);
    }

    // Test 5: Multiple special tokens in sequence
    {
        tokenizer::SpecialTokensInput special_tokens("", "", "<|pad|>");
        auto tokenizer = tokenizer::bpe_train(test_text, 300, pattern,
                                              special_tokens, 0, 1000);

        std::string text = "<|pad|><|pad|><|unk|>";
        auto tokens = tokenizer::encode(text, tokenizer);

        assert(tokens.size() == 3);
        assert(tokens[0] == tokenizer.pad_token_id);
        assert(tokens[1] == tokenizer.pad_token_id);
        assert(tokens[2] == tokenizer.unk_token_id);
    }

    // Test 6: Special tokens don't interfere with regular encoding
    {
        tokenizer::SpecialTokensInput special_tokens("", "<|endoftext|>", "");
        auto tokenizer = tokenizer::bpe_train(test_text, 300, pattern,
                                              special_tokens, 0, 1000);

        std::string text = "hello world";
        auto tokens = tokenizer::encode(text, tokenizer);
        auto decoded = tokenizer::decode(tokens, tokenizer, false);

        assert(decoded == text);
    }

    // Test 7: Save and load tokenizer with special tokens
    {
        tokenizer::SpecialTokensInput special_tokens("", "<|endoftext|>",
                                                     "<|pad|>");
        auto tokenizer = tokenizer::bpe_train(test_text, 300, pattern,
                                              special_tokens, 0, 1000);

        // Save
        tokenizer::save(tokenizer, "/tmp/test_tokenizer.bin");

        // Load
        auto loaded = tokenizer::load("/tmp/test_tokenizer.bin");

        // Verify special tokens are preserved (3 total: EOS, PAD, UNK)
        assert(loaded.special_tokens.size() == 3);
        assert(loaded.unk_token_id == tokenizer.unk_token_id);
        assert(loaded.eos_token_id == tokenizer.eos_token_id);
        assert(loaded.pad_token_id == tokenizer.pad_token_id);

        // Test encoding works the same
        std::string text = "<|endoftext|>hello<|pad|>";
        auto tokens1 = tokenizer::encode(text, tokenizer);
        auto tokens2 = tokenizer::encode(text, loaded);

        assert(tokens1.size() == tokens2.size());
        for (size_t i = 0; i < tokens1.size(); ++i) {
            assert(tokens1[i] == tokens2[i]);
        }
    }
    return 0;
}
