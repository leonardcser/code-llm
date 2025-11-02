#include "lib/tokenizer.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(tokenizer_cpp, m) {
    m.doc() = "Python bindings for the C++ BPE tokenizer";

    // Bind SpecialToken struct
    py::class_<tokenizer::SpecialToken>(m, "SpecialToken")
        .def(py::init<>())
        .def(py::init<std::string, tokenizer::TokenId, bool>(),
             py::arg("content"), py::arg("id"), py::arg("special") = true)
        .def_readwrite("content", &tokenizer::SpecialToken::content)
        .def_readwrite("id", &tokenizer::SpecialToken::id)
        .def_readwrite("special", &tokenizer::SpecialToken::special);

    // Bind SpecialTokensInput struct
    py::class_<tokenizer::SpecialTokensInput>(m, "SpecialTokensInput")
        .def(py::init<>())
        .def(py::init<std::string, std::string, std::string>(),
             py::arg("bos_token"), py::arg("eos_token"), py::arg("pad_token"))
        .def(py::init<std::string, std::string, std::string, std::string, std::string, std::string>(),
             py::arg("bos_token"), py::arg("eos_token"), py::arg("pad_token"),
             py::arg("cursor_token"), py::arg("edit_start_token"), py::arg("edit_end_token"))
        .def_readwrite("bos_token", &tokenizer::SpecialTokensInput::bos_token)
        .def_readwrite("eos_token", &tokenizer::SpecialTokensInput::eos_token)
        .def_readwrite("pad_token", &tokenizer::SpecialTokensInput::pad_token)
        .def_readwrite("cursor_token", &tokenizer::SpecialTokensInput::cursor_token)
        .def_readwrite("edit_start_token", &tokenizer::SpecialTokensInput::edit_start_token)
        .def_readwrite("edit_end_token", &tokenizer::SpecialTokensInput::edit_end_token);

    // Bind Tokenizer struct
    py::class_<tokenizer::Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def_readwrite("ranks", &tokenizer::Tokenizer::ranks)
        .def_readwrite("pattern", &tokenizer::Tokenizer::pattern)
        .def_readwrite("special_tokens", &tokenizer::Tokenizer::special_tokens)
        .def_readwrite("unk_token_id", &tokenizer::Tokenizer::unk_token_id)
        .def_readwrite("bos_token_id", &tokenizer::Tokenizer::bos_token_id)
        .def_readwrite("eos_token_id", &tokenizer::Tokenizer::eos_token_id)
        .def_readwrite("pad_token_id", &tokenizer::Tokenizer::pad_token_id)
        .def_readwrite("cursor_token_id", &tokenizer::Tokenizer::cursor_token_id)
        .def_readwrite("edit_start_token_id", &tokenizer::Tokenizer::edit_start_token_id)
        .def_readwrite("edit_end_token_id", &tokenizer::Tokenizer::edit_end_token_id)
        .def("vocab_size", [](const tokenizer::Tokenizer &tok) {
            // Calculate actual vocab size including special tokens
            tokenizer::TokenId max_id = static_cast<tokenizer::TokenId>(tok.ranks.size());

            // Check special token IDs
            for (const auto &[token_str, st] : tok.special_tokens) {
                if (st.id >= max_id) {
                    max_id = st.id + 1;
                }
            }

            return static_cast<size_t>(max_id);
        });

    // Bind training function
    m.def("bpe_train", &tokenizer::bpe_train,
          py::arg("text"),
          py::arg("vocab_size"),
          py::arg("pattern"),
          py::arg("special_tokens_input") = tokenizer::SpecialTokensInput(),
          py::arg("max_unique_words") = 0,
          py::arg("logging_interval") = 1000,
          "Train a BPE tokenizer on the given text");

    // Bind save/load functions
    m.def("save", &tokenizer::save,
          py::arg("tokenizer"),
          py::arg("filename"),
          "Save tokenizer to a binary file");

    m.def("load", &tokenizer::load,
          py::arg("filename"),
          "Load tokenizer from a binary file");

    // Bind encode function
    m.def("encode", &tokenizer::encode,
          py::arg("text"),
          py::arg("tokenizer"),
          "Encode text into token IDs");

    // Bind decode function
    m.def("decode", &tokenizer::decode,
          py::arg("tokens"),
          py::arg("tokenizer"),
          py::arg("skip_special_tokens") = false,
          "Decode token IDs back into text");

    // Bind visualize function
    m.def("visualize", &tokenizer::visualize,
          py::arg("tokens"),
          py::arg("tokenizer"),
          "Visualize tokens with boundaries");
}
