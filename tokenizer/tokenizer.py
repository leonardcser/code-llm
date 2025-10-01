"""Python wrapper for the C++ BPE tokenizer."""

from pathlib import Path
from typing import List, Union

try:
    from . import tokenizer_cpp
except ImportError:
    import tokenizer_cpp


class Tokenizer:
    """High-level Python wrapper for the C++ BPE tokenizer."""

    def __init__(self, tokenizer_path: Union[str, Path]):
        """Load a tokenizer from a file.

        Args:
            tokenizer_path: Path to the tokenizer binary file (.bin)
        """
        self._tok = tokenizer_cpp.load(str(tokenizer_path))

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._tok.vocab_size()

    @property
    def unk_token_id(self) -> int:
        """Get the unknown token ID."""
        return self._tok.unk_token_id

    @property
    def bos_token_id(self) -> int:
        """Get the beginning-of-sequence token ID."""
        return self._tok.bos_token_id

    @property
    def eos_token_id(self) -> int:
        """Get the end-of-sequence token ID."""
        return self._tok.eos_token_id

    @property
    def pad_token_id(self) -> int:
        """Get the padding token ID."""
        return self._tok.pad_token_id

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        return tokenizer_cpp.encode(text, self._tok)

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back into text.

        Args:
            tokens: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        return tokenizer_cpp.decode(tokens, self._tok, skip_special_tokens)

    def visualize(self, tokens: List[int]) -> str:
        """Visualize tokens with boundaries.

        Args:
            tokens: List of token IDs to visualize

        Returns:
            Visualization string with token boundaries
        """
        return tokenizer_cpp.visualize(tokens, self._tok)


def train(
    text: str,
    vocab_size: int,
    pattern: str,
    bos_token: str = "",
    eos_token: str = "",
    pad_token: str = "",
    max_unique_words: int = 0,
    logging_interval: int = 1000,
) -> Tokenizer:
    """Train a BPE tokenizer on the given text.

    Args:
        text: Training text
        vocab_size: Number of BPE merges (excludes byte tokens and special tokens)
        pattern: Regex pattern for pre-tokenization
        bos_token: Beginning-of-sequence token (empty = unused)
        eos_token: End-of-sequence token
        pad_token: Padding token
        max_unique_words: Limit on unique words to process (0 = no limit)
        logging_interval: Logging frequency during training

    Returns:
        Trained Tokenizer object
    """
    special_tokens_input = tokenizer_cpp.SpecialTokensInput(bos_token, eos_token, pad_token)
    cpp_tok = tokenizer_cpp.bpe_train(
        text, vocab_size, pattern, special_tokens_input, max_unique_words, logging_interval
    )

    # Wrap in our Python class
    tok = Tokenizer.__new__(Tokenizer)
    tok._tok = cpp_tok
    return tok


def save(tokenizer: Tokenizer, path: Union[str, Path]) -> None:
    """Save tokenizer to a binary file.

    Args:
        tokenizer: Tokenizer to save
        path: Output file path
    """
    tokenizer_cpp.save(tokenizer._tok, str(path))


def load(path: Union[str, Path]) -> Tokenizer:
    """Load tokenizer from a binary file.

    Args:
        path: Path to tokenizer binary file

    Returns:
        Loaded Tokenizer object
    """
    return Tokenizer(path)
