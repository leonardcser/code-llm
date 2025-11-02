"""Type stubs for the C++ tokenizer module."""

from typing import Dict

class SpecialToken:
    """Special token metadata."""

    content: str
    id: int
    special: bool

    def __init__(
        self, content: str = "", id: int = 0, special: bool = True
    ) -> None: ...

class SpecialTokensInput:
    """Input struct for specifying special tokens during training."""

    bos_token: str
    eos_token: str
    pad_token: str
    cursor_token: str
    edit_start_token: str
    edit_end_token: str

    def __init__(
        self,
        bos_token: str = "",
        eos_token: str = "",
        pad_token: str = "",
        cursor_token: str = "",
        edit_start_token: str = "",
        edit_end_token: str = "",
    ) -> None: ...

class Tokenizer:
    """BPE tokenizer."""

    ranks: Dict[str, int]
    pattern: str
    special_tokens: Dict[str, SpecialToken]
    unk_token_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    cursor_token_id: int
    edit_start_token_id: int
    edit_end_token_id: int

    def __init__(self) -> None: ...
    def vocab_size(self) -> int: ...

def bpe_train(
    text: str,
    vocab_size: int,
    pattern: str,
    special_tokens_input: SpecialTokensInput = ...,
    max_unique_words: int = 0,
    logging_interval: int = 1000,
) -> Tokenizer:
    """Train a BPE tokenizer on the given text."""
    ...

def save(tokenizer: Tokenizer, filename: str) -> None:
    """Save tokenizer to a binary file."""
    ...

def load(filename: str) -> Tokenizer:
    """Load tokenizer from a binary file."""
    ...

def encode(text: str, tokenizer: Tokenizer) -> list[int]:
    """Encode text into token IDs."""
    ...

def decode(
    tokens: list[int], tokenizer: Tokenizer, skip_special_tokens: bool = False
) -> str:
    """Decode token IDs back into text."""
    ...

def visualize(tokens: list[int], tokenizer: Tokenizer) -> str:
    """Visualize tokens with boundaries."""
    ...
