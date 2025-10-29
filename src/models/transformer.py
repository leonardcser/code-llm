"""Abstract base class for transformer language models."""

from abc import ABC, abstractmethod
import torch
import lightning as L


class Transformer(L.LightningModule, ABC):
    """Abstract base class for transformer-based language models."""

    @abstractmethod
    @torch.no_grad()
    def generate_once(
        self,
        prompt: str,
        tokenizer_path: str,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> tuple[int, str, dict[str, float]]:
        """Predict next token given input prompt.

        Args:
            prompt: Input prompt text
            tokenizer_path: Path to tokenizer binary file
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens

        Returns:
            token_id: Predicted token ID
            token_text: Decoded token text
            probs: Top 10 token probabilities as dict
        """
        ...

    @abstractmethod
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer_path: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        """Generate text completion given a prompt.

        Args:
            prompt: Input prompt text
            tokenizer_path: Path to tokenizer binary file
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k most likely tokens

        Returns:
            Generated text (prompt + completion)
        """
        ...
