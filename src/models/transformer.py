"""Abstract base class for transformer language models."""

from abc import ABC, abstractmethod
import torch
import lightning as L


class Transformer(L.LightningModule, ABC):
    """Abstract base class for transformer-based language models."""

    @abstractmethod
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer_path: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> str:
        """Generate text completion given a prompt.

        Args:
            prompt: Input prompt text
            tokenizer_path: Path to tokenizer binary file
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k most likely tokens
            eos_token_id: EOS token ID to stop generation

        Returns:
            Generated text (prompt + completion)
        """
        ...
