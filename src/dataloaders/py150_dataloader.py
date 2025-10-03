"""Dataset and utilities for Py150 tokenized data."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional


class TokenDataset(Dataset):
    """Dataset for loading tokenized data from binary files with proper document boundary handling."""

    def __init__(self, token_file: str, seq_length: int = 512, eos_token_id: Optional[int] = None):
        """
        Args:
            token_file: Path to binary token file (uint32)
            seq_length: Length of each training sequence
            eos_token_id: Token ID for end-of-sequence (document boundaries)
        """
        self.seq_length = seq_length
        self.eos_token_id = eos_token_id

        # Load tokens from binary file (uint32)
        tokens = np.fromfile(token_file, dtype=np.uint32)
        self.tokens = torch.from_numpy(tokens).long()

        # Calculate number of sequences (need seq_length + 1 tokens per sequence)
        self.n_sequences = (len(self.tokens) - 1) // seq_length

        print(f"Loaded {len(self.tokens)} tokens from {token_file}")
        print(f"Created {self.n_sequences} sequences of length {seq_length}")
        if eos_token_id is not None:
            eos_count = (self.tokens == eos_token_id).sum().item()
            print(f"Found {eos_count} document boundaries (EOS tokens)")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        """Returns input, target sequences, attention mask, and position IDs."""
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1

        # Get sequence + 1 token for target
        if end_idx > len(self.tokens):
            end_idx = len(self.tokens)
            start_idx = end_idx - self.seq_length - 1

        sequence = self.tokens[start_idx:end_idx]

        # Input: sequence[:-1], Target: sequence[1:]
        x = sequence[:-1]
        y = sequence[1:]

        # Create attention mask and position IDs if EOS token is specified
        if self.eos_token_id is not None:
            attention_mask = self._create_attention_mask(x)
            position_ids = self._create_position_ids(x)
            return x, y, attention_mask, position_ids
        else:
            return x, y

    def _create_attention_mask(self, tokens):
        """
        Create attention mask for the sequence with document boundaries.

        For packed sequences with multiple documents separated by EOS tokens,
        this creates a 3D attention mask that prevents cross-document attention.

        Returns a 3D mask (1, seq_len, seq_len) where:
        - True indicates tokens that can be attended to
        - False indicates tokens that should be masked out
        - Causal masking (lower triangular) is combined with document boundaries
        - After batching, becomes (batch_size, 1, seq_len, seq_len)
        """
        seq_len = len(tokens)

        # Create causal mask (lower triangular)
        # Shape: (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        # Find EOS token positions
        eos_positions = (tokens == self.eos_token_id).nonzero(as_tuple=False).squeeze(-1)

        if len(eos_positions) > 0:
            # For each EOS token, block attention from later tokens to this and earlier tokens
            for eos_pos in eos_positions:
                eos_pos = eos_pos.item()
                # All positions after EOS cannot attend to positions up to and including EOS
                if eos_pos + 1 < seq_len:
                    mask[eos_pos + 1:, :eos_pos + 1] = False

        # Reshape to 3D: (1, seq_len, seq_len)
        # When batched, this becomes (batch_size, 1, seq_len, seq_len)
        mask = mask.unsqueeze(0)

        return mask

    def _create_position_ids(self, tokens):
        """
        Create position IDs that reset at document boundaries.

        Returns position IDs that restart from 0 after each EOS token.
        """
        seq_len = len(tokens)
        position_ids = torch.arange(seq_len, dtype=torch.long)

        # Find EOS token positions
        eos_positions = (tokens == self.eos_token_id).nonzero(as_tuple=False).squeeze(-1)

        if len(eos_positions) > 0:
            # Reset positions after each EOS token
            for eos_pos in eos_positions:
                eos_pos = eos_pos.item()
                if eos_pos + 1 < seq_len:
                    # Subtract the offset to reset positions after EOS
                    position_ids[eos_pos + 1:] -= (eos_pos + 1)

        return position_ids
