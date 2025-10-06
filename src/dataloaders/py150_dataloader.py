"""Dataset and utilities for Py150 tokenized data."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional


class TokenDataset(Dataset):
    """Dataset for loading tokenized data from binary files with proper document boundary handling."""

    def __init__(self, token_file: str, seq_length: int = 512, eos_token_id: Optional[int] = None, bos_token_id: Optional[int] = None):
        """
        Args:
            token_file: Path to binary token file (uint32)
            seq_length: Length of each training sequence
            eos_token_id: Token ID for end-of-sequence (document boundaries)
            bos_token_id: Token ID for beginning-of-sequence (document boundaries)
        """
        self.seq_length = seq_length
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id

        # Load tokens from binary file (uint32)
        tokens = np.fromfile(token_file, dtype=np.uint32)
        self.tokens = torch.from_numpy(tokens).long()

        # Calculate number of sequences (need seq_length + 1 tokens per sequence)
        self.n_sequences = (len(self.tokens) - 1) // seq_length

        print(f"Loaded {len(self.tokens)} tokens from {token_file}")
        print(f"Created {self.n_sequences} sequences of length {seq_length}")
        if eos_token_id is not None:
            eos_count = (self.tokens == eos_token_id).sum().item()
            print(f"Found {eos_count} EOS token boundaries")
        if bos_token_id is not None:
            bos_count = (self.tokens == bos_token_id).sum().item()
            print(f"Found {bos_count} BOS token boundaries")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        """Returns input, target sequences, attention mask, and position IDs."""
        # If using BOS, we prepend it to each sequence (attention sink pattern)
        # x will be: [BOS, tok0, tok1, ..., tok(n-2)]
        # y will be: [tok0, tok1, ..., tok(n-1)]
        # where n = seq_length

        if self.bos_token_id is not None:
            # Get seq_length tokens from file (not seq_length + 1)
            start_idx = idx * self.seq_length
            end_idx = start_idx + self.seq_length

            if end_idx > len(self.tokens):
                end_idx = len(self.tokens)
                start_idx = max(0, end_idx - self.seq_length)

            sequence = self.tokens[start_idx:end_idx]

            # Prepend BOS to input
            bos_tensor = torch.tensor([self.bos_token_id], dtype=torch.long)
            x = torch.cat([bos_tensor, sequence[:-1]])  # [BOS] + first (seq_length-1) tokens
            y = sequence  # All seq_length tokens
        else:
            # Original behavior without BOS
            start_idx = idx * self.seq_length
            end_idx = start_idx + self.seq_length + 1

            if end_idx > len(self.tokens):
                end_idx = len(self.tokens)
                start_idx = end_idx - self.seq_length - 1

            sequence = self.tokens[start_idx:end_idx]
            x = sequence[:-1]
            y = sequence[1:]

        # Create attention mask and position IDs if EOS or BOS token is specified
        if self.eos_token_id is not None or self.bos_token_id is not None:
            attention_mask = self._create_attention_mask(x)
            position_ids = self._create_position_ids(x)
            return x, y, attention_mask, position_ids
        else:
            return x, y

    def _create_attention_mask(self, tokens):
        """
        Create attention mask for the sequence with document boundaries.

        Attention sink pattern with BOS:
        - BOS (always at position 0) is visible to ALL tokens (attention sink)
        - EOS tokens mark document boundaries and block cross-document attention
        - Causal masking is maintained

        Returns a 3D mask (1, seq_len, seq_len) where:
        - True indicates tokens that can be attended to
        - False indicates tokens that should be masked out
        - After batching, becomes (batch_size, 1, seq_len, seq_len)
        """
        seq_len = len(tokens)

        # Create causal mask (lower triangular)
        # Shape: (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        # Special handling for BOS token at position 0 (if present)
        # BOS is the attention sink - all tokens can attend to it
        # This is already handled by the causal mask (all can see position 0)

        # Find EOS token positions for document boundaries
        if self.eos_token_id is not None:
            eos_positions = (tokens == self.eos_token_id).nonzero(as_tuple=False).squeeze(-1)
            if len(eos_positions) > 0:
                eos_list = eos_positions.tolist() if eos_positions.dim() > 0 else [eos_positions.item()]

                # For each EOS token, block attention from later tokens to this and earlier tokens
                # Exception: position 0 (BOS) remains visible to all (attention sink)
                for eos_pos in eos_list:
                    if eos_pos + 1 < seq_len:
                        # Block attention to positions up to and including EOS
                        mask[eos_pos + 1:, :eos_pos + 1] = False

                        # BUT: if BOS is present (position 0), keep it visible (attention sink)
                        if self.bos_token_id is not None:
                            mask[eos_pos + 1:, 0] = True

        # Reshape to 3D: (1, seq_len, seq_len)
        # When batched, this becomes (batch_size, 1, seq_len, seq_len)
        mask = mask.unsqueeze(0)

        return mask

    def _create_position_ids(self, tokens):
        """
        Create position IDs for attention sink pattern with BOS.

        Position 0 is always BOS (the attention sink).
        Positions reset to 1 after each EOS token (document boundary).

        Example: [BOS, tok1, tok2, EOS, tok3, tok4]
        Positions: [0,    1,    2,   3,   1,    2  ]
        """
        seq_len = len(tokens)
        position_ids = torch.arange(seq_len, dtype=torch.long)

        # If BOS is present at position 0, handle EOS resets differently
        if self.bos_token_id is not None and self.eos_token_id is not None:
            # Find EOS positions
            eos_positions = (tokens == self.eos_token_id).nonzero(as_tuple=False).squeeze(-1)
            if len(eos_positions) > 0:
                eos_list = eos_positions.tolist() if eos_positions.dim() > 0 else [eos_positions.item()]

                # After each EOS, reset positions to start from 1 (not 0, since 0 is BOS)
                for eos_pos in eos_list:
                    if eos_pos + 1 < seq_len:
                        # Calculate offset: everything after EOS should restart from 1
                        # Current value at eos_pos+1 is eos_pos+1, we want it to be 1
                        # So subtract (eos_pos)
                        position_ids[eos_pos + 1:] -= eos_pos

        elif self.eos_token_id is not None:
            # Original behavior: reset to 0 after EOS (no BOS)
            eos_positions = (tokens == self.eos_token_id).nonzero(as_tuple=False).squeeze(-1)
            if len(eos_positions) > 0:
                eos_list = eos_positions.tolist() if eos_positions.dim() > 0 else [eos_positions.item()]

                for eos_pos in eos_list:
                    if eos_pos + 1 < seq_len:
                        position_ids[eos_pos + 1:] -= (eos_pos + 1)

        return position_ids
