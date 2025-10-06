"""Dataset and utilities for Py150 tokenized data."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional


class TokenDataset(Dataset):
    """Dataset for loading tokenized data from binary files with proper document boundary handling."""

    def __init__(
        self,
        token_file: str,
        seq_length: int = 512,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        split_ratio: float = 0.7,
        split_type: str = "train",
        seed: Optional[int] = None,
        max_tokens: int = 0,
    ):
        """
        Args:
            token_file: Path to binary token file (uint32)
            seq_length: Length of each training sequence
            eos_token_id: Token ID for end-of-sequence (document boundaries)
            bos_token_id: Token ID for beginning-of-sequence (document boundaries)
            split_ratio: Ratio of data to use for training (rest for validation)
            split_type: Either "train" or "val" to specify which split to use
            seed: Random seed for reproducible splitting
            max_tokens: Maximum number of tokens to load (0 for no limit)
        """
        self.seq_length = seq_length
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id

        # Load tokens from binary file (uint32)
        all_tokens = np.fromfile(token_file, dtype=np.uint32)
        all_tokens = torch.from_numpy(all_tokens).long()

        # Truncate tokens if max_tokens is specified
        if max_tokens > 0 and len(all_tokens) > max_tokens:
            print(f"Truncating tokens from {len(all_tokens)} to {max_tokens}")
            all_tokens = all_tokens[:max_tokens]

        # Split tokens based on split_ratio
        split_idx = int(len(all_tokens) * split_ratio)

        if split_type == "train":
            self.tokens = all_tokens[:split_idx]
        elif split_type == "val":
            self.tokens = all_tokens[split_idx:]
        else:
            raise ValueError(f"split_type must be 'train' or 'val', got {split_type}")

        # Calculate number of sequences
        if self.bos_token_id is not None:
            # With BOS each sequence consumes exactly seq_length tokens
            self.n_sequences = len(self.tokens) // self.seq_length
        else:
            # Original behavior without BOS requires seq_length + 1 tokens per sequence
            self.n_sequences = (len(self.tokens) - 1) // self.seq_length

        print(
            f"Loaded {len(self.tokens)} tokens from {token_file} ({split_type} split)"
        )
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
            x = torch.cat(
                [bos_tensor, sequence[:-1]]
            )  # [BOS] + first (seq_length-1) tokens
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
            # Create binary tensor: 1 where EOS, 0 elsewhere
            is_eos = (tokens == self.eos_token_id).long()

            # Cumulative sum gives us document segment IDs
            # After each EOS, the cumsum increments, marking a new document
            doc_ids = torch.cumsum(is_eos, dim=0)

            # Create document boundary mask: (seq_len, seq_len)
            # doc_ids[i] != doc_ids[j] means tokens i and j are in different documents
            # We want to block attention across document boundaries
            doc_mask = doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0)

            # Combine causal mask with document boundary mask
            mask = mask & doc_mask

            # Restore BOS attention sink if present (position 0 visible to all)
            if self.bos_token_id is not None:
                mask[:, 0] = True

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

        # Vectorized position reset using cumulative operations
        if self.eos_token_id is not None:
            # Find indices where EOS tokens appear
            eos_indices = (tokens == self.eos_token_id).nonzero(as_tuple=True)[0]

            if len(eos_indices) > 0:
                # Create offset tensor: stores the value to subtract at each position
                # Strategy: For each position, find the cumulative offset from all EOS tokens before it

                # Create a matrix: (seq_len, num_eos) where entry [i,j] indicates if position i is after EOS j
                is_after_eos = position_ids.unsqueeze(1) > eos_indices.unsqueeze(
                    0
                )  # (seq_len, num_eos)

                # For BOS pattern: offset = eos_pos for each EOS before this position
                # For non-BOS pattern: offset = eos_pos + 1
                if self.bos_token_id is not None:
                    offsets_per_eos = eos_indices.unsqueeze(0)  # (1, num_eos)
                else:
                    offsets_per_eos = (eos_indices + 1).unsqueeze(0)  # (1, num_eos)

                # Sum offsets for all EOS tokens that come before each position
                # Only count EOS tokens that are actually before this position
                total_offset = (is_after_eos * offsets_per_eos).sum(dim=1)  # (seq_len,)

                position_ids = position_ids - total_offset

        return position_ids
