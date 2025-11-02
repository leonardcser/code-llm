"""Dataset and utilities for tokenized data."""

import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple
import os


class TokenDataset(Dataset):
    """Memory-efficient dataset for loading tokenized data from chunked binary files using memory mapping."""

    def __init__(
        self,
        token_dir: str,
        seq_length: int = 512,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        split_ratio: float = 0.7,
        split_type: str = "train",
        max_tokens: int = 0,
    ):
        """
        Args:
            token_dir: Path to directory containing chunk_*.bin files
            seq_length: Length of each training sequence
            eos_token_id: Token ID for end-of-sequence (document boundaries)
            bos_token_id: Token ID for beginning-of-sequence (document boundaries)
            split_ratio: Ratio of data to use for training (rest for validation)
            split_type: Either "train" or "val" to specify which split to use
            max_tokens: Maximum number of tokens to use (0 for no limit)
        """
        self.seq_length = seq_length
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.token_dir = token_dir

        # Find and sort chunk files
        chunk_files = sorted(glob.glob(os.path.join(token_dir, "chunk_*.bin")))
        if not chunk_files:
            raise ValueError(f"No chunk files found in {token_dir}")

        print(f"Found {len(chunk_files)} chunk files in {token_dir}")

        # Build index: map global token position to (chunk_idx, local_position)
        self.chunk_files = chunk_files
        self.chunk_offsets: List[int] = []  # Starting position of each chunk in global space
        self.chunk_sizes: List[int] = []  # Size of each chunk in tokens
        self.memmaps: List[Optional[np.memmap]] = [None] * len(chunk_files)  # Lazy-loaded memmaps

        total_tokens = 0
        for chunk_file in chunk_files:
            # Get file size without loading data
            file_size = os.path.getsize(chunk_file)
            num_tokens = file_size // np.dtype(np.uint32).itemsize

            self.chunk_offsets.append(total_tokens)
            self.chunk_sizes.append(num_tokens)
            total_tokens += num_tokens

            print(f"  {os.path.basename(chunk_file)}: {num_tokens:,} tokens")

        print(f"Total tokens across all chunks: {total_tokens:,}")

        # Apply max_tokens limit
        if max_tokens > 0 and total_tokens > max_tokens:
            print(f"Limiting to {max_tokens:,} tokens (from {total_tokens:,})")
            total_tokens = max_tokens

        # Split into train/val
        split_idx = int(total_tokens * split_ratio)

        if split_type == "train":
            self.start_token = 0
            self.end_token = split_idx
        elif split_type == "val":
            self.start_token = split_idx
            self.end_token = total_tokens
        else:
            raise ValueError(f"split_type must be 'train' or 'val', got {split_type}")

        self.total_tokens = self.end_token - self.start_token

        # Calculate number of sequences
        if self.bos_token_id is not None:
            # With BOS each sequence consumes exactly seq_length tokens
            self.n_sequences = self.total_tokens // self.seq_length
        else:
            # Without BOS requires seq_length + 1 tokens per sequence
            self.n_sequences = (self.total_tokens - 1) // self.seq_length

        print(f"Split: {split_type}, tokens: {self.total_tokens:,}, sequences: {self.n_sequences:,}")

        # Count special tokens (requires scanning, but memory-efficient)
        if eos_token_id is not None or bos_token_id is not None:
            self._count_special_tokens()

    def _count_special_tokens(self):
        """Count special tokens across all chunks (memory-efficient scan)."""
        eos_count = 0
        bos_count = 0

        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            chunk_start = self.chunk_offsets[chunk_idx]
            chunk_end = chunk_start + self.chunk_sizes[chunk_idx]

            # Skip chunks outside our split range
            if chunk_end <= self.start_token or chunk_start >= self.end_token:
                continue

            # Load chunk with memmap
            memmap = self._get_chunk_memmap(chunk_idx)

            # Calculate relevant range within this chunk
            local_start = max(0, self.start_token - chunk_start)
            local_end = min(self.chunk_sizes[chunk_idx], self.end_token - chunk_start)

            chunk_data = memmap[local_start:local_end]

            if self.eos_token_id is not None:
                eos_count += np.sum(chunk_data == self.eos_token_id)
            if self.bos_token_id is not None:
                bos_count += np.sum(chunk_data == self.bos_token_id)

        if self.eos_token_id is not None:
            print(f"Found {eos_count:,} EOS token boundaries")
        if self.bos_token_id is not None:
            print(f"Found {bos_count:,} BOS token boundaries")

    def _get_chunk_memmap(self, chunk_idx: int) -> np.memmap:
        """Lazy-load memory map for a chunk."""
        if self.memmaps[chunk_idx] is None:
            self.memmaps[chunk_idx] = np.memmap(
                self.chunk_files[chunk_idx],
                dtype=np.uint32,
                mode='r'
            )
        return self.memmaps[chunk_idx]

    def _find_chunk(self, global_pos: int) -> Tuple[int, int]:
        """Find which chunk contains global_pos and return (chunk_idx, local_pos)."""
        # Binary search for efficiency
        left, right = 0, len(self.chunk_offsets) - 1

        while left <= right:
            mid = (left + right) // 2
            chunk_start = self.chunk_offsets[mid]
            chunk_end = chunk_start + self.chunk_sizes[mid]

            if global_pos < chunk_start:
                right = mid - 1
            elif global_pos >= chunk_end:
                left = mid + 1
            else:
                # Found it
                return mid, global_pos - chunk_start

        raise ValueError(f"Position {global_pos} not found in any chunk")

    def _get_token_range(self, start_pos: int, length: int) -> np.ndarray:
        """Get a range of tokens, potentially spanning multiple chunks."""
        # Convert to global positions
        global_start = self.start_token + start_pos

        # Find starting chunk
        chunk_idx, _ = self._find_chunk(global_start)

        # Collect tokens (may span multiple chunks)
        tokens = []
        remaining = length
        current_pos = global_start

        while remaining > 0 and chunk_idx < len(self.chunk_files):
            chunk_start = self.chunk_offsets[chunk_idx]
            chunk_size = self.chunk_sizes[chunk_idx]

            # How many tokens to take from this chunk
            local_pos = current_pos - chunk_start
            available = min(chunk_size - local_pos, remaining)

            if available <= 0:
                chunk_idx += 1
                current_pos = self.chunk_offsets[chunk_idx] if chunk_idx < len(self.chunk_files) else current_pos
                continue

            # Get tokens from this chunk
            memmap = self._get_chunk_memmap(chunk_idx)
            chunk_tokens = memmap[local_pos:local_pos + available]
            tokens.append(chunk_tokens)

            remaining -= available
            current_pos += available
            chunk_idx += 1

        # Concatenate if we read from multiple chunks
        if len(tokens) == 1:
            return tokens[0]
        else:
            return np.concatenate(tokens)

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
            length = self.seq_length

            if start_idx + length > self.total_tokens:
                # Handle edge case at end
                start_idx = max(0, self.total_tokens - self.seq_length)

            sequence = self._get_token_range(start_idx, length)
            sequence = torch.from_numpy(sequence.astype(np.int64))

            # Prepend BOS to input
            bos_tensor = torch.tensor([self.bos_token_id], dtype=torch.long)
            x = torch.cat([bos_tensor, sequence[:-1]])  # [BOS] + first (seq_length-1) tokens
            y = sequence  # All seq_length tokens
        else:
            # Original behavior without BOS
            start_idx = idx * self.seq_length
            length = self.seq_length + 1

            if start_idx + length > self.total_tokens:
                # Handle edge case at end
                start_idx = max(0, self.total_tokens - length)

            sequence = self._get_token_range(start_idx, length)
            sequence = torch.from_numpy(sequence.astype(np.int64))
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
                is_after_eos = position_ids.unsqueeze(1) > eos_indices.unsqueeze(0)  # (seq_len, num_eos)

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
