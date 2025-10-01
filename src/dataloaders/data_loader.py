import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional


def _seed_worker(worker_id):
    """Worker initialization function for reproducible data loading."""
    worker_seed_info = torch.utils.data.get_worker_info()
    if worker_seed_info is not None:
        seed = worker_seed_info.seed % (2**32)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


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
        Create attention mask for the sequence.

        Returns a 1D binary mask (shape: seq_len) where 1 indicates tokens to attend to.
        For Qwen3, this should be shape (batch_size, sequence_length) after batching.

        Note: Causal masking is handled internally by the model. Document boundaries
        are handled via position_ids reset.
        """
        seq_len = len(tokens)

        # Return a 1D mask of all ones (all tokens are valid, no padding)
        # Shape: (seq_len,)
        mask = torch.ones(seq_len, dtype=torch.long)

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


def get_dataloaders(
    train_file: str,
    val_file: str,
    seq_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: Optional[int] = None,
    eos_token_id: Optional[int] = None,
):
    """
    Create train and validation dataloaders.

    Args:
        train_file: Path to training tokens binary file
        val_file: Path to validation tokens binary file
        seq_length: Length of each sequence
        batch_size: Batch size
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster data transfer (use only for CUDA)
        seed: Random seed for reproducibility
        eos_token_id: Token ID for end-of-sequence (enables attention masking)

    Returns:
        train_loader, val_loader
    """
    train_dataset = TokenDataset(train_file, seq_length, eos_token_id)
    val_dataset = TokenDataset(val_file, seq_length, eos_token_id)

    # Setup worker initialization and generator for reproducibility
    if seed is not None:
        worker_init_fn = _seed_worker if num_workers > 0 else None
        # Create generator for reproducible shuffling
        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        worker_init_fn = None
        generator = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader
