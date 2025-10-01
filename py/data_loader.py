import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    """Dataset for loading tokenized data from binary files."""

    def __init__(self, token_file: str, seq_length: int = 512):
        """
        Args:
            token_file: Path to binary token file (uint32)
            seq_length: Length of each training sequence
        """
        self.seq_length = seq_length

        # Load tokens from binary file (uint32)
        tokens = np.fromfile(token_file, dtype=np.uint32)
        self.tokens = torch.from_numpy(tokens).long()

        # Calculate number of sequences
        self.n_sequences = len(self.tokens) // seq_length

        print(f"Loaded {len(self.tokens)} tokens from {token_file}")
        print(f"Created {self.n_sequences} sequences of length {seq_length}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        """Returns input and target sequences."""
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

        return x, y


def get_dataloaders(
    train_file: str,
    val_file: str,
    seq_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    Create train and validation dataloaders.

    Args:
        train_file: Path to training tokens binary file
        val_file: Path to validation tokens binary file
        seq_length: Length of each sequence
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader
    """
    train_dataset = TokenDataset(train_file, seq_length)
    val_dataset = TokenDataset(val_file, seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
