"""PyTorch Lightning DataModule for Py150 dataset."""

import lightning as L
import torch
from torch.utils.data import DataLoader
from typing import Optional

from dataloaders.py150_dataloader import TokenDataset


class Py150DataModule(L.LightningDataModule):
    """Lightning DataModule for Py150 tokenized dataset."""

    def __init__(
        self,
        dataset_file: str,
        split_ratio: float = 0.7,
        seq_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        max_tokens: int = 0,
    ):
        """
        Initialize Py150 DataModule.

        Args:
            dataset_file: Path to combined dataset tokens binary file
            split_ratio: Ratio of data to use for training (rest for validation)
            seq_length: Length of each sequence
            batch_size: Batch size
            num_workers: Number of workers for data loading
            pin_memory: Pin memory for faster data transfer (use only for CUDA)
            seed: Random seed for reproducibility
            eos_token_id: Token ID for end-of-sequence (enables attention masking)
            bos_token_id: Token ID for beginning-of-sequence (enables attention masking)
            max_tokens: Maximum number of tokens to load (0 for no limit)
        """
        super().__init__()
        self.save_hyperparameters()

        self.dataset_file = dataset_file
        self.split_ratio = split_ratio
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.max_tokens = max_tokens

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation."""
        if stage == "fit" or stage is None:
            self.train_dataset = TokenDataset(
                self.dataset_file,
                self.seq_length,
                self.eos_token_id,
                self.bos_token_id,
                self.split_ratio,
                "train",
                self.seed,
                self.max_tokens,
            )
            self.val_dataset = TokenDataset(
                self.dataset_file,
                self.seq_length,
                self.eos_token_id,
                self.bos_token_id,
                self.split_ratio,
                "val",
                self.seed,
                self.max_tokens,
            )

    def train_dataloader(self):
        """Create training dataloader."""
        assert self.train_dataset is not None, (
            "Dataset not initialized. Call setup() first."
        )

        # Create generator for reproducible shuffling
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=generator,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        assert self.val_dataset is not None, (
            "Dataset not initialized. Call setup() first."
        )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
