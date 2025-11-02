"""PyTorch Lightning DataModule for tokenized datasets."""

import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

from dataloaders.token_dataloader import TokenDataset


class TokenDataModule(L.LightningDataModule):
    """Lightning DataModule for tokenized datasets."""

    def __init__(
        self,
        dataset_dir: str,
        split_ratio: float = 0.7,
        seq_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        max_tokens: int = 0,
        pad_token_id: Optional[int] = None,
    ):
        """
        Initialize TokenDataModule.

        Args:
            dataset_dir: Path to directory containing chunk_*.bin files
            split_ratio: Ratio of data to use for training (rest for validation)
            seq_length: Length of each sequence
            batch_size: Batch size
            num_workers: Number of workers for data loading
            pin_memory: Pin memory for faster data transfer (use only for CUDA)
            seed: Random seed for reproducibility
            eos_token_id: Token ID for end-of-sequence (enables attention masking)
            bos_token_id: Token ID for beginning-of-sequence (enables attention masking)
            max_tokens: Maximum number of tokens to load (0 for no limit)
            pad_token_id: Token ID for padding (used to pad batches instead of dropping last)
        """
        super().__init__()
        self.save_hyperparameters()

        self.dataset_dir = dataset_dir
        self.split_ratio = split_ratio
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.max_tokens = max_tokens
        self.pad_token_id = pad_token_id

        self.train_dataset = None
        self.val_dataset = None

    def _collate_fn(self, batch):
        """Custom collate function to pad sequences using torch.nn.functional.pad."""
        # Check if dataset returns masks (4 elements) or not (2 elements)
        has_masks = len(batch[0]) == 4

        if has_masks:
            xs, ys, masks, pos_ids = zip(*batch)
        else:
            xs, ys = zip(*batch)

        # Get sequence lengths
        lengths = torch.tensor([x.size(0) for x in xs])
        max_len = lengths.max().item()

        # Check if padding is needed
        needs_padding = (lengths != max_len).any().item()

        if needs_padding and self.pad_token_id is not None:
            # Vectorized padding using F.pad
            # Pad amount for each sequence: (max_len - current_len)
            pad_amounts = max_len - lengths

            # Pad input and target sequences
            xs = [
                F.pad(x, (0, pad_amt.item()), mode="constant", value=self.pad_token_id)
                for x, pad_amt in zip(xs, pad_amounts)
            ]
            ys = [
                F.pad(y, (0, pad_amt.item()), mode="constant", value=self.pad_token_id)
                for y, pad_amt in zip(ys, pad_amounts)
            ]

            if has_masks:
                # Pad attention masks: (1, seq_len, seq_len) -> (1, max_len, max_len)
                # Pad last 2 dimensions: (left, right, top, bottom)
                masks = [
                    F.pad(
                        mask,
                        (0, pad_amt.item(), 0, pad_amt.item()),
                        mode="constant",
                        value=False,
                    )
                    for mask, pad_amt in zip(masks, pad_amounts)
                ]
                # Pad position IDs
                pos_ids = [
                    F.pad(pos_id, (0, pad_amt.item()), mode="constant", value=0)
                    for pos_id, pad_amt in zip(pos_ids, pad_amounts)
                ]

        # Stack tensors
        x = torch.stack(xs)
        y = torch.stack(ys)

        if has_masks:
            attention_mask = torch.stack(masks)
            position_ids = torch.stack(pos_ids)

        # Batch-level padding: pad batch to self.batch_size if needed
        current_batch_size = x.size(0)
        if current_batch_size < self.batch_size and self.pad_token_id is not None:
            num_pad_samples = self.batch_size - current_batch_size
            seq_len = x.size(1)

            # Create padding samples filled with pad_token_id
            pad_x = torch.full(
                (num_pad_samples, seq_len), self.pad_token_id, dtype=x.dtype
            )
            pad_y = torch.full(
                (num_pad_samples, seq_len), self.pad_token_id, dtype=y.dtype
            )

            # Concatenate padding samples
            x = torch.cat([x, pad_x], dim=0)
            y = torch.cat([y, pad_y], dim=0)

            if has_masks:
                # Create padding attention masks (all False)
                pad_attention_mask = torch.zeros(
                    (num_pad_samples, *attention_mask.shape[1:]),
                    dtype=attention_mask.dtype,
                )
                # Create padding position IDs (all zeros)
                pad_position_ids = torch.zeros(
                    (num_pad_samples, seq_len), dtype=position_ids.dtype
                )

                attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=0)
                position_ids = torch.cat([position_ids, pad_position_ids], dim=0)

        if has_masks:
            return x, y, attention_mask, position_ids
        else:
            return x, y

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation."""
        if stage == "fit" or stage is None:
            self.train_dataset = TokenDataset(
                token_dir=self.dataset_dir,
                seq_length=self.seq_length,
                eos_token_id=self.eos_token_id,
                bos_token_id=self.bos_token_id,
                split_ratio=self.split_ratio,
                split_type="train",
                max_tokens=self.max_tokens,
            )
            self.val_dataset = TokenDataset(
                token_dir=self.dataset_dir,
                seq_length=self.seq_length,
                eos_token_id=self.eos_token_id,
                bos_token_id=self.bos_token_id,
                split_ratio=self.split_ratio,
                split_type="val",
                max_tokens=self.max_tokens,
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
            prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=self._collate_fn,
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
            prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=self._collate_fn,
        )
