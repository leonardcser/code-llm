"""Custom Trainer built with Lightning Fabric for Qwen3 training."""

import json
from pathlib import Path
from typing import Optional, Union

import lightning as L
from lightning.fabric.loggers.logger import Logger
import torch
from tqdm import tqdm

from models.transformer import Transformer


class Trainer:
    """Custom Trainer for LightningModule using Fabric.

    This trainer is specifically designed for language model training with features like:
    - Gradient accumulation
    - Gradient clipping
    - Learning rate warmup and scheduling
    - Per-step and per-epoch logging
    - Best checkpoint tracking
    - Support for attention masking
    """

    def __init__(
        self,
        accelerator: str = "auto",
        strategy: str = "auto",
        devices: Union[list[int], str, int] = "auto",
        precision: str = "32-true",
        loggers: Optional[Logger] = None,
        max_epochs: int = 100,
        grad_clip: float = 1.0,
        gradient_accumulation_steps: int = 1,
        log_every_n_steps: Optional[int] = None,
        save_dir: str = "./out/train/checkpoints",
        use_attention_mask: bool = False,
        tokenizer_path: Optional[str] = None,
        val_preview_prompts: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            accelerator: Hardware to run on ("cpu", "cuda", "mps", "auto")
            strategy: Strategy for multi-device training ("dp", "ddp", "fsdp", etc.)
            devices: Number/list of devices to use
            precision: Training precision ("32-true", "16-mixed", "bf16-mixed")
            loggers: Logger(s) for experiment tracking
            max_epochs: Maximum number of training epochs
            grad_clip: Gradient clipping max norm (0 to disable)
            gradient_accumulation_steps: Number of batches to accumulate before optimizer step
            log_every_n_steps: Log training metrics every N steps (None to disable)
            save_dir: Directory for saving checkpoints
            use_attention_mask: Whether batches include attention masks
            tokenizer_path: Path to tokenizer binary file for generating preview completions
            val_preview_prompts: List of prompts to generate completions for during validation
        """
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,  # type: ignore[arg-type]
            loggers=loggers,
        )

        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_every_n_steps = log_every_n_steps
        self.save_dir = Path(save_dir)
        self.use_attention_mask = use_attention_mask
        self.tokenizer_path = tokenizer_path
        self.val_preview_prompts = val_preview_prompts or []

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_train_loss = float("inf")
        self.best_epoch = 0

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        model: Transformer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        hparams: Optional[dict] = None,
    ) -> None:
        """
        Main training loop.

        Args:
            model: Transformer to train
            train_loader: Training data loader
            val_loader: Validation data loader
            hparams: Hyperparameters dictionary for logging
        """
        self.fabric.launch()

        # Get optimizer and scheduler from model
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config["optimizer"]  # type: ignore[index]
        scheduler = optimizer_config["lr_scheduler"]["scheduler"]  # type: ignore[index]

        # Setup model, optimizer, and dataloaders with Fabric
        model, optimizer = self.fabric.setup(model, optimizer)
        train_loader, val_loader = self.fabric.setup_dataloaders(
            train_loader, val_loader
        )

        # Log hyperparameters if provided
        if hparams is not None and self.fabric.logger is not None:
            self.fabric.logger.log_hyperparams(
                hparams, metrics={"hp/val_loss": float("inf")}
            )

        self.fabric.print("\nStarting training...")

        # Initialize loss tracking
        train_loss = 0.0
        val_loss = 0.0

        # Log initial learning rate at step 0
        initial_lr = optimizer.param_groups[0]["lr"]
        self.fabric.log("lr", initial_lr, step=self.global_step)

        # Training loop
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            self.fabric.print(f"\nEpoch {epoch + 1}/{self.max_epochs}")

            # Training phase
            train_loss = self.train_loop(model, optimizer, scheduler, train_loader)
            self.fabric.print(f"Train loss: {train_loss:.4f}")

            # Validation phase
            val_loss = self.val_loop(model, val_loader)
            self.fabric.print(f"Val loss: {val_loss:.4f}")

            # Get current learning rate for logging
            current_lr = optimizer.param_groups[0]["lr"]

            # Log epoch-level metrics
            self.fabric.log_dict(
                {
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                    "lr": current_lr,
                },
                step=self.global_step,
            )

            self.fabric.print(f"Learning rate: {current_lr:.6f}")

            # Save checkpoints
            self._save_checkpoints(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=train_loss,
                val_loss=val_loss,
            )

        # Save final metrics
        self._save_final_metrics(train_loss, val_loss)
        self.fabric.print("\nTraining complete!")

    def train_loop(
        self,
        model: Transformer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: torch.utils.data.DataLoader,
    ) -> float:
        """
        Training loop for one epoch.

        Args:
            model: Transformer to train
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            train_loader: Training data loader

        Returns:
            Average training loss for the epoch
        """
        model.train()
        train_losses = []

        total = len(train_loader)
        pbar = self._create_progress_bar(train_loader, total, "Training")

        for batch_idx, batch in enumerate(pbar):
            # Determine if we're accumulating gradients
            is_accumulating = (batch_idx + 1) % self.gradient_accumulation_steps != 0

            # Use no_backward_sync to skip gradient synchronization during accumulation
            with self.fabric.no_backward_sync(model, enabled=is_accumulating):  # type: ignore[arg-type]
                # Forward pass and compute loss
                loss_output = model.training_step(batch, batch_idx)
                # Ensure loss is a tensor
                if isinstance(loss_output, dict):
                    loss_tensor = loss_output["loss"]
                else:
                    loss_tensor = loss_output
                assert isinstance(loss_tensor, torch.Tensor)

                loss_value = loss_tensor.detach().item()

                if self.gradient_accumulation_steps > 1:
                    loss_for_backward = loss_tensor / self.gradient_accumulation_steps
                else:
                    loss_for_backward = loss_tensor

                # Backward pass
                self.fabric.backward(loss_for_backward)

            # Step optimizer only when accumulation phase is complete
            if not is_accumulating:
                # Gradient clipping
                if self.grad_clip > 0:
                    self.fabric.clip_gradients(
                        model, optimizer, max_norm=self.grad_clip
                    )

                optimizer.step()
                optimizer.zero_grad()

                # Step the scheduler after optimizer step
                scheduler.step()

            train_losses.append(loss_value)
            self._update_progress_bar(pbar, {"loss": loss_value})

            # Log per-step metrics if enabled
            if (
                self.log_every_n_steps is not None
                and (self.global_step + 1) % self.log_every_n_steps == 0
            ):
                self.fabric.log_dict(
                    {"loss/train_step": loss_value},
                    step=self.global_step,
                )

            self.global_step += 1

        # Handle remaining gradients at end of epoch
        if len(train_losses) > 0 and (
            len(train_losses) % self.gradient_accumulation_steps != 0
        ):
            if self.grad_clip > 0:
                self.fabric.clip_gradients(model, optimizer, max_norm=self.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        return sum(train_losses) / len(train_losses) if train_losses else 0.0

    @torch.no_grad()
    def val_loop(
        self,
        model: Transformer,
        val_loader: torch.utils.data.DataLoader,
    ) -> float:
        """
        Validation loop for one epoch.

        Args:
            model: Transformer to validate
            val_loader: Validation data loader

        Returns:
            Average validation loss for the epoch
        """
        model.eval()
        val_losses = []

        total = len(val_loader)
        pbar = self._create_progress_bar(val_loader, total, "Validation")

        for batch_idx, batch in enumerate(pbar):
            # Forward pass and compute loss
            loss_output = model.validation_step(batch, batch_idx)
            # Ensure loss is a tensor
            if isinstance(loss_output, dict):
                loss = loss_output["loss"]
            else:
                loss = loss_output
            assert isinstance(loss, torch.Tensor)
            val_losses.append(loss.item())
            self._update_progress_bar(pbar, {"loss": loss.item()})

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        # Generate preview completions if configured
        if (
            self.tokenizer_path
            and self.val_preview_prompts
            and self.fabric.logger is not None
        ):
            self._generate_validation_previews(model)

        return avg_val_loss

    def _save_checkpoints(
        self,
        model: Transformer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loss: float,
        val_loss: float,
    ) -> None:
        """
        Save latest and best checkpoints.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Learning rate scheduler to save
            train_loss: Current training loss
            val_loss: Current validation loss
        """
        state = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": self.current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "hparams": model.hparams,
        }

        # Save latest checkpoint
        self.fabric.save(self.save_dir / "latest.ckpt", state)

        # Save best checkpoint and update metrics
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_train_loss = train_loss
            self.best_epoch = self.current_epoch
            self.fabric.save(self.save_dir / "best.ckpt", state)
            self.fabric.print(f"Saved best model (val_loss: {val_loss:.4f})")

            # Update hyperparameter metric in logger
            if self.fabric.logger is not None:
                self.fabric.log("hp/val_loss", val_loss, step=self.global_step)  # type: ignore[call-arg]

            # Save metrics for DVC or other tracking
            metrics = {
                "best_train_loss": float(self.best_train_loss),
                "best_val_loss": float(self.best_val_loss),
                "best_epoch": int(self.best_epoch),
            }
            with open(self.save_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    def _save_final_metrics(
        self, final_train_loss: float, final_val_loss: float
    ) -> None:
        """
        Save final metrics after training completes.

        Args:
            final_train_loss: Final training loss
            final_val_loss: Final validation loss
        """
        metrics_path = self.save_dir / "metrics.json"

        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {}

        metrics.update(
            {
                "final_train_loss": float(final_train_loss),
                "final_val_loss": float(final_val_loss),
                "total_epochs": self.max_epochs,
            }
        )

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        self.fabric.print(f"\nSaved metrics to {metrics_path}")

    def _create_progress_bar(
        self,
        iterable: torch.utils.data.DataLoader,
        total: int,
        desc: str,
    ):
        """
        Create progress bar (only on rank 0).

        Args:
            iterable: Iterable to wrap
            total: Total number of items
            desc: Description for progress bar

        Returns:
            Progress bar or original iterable
        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, desc=desc)
        return iterable

    def _update_progress_bar(self, pbar, metrics: dict) -> None:
        """
        Update progress bar with metrics.

        Args:
            pbar: Progress bar to update
            metrics: Dictionary of metrics to display
        """
        if isinstance(pbar, tqdm):
            pbar.set_postfix(metrics)

    def _generate_validation_previews(self, model: Transformer) -> None:
        """
        Generate text completion previews and log to TensorBoard.

        Args:
            model: Model to generate completions with
        """
        if not self.fabric.is_global_zero:
            return

        if self.tokenizer_path is None:
            self.fabric.print(
                "[WARN] Skipping validation previews (no tokenizer path provided)"
            )
            return

        for i, prompt in enumerate(self.val_preview_prompts):
            try:
                # Generate completion
                completion = model.generate(
                    prompt=prompt,
                    tokenizer_path=self.tokenizer_path,
                    max_new_tokens=50,
                    temperature=0.3,
                    top_k=50,
                )

                # Format as markdown for better readability
                formatted_text = f"**Prompt:** `{prompt}`\n\n**Completion:**\n```python\n{completion}\n```"

                # Log to TensorBoard
                self.fabric.logger.experiment.add_text(  # type: ignore[union-attr]
                    f"validation_previews/prompt_{i}",
                    formatted_text,
                    global_step=self.global_step,
                )

            except Exception as e:
                self.fabric.print(
                    f"  Warning: Failed to generate preview for {prompt!r}: {e}"
                )
