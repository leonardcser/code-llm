import torch
import yaml
import time
import json
from pathlib import Path
import lightning as L
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from tqdm import tqdm

from models.qwen3 import Qwen3
from dataloaders.py150 import Py150DataModule


def train_epoch(
    fabric,
    model,
    optimizer,
    train_loader,
    grad_clip,
    grad_accum_steps,
    max_batches=None,
    log_every_n_steps=None,
    epoch=0,
):
    """Train for one epoch."""
    model.train()
    train_losses = []
    global_step = epoch * len(train_loader)

    total = min(len(train_loader), max_batches) if max_batches is not None else None
    pbar = tqdm(train_loader, desc="Training", total=total)
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Determine if we're accumulating gradients
        is_accumulating = (batch_idx + 1) % grad_accum_steps != 0

        # Use no_backward_sync to skip gradient synchronization during accumulation
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            # Call LightningModule's training_step
            loss = model.training_step(batch, batch_idx)
            # Fabric backward (handles precision automatically)
            fabric.backward(loss)

        # Step optimizer only when accumulation phase is complete
        if not is_accumulating:
            # Gradient clipping
            if grad_clip > 0:
                fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

            optimizer.step()
            optimizer.zero_grad()

        train_losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})

        # Log per-step metrics if enabled
        current_step = global_step + batch_idx
        if (
            log_every_n_steps is not None
            and (current_step + 1) % log_every_n_steps == 0
        ):
            current_lr = optimizer.param_groups[0]["lr"]
            fabric.log_dict(
                {
                    "train_loss_step": loss.item(),
                    "lr_step": current_lr,
                },
                step=current_step,
            )

    # Handle remaining gradients
    if len(train_losses) > 0 and (len(train_losses) % grad_accum_steps != 0):
        if grad_clip > 0:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    return sum(train_losses) / len(train_losses) if train_losses else 0.0


@torch.no_grad()
def validate(fabric, model, val_loader, max_batches=None):
    """Validate the model."""
    model.eval()
    val_losses = []

    total = min(len(val_loader), max_batches) if max_batches is not None else None
    pbar = tqdm(val_loader, desc="Validation", total=total)
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Call LightningModule's validation_step
        loss = model.validation_step(batch, batch_idx)
        val_losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})

    return sum(val_losses) / len(val_losses) if val_losses else 0.0


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    data_params = params["data"]
    model_params = params["model"]
    training_params = params["training"]
    other_params = params["other"]

    # Set random seeds for reproducibility
    seed = training_params["seed"]
    L.seed_everything(seed, workers=True)

    # Set matmul precision for better MPS performance
    torch.set_float32_matmul_precision("high")

    # Setup directories
    save_dir = Path(other_params["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    prefix = training_params["prefix"]
    run_name = f"{prefix}_{int(time.time())}"

    # Determine accelerator and precision
    if torch.backends.mps.is_available():
        accelerator = "mps"
        precision = "bf16-mixed" if training_params["use_amp"] else "32-true"
    elif torch.cuda.is_available():
        accelerator = "cuda"
        precision = "16-mixed" if training_params["use_amp"] else "32-true"
    else:
        accelerator = "cpu"
        precision = "32-true"

    # Initialize Fabric
    logger = TensorBoardLogger(
        root_dir=str(Path(other_params["log_dir"])),
        name=run_name,
    )

    # Multi-device configuration
    devices = training_params["devices"]
    strategy = training_params["strategy"]

    fabric = L.Fabric(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        loggers=logger,
    )
    fabric.launch()

    fabric.print(
        f"Using {accelerator.upper()} with {devices} device(s), strategy: {strategy}, precision: {precision}"
    )

    # Create data module
    fabric.print("\nLoading data...")
    eos_token_id = data_params.get("eos_token_id")
    use_attention_mask = eos_token_id is not None

    if use_attention_mask:
        fabric.print(f"Using attention masking with EOS token ID: {eos_token_id}")

    data_module = Py150DataModule(
        train_file=data_params["train_file"],
        val_file=data_params["val_file"],
        seq_length=data_params["seq_length"],
        batch_size=training_params["batch_size"],
        num_workers=data_params["num_workers"],
        pin_memory=False,  # Don't use pin_memory on MPS
        seed=seed,
        eos_token_id=eos_token_id,
    )
    data_module.setup("fit")

    # Create model
    fabric.print("\nCreating model...")
    model = Qwen3(
        vocab_size=data_params["vocab_size"],
        hidden_size=model_params["hidden_size"],
        num_hidden_layers=model_params["num_hidden_layers"],
        num_attention_heads=model_params["num_attention_heads"],
        num_key_value_heads=model_params["num_key_value_heads"],
        intermediate_size=model_params["intermediate_size"],
        max_position_embeddings=model_params["max_position_embeddings"],
        rope_theta=model_params["rope_theta"],
        attention_dropout=model_params["attention_dropout"],
        rms_norm_eps=model_params["rms_norm_eps"],
        use_sliding_window=model_params["use_sliding_window"],
        sliding_window=model_params["sliding_window"],
        lr=training_params["lr"],
        weight_decay=training_params["weight_decay"],
        warmup_steps=training_params["warmup_steps"],
        scheduler_t_max=training_params["scheduler_t_max"],
        use_attention_mask=use_attention_mask,
    )

    # Compile model if requested (before fabric.setup)
    compile_mode = training_params.get("compile_mode")
    if compile_mode:
        fabric.print(f"\nCompiling model with mode: {compile_mode}")
        model.model = torch.compile(model.model, mode=compile_mode)  # type: ignore[assignment]

    total_params, trainable_params = model.get_parameter_counts()
    fabric.print(f"\nTotal parameters: {total_params:,}")
    fabric.print(f"Trainable parameters: {trainable_params:,}")
    fabric.print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Get optimizer and scheduler from LightningModule
    optimizer_config = model.configure_optimizers()
    optimizer = optimizer_config["optimizer"]
    scheduler = optimizer_config["lr_scheduler"]["scheduler"]

    # Get dataloaders from DataModule
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Setup with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # Training configuration
    epochs = training_params["epochs"]
    grad_clip = training_params["grad_clip"]
    grad_accum_steps = training_params.get("gradient_accumulation_steps", 1)
    max_batches_per_epoch = training_params.get("max_batches_per_epoch")
    log_every_n_steps = training_params.get("log_every_n_steps")

    # Validate scheduler T_max matches epochs
    scheduler_t_max = training_params["scheduler_t_max"]
    assert scheduler_t_max == epochs, (
        f"scheduler_t_max ({scheduler_t_max}) must equal epochs ({epochs}) "
        "for CosineAnnealingLR to work correctly"
    )

    # Training loop
    fabric.print("\nStarting training...")
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    train_loss = 0
    val_loss = 0

    for epoch in range(epochs):
        fabric.print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_epoch(
            fabric,
            model,
            optimizer,
            train_loader,
            grad_clip,
            grad_accum_steps,
            max_batches_per_epoch,
            log_every_n_steps,
            epoch,
        )
        fabric.print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(
            fabric,
            model,
            val_loader,
            max_batches_per_epoch,
        )
        fabric.print(f"Val loss: {val_loss:.4f}")

        # Get current LR before stepping scheduler
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics via Fabric
        fabric.log_dict(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            },
            step=epoch,
        )

        # Update learning rate
        scheduler.step()
        fabric.print(f"Learning rate: {current_lr:.6f}")

        # Save checkpoint via Fabric
        state = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "hparams": model.hparams,
        }

        # Save latest checkpoint
        fabric.save(save_dir / "latest.ckpt", state)

        # Save best checkpoint and metrics
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            fabric.save(save_dir / "best.ckpt", state)
            fabric.print(f"Saved best model (val_loss: {val_loss:.4f})")

            # Save DVC metrics for best model
            metrics_path = save_dir / "metrics.json"
            metrics = {
                "best_train_loss": float(best_train_loss),
                "best_val_loss": float(best_val_loss),
                "best_epoch": int(best_epoch),
            }
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

    # Update final metrics after training completes
    metrics_path = save_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics.update(
        {
            "final_train_loss": float(train_loss),
            "final_val_loss": float(val_loss),
            "total_epochs": epochs,
        }
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    fabric.print(f"\nSaved metrics to {metrics_path}")

    fabric.print("\nTraining complete!")


if __name__ == "__main__":
    main()
