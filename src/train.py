import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from pathlib import Path
from tqdm import tqdm
import yaml
import time
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from models.model import create_qwen3_model, count_parameters
from dataloaders.data_loader import get_dataloaders


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    grad_clip=1.0,
    gradient_accumulation_steps=1,
    max_batches=None,
    scaler=None,
    use_amp=False,
    use_attention_mask=False,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(
        train_loader,
        desc="Training",
        total=max_batches if max_batches else len(train_loader),
    )
    step = -1
    for step, batch in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break

        # Handle both 2-tuple and 4-tuple returns from dataloader
        if use_attention_mask:
            x, y, attention_mask, position_ids = batch
            x = x.to(device)
            y = y.to(device)
            attention_mask = attention_mask.to(device)
            position_ids = position_ids.to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            attention_mask = None
            position_ids = None

        # Forward pass with mixed precision (returns ModelOutput with .logits)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(x, attention_mask=attention_mask, position_ids=position_ids)
            logits = outputs.logits

            # Calculate loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Only update weights every gradient_accumulation_steps
        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp and scaler is not None:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * gradient_accumulation_steps
        pbar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

    # Apply any remaining accumulated gradients
    if (step + 1) % gradient_accumulation_steps != 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

    num_batches = max_batches if max_batches is not None else len(train_loader)
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model, val_loader, device, max_batches=None, use_amp=False, use_attention_mask=False
):
    """Validate the model."""
    model.eval()
    total_loss = 0

    pbar = tqdm(
        val_loader,
        desc="Validation",
        total=max_batches if max_batches else len(val_loader),
    )
    for step, batch in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break

        # Handle both 2-tuple and 4-tuple returns from dataloader
        if use_attention_mask:
            x, y, attention_mask, position_ids = batch
            x = x.to(device)
            y = y.to(device)
            attention_mask = attention_mask.to(device)
            position_ids = position_ids.to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            attention_mask = None
            position_ids = None

        # Forward pass with mixed precision (returns ModelOutput with .logits)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(x, attention_mask=attention_mask, position_ids=position_ids)
            logits = outputs.logits

            # Calculate loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    num_batches = max_batches if max_batches is not None else len(val_loader)
    return total_loss / num_batches


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    data_params = params["data"]
    model_params = params["model"]
    training_params = params["training"]
    other_params = params["other"]

    # Set random seeds for reproducibility
    seed = training_params.get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device (use MPS on Apple Silicon, CUDA if available, else CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create data loaders with attention masking if EOS token is specified
    print("\nLoading data...")
    eos_token_id = data_params.get("eos_token_id")
    use_attention_mask = eos_token_id is not None

    if use_attention_mask:
        print(f"Using attention masking with EOS token ID: {eos_token_id}")

    train_loader, val_loader = get_dataloaders(
        data_params["train_file"],
        data_params["val_file"],
        seq_length=data_params["seq_length"],
        batch_size=training_params["batch_size"],
        num_workers=data_params["num_workers"],
        seed=seed,
        eos_token_id=eos_token_id,
    )

    # Create model
    print("\nCreating model...")
    model = create_qwen3_model(
        vocab_size=data_params["vocab_size"],
        hidden_size=model_params["hidden_size"],
        num_hidden_layers=model_params["num_hidden_layers"],
        num_attention_heads=model_params["num_attention_heads"],
        num_key_value_heads=model_params["num_key_value_heads"],
        intermediate_size=model_params["intermediate_size"],
        max_position_embeddings=model_params["max_position_embeddings"],
        rope_theta=model_params.get("rope_theta", 10000.0),
        attention_dropout=model_params.get("attention_dropout", 0.1),
        rms_norm_eps=model_params.get("rms_norm_eps", 1e-6),
        use_sliding_window=model_params.get("use_sliding_window", False),
        sliding_window=model_params.get("sliding_window", 4096),
    ).to(device)

    # Compile model if requested
    compile_mode = training_params.get("compile_mode")
    if compile_mode:
        print(f"\nCompiling model with mode: {compile_mode}")
        model = torch.compile(model, mode=compile_mode)

    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Create optimizer (only optimize trainable parameters)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_params["lr"],
        weight_decay=training_params["weight_decay"],
    )

    # Setup mixed precision training (only supported on CUDA)
    use_amp = training_params.get("use_amp", False) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None
    if use_amp:
        print("\nUsing automatic mixed precision training (AMP)")
    elif training_params.get("use_amp", False) and device.type != "cuda":
        print(f"\nWarning: AMP is not supported on {device.type}, disabling AMP")

    # Learning rate scheduler with warmup
    warmup_steps = training_params.get("warmup_steps", 0)
    t_max = training_params.get("scheduler_t_max", training_params["epochs"])

    if warmup_steps > 0:
        # Warmup scheduler: linearly increase LR from 0 to target LR
        warmup_scheduler = LambdaLR(
            optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_steps)
        )
        # Cosine annealing after warmup
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=t_max - warmup_steps)
        # Combine warmup + cosine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        print(
            f"\nUsing warmup ({warmup_steps} steps) + CosineAnnealing (T_max={t_max - warmup_steps})"
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
        print(f"\nUsing CosineAnnealing (T_max={t_max})")

    save_dir = Path(other_params["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"llm_{int(time.time())}"
    log_dir = Path(other_params["log_dir"]) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")

    for epoch in range(training_params["epochs"]):
        print(f"\nEpoch {epoch + 1}/{training_params['epochs']}")

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training_params["grad_clip"],
            training_params.get("gradient_accumulation_steps", 1),
            training_params.get("max_batches_per_epoch"),
            scaler=scaler,
            use_amp=use_amp,
            use_attention_mask=use_attention_mask,
        )
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(
            model,
            val_loader,
            device,
            training_params.get("max_batches_per_epoch"),
            use_amp=use_amp,
            use_attention_mask=use_attention_mask,
        )
        print(f"Val loss: {val_loss:.4f}")

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)

        # Update learning rate
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        # Flush logs to disk
        writer.flush()

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "params": params,
        }

        # Save latest checkpoint
        torch.save(checkpoint, save_dir / "latest.pt")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, save_dir / "best.pt")
            print(f"Saved best model (val_loss: {val_loss:.4f})")

    print("\nTraining complete!")
    writer.close()


if __name__ == "__main__":
    main()
