import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import argparse

from model import create_model_with_lora, count_parameters
from data_loader import get_dataloaders


def train_epoch(model, train_loader, optimizer, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc="Training")
    for _, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)

        # Calculate loss
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0

    pbar = tqdm(val_loader, desc="Validation")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)

        # Calculate loss
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train Transformer LM with LoRA")

    # Data arguments
    parser.add_argument(
        "--train_file",
        type=str,
        default="../out/train.bin",
        help="Path to training tokens",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="../out/val.bin",
        help="Path to validation tokens",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50256,
        help="Vocabulary size (256 bytes + BPE merges + special tokens)",
    )

    # Model arguments
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_encoder_layers", type=int, default=6, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=6, help="Number of decoder layers"
    )
    parser.add_argument(
        "--dim_feedforward", type=int, default=2048, help="Feedforward dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")

    # LoRA arguments
    parser.add_argument(
        "--use_lora", action="store_true", default=True, help="Use LoRA"
    )
    parser.add_argument(
        "--no_lora", dest="use_lora", action="store_false", help="Disable LoRA"
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping"
    )

    # Other arguments
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of data loading workers"
    )

    args = parser.parse_args()

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

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(
        args.train_file,
        args.val_file,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create model
    print("\nCreating model...")
    model = create_model_with_lora(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=args.seq_length,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Create optimizer (only optimize trainable parameters)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"Val loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        }

        # Save latest checkpoint
        torch.save(checkpoint, save_dir / "latest.pt")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, save_dir / "best.pt")
            print(f"Saved best model (val_loss: {val_loss:.4f})")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
