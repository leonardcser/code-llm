import math
import time
from pathlib import Path

import torch
import yaml
import lightning as L
from lightning.fabric.loggers.tensorboard import TensorBoardLogger

from models.qwen3 import Qwen3
from dataloaders.py150 import Py150DataModule
from trainers.trainer import Trainer


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

    # Initialize logger
    logger = TensorBoardLogger(
        root_dir=str(Path(other_params["log_dir"])),
        name=run_name,
    )

    # Multi-device configuration
    devices = training_params["devices"]
    strategy = training_params["strategy"]

    print(
        f"Using {accelerator.upper()} with {devices} device(s), strategy: {strategy}, precision: {precision}"
    )

    # Create data module
    print("\nLoading data...")
    eos_token_id = data_params.get("eos_token_id")
    bos_token_id = data_params.get("bos_token_id")
    use_attention_mask = eos_token_id is not None or bos_token_id is not None

    if use_attention_mask:
        if eos_token_id is not None:
            print(f"Using attention masking with EOS token ID: {eos_token_id}")
        if bos_token_id is not None:
            print(f"Using attention masking with BOS token ID: {bos_token_id}")

    data_module = Py150DataModule(
        train_file=data_params["train_file"],
        val_file=data_params["val_file"],
        seq_length=data_params["seq_length"],
        batch_size=training_params["batch_size"],
        num_workers=data_params["num_workers"],
        pin_memory=False,  # Don't use pin_memory on MPS
        seed=seed,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    )
    data_module.setup("fit")

    # Create model
    print("\nCreating model...")
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
        scheduler_t_max_steps=training_params["scheduler_t_max_steps"],
        use_attention_mask=use_attention_mask,
    )

    # Compile model if requested
    compile_mode = training_params.get("compile_mode")
    if compile_mode:
        print(f"\nCompiling model with mode: {compile_mode}")
        model.model = torch.compile(model.model, mode=compile_mode)  # type: ignore[assignment]

    total_params, trainable_params = model.get_parameter_counts()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Get dataloaders from DataModule
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Training configuration
    epochs = training_params["epochs"]
    grad_clip = training_params["grad_clip"]
    grad_accum_steps = training_params.get("gradient_accumulation_steps", 1)
    max_batches_per_epoch = training_params.get("max_batches_per_epoch")
    log_every_n_steps = training_params.get("log_every_n_steps")

    # Calculate scheduler_t_max_steps if not provided
    scheduler_t_max_steps = training_params.get("scheduler_t_max_steps")
    if scheduler_t_max_steps is None:
        batches_per_epoch = (
            max_batches_per_epoch
            if max_batches_per_epoch is not None
            else len(train_loader)
        )
        optimizer_steps_per_epoch = math.ceil(batches_per_epoch / grad_accum_steps)
        scheduler_t_max_steps = epochs * optimizer_steps_per_epoch
        print(
            f"\nAuto-calculated scheduler_t_max_steps: {scheduler_t_max_steps} "
            f"({epochs} epochs × {optimizer_steps_per_epoch} optimizer steps per epoch)"
        )

    # Update model with calculated scheduler_t_max_steps
    model.scheduler_t_max_steps = scheduler_t_max_steps

    # Log hyperparameters to TensorBoard
    hparams_dict = {
        # Model params
        "model/vocab_size": data_params["vocab_size"],
        "model/hidden_size": model_params["hidden_size"],
        "model/num_hidden_layers": model_params["num_hidden_layers"],
        "model/num_attention_heads": model_params["num_attention_heads"],
        "model/num_key_value_heads": model_params["num_key_value_heads"],
        "model/intermediate_size": model_params["intermediate_size"],
        "model/max_position_embeddings": model_params["max_position_embeddings"],
        "model/rope_theta": model_params["rope_theta"],
        "model/attention_dropout": model_params["attention_dropout"],
        "model/use_sliding_window": model_params["use_sliding_window"],
        "model/sliding_window": model_params["sliding_window"],
        # Training params
        "training/lr": training_params["lr"],
        "training/batch_size": training_params["batch_size"],
        "training/epochs": training_params["epochs"],
        "training/weight_decay": training_params["weight_decay"],
        "training/grad_clip": training_params["grad_clip"],
        "training/gradient_accumulation_steps": grad_accum_steps,
        "training/warmup_steps": training_params["warmup_steps"],
        "training/use_amp": training_params["use_amp"],
        "training/seed": training_params["seed"],
        "training/devices": training_params["devices"],
        "training/strategy": training_params["strategy"],
        # Data params
        "data/seq_length": data_params["seq_length"],
        "data/num_workers": data_params["num_workers"],
        # System
        "system/accelerator": accelerator,
        "system/precision": precision,
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
    }
    if compile_mode:
        hparams_dict["training/compile_mode"] = compile_mode

    # Initialize Trainer
    trainer = Trainer(
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        precision=precision,
        loggers=logger,
        max_epochs=epochs,
        grad_clip=grad_clip,
        gradient_accumulation_steps=grad_accum_steps,
        max_batches_per_epoch=max_batches_per_epoch,
        log_every_n_steps=log_every_n_steps,
        save_dir=str(save_dir),
        use_attention_mask=use_attention_mask,
    )

    # Train the model
    trainer.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        hparams=hparams_dict,
    )


if __name__ == "__main__":
    main()
