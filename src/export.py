"""Export trained model to serving-optimized format."""

import json
from pathlib import Path

import torch
import yaml

from models.qwen3 import Qwen3


def export_model(
    checkpoint_path: str,
    output_dir: str,
    params_path: str = "params.yaml",
):
    """Export Lightning checkpoint to HuggingFace format for serving.

    Args:
        checkpoint_path: Path to Lightning checkpoint (.ckpt file)
        output_dir: Directory to save exported model
        params_path: Path to params.yaml file
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load params to get model configuration
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    tokenize_params = params["tokenize"]

    # Load Lightning checkpoint (weights_only=False needed for Lightning checkpoints)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract hyperparameters from checkpoint
    hparams = checkpoint["hparams"]

    # Create Lightning module with saved hyperparameters
    model = Qwen3(**hparams)

    # Load state dict
    # Handle torch.compile() wrapper (_orig_mod prefix) if present
    state_dict = checkpoint["model"]
    if any(k.startswith("model._orig_mod.") for k in state_dict.keys()):
        # Strip _orig_mod prefix from compiled model state dict
        state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Extract the underlying HuggingFace model
    hf_model = model.model

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to {output_dir}...")

    # Save HuggingFace model (config.json + pytorch_model.bin)
    hf_model.save_pretrained(output_dir)

    # Save tokenizer path reference
    tokenizer_info = {
        "tokenizer_path": tokenize_params["tok_file"],
        "vocab_size": tokenize_params["vocab_size"],
        "bos_token": tokenize_params["bos_token"],
        "eos_token": tokenize_params["eos_token"],
        "pad_token": tokenize_params["pad_token"],
        "cursor_token": tokenize_params.get("cursor_token"),
        "edit_start_token": tokenize_params.get("edit_start_token"),
        "edit_end_token": tokenize_params.get("edit_end_token"),
    }

    with open(output_path / "tokenizer_info.json", "w") as f:
        json.dump(tokenizer_info, f, indent=2)

    # Save export metadata
    metadata = {
        "source_checkpoint": checkpoint_path,
        "model_architecture": "Qwen3ForCausalLM",
        "framework": "transformers",
        "export_format": "huggingface",
        "parameters": {
            "vocab_size": hparams["vocab_size"],
            "hidden_size": hparams["hidden_size"],
            "num_hidden_layers": hparams["num_hidden_layers"],
            "num_attention_heads": hparams["num_attention_heads"],
            "num_key_value_heads": hparams["num_key_value_heads"],
            "intermediate_size": hparams["intermediate_size"],
            "max_position_embeddings": hparams["max_position_embeddings"],
        },
    }

    with open(output_path / "export_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("✓ Model exported successfully!")
    print(f"  - Model weights: {output_dir}/pytorch_model.bin")
    print(f"  - Model config: {output_dir}/config.json")
    print(f"  - Tokenizer info: {output_dir}/tokenizer_info.json")
    print(f"  - Export metadata: {output_dir}/export_metadata.json")
    print("\nTo load this model for inference:")
    print("  from transformers import Qwen3ForCausalLM")
    print(f"  model = Qwen3ForCausalLM.from_pretrained('{output_dir}')")


def main():
    """Export best checkpoint from training."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    training_params = params["training"]
    checkpoint_path = Path(training_params["save_dir"]) / "best.ckpt"

    # Default export directory
    export_dir = "out/export"

    export_model(
        checkpoint_path=str(checkpoint_path),
        output_dir=export_dir,
    )


if __name__ == "__main__":
    main()
