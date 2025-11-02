"""Inference script for trained transformer models."""

import argparse
import time
import lightning as L
import yaml
from typing import cast

from models.qwen3 import Qwen3
from models.transformer import Transformer


def load_model(checkpoint_path: str, fabric: L.Fabric):
    """Load model from checkpoint using Fabric."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load checkpoint using Fabric (handles device placement automatically)
    checkpoint = fabric.load(checkpoint_path)

    hparams = checkpoint["hparams"]

    # Load Qwen3 model
    print("Using model: Qwen3")
    lightning_module = Qwen3(**hparams)

    # Load state dict into reconstructed module
    # Handle torch.compile() wrapper (_orig_mod prefix) if present
    state_dict = checkpoint["model"]
    if any(k.startswith("model._orig_mod.") for k in state_dict.keys()):
        # Strip _orig_mod prefix from compiled model state dict
        state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    lightning_module.load_state_dict(state_dict)

    # Setup Lightning module with Fabric for inference
    lightning_module = fabric.setup(lightning_module)
    lightning_module.eval()

    print(
        f"Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f})"
    )
    return lightning_module, hparams


def generate_text(
    lightning_module: Transformer,
    tokenizer_path: str,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    """Generate text using the Lightning module.

    Args:
        lightning_module: Lightning module (Qwen3)
        tokenizer_path: Path to tokenizer binary file
        prompt: Initial prompt text
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: If set, only sample from top k most likely tokens

    Returns:
        Generated text
    """
    return cast(Qwen3, lightning_module).generate(
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )


def main():
    # Load configuration from params.yaml
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        print("Loaded configuration from params.yaml")
    except Exception as e:
        print(f"Warning: Could not load params.yaml: {e}")
        params = {}

    # Extract default values from params
    tokenize_params = params.get("tokenize", {})
    data_params = params.get("data", {})
    training_params = params.get("training", {})

    default_checkpoint = training_params.get("save_dir", "out/train") + "/best.ckpt"
    default_tokenizer = tokenize_params.get("tok_file", "out/tokenize/tok.bin")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Inference with trained transformer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_checkpoint,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=default_tokenizer,
        help="Path to tokenizer binary file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Input prompt text",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    args = parser.parse_args()

    # Initialize Fabric for inference
    fabric = L.Fabric(accelerator="auto", devices=1)
    fabric.launch()

    # Load model
    lightning_module, _hparams = load_model(args.checkpoint, fabric)
    print(f"\nTokenizer: {args.tokenizer}")

    # Text generation with next token prediction
    print("\n" + "=" * 80)
    print("NEXT TOKEN PREDICTION - Single Token Demo")
    print("=" * 80)

    # Use provided prompt or default example
    test_text = args.prompt if args.prompt else "def hello():"
    print(f"\nInput text: {test_text!r}")

    token_id, token_text, top_10 = lightning_module.generate_once(
        prompt=test_text,
        tokenizer_path=args.tokenizer,
        temperature=args.temperature,
    )

    print(f"\nPredicted next token: {token_text!r} (ID: {token_id})")
    print("\nTop 10 most likely tokens:")
    for i, (token, prob) in enumerate(top_10.items(), 1):
        print(f"  {i:2d}. {token!r:20s} {prob * 100:6.2f}%")

    # Text generation with KV cache optimization
    print("\n" + "=" * 80)
    print("TEXT GENERATION (Optimized with KV Cache)")
    print("=" * 80)

    prompt = args.prompt if args.prompt else "def sum_list(nums: List[int]"
    print(f"\nPrompt: {prompt!r}")
    print(
        f"Generating {args.max_tokens} tokens with temperature={args.temperature}, top_k={args.top_k}"
    )

    # Timed generation with KV cache
    print("\n[Generating...]")
    start_time = time.time()
    generated = generate_text(
        lightning_module,
        args.tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    generation_time = time.time() - start_time

    print("\nGenerated text:")
    print("-" * 80)
    print(generated)
    print("-" * 80)
    print(f"\n✓ Generation completed in {generation_time:.3f}s")
    print(f"  Tokens/second: {args.max_tokens / generation_time:.1f}")


if __name__ == "__main__":
    main()
