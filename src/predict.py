"""Predict next token using trained transformer model."""

import time
import torch
import lightning as L

from models.qwen3 import Qwen3
import tokenizer as tok

from models.transformer import Transformer

# TODO: FUTURE ENHANCEMENT - HuggingFace Tokenizer Integration
# ============================================================
# Currently using custom C++ tokenizer directly, which works fine for manual inference.
#
# OPTIONAL: Create a HuggingFace PreTrainedTokenizerFast wrapper to enable:
#   1. Using model.generate() instead of custom sampling loop
#   2. Integration with HF pipelines and ecosystem
#   3. Automatic padding/truncation utilities
#
# Implementation approach:
#   - Create src/tokenizers/custom_tokenizer.py
#   - Subclass PreTrainedTokenizerFast
#   - Wrap C++ tokenizer encode/decode methods
#   - Map special token IDs (bos, eos, pad, unk)
#   - Register with AutoTokenizer if desired
#
# For now, keeping current approach as it works perfectly for training and
# custom inference. Only create wrapper if you need model.generate() later.


def load_model(checkpoint_path: str, fabric: L.Fabric):
    """Load model from checkpoint using Fabric."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load checkpoint using Fabric (handles device placement automatically)
    checkpoint = fabric.load(checkpoint_path)

    hparams = checkpoint["hparams"]
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
    """Generate text using the Lightning module's generate method.

    Args:
        lightning_module: Lightning module (Qwen3) with generate method
        tokenizer_path: Path to tokenizer binary file
        prompt: Initial prompt text
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: If set, only sample from top k most likely tokens

    Returns:
        Generated text (prompt + generated tokens)
    """
    return lightning_module.generate(
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )


def main():
    # Configuration
    checkpoint_path = "out/train/checkpoints/best.ckpt"
    tokenizer_path = "out/tokenize/tok.bin"

    # Initialize Fabric for inference
    fabric = L.Fabric(accelerator="auto", devices=1)
    fabric.launch()

    # Load model
    lightning_module, _hparams = load_model(checkpoint_path, fabric)
    print(f"\nTokenizer will be loaded from {tokenizer_path}")

    # Enable torch.compile for 2-4x additional speedup during generation
    # Note: First run will be slower due to compilation, subsequent runs are much faster
    # print("\nEnabling torch.compile for optimized inference...")
    # try:
    #     lightning_module.model.forward = torch.compile(lightning_module.model.forward, mode="reduce-overhead", fullgraph=True)
    #     print("✓ torch.compile enabled (first generation will compile, subsequent runs will be fast)")
    # except Exception as e:
    #     print(f"⚠ torch.compile failed (continuing without it): {e}")

    # Example prediction
    print("\n" + "=" * 80)
    print("NEXT TOKEN PREDICTION (Single Token Demo)")
    print("=" * 80)

    test_text = "def hello():"
    print(f"\nInput text: {test_text!r}")

    token_id, token_text, top_10 = lightning_module.generate_once(
        prompt=test_text,
        tokenizer_path=tokenizer_path,
        temperature=0.5,
    )

    print(f"\nPredicted next token: {token_text!r} (ID: {token_id})")
    print("\nTop 10 most likely tokens:")
    for i, (token, prob) in enumerate(top_10.items(), 1):
        print(f"  {i:2d}. {token!r:20s} {prob * 100:6.2f}%")

    # Text generation with KV cache optimization
    print("\n" + "=" * 80)
    print("TEXT GENERATION (Optimized with KV Cache)")
    print("=" * 80)

    prompt = "def sum_list(nums: List[int]"
    print(f"\nPrompt: {prompt!r}")
    print("Generating 100 tokens with temperature=0.5, top_k=50")

    # Timed generation with KV cache
    print("\n[Generating with KV cache...]")
    start_time = time.time()
    generated = generate_text(
        lightning_module,
        tokenizer_path,
        prompt,
        max_tokens=100,
        temperature=0.5,
        top_k=50,
    )
    generation_time = time.time() - start_time

    print("\nGenerated text:")
    print("-" * 80)
    print(generated)
    print("-" * 80)
    print(f"\n✓ Generation completed in {generation_time:.3f}s")
    print(f"  Tokens/second: {100 / generation_time:.1f}")

    print("\n" + "=" * 80)
    print("TEXT GENERATION (Step-by-Step Token Prediction)")
    print("=" * 80)

    prompt = "def sum_list(nums: List[int]"
    print(f"\nPrompt: {prompt!r}")
    print("Generating 50 tokens step by step with temperature=0.5, top_k=50\n")

    max_steps = 50
    generated_text = prompt

    for step in range(max_steps):
        token_id, token_text, top_10 = lightning_module.generate_once(
            prompt=generated_text,
            tokenizer_path=tokenizer_path,
            temperature=0.5,
            top_k=50,
        )

        generated_text += token_text

        print(f"Step {step + 1:02d}: Token: {token_text!r} (ID: {token_id})")
        print("  Top 5 likely tokens:")
        for i, (tok_str, prob) in enumerate(list(top_10.items())[:5], 1):
            print(f"    {i}. {tok_str!r:10s} {prob * 100:6.2f}%")
        print("-" * 50)

    print("\nGenerated text:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)


if __name__ == "__main__":
    main()
