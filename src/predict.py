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


@torch.no_grad()
def predict_next_token(
    lightning_module: Transformer,
    tokenizer: tok.Tokenizer,
    text: str,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> tuple[int, str, dict[str, float]]:
    """Predict next token given input text.

    Args:
        lightning_module: Lightning module (Qwen3)
        tokenizer: Tokenizer for encoding/decoding
        text: Input text
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k most likely tokens

    Returns:
        token_id: Predicted token ID
        token_text: Decoded token text
        probs: Top 10 token probabilities as dict
    """
    # Encode input text
    tokens = tok.encode(text, tokenizer)

    # Convert to tensor
    x = torch.tensor([tokens], dtype=torch.long, device=lightning_module.device)

    # Get model predictions (returns ModelOutput with .logits)
    outputs = lightning_module(x)
    logits = outputs.logits

    # Get logits for the last position (next token)
    next_token_logits = logits[0, -1, :]

    # Apply temperature
    if temperature != 1.0:
        next_token_logits = next_token_logits / temperature

    # Get probabilities
    probs = torch.softmax(next_token_logits, dim=-1)

    # Apply top-k filtering if specified
    if top_k is not None:
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        # Sample from top-k
        sample_idx = torch.multinomial(top_k_probs, num_samples=1)
        token_id = int(top_k_indices[sample_idx].item())
    else:
        # Sample from full distribution
        token_id = int(torch.multinomial(probs, num_samples=1).item())

    # Decode token
    token_text = tok.decode([token_id], tokenizer)

    # Get top 10 probabilities for display
    top_10_probs, top_10_indices = torch.topk(probs, k=10)
    top_10_dict = {
        tok.decode([int(idx.item())], tokenizer): prob.item()
        for idx, prob in zip(top_10_indices, top_10_probs)
    }

    return token_id, token_text, top_10_dict


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

    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = tok.load(tokenizer_path)
    print(f"Tokenizer loaded (vocab_size: {tokenizer.vocab_size()})")

    # Load model
    lightning_module, _hparams = load_model(checkpoint_path, fabric)

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

    token_id, token_text, top_10 = predict_next_token(
        lightning_module, tokenizer, test_text, temperature=0.5
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


if __name__ == "__main__":
    main()
