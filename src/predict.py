"""Predict next token using trained transformer model."""

import sys
import time
import torch
from pathlib import Path
from transformers import StaticCache

# Add tokenizer build directory to Python path
tokenizer_build_path = Path(__file__).parent.parent / "tokenizer" / "build"
sys.path.insert(0, str(tokenizer_build_path))

# Also add tokenizer directory for the Python wrapper
tokenizer_path = Path(__file__).parent.parent / "tokenizer"
sys.path.insert(0, str(tokenizer_path))

from tokenizer import Tokenizer
from models.model import create_qwen3_model

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


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract params from checkpoint
    params = checkpoint["params"]
    data_params = params["data"]
    model_params = params["model"]

    # Create Qwen3 model
    model = create_qwen3_model(
        vocab_size=data_params["vocab_size"],
        hidden_size=model_params["hidden_size"],
        num_hidden_layers=model_params["num_hidden_layers"],
        num_attention_heads=model_params["num_attention_heads"],
        num_key_value_heads=model_params.get(
            "num_key_value_heads", model_params["num_attention_heads"]
        ),
        intermediate_size=model_params["intermediate_size"],
        max_position_embeddings=model_params["max_position_embeddings"],
        rope_theta=model_params.get("rope_theta", 10000.0),
        attention_dropout=model_params.get("attention_dropout", 0.1),
        hidden_dropout=model_params.get("hidden_dropout", 0.1),
        rms_norm_eps=model_params.get("rms_norm_eps", 1e-6),
        use_sliding_window=model_params.get("use_sliding_window", False),
        sliding_window=model_params.get("sliding_window", 4096),
    )

    # Load weights
    state_dict = checkpoint["model_state_dict"]
    # Remove _orig_mod. prefix if present (from torch.compile)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(
        f"Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f})"
    )
    return model, params


@torch.no_grad()
def predict_next_token(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    text: str,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> tuple[int, str, dict[str, float]]:
    """Predict next token given input text.

    Args:
        model: Trained transformer model
        tokenizer: Tokenizer for encoding/decoding
        text: Input text
        device: Device to run inference on
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k most likely tokens

    Returns:
        token_id: Predicted token ID
        token_text: Decoded token text
        probs: Top 10 token probabilities as dict
    """
    # Encode input text
    tokens = tokenizer.encode(text)

    # Convert to tensor
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    # Get model predictions (returns ModelOutput with .logits)
    outputs = model(x)
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
        token_id = top_k_indices[sample_idx].item()
    else:
        # Sample from full distribution
        token_id = torch.multinomial(probs, num_samples=1).item()

    # Decode token
    token_text = tokenizer.decode([token_id])

    # Get top 10 probabilities for display
    top_10_probs, top_10_indices = torch.topk(probs, k=10)
    top_10_dict = {
        tokenizer.decode([idx.item()]): prob.item()
        for idx, prob in zip(top_10_indices, top_10_probs)
    }

    return token_id, token_text, top_10_dict


def generate_text(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    use_cache: bool = True,
    use_compile: bool = False,
) -> str:
    """Generate text using model.generate() with KV caching for optimal performance.

    This implementation uses HuggingFace's built-in generation with StaticCache
    for much faster inference compared to manual token-by-token generation.

    Args:
        model: Trained transformer model (Qwen2ForCausalLM)
        tokenizer: Tokenizer for encoding/decoding
        prompt: Initial prompt text
        device: Device to run inference on
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: If set, only sample from top k most likely tokens
        use_cache: Whether to use KV caching (default: True)
        use_compile: Whether to use torch.compile (default: False, set True in main)

    Returns:
        Generated text (prompt + generated tokens)
    """
    # Encode prompt with custom tokenizer
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Initialize StaticCache for best performance with torch.compile
    cache = None
    if use_cache:
        cache = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=model.config.max_position_embeddings,
            device=device,
            dtype=model.dtype,
        )

    # Create attention mask (all ones since we have a single prompt, no padding)
    attention_mask = torch.ones_like(input_tensor)

    # Generate tokens using model.generate() with KV caching
    # This is 10-100x faster than manual loop due to cached attention
    with torch.no_grad():
        outputs = model.generate(
            input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k if top_k else 50,  # Default to top_k=50 if None
            do_sample=True,
            past_key_values=cache,
            use_cache=use_cache,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode generated tokens with custom tokenizer
    generated_ids = outputs[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    return generated_text


def main():
    # Configuration
    checkpoint_path = "out/train/checkpoints/best.pt"
    tokenizer_path = "out/tokenize/tok.bin"

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer(tokenizer_path)
    print(f"Tokenizer loaded (vocab_size: {tokenizer.vocab_size})")

    # Load model
    model, _params = load_model(checkpoint_path, device)

    # Enable torch.compile for 2-4x additional speedup during generation
    # Note: First run will be slower due to compilation, subsequent runs are much faster
    # print("\nEnabling torch.compile for optimized inference...")
    # try:
    #     model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
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
        model, tokenizer, test_text, device, temperature=0.6
    )

    print(f"\nPredicted next token: {token_text!r} (ID: {token_id})")
    print("\nTop 10 most likely tokens:")
    for i, (token, prob) in enumerate(top_10.items(), 1):
        print(f"  {i:2d}. {token!r:20s} {prob * 100:6.2f}%")

    # Text generation with KV cache optimization
    print("\n" + "=" * 80)
    print("TEXT GENERATION (Optimized with StaticCache)")
    print("=" * 80)

    prompt = "def calculate("
    print(f"\nPrompt: {prompt!r}")
    print(f"Generating {100} tokens with temperature=0.6, top_k=50")

    # Timed generation with KV cache
    print("\n[Generating with StaticCache...]")
    start_time = time.time()
    generated = generate_text(
        model, tokenizer, prompt, device, max_tokens=100, temperature=0.6, top_k=50
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
