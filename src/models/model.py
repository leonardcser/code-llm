"""Qwen3 model creation using transformers library."""

from transformers import Qwen2Config, Qwen2ForCausalLM


def create_qwen3_model(
    vocab_size: int,
    hidden_size: int = 512,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 8,
    num_key_value_heads: int = 8,
    intermediate_size: int = 2048,
    max_position_embeddings: int = 2048,
    rope_theta: float = 10000.0,
    attention_dropout: float = 0.1,
    hidden_dropout: float = 0.1,
    rms_norm_eps: float = 1e-6,
    use_sliding_window: bool = False,
    sliding_window: int = 4096,
):
    """Create a Qwen3 model from scratch (random initialization).

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension size
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        intermediate_size: FFN intermediate dimension
        max_position_embeddings: Maximum sequence length
        rope_theta: RoPE theta parameter
        attention_dropout: Attention dropout rate
        hidden_dropout: Hidden layer dropout rate
        rms_norm_eps: RMSNorm epsilon
        use_sliding_window: Whether to use sliding window attention
        sliding_window: Sliding window size

    Returns:
        Qwen3 model initialized from scratch
    """
    # Create Qwen3 configuration
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        attention_dropout=attention_dropout,
        hidden_dropout_prob=hidden_dropout,
        rms_norm_eps=rms_norm_eps,
        use_sliding_window=use_sliding_window,
        sliding_window=sliding_window,
        tie_word_embeddings=False,
    )

    # Initialize model from config (random weights)
    model = Qwen2ForCausalLM(config)

    # Ensure token embeddings match vocab size
    model.resize_token_embeddings(vocab_size)

    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
