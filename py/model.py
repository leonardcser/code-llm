import torch
import torch.nn as nn
from typing import Optional
from peft import LoraConfig, get_peft_model


class TransformerLM(nn.Module):
    """Transformer Language Model wrapper using nn.Transformer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()

        self.d_model = d_model

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)

        # Transformer (decoder-only for language modeling)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(
        self, x: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input token indices [batch_size, seq_length]
            tgt_mask: Target mask [seq_length, seq_length]

        Returns:
            logits: [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = x.shape

        # Token embeddings
        token_emb = self.token_embedding(x)  # [B, S, D]

        # Positional embeddings
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)  # [1, S]
        pos_emb = self.pos_embedding(positions)  # [1, S, D]

        # Combine embeddings
        embeddings = self.dropout(token_emb + pos_emb)

        # Create causal mask
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_length, device=x.device
            )

        # Use decoder-only mode (encoder processes same input)
        # For decoder-only LM, use the same embeddings for both encoder and decoder
        memory = self.transformer.encoder(embeddings, mask=tgt_mask, is_causal=True)
        output = self.transformer.decoder(
            embeddings, memory, tgt_mask=tgt_mask, tgt_is_causal=True
        )

        # Output projection
        logits = self.output_proj(output)

        return logits


def create_model_with_lora(
    vocab_size: int,
    d_model: int = 512,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    max_seq_length: int = 512,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
):
    """
    Create TransformerLM with optional LoRA.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout rate
        max_seq_length: Maximum sequence length
        use_lora: Whether to apply LoRA
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout

    Returns:
        model: TransformerLM with optional LoRA applied
    """
    # Create base model
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_length=max_seq_length,
    )

    if use_lora:
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=[
                "out_proj",
                "linear1",
                "linear2",
            ],  # Apply to attention and FFN
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,  # Custom model
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
