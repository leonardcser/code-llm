"""PyTorch Lightning module for Qwen3 model training."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from transformers import Qwen3Config, Qwen3ForCausalLM

from models.transformer import Transformer


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
        rms_norm_eps: RMSNorm epsilon
        use_sliding_window: Whether to use sliding window attention
        sliding_window: Sliding window size

    Returns:
        Qwen3 model initialized from scratch
    """
    # Create Qwen3 configuration
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        attention_dropout=attention_dropout,
        rms_norm_eps=rms_norm_eps,
        use_sliding_window=use_sliding_window,
        sliding_window=sliding_window,
        tie_word_embeddings=False,
    )

    # Initialize model from config (random weights)
    model = Qwen3ForCausalLM(config)

    # Ensure token embeddings match vocab size
    model.resize_token_embeddings(vocab_size)

    return model


class Qwen3(Transformer):
    """Lightning module wrapper for Qwen3 causal language model."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.1,
        rms_norm_eps: float = 1e-6,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        scheduler_t_max_steps: int | None = None,
        use_attention_mask: bool = False,
    ):
        """
        Initialize Lightning Qwen3 module.

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
            rms_norm_eps: RMSNorm epsilon
            use_sliding_window: Whether to use sliding window attention
            sliding_window: Sliding window size
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps for learning rate
            scheduler_t_max_steps: T_max for CosineAnnealingLR (in steps). Will be auto-calculated if None.
            use_attention_mask: Whether to use attention masking for document boundaries
        """
        super().__init__()
        self.save_hyperparameters()

        # Create Qwen3 model
        self.model = create_qwen3_model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.scheduler_t_max_steps = scheduler_t_max_steps
        self.use_attention_mask = use_attention_mask

    def forward(self, x, attention_mask=None, position_ids=None):
        """Forward pass through the model."""
        return self.model(x, attention_mask=attention_mask, position_ids=position_ids)

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        """Training step - compute loss and log metrics."""
        # Handle both 2-tuple and 4-tuple returns from dataloader
        if self.use_attention_mask:
            x, y, attention_mask, position_ids = batch
        else:
            x, y = batch
            attention_mask = None
            position_ids = None

        # Forward pass (returns ModelOutput with .logits)
        outputs = self(x, attention_mask=attention_mask, position_ids=position_ids)
        logits = outputs.logits

        # Calculate loss
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        return loss

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        """Validation step - compute loss and log metrics."""
        # Handle both 2-tuple and 4-tuple returns from dataloader
        if self.use_attention_mask:
            x, y, attention_mask, position_ids = batch
        else:
            x, y = batch
            attention_mask = None
            position_ids = None

        # Forward pass (returns ModelOutput with .logits)
        outputs = self(x, attention_mask=attention_mask, position_ids=position_ids)
        logits = outputs.logits

        # Calculate loss
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        return loss

    def configure_optimizers(self):  # type: ignore[override]
        """Configure optimizer and learning rate scheduler."""
        # Create optimizer (only optimize trainable parameters)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Setup learning rate scheduler with warmup
        # Note: scheduler_t_max_steps must be set by train.py before calling this
        assert self.scheduler_t_max_steps is not None, (
            "scheduler_t_max_steps must be set before configure_optimizers is called"
        )
        t_max_steps: int = self.scheduler_t_max_steps  # Type narrowing for Pyright

        if self.warmup_steps > 0:
            # Warmup scheduler: linearly increase LR from 0 to target LR
            warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(1.0, (step + 1) / self.warmup_steps),
            )
            # Cosine annealing after warmup
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=t_max_steps - self.warmup_steps
            )
            # Combine warmup + cosine
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def get_parameter_counts(self):
        """Get total and trainable parameter counts."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer_path: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        """Generate text completion given a prompt.

        Args:
            prompt: Input prompt text
            tokenizer_path: Path to tokenizer binary file
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k most likely tokens

        Returns:
            Generated text (prompt + completion)
        """
        import torch
        import tokenizer as tok

        # Load tokenizer
        tokenizer = tok.load(tokenizer_path)

        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Create attention mask
        attention_mask = torch.ones_like(input_tensor)

        # Generate tokens with KV caching (DynamicCache used internally)
        outputs = self.model.generate(
            input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k else 50,
            do_sample=True,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode generated tokens
        generated_ids = outputs[0].tolist()
        generated_text = tokenizer.decode(generated_ids)

        return generated_text
