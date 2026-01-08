"""OpenAI-compatible API server for code completion."""

import argparse
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Qwen3ForCausalLM

# Try to import the tokenizer module
try:
    import tokenizer as tok
except ImportError:
    tok = None


# Global model storage
model_state = {"model": None, "tokenizer": None, "config": None, "eos_token": None}


class CompletionRequest(BaseModel):
    """OpenAI Completion API request."""

    model: str = "code-llm"
    prompt: str | list[str]
    temperature: float = 0.8
    max_tokens: int = 100
    top_k: int = 50
    stop: list[str] | None = None
    n: int = 1
    echo: bool = False
    uncertainty_threshold: float = 0  # Min confidence for accepting completions


class ModelInfo(BaseModel):
    """Model information response."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "user"


class ModelListResponse(BaseModel):
    """List models response."""

    object: str = "list"
    data: list[ModelInfo]


def encode_text(text: str, tokenizer) -> list[int]:
    """Encode text using the custom tokenizer.

    Args:
        text: Input text
        tokenizer: Loaded tokenizer

    Returns:
        List of token IDs
    """
    if tok is None:
        raise RuntimeError("Tokenizer module not available")

    tokens = tok.encode(text, tokenizer)
    # Prepend BOS token
    bos_token_id = tokenizer.bos_token_id
    return [bos_token_id] + tokens


def decode_tokens(token_ids: list[int], tokenizer) -> str:
    """Decode tokens using the custom tokenizer.

    Args:
        token_ids: List of token IDs
        tokenizer: Loaded tokenizer

    Returns:
        Decoded text
    """
    if tok is None:
        raise RuntimeError("Tokenizer module not available")

    return tok.decode(token_ids, tokenizer)


@torch.no_grad()
def generate_completion(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    stop: list[str] | None = None,
    uncertainty_threshold: float = 0.4,
) -> tuple[str, int, int]:
    """Generate code completion.

    Args:
        prompt: Input code prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        stop: Stop sequences
        uncertainty_threshold: Minimum confidence threshold (0-1). If max probability
            falls below this for any token, return empty completion.

    Returns:
        Tuple of (generated_text, prompt_tokens, completion_tokens)
    """
    model = model_state["model"]
    tokenizer = model_state["tokenizer"]

    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")

    # Encode prompt
    input_ids = encode_text(prompt, tokenizer)
    prompt_tokens = len(input_ids)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_tensor)

    # Start timing
    start_time = time.time()

    # Generate with output scores to check confidence
    outputs = model.generate(
        input_tensor,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=temperature > 0,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Calculate generation time
    generation_time = time.time() - start_time

    # Check uncertainty - examine the confidence of generated tokens
    generated_ids = outputs.sequences[0].tolist()

    # Check if we have scores (only for newly generated tokens)
    if hasattr(outputs, "scores") and len(outputs.scores) > 0:
        min_confidence = 1.0
        for score in outputs.scores:
            # Get probabilities for the batch
            probs = torch.softmax(score[0], dim=-1)
            max_prob = probs.max().item()
            min_confidence = min(min_confidence, max_prob)

        # If any token has low confidence, return empty completion
        if min_confidence < uncertainty_threshold:
            print(
                f"Low confidence detected ({min_confidence:.3f} < {uncertainty_threshold:.3f}), returning empty completion"
            )
            return "", prompt_tokens, 0

    # Decode
    full_text = decode_tokens(generated_ids, tokenizer)

    # Extract only the completion (remove prompt)
    prompt_text = decode_tokens(input_ids, tokenizer)
    completion = full_text[len(prompt_text) :]

    # Get cached EOS token string
    eos_text = model_state["eos_token"]

    # Ensure we end with EOS token if it's not already there
    if eos_text and not completion.endswith(eos_text):
        # If we hit max_tokens without EOS, append it
        completion += eos_text

    # Handle stop sequences (before EOS removal)
    if stop:
        for stop_seq in stop:
            if stop_seq in completion:
                completion = completion[: completion.index(stop_seq)]
                break

    # Remove the EOS token from the final output (it's implicit)
    if eos_text and completion.endswith(eos_text):
        completion = completion[: -len(eos_text)]

    completion_tokens = len(generated_ids) - prompt_tokens

    # Calculate and log tokens per second
    tokens_per_second = (
        completion_tokens / generation_time if generation_time > 0 else 0
    )
    print(
        f"Generated {completion_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tok/s)"
    )

    return completion, prompt_tokens, completion_tokens


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on startup, cleanup on shutdown."""
    # Load model
    model_dir = app.state.model_dir
    print(f"Loading model from {model_dir}...")

    # Load HuggingFace model
    model = Qwen3ForCausalLM.from_pretrained(model_dir, device_map="auto")

    # Move to GPU if available
    model.eval()

    # Load tokenizer info
    tokenizer_info_path = Path(model_dir) / "tokenizer_info.json"
    with open(tokenizer_info_path) as f:
        tokenizer_info = json.load(f)

    # Load custom tokenizer
    tokenizer_path = tokenizer_info["tokenizer_path"]
    print(f"Loading tokenizer from {tokenizer_path}...")

    if tok is None:
        raise RuntimeError(
            "Tokenizer module not available. Please ensure tokenizer is built."
        )

    tokenizer = tok.load(tokenizer_path)
    print("Tokenizer loaded")

    # Load export metadata
    metadata_path = Path(model_dir) / "export_metadata.json"
    with open(metadata_path) as f:
        config = json.load(f)

    # Cache EOS token string from special_tokens map
    eos_token = None
    for token_str, special_token in tokenizer.special_tokens.items():
        if special_token.id == tokenizer.eos_token_id:
            eos_token = token_str
            break

    # Store in global state
    model_state["model"] = model
    model_state["tokenizer"] = tokenizer
    model_state["config"] = config
    model_state["eos_token"] = eos_token

    print("\n" + "=" * 80)
    print("Code completion server ready!")
    print("=" * 80)

    yield

    # Cleanup
    model_state["model"] = None
    model_state["tokenizer"] = None
    model_state["config"] = None
    model_state["eos_token"] = None


# Create FastAPI app
app = FastAPI(
    title="Code Completion API",
    description="OpenAI-compatible API for code completion",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Code Completion API Server",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/v1/models")
async def list_models() -> ModelListResponse:
    """List available models (OpenAI-compatible)."""
    config = model_state["config"]
    if config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_id = "code-llm"
    return ModelListResponse(
        data=[
            ModelInfo(
                id=model_id,
                created=int(time.time()),
            )
        ]
    )


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest) -> dict[str, Any]:
    """Create code completion (OpenAI-compatible)."""
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Handle single prompt or list of prompts
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

    # Generate completions for all prompts
    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, prompt in enumerate(prompts):
        try:
            completion_text, prompt_tokens, completion_tokens = generate_completion(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                stop=request.stop,
                uncertainty_threshold=request.uncertainty_threshold,
            )

            # Include prompt in output if echo=True
            output_text = prompt + completion_text if request.echo else completion_text

            choices.append(
                {
                    "index": idx,
                    "text": output_text,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            )

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    # Build response
    response = {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }

    return response


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


def main():
    """Start the API server."""
    parser = argparse.ArgumentParser(description="Start code completion API server")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="out/export",
        help="Directory containing exported model",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    args = parser.parse_args()

    # Store model directory in app state
    app.state.model_dir = args.model_dir

    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
