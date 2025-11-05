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
model_state = {"model": None, "tokenizer": None, "config": None}


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
) -> tuple[str, int, int]:
    """Generate code completion.

    Args:
        prompt: Input code prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        stop: Stop sequences

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

    # Generate
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
    )

    # Decode
    generated_ids = outputs[0].tolist()
    full_text = decode_tokens(generated_ids, tokenizer)

    # Extract only the completion (remove prompt)
    prompt_text = decode_tokens(input_ids, tokenizer)
    completion = full_text[len(prompt_text) :]

    # Handle stop sequences
    if stop:
        for stop_seq in stop:
            if stop_seq in completion:
                completion = completion[: completion.index(stop_seq)]
                break

    completion_tokens = len(generated_ids) - prompt_tokens

    return completion, prompt_tokens, completion_tokens


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on startup, cleanup on shutdown."""
    # Load model
    model_dir = app.state.model_dir
    print(f"Loading model from {model_dir}...")

    # Load HuggingFace model
    model = Qwen3ForCausalLM.from_pretrained(model_dir)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

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

    # Store in global state
    model_state["model"] = model
    model_state["tokenizer"] = tokenizer
    model_state["config"] = config

    print("\n" + "=" * 80)
    print("Code completion server ready!")
    print("=" * 80)

    yield

    # Cleanup
    model_state["model"] = None
    model_state["tokenizer"] = None
    model_state["config"] = None


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

    print("=" * 80)
    print("Code Completion API Server")
    print("=" * 80)
    print(f"Model directory: {args.model_dir}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    print(f"Health: http://{args.host}:{args.port}/health")
    print("=" * 80)
    print("\nExample usage:")
    print('  curl -X POST http://localhost:8000/v1/completions \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"model": "code-llm", "prompt": "def factorial(n):", "max_tokens": 50}\'')
    print("=" * 80)

    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
