"""Reverse proxy server for vllm with custom tokenizer."""

import subprocess
import sys
import time
from pathlib import Path

import httpx
import tokenizer as tok
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Load configuration from params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Configuration
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8001
PROXY_HOST = "0.0.0.0"
PROXY_PORT = 8000
MODEL_PATH = "out/export"
TOKENIZER_PATH = params["tokenize"]["tok_file"]
BOS_TOKEN_ID = params["data"]["bos_token_id"]

# Global tokenizer instance
tokenizer = None
vllm_process = None

app = FastAPI(title="Code LLM Proxy Server")


def load_tokenizer():
    """Load the custom tokenizer."""
    global tokenizer
    if tokenizer is None:
        print(f"Loading tokenizer from {TOKENIZER_PATH}...")
        tokenizer = tok.load(TOKENIZER_PATH)
        print("Tokenizer loaded successfully")
    return tokenizer


def start_vllm_server():
    """Start vllm server as subprocess."""
    global vllm_process

    cmd = [
        "uv",
        "run",
        "vllm",
        "serve",
        MODEL_PATH,
        "--served-model-name",
        "codellm",
        "--skip-tokenizer-init",
        "--host",
        VLLM_HOST,
        "--port",
        str(VLLM_PORT),
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--quantization",
        "fp8",
        "--speculative-config.method",
        "ngram",
        "--speculative-config.num_speculative_tokens",
        "8",
        "--speculative-config.prompt_lookup_max",
        "4",
        "--speculative-config.prompt_lookup_min",
        "2",
    ]

    print(f"Starting vllm server: {' '.join(cmd)}")
    vllm_process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Wait for server to be ready
    print("Waiting for vllm server to start...")
    for i in range(60):  # Wait up to 60 seconds
        try:
            response = httpx.get(f"http://{VLLM_HOST}:{VLLM_PORT}/health")
            if response.status_code == 200:
                print("vllm server is ready!")
                return
        except httpx.ConnectError:
            pass
        time.sleep(1)

    raise RuntimeError("vllm server failed to start within 60 seconds")


@app.on_event("startup")
async def startup_event():
    """Initialize tokenizer and start vllm server on startup."""
    load_tokenizer()
    start_vllm_server()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup vllm process on shutdown."""
    global vllm_process
    if vllm_process:
        print("Shutting down vllm server...")
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            vllm_process.kill()
            vllm_process.wait()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/v1/completions")
async def completions_proxy(request: Request):
    """Proxy completions requests with custom tokenizer."""
    global tokenizer

    # Parse request body
    body = await request.json()

    # Extract prompt
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' field")

    # Encode prompt with custom tokenizer
    if isinstance(prompt, str):
        token_ids = tok.encode(prompt, tokenizer)
        # Prepend BOS token
        token_ids = [BOS_TOKEN_ID] + token_ids
        prompt_token_ids = [token_ids]
    elif isinstance(prompt, list):
        # Handle batch of prompts
        prompt_token_ids = []
        for p in prompt:
            token_ids = tok.encode(p, tokenizer)
            token_ids = [BOS_TOKEN_ID] + token_ids
            prompt_token_ids.append(token_ids)
    else:
        raise HTTPException(status_code=400, detail="Invalid 'prompt' type")

    # Add token IDs (keep prompt for vllm validation)
    body["prompt_token_ids"] = prompt_token_ids

    # Check if streaming is requested
    stream = body.get("stream", False)

    # Forward request to vllm
    vllm_url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/completions"

    async with httpx.AsyncClient(timeout=300.0) as client:
        if stream:
            # Handle streaming response
            async def stream_generator():
                async with client.stream("POST", vllm_url, json=body) as response:
                    async for chunk in response.aiter_text():
                        yield chunk

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            # Handle non-streaming response
            response = await client.post(vllm_url, json=body)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, detail=response.text
                )

            # Return the response as-is (vllm returns text in OpenAI format)
            return JSONResponse(content=response.json())


def main():
    """Run the proxy server."""
    print(f"Starting proxy server on {PROXY_HOST}:{PROXY_PORT}")
    print(f"Proxying to vllm server at {VLLM_HOST}:{VLLM_PORT}")
    print(f"Model: {MODEL_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")

    uvicorn.run(
        app,
        host=PROXY_HOST,
        port=PROXY_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
