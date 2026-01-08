# CodeLLM

Training a coding large language model from scratch using a custom C++ tokenizer
and PyTorch.

## Example Generation

Given the prompt `def fib(`, the model generates:

```python
def fib( n ):
    if n == 1:
        return 1
    else:
        return fib( n-1 ) + fib( n-2 )
```

*Generated in 0.2s at 496 tokens/second on NVIDIA RTX 5070 Ti*

## Training Progress

![Perplexity over training steps](docs/perplexity.png)

*Trained for 2 days on NVIDIA RTX 5070 Ti on Python code*

---

This project is a work in progress. See the detailed journal at
[docs/journal.pdf](docs/journal.pdf)

## Getting Started

```bash
uv sync
```

This project uses DVC. In order to execute the pipeline, run:

```bash
uv run dvc repro
```

### Tokenizer

The tokenizer is an extra module, it implements BPE (byte-pair-encoding)
training and encoding/decoding.

A python binding is available as well as the options to compile the executables:

```bash
cmake -S tokenizer -B tokenizer/build -DCMAKE_BUILD_TYPE=Release && cmake --build tokenizer/build -j$(nproc)
```

This produces the following executables:

- `./tokenizer/build/tokenize`
- `./tokenizer/build/tokenize-cli`
- `./tokenizer/build/encoding`
- `./tokenizer/build/visualize`

## Development

### Testing

To run the python test, run:

```bash
uv run pytest src
```

In order to run the ./tokenizer tests, make sure you compiled it and then run:

```bash
cd ./tokenizer/build
make test
```
