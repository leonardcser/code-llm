# CodeLLM

Training a coding large language model from scratch using a custom cpp tokenizer
and torch.

This project is a work in progress. See the updates in
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

A python binding is avaiable as well as the options to compile the executables:

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
