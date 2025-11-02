"""Train BPE tokenizer using HuggingFace tokenizers library and tokenize dataset."""

import glob
import multiprocessing as mp
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tqdm import tqdm


def load_file_paths(dataset_path: str, glob_pattern: str) -> list[str]:
    """Load all file paths matching the pattern."""
    pattern = str(Path(dataset_path) / "**" / glob_pattern)
    paths = glob.glob(pattern, recursive=True)
    return [p for p in paths if Path(p).is_file()]


def text_iterator(
    file_paths: list[str],
    batch_size: int = 1000,
    add_eos: bool = False,
    eos_token: str = ""
) -> Iterator[list[str]]:
    """
    Iterator that yields batches of text from files.

    Args:
        file_paths: List of file paths to read
        batch_size: Number of documents per batch
        add_eos: Whether to add EOS token after each document
        eos_token: The EOS token string
    """
    batch = []

    for file_path in tqdm(file_paths, desc="Reading files", unit="file"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if content.strip():
                if add_eos and eos_token:
                    content += eos_token
                batch.append(content)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        except Exception as e:
            tqdm.write(f"Warning: Error reading {file_path}: {e}")
            continue

    # Yield remaining batch
    if batch:
        yield batch


def create_tokenizer(pattern: str) -> Tokenizer:
    """
    Create and configure a BPE tokenizer.

    Args:
        pattern: Regex pattern for splitting text
    """
    # Create BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Configure normalizer: NFD Unicode normalization + Strip Accents
    tokenizer.normalizer = normalizers.Sequence([  # type: ignore[assignment]
        normalizers.NFD(),
        normalizers.StripAccents(),
    ])

    # Configure pre-tokenizer with the custom regex pattern
    tokenizer.pre_tokenizer = pre_tokenizers.Split(  # type: ignore[assignment]
        pattern=pattern,
        behavior="isolated"  # Don't merge across splits
    )

    # Configure decoder
    tokenizer.decoder = decoders.BPEDecoder()  # type: ignore[assignment]

    return tokenizer


def train_tokenizer(
    tokenizer: Tokenizer,
    file_paths: list[str],
    vocab_size: int,
    special_tokens: dict,
) -> Tokenizer:
    """
    Train the tokenizer on the dataset.

    Args:
        tokenizer: Tokenizer to train
        file_paths: List of file paths to train on
        vocab_size: Target vocabulary size
        special_tokens: Dictionary of special token names to values
    """
    # Prepare special tokens list
    special_tokens_list = [
        special_tokens.get("unk_token", "<|unk|>"),
        special_tokens.get("bos_token", "<|startoftext|>"),
        special_tokens.get("eos_token", "<|endoftext|>"),
        special_tokens.get("pad_token", "<|pad|>"),
        special_tokens.get("cursor_token", "<|cursor|>"),
        special_tokens.get("edit_start_token", "<|edit_start|>"),
        special_tokens.get("edit_end_token", "<|edit_end|>"),
    ]

    # Filter out empty special tokens
    special_tokens_list = [t for t in special_tokens_list if t]

    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens_list,
        show_progress=True,
        initial_alphabet=[]  # Don't add default alphabet
    )

    # Train from iterator
    print(f"Training tokenizer on {len(file_paths)} files...")
    iterator = text_iterator(file_paths, batch_size=1000, add_eos=False, eos_token="")
    tokenizer.train_from_iterator(iterator, trainer=trainer, length=len(file_paths))

    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, output_path: str):
    """Save tokenizer to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")


def load_tokenizer(path: str) -> Tokenizer:
    """Load tokenizer from JSON file."""
    return Tokenizer.from_file(path)


def encode_file(args: tuple[str, str, str]) -> list[int]:
    """
    Encode a single file (for multiprocessing).

    Args:
        args: Tuple of (file_path, tokenizer_json_str, eos_token)

    Returns:
        List of token IDs
    """
    file_path, tokenizer_json_str, eos_token = args

    try:
        # Load tokenizer from JSON string
        tokenizer = Tokenizer.from_str(tokenizer_json_str)

        # Read file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if not content.strip():
            return []

        # Add EOS token
        if eos_token:
            content += eos_token

        # Encode
        encoding = tokenizer.encode(content)
        return encoding.ids

    except Exception as e:
        tqdm.write(f"Warning: Error encoding {file_path}: {e}")
        return []


def encode_dataset(
    tokenizer: Tokenizer,
    file_paths: list[str],
    output_path: str,
    eos_token: str,
    num_workers: int | None = None
):
    """
    Encode entire dataset and save to binary file using multiprocessing.

    Args:
        tokenizer: Trained tokenizer
        file_paths: List of file paths to encode
        output_path: Path to save encoded tokens
        eos_token: EOS token to add between documents
        num_workers: Number of worker processes (default: CPU count)
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"\nEncoding {len(file_paths)} files using {num_workers} workers...")

    # Serialize tokenizer to JSON string for multiprocessing
    tokenizer_json_str = tokenizer.to_str()

    # Prepare arguments for multiprocessing
    args_list = [(fp, tokenizer_json_str, eos_token) for fp in file_paths]

    # Use multiprocessing to encode files in parallel
    all_tokens = []
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(encode_file, args_list, chunksize=100),
            total=len(file_paths),
            desc="Encoding files",
            unit="file"
        ))

    # Flatten results
    for tokens in results:
        all_tokens.extend(tokens)

    print(f"Total tokens: {len(all_tokens)}")

    # Save as uint32 binary file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokens_array = np.array(all_tokens, dtype=np.uint32)
    tokens_array.tofile(output_path)
    print(f"Tokens saved to {output_path}")


def main():
    """Main entry point for tokenization."""
    # Load configuration
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    tokenize_params = params["tokenize"]

    # Extract parameters
    seed = tokenize_params["seed"]
    dataset_path = tokenize_params["dataset_path"]
    glob_pattern = tokenize_params["glob_pattern"]
    vocab_size = tokenize_params["vocab_size"]
    pattern = tokenize_params["pattern"]
    tok_file = tokenize_params["tok_file"]
    dataset_file = tokenize_params["dataset_file"]
    num_workers = tokenize_params.get("num_workers", 0)  # 0 means use CPU count

    # Special tokens
    special_tokens = {
        "unk_token": "<|unk|>",  # UNK is automatically added by BPE
        "bos_token": tokenize_params["bos_token"],
        "eos_token": tokenize_params["eos_token"],
        "pad_token": tokenize_params["pad_token"],
        "cursor_token": tokenize_params["cursor_token"],
        "edit_start_token": tokenize_params["edit_start_token"],
        "edit_end_token": tokenize_params["edit_end_token"],
    }

    # Set random seed
    random.seed(seed)

    # Create output directory
    Path("out/tokenize").mkdir(parents=True, exist_ok=True)

    # Load file paths
    print("Loading file paths...")
    file_paths = load_file_paths(dataset_path, glob_pattern)
    random.shuffle(file_paths)
    print(f"Total files: {len(file_paths)}")

    # Create tokenizer
    print("\nCreating tokenizer...")
    tokenizer = create_tokenizer(pattern)

    # Train tokenizer
    print("\nTraining tokenizer...")
    tokenizer = train_tokenizer(
        tokenizer,
        file_paths,
        vocab_size,
        special_tokens,
    )

    # Save tokenizer
    save_tokenizer(tokenizer, tok_file)

    # Print vocab info
    vocab_size_actual = tokenizer.get_vocab_size()
    print(f"\nVocabulary size: {vocab_size_actual}")

    # Encode dataset with multiprocessing
    encode_dataset(
        tokenizer,
        file_paths,
        dataset_file,
        special_tokens["eos_token"],
        num_workers if num_workers > 0 else None
    )

    print("\nTokenization complete!")


if __name__ == "__main__":
    main()
