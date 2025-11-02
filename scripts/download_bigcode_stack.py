#!/usr/bin/env python3
"""
Download script for BigCode The Stack Dedup dataset.

This script downloads files from the bigcode/the-stack-dedup dataset
and saves them to the local filesystem.
"""

import argparse
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm


def parse_size(size_str: str) -> int:
    """
    Parse a size string like '1GB', '500MB', '1.5GB' into bytes.

    Args:
        size_str: Size string with optional suffix (B, KB, MB, GB, TB)

    Returns:
        Size in bytes
    """
    size_str = size_str.strip().upper()
    match = re.match(r"^([\d.]+)\s*([KMGT]?B?)$", size_str)

    if not match:
        raise ValueError(f"Invalid size format: {size_str}")

    number = float(match.group(1))
    unit = match.group(2) or "B"

    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }

    return int(number * units.get(unit, 1))


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def download_sample(
    sample: dict, output_dir: Path, index: int
) -> Optional[Tuple[str, int]]:
    """
    Download and save a single sample from the dataset.

    Args:
        sample: The sample dictionary from the dataset
        output_dir: Directory to save the file
        index: Index of the sample for naming

    Returns:
        Tuple of (path to saved file, size in bytes) or None if failed
    """
    try:
        # Create a filename from the hexsha or use index
        hexsha = sample.get("hexsha", f"file_{index}")
        ext = sample.get("ext", "txt")
        filename = f"{hexsha}.{ext}"

        # Create output path
        output_path = output_dir / filename

        # Save the content
        content = sample.get("content", "")
        output_path.write_text(content, encoding="utf-8")

        # Calculate size
        file_size = len(content.encode("utf-8"))

        return (str(output_path), file_size)
    except Exception as e:
        print(f"Error saving sample {index}: {e}")
        return None


def download_sequential(dataset, output_dir: Path, max_size: Optional[int]):
    """Download files sequentially."""
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    total_size = 0
    with tqdm(total=max_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for i, sample in enumerate(dataset):
            result = download_sample(sample, output_dir, i)
            if result:
                _, file_size = result
                count += 1
                total_size += file_size
                pbar.update(file_size)

                if max_size and total_size >= max_size:
                    break

    return count, total_size


def download_parallel(
    dataset, output_dir: Path, max_size: Optional[int], num_workers: int
):
    """Download files in parallel."""
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    total_size = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        samples_iter = iter(enumerate(dataset))

        # Submit initial batch of jobs
        for _ in range(num_workers * 2):
            try:
                i, sample = next(samples_iter)
                future = executor.submit(download_sample, sample, output_dir, i)
                futures.append(future)
            except StopIteration:
                break

        with tqdm(
            total=max_size, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            while futures:
                # Wait for next future to complete
                done_futures = []
                for future in futures[:]:
                    if future.done():
                        done_futures.append(future)
                        futures.remove(future)

                # Process completed futures
                for future in done_futures:
                    result = future.result()
                    if result:
                        _, file_size = result
                        count += 1
                        total_size += file_size
                        pbar.update(file_size)

                # Check if we've reached the size limit
                if max_size and total_size >= max_size:
                    # Cancel remaining futures
                    for future in futures:
                        future.cancel()
                    break

                # Submit new jobs to keep workers busy
                if not max_size or total_size < max_size:
                    try:
                        i, sample = next(samples_iter)
                        future = executor.submit(download_sample, sample, output_dir, i)
                        futures.append(future)
                    except StopIteration:
                        pass

    return count, total_size


def main():
    parser = argparse.ArgumentParser(
        description="Download files from BigCode The Stack v2 Dedup dataset"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="",
        help="Programming language to download (default: python)",
    )
    parser.add_argument(
        "-j",
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel download workers (default: sequential)",
    )
    parser.add_argument(
        "--max-size",
        type=str,
        default=None,
        help="Maximum total size to download (e.g., '1GB', '500MB', '1.5GB'). Default: unlimited",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/bigcode-the-stack-dedup/<language>)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download (default: train)",
    )

    args = parser.parse_args()

    # Parse max_size if provided
    max_size = None
    if args.max_size:
        max_size = parse_size(args.max_size)

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent
        root_dir = script_dir.parent
        output_dir = root_dir / "data" / "bigcode-the-stack-dedup" / args.language

    print(f"Downloading {args.language} files from bigcode/the-stack-dedup")
    print(f"Output directory: {output_dir}")
    print(f"Split: {args.split}")
    if max_size is not None:
        print(f"Max size: {format_size(max_size)}")
    if args.parallel:
        print(f"Parallel workers: {args.parallel}")

    # Load dataset in streaming mode
    print("\nLoading dataset...")
    ds = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir=f"data/{args.language}" if args.language else None,
        streaming=True,
        split=args.split,
    )

    # Download files
    if args.parallel:
        count, total_size = download_parallel(ds, output_dir, max_size, args.parallel)
    else:
        count, total_size = download_sequential(ds, output_dir, max_size)

    print(f"\nDownloaded {count} files ({format_size(total_size)}) to {output_dir}")


if __name__ == "__main__":
    main()
