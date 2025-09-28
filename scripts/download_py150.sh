#!/usr/bin/env bash

set -xe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
OUT_DIR="$ROOT_DIR/data/py150"

mkdir -p "$OUT_DIR"
wget -O "$OUT_DIR/py150.tar.gz" \
    http://files.srl.inf.ethz.ch/data/py150_files.tar.gz

# First untar py150.tar.gz
tar -xzf "$OUT_DIR/py150.tar.gz" -C "$OUT_DIR"
rm "$OUT_DIR/py150.tar.gz"

# Then untar nested data.tar.gz
tar -xzf "$OUT_DIR/data.tar.gz" -C "$OUT_DIR"
rm "$OUT_DIR/data.tar.gz"
