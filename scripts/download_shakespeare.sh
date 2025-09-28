#!/usr/bin/env bash

set -xe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
OUT_DIR="$ROOT_DIR/data/shakespeare"

mkdir -p "$OUT_DIR"
wget -O "$OUT_DIR/shakespeare.txt" \
  https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/b363733fe79c29068604feec8627627c222923ee/shakespeare.txt
