#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/download_py150.sh"
"$SCRIPT_DIR/download_shakespeare.sh"
