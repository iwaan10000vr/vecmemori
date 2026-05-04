#!/usr/bin/env bash
# Download embedding model for vecmemori.
# Default: cl-nagoya/ruri-v3-310m (~1.2 GB)
#
# Usage:
#   bash scripts/download_model.sh                  # ruri-v3-310m (default)
#   bash scripts/download_model.sh <model-name>     # custom model
#
# Output: ~/.cache/vecmemori/models/<model-name>/  (or $VECMEMORI_MODELS_DIR/<model-name>/)

set -euo pipefail

MODEL="${1:-cl-nagoya/ruri-v3-310m}"
MODEL_SHORT="${MODEL##*/}"
INSTALL_DIR="${VECMEMORI_MODELS_DIR:-$HOME/.cache/vecmemori/models}"

if [ -d "$INSTALL_DIR/$MODEL_SHORT" ]; then
    echo "✓ Model already exists at $INSTALL_DIR/$MODEL_SHORT"
    echo "  Delete it to re-download."
    exit 0
fi

echo "Downloading $MODEL → $INSTALL_DIR/$MODEL_SHORT ..."
mkdir -p "$INSTALL_DIR"

# Prefer huggingface-hub if installed, fall back to git clone
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$MODEL" --local-dir "$INSTALL_DIR/$MODEL_SHORT"
elif command -v git &>/dev/null; then
    GIT_LFS_SKIP_SMUDGE=1 git clone "https://huggingface.co/$MODEL" "$INSTALL_DIR/$MODEL_SHORT"
    (cd "$INSTALL_DIR/$MODEL_SHORT" && git lfs pull)
else
    echo "ERROR: Neither huggingface-cli nor git-lfs found."
    echo "Install one of:"
    echo "  pip install huggingface-hub"
    echo "  apt install git-lfs"
    exit 1
fi

echo "✓ Done: $INSTALL_DIR/$MODEL_SHORT"
echo "  Size: $(du -sh "$INSTALL_DIR/$MODEL_SHORT" | cut -f1)"
