#!/usr/bin/env bash
# Fetch and extract the ONNX all-MiniLM-L6-v2 embedding model used by the API.
#
# Why this exists rather than letting fastembed download on first use:
#   - fastembed's HuggingFace repo (qdrant/all-MiniLM-L6-v2-onnx) omits
#     special_tokens_map.json, which fastembed requires. Loading straight from
#     the Hub cache can therefore leave a snapshot it then refuses to load.
#     The mirror tarball below contains the complete file set.
#   - Fetching at build time keeps production off the network on a cold start
#     and makes container startup deterministic.
#
# Destination: backend/models/all-MiniLM-L6-v2-onnx (gitignored).
# Used locally and by backend/Dockerfile.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/backend/models/all-MiniLM-L6-v2-onnx"
URL="https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz"

if [ -f "${DEST}/model.onnx" ] && [ -f "${DEST}/special_tokens_map.json" ]; then
    echo "Embedding model already present at ${DEST}"
    exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Downloading ONNX embedding model..."
curl -fsSL --retry 3 -o "${TMP_DIR}/model.tar.gz" "${URL}"

# --strip-components=1 drops the tarball's wrapper directory; --exclude drops
# the macOS resource-fork entries (._*) that would otherwise be copied in.
tar xzf "${TMP_DIR}/model.tar.gz" -C "${TMP_DIR}" --strip-components=1 --exclude='._*' 2>/dev/null

mkdir -p "${DEST}"
cp "${TMP_DIR}"/* "${DEST}/"

echo "Embedding model ready at ${DEST}"
