#!/bin/bash
# Download the FINSABER SP500 price dataset to data/sp500_prices.csv.
#
# The CSV is an external, publicly hosted benchmark dataset (not produced by
# the authors of this submission). License and citation information are on
# the Hugging Face dataset page.
#
# Usage:
#   bash scripts/download_data.sh             # download if missing
#   bash scripts/download_data.sh --force     # re-download even if present
#
# Requires `wget` or `curl` on PATH.

set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://huggingface.co/datasets/waylonli/FINSABER-data/resolve/main/data/price/all_sp500_prices_2000_2024_delisted_include.csv"
DEST="data/sp500_prices.csv"
EXPECTED_MIN_BYTES=200000000   # ~190 MB; the real file is ~253 MB

FORCE=0
if [ "${1:-}" = "--force" ]; then
    FORCE=1
fi

mkdir -p data

# Idempotent skip if a sane copy already exists.
if [ -s "$DEST" ] && [ "$FORCE" -eq 0 ]; then
    SIZE=$(stat -Lc %s "$DEST" 2>/dev/null || stat -Lf %z "$DEST")
    if [ "$SIZE" -gt "$EXPECTED_MIN_BYTES" ]; then
        echo "[skip] $DEST already present ($((SIZE / 1024 / 1024)) MB)."
        echo "       Pass --force to re-download."
        exit 0
    else
        echo "[warn] $DEST exists but is only $SIZE bytes; re-downloading…"
    fi
fi

echo "[download] $URL"
echo "[target]   $DEST"

if command -v wget >/dev/null 2>&1; then
    wget --show-progress -O "$DEST" "$URL"
elif command -v curl >/dev/null 2>&1; then
    curl -L --progress-bar -o "$DEST" "$URL"
else
    echo "[error] Neither wget nor curl is installed. Install one of them and retry." >&2
    exit 1
fi

# Sanity-check size after download.
SIZE=$(stat -Lc %s "$DEST" 2>/dev/null || stat -Lf %z "$DEST")
if [ "$SIZE" -lt "$EXPECTED_MIN_BYTES" ]; then
    echo "[error] Downloaded file is only $SIZE bytes; expected > $EXPECTED_MIN_BYTES." >&2
    echo "        The download may have been interrupted or the URL may have changed." >&2
    rm -f "$DEST"
    exit 1
fi

echo "[done] $DEST ($((SIZE / 1024 / 1024)) MB)"
echo "Next: python scripts/prepare_data.py"
