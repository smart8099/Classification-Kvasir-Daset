#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-"$PWD/.data"}
mkdir -p "$DATA_ROOT"

echo "This is a placeholder download script."
echo "Please set the dataset URLs or manually download the datasets listed below into: $DATA_ROOT"
echo

echo "Required public datasets (per paper):"
echo "- Kvasir"
echo "- CVC-300"
echo "- CVC-ClinicDB"
echo "- CVC-ColonDB"
echo "- ETIS-LaribPolypDB"
echo "- SUN-SEG"
echo

echo "If you provide URLs, replace this script's placeholders or export URL_* env vars."

# Example (replace with real URLs):
# curl -L "$URL_KVASIR" -o "$DATA_ROOT/kvasir.zip"
# unzip -q "$DATA_ROOT/kvasir.zip" -d "$DATA_ROOT/kvasir"
