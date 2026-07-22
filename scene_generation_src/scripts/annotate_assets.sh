#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <asset_name> [data_root]"
  echo "Requires AZURE_OPENAI_URL and OPENAI_API_KEY."
  exit 2
fi

ASSET_NAME="$1"
DATA_ROOT="${2:-${BENCHMARKING_DATA_CACHE:-}}"
MODEL="${MODEL:-gpt-4o}"

if [[ -z "$DATA_ROOT" ]]; then
  echo "Set BENCHMARKING_DATA_CACHE or pass data_root as the second argument."
  exit 2
fi

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAPPING="$DATA_ROOT/processed_asset/$ASSET_NAME/mapping.jsonl"
OVERVIEW="$DATA_ROOT/processed_asset/${ASSET_NAME}_preview/preview_all.png"
ANNOTATION="$DATA_ROOT/processed_asset/$ASSET_NAME/${ASSET_NAME}_annotation.json"

python "$PACKAGE_ROOT/asset_annotation1_tag_library_light.py" \
  --overview "$OVERVIEW" \
  --mapping "$MAPPING" \
  --out "$ANNOTATION" \
  --folder_name "$ASSET_NAME" \
  --model "$MODEL"

python "$PACKAGE_ROOT/asset_annotation1_tag_library2_light.py" \
  --overview "$OVERVIEW" \
  --mapping "$MAPPING" \
  --annotation "$ANNOTATION" \
  --model "$MODEL"

python "$PACKAGE_ROOT/asset_annotation2_assign_tag_light.py" \
  --mapping "$MAPPING" \
  --annotation "$ANNOTATION" \
  --model "$MODEL" \
  --resume
