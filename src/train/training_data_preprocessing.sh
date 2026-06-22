#!/usr/bin/env bash

# Replace the paths below with your own local paths before running.
# Example:
#   SRC_ROOT="/path/to/VIEW2SPACE_DEV/src"
#   DATA_ROOT="/path/to/view2space-release-debug"
#   OUT_JSONL="/path/to/output/train.jsonl"

SRC_ROOT="/path/to/VIEW2SPACE_DEV/src"
DATA_ROOT="/path/to/view2space-release-debug"
OUT_JSONL="/path/to/output/train.jsonl"

python "$SRC_ROOT/build_qwen3vl_train_jsonl.py" \
  --in_overall_jsonl "$DATA_ROOT/overall.jsonl" \
  --out_train_jsonl "$OUT_JSONL" \
  --scenes_root "$DATA_ROOT"
