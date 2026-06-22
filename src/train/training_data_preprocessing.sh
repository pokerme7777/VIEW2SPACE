#!/usr/bin/env bash

# Replace the paths below with your own local paths before running.
# Example:
#   SRC_ROOT="/path/to/VIEW2SPACE_DEV/src"
#   DATA_ROOT="/path/to/your/data_root"
#   OUT_JSONL="/path/to/output/train.jsonl"

SUBSET_NAME=FILL_SUBSET_NAME  # e.g., "VIEW2SPACE_TRAIN"
SRC_ROOT="/path/to/VIEW2SPACE_DEV/src"
DATA_ROOT="/path/to/your/data_root"
OUT_JSONL="/path/to/output/predictions.jsonl"

python "$SRC_ROOT/build_qwen3vl_train_jsonl.py" \
  --in_overall_jsonl "$DATA_ROOT/cot_training/$SUBSET_NAME/overall.jsonl" \
  --out_train_jsonl "$OUT_JSONL" \
  --scenes_root "$DATA_ROOT/scenes/train_set_scenes"
