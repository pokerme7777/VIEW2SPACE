#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash run_all_subsets.sh MODEL_DIR DATASET_ROOT RESULT_ROOT [EXTRA_TEST_ARGS...]"
  echo "Example:"
  echo "  bash run_all_subsets.sh /path/to/model /path/to/view2space-v1-release /path/to/results"
  exit 1
fi

MODEL_DIR=$1
DATASET_ROOT=$2
RESULT_ROOT=$3
shift 3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_TEST_ARGS=(
  --batch_size 16
  --num_workers 6
  --prefetch_factor 4
  --max_new_tokens 4096
  --dtype bf16
)

for SUBSET in count detect mcq; do
  INPUT_JSONL="$DATASET_ROOT/$SUBSET/overall.jsonl"
  OUTPUT_DIR="$RESULT_ROOT/$SUBSET"

  echo "[RUN] subset=$SUBSET"
  python "$SCRIPT_DIR/test_qwen3vl.py" \
    --model_dir "$MODEL_DIR" \
    --in_overall_jsonl "$INPUT_JSONL" \
    --scenes_root "$DATASET_ROOT" \
    --result_dir "$OUTPUT_DIR" \
    "${DEFAULT_TEST_ARGS[@]}" \
    "$@"

  EVAL_ARGS=(
    --predictions_jsonl "$OUTPUT_DIR/predictions.jsonl"
    --task_mode "$SUBSET"
  )
  if [[ "$SUBSET" == "detect" ]]; then
    EVAL_ARGS+=(--scale_1000)
  fi

  python "$SCRIPT_DIR/evaluation_qwen3vl.py" "${EVAL_ARGS[@]}"
done
