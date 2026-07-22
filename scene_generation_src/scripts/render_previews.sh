#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <asset_name> [extra blenderkit_preview.py args]"
  echo "Example: BLENDER_BIN=blender $0 snowman --data-root /path/to/cache --res 768"
  exit 2
fi

ASSET_NAME="$1"
shift

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLENDER_BIN="${BLENDER_BIN:-blender}"
DATA_ROOT="${BENCHMARKING_DATA_CACHE:-}"
EXTRA_ARGS=("$@")

for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
  if [[ "${EXTRA_ARGS[$i]}" == "--data-root" && $((i + 1)) -lt ${#EXTRA_ARGS[@]} ]]; then
    DATA_ROOT="${EXTRA_ARGS[$((i + 1))]}"
  fi
done

"$BLENDER_BIN" --background --python "$PACKAGE_ROOT/blenderkit_preview.py" -- -S "$ASSET_NAME" "${EXTRA_ARGS[@]}"

VIS_ARGS=(-S "$ASSET_NAME")
if [[ -n "$DATA_ROOT" ]]; then
  VIS_ARGS+=(--data-root "$DATA_ROOT")
fi
python "$PACKAGE_ROOT/visualize.py" "${VIS_ARGS[@]}"
