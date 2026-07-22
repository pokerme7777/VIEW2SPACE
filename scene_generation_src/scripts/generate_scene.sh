#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <scene_name> [extra scene_pipeline_grid.py args]"
  echo "Example: BLENDER_BIN=blender $0 example_grid_scene --data-root /path/to/cache"
  exit 2
fi

SCENE_NAME="$1"
shift

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLENDER_BIN="${BLENDER_BIN:-blender}"
CONFIG_PATH="${SCENE_CONFIG:-$PACKAGE_ROOT/config/scene_config.json}"

"$BLENDER_BIN" --background --python "$PACKAGE_ROOT/scene_pipeline_grid.py" -- \
  --scene-name "$SCENE_NAME" \
  --config "$CONFIG_PATH" \
  "$@"
