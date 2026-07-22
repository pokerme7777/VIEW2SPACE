#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PREPROCESS_ASSETS=(snowman table cup fence materials sky)
PREVIEW_ASSETS=(snowman table cup fence)

for asset_name in "${PREPROCESS_ASSETS[@]}"; do
  echo "[prepare_example_assets] Preprocess: $asset_name"
  "$PACKAGE_ROOT/scripts/preprocess_asset.sh" "$asset_name" "$@"
done

for asset_name in "${PREVIEW_ASSETS[@]}"; do
  echo "[prepare_example_assets] Render previews: $asset_name"
  "$PACKAGE_ROOT/scripts/render_previews.sh" "$asset_name" "$@"
done
