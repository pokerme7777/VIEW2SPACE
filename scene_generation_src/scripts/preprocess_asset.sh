#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <source_name> [--target-name NAME] [--data-root PATH] [--config PATH] [--scale VALUE] [--facing VALUE]"
  exit 2
fi

SOURCE_NAME="$1"
shift

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$PACKAGE_ROOT/preprocess_cache.py" -S "$SOURCE_NAME" "$@"
