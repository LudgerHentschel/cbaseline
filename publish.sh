#!/usr/bin/env zsh

set -euo pipefail

if [[ ! -d dist ]]; then
  echo "dist/ does not exist. Run ./release.sh first." >&2
  exit 1
fi

python -m twine check dist/*
python -m twine upload dist/*

echo "Published versions:"
pip index versions cbaseline
