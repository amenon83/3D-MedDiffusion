#!/usr/bin/env bash
# Run pip install using temp and cache on the SSD (avoids "No space left on device" on main drive).
# Usage: from repo root, with conda env activated (do NOT use sudo):
#   bash scripts/pip_install_ssd.sh
# or: bash scripts/pip_install_ssd.sh -r requirements.txt

set -e
SSD_BASE="/media/liyong/f001be89-9498-4619-9827-7607b8ac9501/home/liyong/Arnav"
export TMPDIR="${SSD_BASE}/tmp"
export PIP_CACHE_DIR="${SSD_BASE}/pip_cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
echo "TMPDIR=$TMPDIR"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
pip install "${@:--r requirements.txt}"
