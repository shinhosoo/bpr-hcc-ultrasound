#!/usr/bin/env bash
# Install python packages needed by the unified evaluation & visualization scripts.
set -e
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 -m pip install --upgrade pip
python3 -m pip install -r "$HERE/requirements_eval.txt"
echo "[install_eval_requirements] done."
