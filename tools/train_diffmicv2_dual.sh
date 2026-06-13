#!/usr/bin/env bash
# 설계 2 학습만 (test/viz 제외). run_diffmicv2_dual.sh 에 위임.
# Usage: bash tools/train_diffmicv2_dual.sh [tag]
set -e
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec env STAGE_ONLY=train bash "$HERE/run_diffmicv2_dual.sh" "${1:-bpr_diffmicv2_dual}"
