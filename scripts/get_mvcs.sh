#!/bin/bash
# The multi-norm MVCS baselines run the MVCS authors' own code. This fetches it at the commit the
# experiments used into third_party/MVCS (not part of this repository).
set -e
cd "$(dirname "$0")/.."
if [ ! -d third_party/MVCS ]; then
  git clone https://github.com/ElSacho/MVCS.git third_party/MVCS
fi
git -C third_party/MVCS checkout -q 9e9ec51
echo "third_party/MVCS at $(git -C third_party/MVCS rev-parse --short HEAD)"
