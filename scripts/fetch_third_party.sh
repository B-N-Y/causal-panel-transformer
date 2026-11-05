#!/usr/bin/env bash
# Fetches the competing methods' code at the commits used in the paper into third_party/.
# The code is not redistributed here; each repository keeps its own license.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/third_party"
fetch () {  # name url commit
  local dir="$ROOT/third_party/$1"
  if [ ! -d "$dir/.git" ]; then git clone --quiet "$2" "$dir"; fi
  git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet "$3"
  echo "$1 -> $(git -C "$dir" rev-parse --short HEAD)"
}
fetch tdrl  https://github.com/weirayao/tdrl.git               655ea52
fetch nctrl https://github.com/xiangchensong/nctrl.git          d2540bb
fetch isds  https://github.com/charlio23/identifiable-SDS.git   b6c11a5
