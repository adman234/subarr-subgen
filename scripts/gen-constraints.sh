#!/usr/bin/env bash
# Regenerate docker/constraints.txt from a PUBLISHED image (subarr-subgen#73).
#
# Deliberately reads an image you have already verified rather than doing a
# fresh resolve: the point of the file is to pin what is proven to work, not
# whatever the index offers today. Run it after a rev you are happy with, look
# at the diff, and commit it as its own change.
#
#   bash scripts/gen-constraints.sh ghcr.io/coaxk/subarr-subgen:2026.08.1-r13 \
#       > docker/constraints.txt
#
# ⚠️ It prints ONLY the pins. The explanatory header in docker/constraints.txt
# is hand-maintained, so redirecting straight over the file drops it -- diff
# before committing, or splice the pins in under the existing header.
set -euo pipefail

IMG="${1:-}"
if [[ -z "$IMG" ]]; then
  echo "usage: $0 <image-ref>" >&2
  exit 2
fi

# pip is purged from the shipped image, so importlib.metadata does the reading.
# Ranking by sys.path matters: setuptools is installed TWICE (apt 59.6.0 under
# /usr/lib, pip 84.0.0 under /usr/local) and only the /usr/local copy imports.
# Taking whatever importlib.metadata yields last picks the shadowed one and
# would pin the vulnerable version the Dockerfile exists to avoid.
docker run --rm --entrypoint python3 "$IMG" -c '
import sys
from importlib.metadata import distributions

def rank(p):
    p = str(p)
    for i, entry in enumerate(sys.path):
        if entry and p.startswith(entry):
            return i
    return len(sys.path)

best = {}
for d in distributions():
    n = d.metadata["Name"]
    if not n or not d.version:
        continue
    r = rank(d._path)
    key = n.lower()
    if key not in best or r < best[key][0]:
        best[key] = (r, n, d.version)

# See the DELIBERATE OMISSIONS block in docker/constraints.txt before changing.
SKIP = {"torch", "torchaudio", "wheel"}
for key in sorted(best):
    if key in SKIP:
        continue
    _, n, v = best[key]
    print(f"{n}=={v}")
'
