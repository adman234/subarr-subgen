#!/usr/bin/env bash
# One-shot build: sync submodule, apply patches, docker build.
# Produces a local `subarr-subgen:dev` image.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> syncing upstream submodule"
git submodule update --init --recursive

echo "==> applying patches"
./scripts/apply-patches.sh

echo "==> staging build context"
# Copy the patched upstream tree (minus its .git) into ./build/ so the
# Dockerfile has a clean context.
rm -rf build/
mkdir -p build/
cp -a upstream/. build/
# [#73] Our constraints file is NOT part of the upstream tree, so it has to be
# staged alongside it or the Dockerfile's COPY fails the build.
cp docker/constraints.txt build/
# Strip submodule's .git so it doesn't bloat the image build.
rm -rf build/.git

UPSTREAM_VERSION="$(grep -E "^subgen_version = " build/subgen.py | head -1 | sed -E "s/.*'([^']+)'.*/\1/")"
PATCH_REV="$(git rev-parse --short HEAD)"
IMAGE_TAG="subarr-subgen:dev-${UPSTREAM_VERSION}-${PATCH_REV}"

echo "==> docker build (upstream=$UPSTREAM_VERSION patch_rev=$PATCH_REV)"
docker build \
  --build-arg "UPSTREAM_VERSION=$UPSTREAM_VERSION" \
  --build-arg "PATCH_REV=$PATCH_REV" \
  --build-arg "RELEASE_TAG=dev-${PATCH_REV}" \
  --build-arg "APT_REFRESH=$(date -u +%Y-%m-%d)" \
  -f docker/Dockerfile \
  -t "$IMAGE_TAG" \
  -t "subarr-subgen:dev" \
  build/

echo
echo "built: $IMAGE_TAG"
echo "also tagged: subarr-subgen:dev"
