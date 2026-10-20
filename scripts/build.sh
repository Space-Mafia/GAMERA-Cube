#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# Do not let another planetary experiment leak into build-time tests.
for key in $(compgen -v); do
  case "$key" in GAMERA_CS_*|GAMERA_NO_*|IO_*|MARS_*) unset "$key";; esac
done
cmake -S "$root/cubedsphere" -B "$root/build" \
  -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -fno-fast-math' \
  -DGAMERA_REQUIRE_PRODUCTION=ON "$@"
cmake --build "$root/build" --parallel "${BUILD_JOBS:-4}"
ctest --test-dir "$root/build" --output-on-failure --timeout 600 -j 2 \
  -E 'mpi|tiled|mi_shell'
