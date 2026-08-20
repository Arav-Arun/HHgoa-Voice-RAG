#!/usr/bin/env sh
# Fetch the prebuilt index, then serve. Always proceeds to uvicorn: a missing
# index surfaces as /health reporting 0 chunks, which is diagnosable, whereas a
# container that refuses to boot is not.
set -e

if [ -n "$INDEX_URL" ] && [ ! -f "${INDEX_DIR:-data/index}/chunks.json" ]; then
  echo "[entrypoint] fetching index from $INDEX_URL"
  mkdir -p "${INDEX_DIR:-data/index}"
  curl -fsSL "$INDEX_URL" | tar -xz -C "$(dirname "${INDEX_DIR:-data/index}")" \
    || echo "[entrypoint] index fetch failed; serving without one"
fi

# --no-sync: the venv is already built into the image, and re-syncing on
# every boot rebuilds the project and needs a writable /app.
exec uv run --no-sync python -m core.cli serve
