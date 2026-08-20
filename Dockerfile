# Linux CPU image. TORCH_DEVICE defaults to cpu, which is both what this box has
# and what measured fastest locally, so container numbers match the docs.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TORCH_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first: this layer only rebuilds when the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

COPY . .

# Bake the models into the image. Downloading them at boot would add ~60s to
# every cold start and make the service depend on Hugging Face being reachable.
RUN uv run python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small', device='cpu'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', max_length=256, device='cpu')"

# The index is gitignored and container disks are ephemeral, so it is fetched at
# boot from INDEX_URL. Missing index degrades to a clear /health error rather
# than a container that will not start.
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
