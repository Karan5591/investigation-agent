from __future__ import annotations

import hashlib
import math
from typing import Sequence

from app.config import get_settings

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


class EmbeddingService:
    """Wrap the configured model; deterministic vectors are limited to local test mode."""

    def __init__(
        self,
        model_name: str | None = None,
        dim: int | None = None,
        *,
        allow_fallback: bool = True,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.dim = dim or settings.embedding_dim
        self.allow_fallback = allow_fallback
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None

    def embed(self, text: str) -> list[float]:
        if self.model is not None:
            return self.model.encode(text).tolist()
        if not self.allow_fallback:
            raise RuntimeError(
                f"Embedding model '{self.model_name}' is unavailable; install/download it before using Elasticsearch."
            )
        return self._hash_vector(text, dim=self.dim)

    def _hash_vector(self, text: str, dim: int = 128) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        vector: list[float] = []
        for i in range(dim):
            token = digest[(i * 2) % len(digest) : (i * 2 + 2) % (len(digest) + 1)]
            if not token:
                token = digest[:2]
            vector.append(int(token, 16) / 255.0)
        return vector


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
