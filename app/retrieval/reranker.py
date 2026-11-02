from __future__ import annotations

from typing import Iterable, Sequence
from app.config import get_settings
from app.retrieval.embeddings import EmbeddingService, cosine_similarity
from app.retrieval.search import SearchHit

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None


class Reranker:
    """Optionally use BGE Cross-Encoder; otherwise use vector similarity fallback."""

    def __init__(self, embedding_service: EmbeddingService | None = None, model_name: str | None = None):
        settings = get_settings()
        self.embedding_service = embedding_service or EmbeddingService()
        self.model_name = model_name or settings.reranker_model
        self.cross_encoder = None
        if settings.reranker_enabled and CrossEncoder is not None:
            try:
                self.cross_encoder = CrossEncoder(self.model_name)
            except Exception:
                self.cross_encoder = None

    def rerank(self, query: str, hits: Iterable[SearchHit], top_n: int | None = None) -> list[SearchHit]:
        hit_list = list(hits)
        if not hit_list:
            return []

        # 1. Preferred path: Real BGE CrossEncoder
        if self.cross_encoder is not None:
            try:
                pairs = [[query, hit.text] for hit in hit_list]
                scores = self.cross_encoder.predict(pairs)
                for hit, score in zip(hit_list, scores):
                    hit.score = float(score)
                ranked = sorted(hit_list, key=lambda item: item.score, reverse=True)
                return ranked[:top_n] if top_n else ranked
            except Exception:
                pass

        # 2. Fallback path: combined vector cosine similarity and lexical base score
        query_embedding = self.embedding_service.embed(query)
        scored: list[tuple[float, SearchHit]] = []
        for hit in hit_list:
            base = float(hit.score)
            vector_bonus = 0.0
            if hit.metadata and hit.metadata.get("embedding"):
                vector_bonus = cosine_similarity(query_embedding, hit.metadata["embedding"])
            total = base + vector_bonus
            scored.append((total, hit))

        ranked = [hit for _, hit in sorted(scored, key=lambda item: item[0], reverse=True)]
        return ranked[:top_n] if top_n else ranked
