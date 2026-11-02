from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

try:
    from elasticsearch import Elasticsearch
except Exception:  # pragma: no cover
    Elasticsearch = None

from app.retrieval.embeddings import EmbeddingService, cosine_similarity


@dataclass
class SearchHit:
    doc_id: str
    filename: str
    text: str
    score: float
    source: str
    page: int | None = None
    metadata: dict[str, Any] | None = None


class SearchClient:
    """Elasticsearch retrieval, with an explicit in-memory mode for isolated tests."""

    def __init__(self, url: str, index: str, *, use_remote: bool = False):
        self.url = url
        self.index = index
        self._memory: list[dict[str, Any]] = []
        self._es = None
        self._use_remote = use_remote
        self._index_ready = False
        self._embedding_dim: int | None = None
        self.embedding_service = EmbeddingService(allow_fallback=not use_remote)
        if use_remote and Elasticsearch is None:
            raise RuntimeError("Elasticsearch support is unavailable; install the elasticsearch package.")
        if use_remote:
            self._es = Elasticsearch(url)

    def _ensure_index(self, embedding_dim: int) -> None:
        if not self._use_remote:
            return
        if self._index_ready:
            if embedding_dim != self._embedding_dim:
                raise RuntimeError(
                    f"Embedding dimension changed from {self._embedding_dim} to {embedding_dim}; use a new index."
                )
            return

        mappings = {
            "properties": {
                "doc_id": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "text": {"type": "text"},
                "source": {"type": "keyword"},
                "page": {"type": "integer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine",
                },
                "method": {"type": "keyword"},
                "ocr_confidence": {"type": "float"},
                "needs_review": {"type": "boolean"},
            }
        }
        if not self._es.indices.exists(index=self.index):
            self._es.indices.create(index=self.index, mappings=mappings)
        else:
            existing = self._es.indices.get_mapping(index=self.index)
            properties = existing.get(self.index, {}).get("mappings", {}).get("properties", {})
            vector_mapping = properties.get("embedding")
            if vector_mapping is None:
                self._es.indices.put_mapping(index=self.index, properties={"embedding": mappings["properties"]["embedding"]})
            elif vector_mapping.get("type") != "dense_vector" or vector_mapping.get("dims") != embedding_dim:
                raise RuntimeError(
                    f"Elasticsearch index '{self.index}' has an incompatible embedding mapping; "
                    "set ES_INDEX to a new index and re-seed the corpus."
                )
        self._embedding_dim = embedding_dim
        self._index_ready = True

    def index_document(
        self,
        doc_id: str,
        filename: str,
        text: str,
        source: str,
        page: int | None = None,
        embedding: Sequence[float] | None = None,
        **metadata: Any,
    ) -> None:
        payload = {
            "doc_id": doc_id,
            "filename": filename,
            "text": text,
            "source": source,
            "page": page,
            "embedding": list(embedding) if embedding is not None else self.embedding_service.embed(text),
            **metadata,
        }
        if self._use_remote:
            self._ensure_index(len(payload["embedding"]))
            self._es.index(index=self.index, document=payload, id=f"{doc_id}:{page or 0}")
            return
        self._memory.append(payload)

    def search(self, query: str, top_k: int = 5, *, embedding: Sequence[float] | None = None) -> list[SearchHit]:
        query_embedding = list(embedding) if embedding is not None else self.embedding_service.embed(query)
        if self._use_remote:
            self._ensure_index(len(query_embedding))
            results = self._es.search(
                index=self.index,
                knn={
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": top_k,
                    "num_candidates": max(top_k * 10, 20),
                    "boost": 1.0,
                },
                query={
                    "match": {
                        "text": {
                            "query": query,
                            "boost": 0.5,
                        }
                    }
                },
                size=top_k,
            )
            hits = []
            for item in results.get("hits", {}).get("hits", []):
                source = item.get("_source", {})
                hits.append(
                    SearchHit(
                        doc_id=source.get("doc_id", ""),
                        filename=source.get("filename", ""),
                        text=source.get("text", ""),
                        score=float(item.get("_score", 0.0)),
                        source=source.get("source", "corpus"),
                        page=source.get("page"),
                        metadata=source,
                    )
                )
            return hits

        matches: list[SearchHit] = []
        for item in self._memory:
            text = str(item.get("text", ""))
            lexical_score = 0.0
            query_terms = query.lower().split()
            for term in query_terms:
                if term in text.lower():
                    lexical_score += 1.0
            vector_embedding = item.get("embedding")
            vector_score = 0.0
            if vector_embedding:
                vector_score = cosine_similarity(query_embedding, vector_embedding)
            total_score = lexical_score + vector_score
            if total_score > 0:
                matches.append(
                    SearchHit(
                        doc_id=item["doc_id"],
                        filename=item["filename"],
                        text=text,
                        score=float(total_score),
                        source=item.get("source", "corpus"),
                        page=item.get("page"),
                        metadata=item,
                    )
                )
        return sorted(matches, key=lambda item: item.score, reverse=True)[:top_k]

    def search_with_status(self, query: str, top_k: int = 5, *, embedding: Sequence[float] | None = None) -> tuple[list[SearchHit], str | None]:
        try:
            hits = self.search(query, top_k=top_k, embedding=embedding)
            return hits, None
        except Exception as exc:  # pragma: no cover - connector/tool failure
            return [], str(exc)
