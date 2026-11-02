from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_downloaded_models(monkeypatch):
    from app.retrieval import embeddings, reranker

    monkeypatch.setattr(embeddings, "SentenceTransformer", None)
    monkeypatch.setattr(reranker, "CrossEncoder", None)
