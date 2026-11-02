from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import main
from app.evidence.models import EvidenceChunk, EvidenceStatus, InvestigationResult


def test_investigate_endpoint_returns_evidence_and_rejects_server_paths(monkeypatch):
    class StubAgent:
        def investigate(self, question: str, uploaded_file=None, top_k: int = 5):
            return InvestigationResult(
                question=question,
                status=EvidenceStatus.READY,
                answer="The evidence supports opening a review.",
                confidence=0.82,
                evidence=[
                    EvidenceChunk(
                        doc_id="doc-1",
                        filename="policy.txt",
                        source="corpus",
                        page=2,
                        text="The policy supports a review.",
                        score=0.9,
                    )
                ],
                sources=["policy.txt"],
                trace=["search_documents", "finish"],
            )

    monkeypatch.setattr(main, "build_agent", lambda: StubAgent())
    monkeypatch.setattr(main.app.router, "on_startup", [])

    with TestClient(main.app) as client:
        response = client.post(
            "/investigate",
            json={"question": "Does the policy support review?", "top_k": 3},
        )
        unsafe_request = client.post(
            "/investigate",
            json={"question": "Read a local file", "uploaded_file": "/etc/passwd"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["evidence"][0]["filename"] == "policy.txt"
    assert payload["evidence"][0]["page"] == 2
    assert unsafe_request.status_code == 422
