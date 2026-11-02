from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db.database import Database
from app.db.models import DocSource
from app.db.repository import DocumentRepository
from app.ingestion.service import IngestionService
from app.retrieval.search import SearchClient


@dataclass
class ToolResult:
    ok: bool
    payload: dict[str, Any] | None = None
    error: str | None = None


class InvestigationToolRegistry:
    def __init__(self, db: Database, search: SearchClient, ingestion: IngestionService):
        self.repo = DocumentRepository(db)
        self.search = search
        self.ingestion = ingestion

    def search_documents(self, query: str, top_k: int = 5) -> ToolResult:
        try:
            return ToolResult(ok=True, payload={"hits": [hit.__dict__ for hit in self.search.search(query, top_k)]})
        except Exception as exc:  # pragma: no cover
            return ToolResult(ok=False, error=str(exc))

    def fetch_document_metadata(self, document_id: str) -> ToolResult:
        document = self.repo.get(document_id)
        if document is None:
            return ToolResult(ok=False, error=f"Document {document_id} not found")
        return ToolResult(ok=True, payload={"document": document.__dict__})

    def process_new_upload(self, file_path: str, original_name: str | None = None, index_document: bool = True) -> ToolResult:
        try:
            doc, created = self.ingestion.register_upload(file_path, original_name=original_name, source=DocSource.UPLOAD)
            if not created:
                return ToolResult(ok=True, payload={"document_id": doc.id, "duplicate": True})
            result = self.ingestion.process_document(doc.id, reindex=index_document)
            return ToolResult(
                ok=result.get("status") != "failed",
                payload={"document_id": doc.id, "duplicate": False, "processing": result, "indexed": index_document},
                error=result.get("reason") if result.get("status") == "failed" else None,
            )
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
