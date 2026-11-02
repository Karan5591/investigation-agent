from __future__ import annotations

from app.db.database import Database
from app.db.models import DocSource, DocStatus
from app.ingestion.service import IngestionService
from app.retrieval.search import SearchClient


def test_text_upload_flows_through_sqlite_and_search_index(tmp_path):
    upload = tmp_path / "inspection.txt"
    upload.write_text("Asset EQ-77 safety assessment passed.", encoding="utf-8")
    search = SearchClient("http://localhost:9200", "test-index", use_remote=False)
    ingestion = IngestionService(
        db_path=str(tmp_path / "integration.db"),
        search_client=search,
    )

    document, created = ingestion.register_upload(
        str(upload),
        original_name=upload.name,
        source=DocSource.UPLOAD,
    )
    result = ingestion.process_document(document.id)
    saved_document = ingestion.repo.get(document.id)
    saved_pages = ingestion.repo.get_pages(document.id)
    hits = search.search("safety assessment", top_k=3)

    assert created is True
    assert result["status"] == "indexed"
    assert saved_document.status == DocStatus.INDEXED
    assert len(saved_pages) == 1
    assert saved_pages[0].text == "Asset EQ-77 safety assessment passed."
    assert any(hit.filename == upload.name and hit.page == 1 for hit in hits)