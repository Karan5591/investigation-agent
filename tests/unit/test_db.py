import time

import pytest

from app.db.database import Database
from app.db.models import DocSource, DocStatus, ExtractionMethod, Page
from app.db.repository import DocumentRepository


@pytest.fixture
def repo(tmp_path):
    return DocumentRepository(Database(str(tmp_path / "test.db")))


def _register(repo, name="a.pdf", content_hash="hash-a", **kw):
    return repo.register(name, f"/data/{name}", content_hash, "application/pdf", **kw)


def test_register_new_document(repo):
    doc, created = _register(repo)
    assert created is True
    assert doc.status == DocStatus.REGISTERED
    assert doc.source == DocSource.UPLOAD
    assert doc.needs_ocr is None
    assert doc.chunk_count == 0


def test_duplicate_upload_returns_existing_record(repo):
    first, created1 = _register(repo, name="a.pdf", content_hash="same")
    second, created2 = _register(repo, name="renamed.pdf", content_hash="same")
    assert created1 is True
    assert created2 is False
    assert second.id == first.id
    assert len(repo.list_documents()) == 1


def test_claim_is_atomic(repo):
    doc, _ = _register(repo)
    assert repo.claim_for_processing(doc.id) is True
    assert repo.claim_for_processing(doc.id) is False  # second worker loses
    assert repo.get(doc.id).status == DocStatus.PROCESSING


def test_failed_document_can_be_retried(repo):
    doc, _ = _register(repo)
    repo.claim_for_processing(doc.id)
    repo.mark_failed(doc.id, "ocr crashed")
    failed = repo.get(doc.id)
    assert failed.status == DocStatus.FAILED
    assert failed.error == "ocr crashed"

    assert repo.claim_for_processing(doc.id) is True
    assert repo.get(doc.id).error is None


def test_indexed_document_cannot_be_reclaimed(repo):
    doc, _ = _register(repo)
    repo.claim_for_processing(doc.id)
    repo.mark_indexed(doc.id, chunk_count=7)
    indexed = repo.get(doc.id)
    assert indexed.status == DocStatus.INDEXED
    assert indexed.chunk_count == 7
    assert repo.claim_for_processing(doc.id) is False


def test_ocr_info_and_page_count(repo):
    doc, _ = _register(repo)
    repo.set_page_count(doc.id, 3)
    repo.set_ocr_info(doc.id, needs_ocr=True, avg_confidence=62.5)
    updated = repo.get(doc.id)
    assert updated.page_count == 3
    assert updated.needs_ocr is True
    assert updated.ocr_avg_confidence == 62.5


def test_save_pages_replaces_and_keeps_review_flag(repo):
    doc, _ = _register(repo)
    repo.save_pages(doc.id, [
        Page(doc.id, 1, ExtractionMethod.OCR, "blurry text", 41.0, needs_review=True),
        Page(doc.id, 2, ExtractionMethod.NATIVE, "clean text"),
    ])
    pages = repo.get_pages(doc.id)
    assert [p.page_number for p in pages] == [1, 2]
    assert pages[0].needs_review is True
    assert pages[0].ocr_confidence == 41.0
    assert pages[1].method == ExtractionMethod.NATIVE

    # Reprocessing replaces, it does not duplicate.
    repo.save_pages(doc.id, [Page(doc.id, 1, ExtractionMethod.NATIVE, "new")])
    assert len(repo.get_pages(doc.id)) == 1


def test_reset_stale_processing_recovers_interrupted_documents(repo):
    stuck, _ = _register(repo, name="stuck.pdf", content_hash="h1")
    done, _ = _register(repo, name="done.pdf", content_hash="h2")
    repo.claim_for_processing(stuck.id)
    repo.claim_for_processing(done.id)
    repo.mark_indexed(done.id, 3)

    time.sleep(0.01)
    assert repo.reset_stale_processing(older_than_seconds=0) == 1

    assert repo.get(stuck.id).status == DocStatus.REGISTERED
    assert repo.get(done.id).status == DocStatus.INDEXED


def test_recent_processing_is_not_reset(repo):
    doc, _ = _register(repo)
    repo.claim_for_processing(doc.id)
    assert repo.reset_stale_processing(older_than_seconds=3600) == 0
    assert repo.get(doc.id).status == DocStatus.PROCESSING


def test_list_filters(repo):
    _register(repo, name="c.pdf", content_hash="c", source=DocSource.CORPUS)
    _register(repo, name="u.pdf", content_hash="u", source=DocSource.UPLOAD)
    assert len(repo.list_documents(source=DocSource.CORPUS)) == 1
    assert len(repo.list_documents(status=DocStatus.REGISTERED)) == 2
    assert repo.list_documents(status=DocStatus.INDEXED) == []
