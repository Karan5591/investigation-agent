from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.db.database import Database
from app.db.models import DocSource, DocStatus, ExtractionMethod, Page
from app.db.repository import DocumentRepository
from app.ocr.processor import OCRProcessor
from app.retrieval.search import SearchClient


class IngestionService:
    def __init__(
        self,
        db_path: str | None = None,
        es_url: str | None = None,
        es_index: str | None = None,
        search_client: SearchClient | None = None,
    ):
        settings = get_settings()
        self.db = Database(db_path or settings.sqlite_path)
        self.repo = DocumentRepository(self.db)
        self.search_client = search_client or SearchClient(
            es_url or settings.es_url,
            es_index or settings.es_index,
            use_remote=settings.es_enabled,
        )
        self.ocr = OCRProcessor(lang=settings.ocr_lang, min_confidence=settings.ocr_min_confidence)

    def register_upload(self, file_path: str, original_name: str | None = None, source: DocSource = DocSource.UPLOAD) -> tuple[Any, bool]:
        path = Path(file_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return self.repo.register(
            filename=original_name or path.name,
            file_path=str(path),
            content_hash=digest,
            mime_type="application/octet-stream",
            source=source,
        )

    def seed_sample_documents(self, directory: str | Path | None = None) -> list[str]:
        sample_dir = Path(directory) if directory else Path(__file__).resolve().parents[2] / "data" / "sample_docs"
        if not sample_dir.exists():
            return []

        results: list[str] = []
        for file_path in sorted(sample_dir.iterdir()):
            if not file_path.is_file():
                continue

            existing = self.repo.get_by_hash(hashlib.sha256(file_path.read_bytes()).hexdigest())
            if existing is None:
                document, created = self.register_upload(str(file_path), original_name=file_path.name, source=DocSource.CORPUS)
            else:
                document = existing
                created = False

            if created or document.status not in {DocStatus.INDEXED, DocStatus.NEEDS_REVIEW}:
                result = self.process_document(document.id)
                results.append(f"{document.filename}: {result.get('status', 'unknown')}")
            elif document.status == DocStatus.INDEXED:
                self._index_pages(document, self.repo.get_pages(document.id))
                results.append(f"{document.filename}: reindexed")
        return results

    def process_document(self, doc_id: str, *, reindex: bool = True) -> dict[str, Any]:
        document = self.repo.get(doc_id)
        if document is None:
            raise ValueError(f"Document {doc_id} not found")

        if not self.repo.claim_for_processing(doc_id):
            return {"status": document.status.value, "already_claimed": True}

        try:
            file_path = document.file_path
            extracted_pages = self.ocr.extract_pages(file_path)
            ocr_confidences = [confidence for text, used_ocr, confidence in extracted_pages if used_ocr and confidence is not None]
            used_ocr = bool(ocr_confidences)
            confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else None
            self.repo.set_ocr_info(doc_id, needs_ocr=used_ocr, avg_confidence=confidence)
            if any(text.strip() for text, _, _ in extracted_pages):
                pages = [
                    Page(
                        document_id=doc_id,
                        page_number=index,
                        method=ExtractionMethod.OCR if page_used_ocr else ExtractionMethod.NATIVE,
                        text=text,
                        ocr_confidence=page_confidence,
                        needs_review=(
                            page_used_ocr
                            and (page_confidence is None or page_confidence < self.ocr.min_confidence)
                        ),
                    )
                    for index, (text, page_used_ocr, page_confidence) in enumerate(extracted_pages, start=1)
                ]
                self.repo.save_pages(doc_id, pages)
                page_count = len(pages)
                self.repo.set_page_count(doc_id, page_count)
                indexable_pages = [page for page in pages if not page.needs_review and page.text.strip()]
                if reindex:
                    if indexable_pages:
                        self._index_pages(document, indexable_pages)
                        self.repo.mark_indexed(doc_id, chunk_count=len(indexable_pages))
                        status_value = DocStatus.INDEXED.value
                    else:
                        self.repo.mark_needs_review(
                            doc_id,
                            "OCR confidence is below the configured review threshold.",
                        )
                        status_value = DocStatus.NEEDS_REVIEW.value
                else:
                    status_value = "extracted_in_memory"

                return {
                    "status": status_value,
                    "pages": page_count,
                    "indexed_pages": len(indexable_pages) if reindex else 0,
                    "used_ocr": used_ocr,
                    "confidence": confidence,
                    "needs_review_pages": sum(page.needs_review for page in pages),
                }

            self.repo.mark_failed(doc_id, "No text could be extracted from the document.")
            return {"status": "failed", "reason": "No text extracted"}
        except Exception as exc:  # pragma: no cover - operational safety
            self.repo.mark_failed(doc_id, str(exc))
            return {"status": "failed", "reason": str(exc)}

    def _split_pages(
        self,
        text: str,
        *,
        method: ExtractionMethod = ExtractionMethod.NATIVE,
        confidence: float | None = None,
        needs_review: bool = False,
    ) -> list[Page]:
        paragraphs = [section.strip() for section in text.split("\n\n") if section.strip()]
        if not paragraphs:
            paragraphs = [text.strip() or "Empty document"]
        pages: list[Page] = []
        for index, paragraph in enumerate(paragraphs, start=1):
            pages.append(
                Page(
                    document_id="",
                    page_number=index,
                    method=method,
                    text=paragraph,
                    ocr_confidence=confidence,
                    needs_review=needs_review,
                )
            )
        return pages

    def _index_pages(self, document, pages: list[Page]) -> None:
        for page in pages:
            if page.needs_review:
                continue
            page.document_id = document.id
            self.search_client.index_document(
                doc_id=document.id,
                filename=document.filename,
                text=page.text,
                source=document.source.value,
                page=page.page_number,
                method=page.method.value,
                ocr_confidence=page.ocr_confidence,
                needs_review=page.needs_review,
            )
