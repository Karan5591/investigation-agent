import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from app.db.database import Database
from app.db.models import DocSource, DocStatus, Document, ExtractionMethod, Page


def _now() -> str:
    # Fixed-width ISO format so string comparison matches time ordering.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _to_document(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        filename=row["filename"],
        file_path=row["file_path"],
        content_hash=row["content_hash"],
        mime_type=row["mime_type"],
        source=DocSource(row["source"]),
        status=DocStatus(row["status"]),
        page_count=row["page_count"],
        needs_ocr=None if row["needs_ocr"] is None else bool(row["needs_ocr"]),
        ocr_avg_confidence=row["ocr_avg_confidence"],
        chunk_count=row["chunk_count"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class DocumentRepository:
    def __init__(self, db: Database):
        self.db = db

    # ---------- registration / lookup ----------

    def register(
        self,
        filename: str,
        file_path: str,
        content_hash: str,
        mime_type: Optional[str] = None,
        source: DocSource = DocSource.UPLOAD,
    ) -> tuple[Document, bool]:
        """Register a file. Returns (document, created).
        If the same content was registered before, returns the existing record
        with created=False, so a duplicate upload is never processed twice."""
        doc_id = str(uuid.uuid4())
        now = _now()
        try:
            with self.db.connect() as conn:
                conn.execute(
                    """INSERT INTO documents
                       (id, filename, file_path, content_hash, mime_type, source,
                        status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (doc_id, filename, file_path, content_hash, mime_type,
                     source.value, DocStatus.REGISTERED.value, now, now),
                )
            return self.get(doc_id), True
        except sqlite3.IntegrityError:
            # UNIQUE(content_hash) fired: safe even if two uploads race.
            existing = self.get_by_hash(content_hash)
            if existing is None:  # pragma: no cover - would be some other constraint
                raise
            return existing, False

    def get(self, doc_id: str) -> Optional[Document]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return _to_document(row) if row else None

    def get_by_hash(self, content_hash: str) -> Optional[Document]:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return _to_document(row) if row else None

    def list_documents(
        self,
        status: Optional[DocStatus] = None,
        source: Optional[DocSource] = None,
    ) -> list[Document]:
        query, params = "SELECT * FROM documents WHERE 1=1", []
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if source:
            query += " AND source = ?"
            params.append(source.value)
        query += " ORDER BY created_at"
        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_to_document(r) for r in rows]

    # ---------- processing state ----------

    def claim_for_processing(self, doc_id: str) -> bool:
        """Atomically move registered/failed -> processing.
        Returns False if another worker already claimed it or it is already indexed."""
        with self.db.connect() as conn:
            cur = conn.execute(
                """UPDATE documents SET status = ?, error = NULL, updated_at = ?
                   WHERE id = ? AND status IN (?, ?)""",
                (DocStatus.PROCESSING.value, _now(), doc_id,
                 DocStatus.REGISTERED.value, DocStatus.FAILED.value),
            )
            return cur.rowcount == 1

    def mark_indexed(self, doc_id: str, chunk_count: int) -> None:
        self._update(doc_id, status=DocStatus.INDEXED.value, chunk_count=chunk_count, error=None)

    def mark_needs_review(self, doc_id: str, error: str | None = None) -> None:
        self._update(doc_id, status=DocStatus.NEEDS_REVIEW.value, chunk_count=0, error=error)

    def mark_failed(self, doc_id: str, error: str) -> None:
        self._update(doc_id, status=DocStatus.FAILED.value, error=error)

    def set_page_count(self, doc_id: str, page_count: int) -> None:
        self._update(doc_id, page_count=page_count)

    def set_ocr_info(
        self, doc_id: str, needs_ocr: bool, avg_confidence: Optional[float] = None
    ) -> None:
        self._update(doc_id, needs_ocr=int(needs_ocr), ocr_avg_confidence=avg_confidence)

    def reset_stale_processing(self, older_than_seconds: float) -> int:
        """Recover from interrupted processing: documents stuck in 'processing'
        for longer than the threshold go back to 'registered' so they can be retried."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)).isoformat(
            timespec="microseconds"
        )
        with self.db.connect() as conn:
            cur = conn.execute(
                """UPDATE documents SET status = ?, error = ?, updated_at = ?
                   WHERE status = ? AND updated_at < ?""",
                (DocStatus.REGISTERED.value, "interrupted; reset for retry", _now(),
                 DocStatus.PROCESSING.value, cutoff),
            )
            return cur.rowcount

    # ---------- pages ----------

    def save_pages(self, doc_id: str, pages: Iterable[Page]) -> None:
        """Replace all stored pages for a document (idempotent on reprocessing)."""
        with self.db.connect() as conn:
            conn.execute("DELETE FROM pages WHERE document_id = ?", (doc_id,))
            conn.executemany(
                """INSERT INTO pages
                   (document_id, page_number, method, text, ocr_confidence, needs_review)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (doc_id, p.page_number, p.method.value, p.text,
                     p.ocr_confidence, int(p.needs_review))
                    for p in pages
                ],
            )

    def get_pages(self, doc_id: str) -> list[Page]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pages WHERE document_id = ? ORDER BY page_number", (doc_id,)
            ).fetchall()
        return [
            Page(
                document_id=r["document_id"],
                page_number=r["page_number"],
                method=ExtractionMethod(r["method"]),
                text=r["text"],
                ocr_confidence=r["ocr_confidence"],
                needs_review=bool(r["needs_review"]),
            )
            for r in rows
        ]

    # ---------- internals ----------

    def _update(self, doc_id: str, **fields) -> None:
        fields["updated_at"] = _now()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self.db.connect() as conn:
            conn.execute(
                f"UPDATE documents SET {assignments} WHERE id = ?",
                (*fields.values(), doc_id),
            )
