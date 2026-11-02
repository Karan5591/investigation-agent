import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id                 TEXT PRIMARY KEY,
    filename           TEXT NOT NULL,
    file_path          TEXT NOT NULL,
    content_hash       TEXT NOT NULL UNIQUE,   -- duplicate-upload detection
    mime_type          TEXT,
    source             TEXT NOT NULL,          -- corpus | upload
    status             TEXT NOT NULL,          -- registered | processing | indexed | failed
    page_count         INTEGER,
    needs_ocr          INTEGER,                -- NULL = undecided, 0 = no, 1 = yes
    ocr_avg_confidence REAL,
    chunk_count        INTEGER NOT NULL DEFAULT 0,
    error              TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

CREATE TABLE IF NOT EXISTS pages (
    document_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number    INTEGER NOT NULL,
    method         TEXT NOT NULL,              -- native | ocr
    text           TEXT NOT NULL,
    ocr_confidence REAL,
    needs_review   INTEGER NOT NULL DEFAULT 0, -- 1 = unverified, do not treat as fact
    PRIMARY KEY (document_id, page_number)
);
"""


class Database:
    """Thin SQLite wrapper. Opens a short-lived connection per operation, which
    keeps it safe to use from FastAPI worker threads. Use a real file path
    (not ':memory:'), because each connection would otherwise get its own database."""

    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")  # better read/write concurrency
            conn.executescript(SCHEMA)
