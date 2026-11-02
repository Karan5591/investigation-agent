from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DocStatus(str, Enum):
    REGISTERED = "registered"   # known, nothing done yet
    PROCESSING = "processing"   # a worker has claimed it
    INDEXED = "indexed"         # chunks are in Elasticsearch
    NEEDS_REVIEW = "needs_review"  # extracted text is too uncertain to index
    FAILED = "failed"           # last attempt failed; see `error`


class DocSource(str, Enum):
    CORPUS = "corpus"   # part of the standing knowledge collection
    UPLOAD = "upload"   # submitted by the user during an investigation


class ExtractionMethod(str, Enum):
    NATIVE = "native"   # text taken directly from the file (e.g. PDF text layer)
    OCR = "ocr"         # text recognised from an image


@dataclass
class Document:
    id: str
    filename: str
    file_path: str
    content_hash: str
    mime_type: Optional[str]
    source: DocSource
    status: DocStatus
    page_count: Optional[int]
    needs_ocr: Optional[bool]            # None = not decided yet
    ocr_avg_confidence: Optional[float]  # 0-100, None if OCR was not used
    chunk_count: int
    error: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class Page:
    document_id: str
    page_number: int                     # 1-based
    method: ExtractionMethod
    text: str
    ocr_confidence: Optional[float] = None  # 0-100, only for OCR pages
    needs_review: bool = False              # True = treat as unverified evidence
