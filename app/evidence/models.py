from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EvidenceStatus(str, Enum):
    READY = "ready"
    INSUFFICIENT = "insufficient"
    CONTRADICTORY = "contradictory"
    FAILED = "failed"


@dataclass
class EvidenceChunk:
    doc_id: str
    filename: str
    source: str
    page: Optional[int] = None
    text: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvestigationResult:
    question: str
    status: EvidenceStatus
    answer: Optional[str] = None
    confidence: float = 0.0
    evidence: list[EvidenceChunk] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
