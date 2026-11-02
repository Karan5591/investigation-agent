from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class InvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)


class EvidenceReference(BaseModel):
    doc_id: str
    filename: str
    page: int | None = None
    source: str
    snippet: str
    score: float


class InvestigationResponse(BaseModel):
    status: Literal["ready", "insufficient", "contradictory", "failed"]
    answer: str | None = None
    confidence: float = 0.0
    missing_information: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    duplicate: bool = False
