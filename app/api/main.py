from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.agent.investigator import InvestigationAgent
from app.api.schemas import InvestigationRequest, InvestigationResponse, UploadResponse
from app.config import get_settings
from app.db.database import Database
from app.db.models import DocSource
from app.ingestion.service import IngestionService
from app.llm.client import LLMClient
from app.retrieval.search import SearchClient

settings = get_settings()
app = FastAPI(title="Investigation Agent", version="0.1.0")


def ensure_demo_corpus() -> None:
    sample_dir = Path(__file__).resolve().parents[2] / "data" / "sample_docs"
    if not sample_dir.exists():
        return

    ingestion = IngestionService(db_path=settings.sqlite_path, es_url=settings.es_url, es_index=settings.es_index)
    ingestion.repo.reset_stale_processing(settings.processing_stale_seconds)
    ingestion.seed_sample_documents(sample_dir)


@app.on_event("startup")
def start_event() -> None:
    ensure_demo_corpus()


def build_agent() -> InvestigationAgent:
    db = Database(settings.sqlite_path)
    search = SearchClient(settings.es_url, settings.es_index, use_remote=settings.es_enabled)
    return InvestigationAgent(db=db, search_client=search, llm=LLMClient())


def _store_upload(file: UploadFile) -> tuple[Path, str]:
    original_name = file.filename or "upload"
    safe_name = Path(original_name.replace("\\", "/")).name
    if safe_name in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid upload filename")

    destination = Path(settings.upload_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{uuid.uuid4().hex}_{safe_name}"
    if target.resolve().parent != destination:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    with target.open("wb") as handle:
        handle.write(file.file.read())
    return target, safe_name


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/investigate", response_model=InvestigationResponse)
def investigate(request: InvestigationRequest) -> InvestigationResponse:
    agent = build_agent()
    result = agent.investigate(
        question=request.question,
        top_k=request.top_k,
    )
    return InvestigationResponse(
        status=result.status.value,
        answer=result.answer,
        confidence=result.confidence,
        missing_information=result.missing_information,
        warnings=result.warnings,
        evidence=[
            {
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "page": chunk.page,
                "source": chunk.source,
                "snippet": chunk.text[:250],
                "score": chunk.score,
            }
            for chunk in result.evidence
        ],
        sources=result.sources,
        trace=result.trace,
        failure_reason=result.failure_reason,
    )


@app.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    target, original_name = _store_upload(file)
    ingestion = IngestionService(db_path=settings.sqlite_path, es_url=settings.es_url, es_index=settings.es_index)
    document, created = ingestion.register_upload(str(target), original_name=original_name, source=DocSource.UPLOAD)
    result = ingestion.process_document(document.id)
    return UploadResponse(
        document_id=document.id,
        filename=document.filename,
        status=result.get("status", "registered"),
        message="Document uploaded and processed." if created else "Duplicate upload reused.",
        duplicate=not created,
    )


@app.post("/investigate-upload")
def investigate_with_upload(question: str = Form(...), file: UploadFile | None = File(default=None)) -> InvestigationResponse:
    agent = build_agent()
    uploaded_path = None
    original_name = None
    if file is not None:
        target, original_name = _store_upload(file)
        uploaded_path = str(target)
    result = agent.investigate(
        question=question,
        uploaded_file=uploaded_path,
        uploaded_filename=original_name,
    )
    return InvestigationResponse(
        status=result.status.value,
        answer=result.answer,
        confidence=result.confidence,
        missing_information=result.missing_information,
        warnings=result.warnings,
        evidence=[
            {
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "page": chunk.page,
                "source": chunk.source,
                "snippet": chunk.text[:250],
                "score": chunk.score,
            }
            for chunk in result.evidence
        ],
        sources=result.sources,
        trace=result.trace,
        failure_reason=result.failure_reason,
    )
