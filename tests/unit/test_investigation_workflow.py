from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi import UploadFile
from pydantic import ValidationError

from app.agent.investigator import InvestigationAgent, ToolExecutionError
from app.api import main as api_main
from app.api.schemas import InvestigationRequest
from app.db.database import Database
from app.db.models import DocSource
from app.db.repository import DocumentRepository
from app.evidence.models import EvidenceChunk, EvidenceStatus
from app.ingestion.service import IngestionService
from app.llm.client import LLMClient, LLMClientError
from app.ocr.processor import OCRProcessor
from app.retrieval import search as search_module
from app.retrieval.search import SearchClient


class FakeLLM(LLMClient):
    def __init__(self, planned_actions=None):
        super().__init__()
        self.planned_actions = list(planned_actions or [])
        self.planner_calls = 0

    def decide_required_info(self, question: str) -> list[str]:
        return ["incident timeline", "contract threshold"]

    def plan_action(self, prompt: str) -> dict:
        self.planner_calls += 1
        if self.planned_actions:
            return self.planned_actions.pop(0)
        question = next(line.removeprefix("Question: ") for line in prompt.splitlines() if line.startswith("Question: "))
        if self.planner_calls == 1:
            max_top_k = int(next(
                line.rsplit(" ", 1)[-1].rstrip(".")
                for line in prompt.splitlines()
                if line.startswith("Search top_k must be between")
            ))
            return {
                "required_information": self.decide_required_info(question),
                "action": "search_documents",
                "arguments": {"query": question, "top_k": max_top_k},
            }
        return {"required_information": self.decide_required_info(question), "action": "finish", "arguments": {}}

    def generate(self, prompt: str, *, schema=None) -> dict:
        return {
            "answer": "The outage exceeded four hours, so the policy supports a service credit review.",
            "confidence": 0.86,
            "needs_more_evidence": False,
            "missing_information": [],
            "sources": ["incident_log.txt"],
            "reasoning": "The timeline and SLA wording align.",
        }


def test_investigation_returns_ready_answer(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    search = SearchClient("http://localhost:9200", "doc_chunks")
    search.index_document(
        "doc-1",
        "incident_log.txt",
        "The outage began at 02:15 and recovery completed at 06:40. This exceeded the four-hour threshold.",
        "corpus",
        page=1,
    )
    search.index_document(
        "doc-2",
        "company_policy.txt",
        "Any outage that exceeds four hours may qualify for a service credit review.",
        "corpus",
        page=2,
    )

    agent = InvestigationAgent(db=db, search_client=search, llm=FakeLLM())
    result = agent.investigate("Did the outage qualify for a service credit review?", top_k=5)

    assert result.status == EvidenceStatus.READY
    assert result.answer is not None
    assert result.confidence >= 0.8
    assert len(result.evidence) >= 2


def test_investigation_abstains_when_evidence_is_missing(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    search = SearchClient("http://localhost:9200", "doc_chunks")
    llm = FakeLLM()
    agent = InvestigationAgent(db=db, search_client=search, llm=llm)

    result = agent.investigate("This question is intentionally unsupported and should fail the evidence check.", top_k=2)

    assert result.status == EvidenceStatus.INSUFFICIENT
    assert result.answer is None
    assert result.missing_information


def test_unrelated_numbers_do_not_count_as_contradictions(tmp_path):
    agent = InvestigationAgent(
        db=Database(str(tmp_path / "test.db")),
        search_client=SearchClient("http://localhost:9200", "doc_chunks"),
        llm=FakeLLM(),
    )
    evidence = [
        EvidenceChunk(
            doc_id="policy",
            filename="company_policy.txt",
            source="corpus",
            text="Report within 24 hours. Credit review applies after a 4-hour SLA breach.",
        ),
        EvidenceChunk(
            doc_id="incident",
            filename="incident_log.txt",
            source="corpus",
            text="Incident occurred on 2026-10-12, from 02:15 until 06:40.",
        ),
    ]

    assert not agent._detect_contradictions("What is the office address in Tokyo?", evidence)


def test_opposing_credit_eligibility_claims_are_contradictory(tmp_path):
    agent = InvestigationAgent(
        db=Database(str(tmp_path / "test.db")),
        search_client=SearchClient("http://localhost:9200", "doc_chunks"),
        llm=FakeLLM(),
    )
    evidence = [
        EvidenceChunk(
            doc_id="policy",
            filename="company_policy.txt",
            source="corpus",
            text="This incident qualifies for a service credit review.",
        ),
        EvidenceChunk(
            doc_id="vendor",
            filename="vendor_contract.txt",
            source="corpus",
            text="This incident does not qualify for a service credit review.",
        ),
    ]

    assert agent._detect_contradictions("Did this incident qualify for a service credit review?", evidence)


def test_conflicting_duration_reports_for_same_incident_are_contradictory(tmp_path):
    agent = InvestigationAgent(
        db=Database(str(tmp_path / "test.db")),
        search_client=SearchClient("http://localhost:9200", "doc_chunks"),
        llm=FakeLLM(),
    )
    evidence = [
        EvidenceChunk(
            doc_id="operations-review",
            filename="incident_duration_review.txt",
            source="corpus",
            text="Incident INC-2041: the outage duration was 3 hours.",
        ),
        EvidenceChunk(
            doc_id="vendor-review",
            filename="vendor_duration_review.txt",
            source="corpus",
            text="Incident INC-2041: the outage lasted 5 hours.",
        ),
    ]

    assert agent._detect_contradictions("What duration did the reports give for incident INC-2041?", evidence)


def test_sample_contradiction_claims_keep_incident_id_in_same_chunk(tmp_path):
    sample_dir = Path(__file__).resolve().parents[2] / "data" / "sample_docs"
    filenames = ["incident_duration_review.txt", "vendor_duration_review.txt"]
    claim_chunks = []
    for filename in filenames:
        text = (sample_dir / filename).read_text(encoding="utf-8")
        claim = next(part for part in text.split("\n\n") if "duration" in part.lower() or "lasted" in part.lower())
        claim_chunks.append(
            EvidenceChunk(
                doc_id=filename,
                filename=filename,
                source="corpus",
                text=claim,
            )
        )

    agent = InvestigationAgent(
        db=Database(str(tmp_path / "test.db")),
        search_client=SearchClient("http://localhost:9200", "doc_chunks"),
        llm=FakeLLM(),
    )
    assert agent._detect_contradictions("What duration did reports give for incident INC-2041?", claim_chunks)


def test_service_credit_question_with_threshold_support_is_not_marked_contradictory(tmp_path):
    agent = InvestigationAgent(
        db=Database(str(tmp_path / "test.db")),
        search_client=SearchClient("http://localhost:9200", "doc_chunks"),
        llm=FakeLLM(),
    )
    evidence = [
        EvidenceChunk(
            doc_id="policy",
            filename="company_policy.txt",
            source="corpus",
            text="Service credit eligibility is triggered when recovery exceeds four hours from the first confirmed outage.",
        ),
        EvidenceChunk(
            doc_id="upload",
            filename="uploaded_note.txt",
            source="corpus",
            text="Recovery was completed at 06:40, which is more than four hours. The final internal decision was to approve a credit review.",
        ),
        EvidenceChunk(
            doc_id="incident-review",
            filename="incident_duration_review.txt",
            source="corpus",
            text="Incident INC-2041: the outage duration was 3 hours.",
        ),
        EvidenceChunk(
            doc_id="vendor-review",
            filename="vendor_duration_review.txt",
            source="corpus",
            text="Incident INC-2041: the outage lasted 5 hours.",
        ),
    ]

    assert not agent._detect_contradictions("Did the outage qualify for a service credit review?", evidence)


def test_malformed_model_output_fails_instead_of_using_a_canned_answer():
    client = LLMClient()
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="not valid JSON"))]
                )
            )
        )
    )

    with pytest.raises(LLMClientError, match="LLM returned malformed output or failed"):
        client.generate(
            "Question: Did the outage qualify for a service credit review?\n"
            "Evidence: Recovery exceeded the 4 hours SLA threshold."
        )


def test_missing_llm_configuration_is_an_explicit_failure():
    client = LLMClient()
    client.client = None

    with pytest.raises(LLMClientError, match="LLM is not configured"):
        client.generate("Question: unsupported\nEvidence: none")


def test_local_llm_uses_base_url_without_an_api_key(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("app.llm.client.get_settings", lambda: SimpleNamespace(
        openai_api_key="",
        openai_model="qwen2.5:7b",
        openai_base_url="http://localhost:11434/v1",
    ))
    monkeypatch.setattr("app.llm.client.OpenAI", FakeOpenAI)

    client = LLMClient()

    assert client.client is not None
    assert captured == {"api_key": "local", "base_url": "http://localhost:11434/v1"}


def test_investigation_normalizes_string_confidence_and_missing_information(tmp_path):
    class TypedResponseLLM(FakeLLM):
        def generate(self, prompt: str, *, schema=None) -> dict:
            return {
                "answer": "The approval status is not established.",
                "confidence": "high",
                "needs_more_evidence": "true",
                "missing_information": "Signed approval record",
                "reasoning": "The approval record is absent.",
            }

    db = Database(str(tmp_path / "test.db"))
    search = SearchClient("http://localhost:9200", "doc_chunks")
    search.index_document("doc-1", "incident.txt", "The incident was reviewed.", "corpus", page=1)
    agent = InvestigationAgent(db=db, search_client=search, llm=TypedResponseLLM())

    result = agent.investigate("Was approval documented?", top_k=1)

    assert result.status == EvidenceStatus.INSUFFICIENT
    assert result.confidence == 0.85
    assert "Signed approval record" in result.missing_information
    assert result.warnings == ["The approval record is absent."]


def test_search_tool_failure_returns_failed_not_insufficient(tmp_path):
    class FailedSearchClient(SearchClient):
        def search(self, query, top_k=5, *, embedding=None):
            raise ConnectionError("Elasticsearch is unavailable")

    agent = InvestigationAgent(
        db=Database(str(tmp_path / "test.db")),
        search_client=FailedSearchClient("http://localhost:9200", "doc_chunks"),
        llm=FakeLLM(),
    )

    result = agent.investigate("Did the outage qualify for a credit review?")

    assert result.status == EvidenceStatus.FAILED
    assert result.failure_reason == "Elasticsearch is unavailable"


def test_search_tool_timeout_returns_failed_not_insufficient(tmp_path):
    class TimedOutSearchClient(SearchClient):
        def search(self, query, top_k=5, *, embedding=None):
            raise TimeoutError("Elasticsearch search timed out")

    agent = InvestigationAgent(
        db=Database(str(tmp_path / "test.db")),
        search_client=TimedOutSearchClient("http://localhost:9200", "doc_chunks"),
        llm=FakeLLM(),
    )

    result = agent.investigate("Did the outage qualify for a credit review?")

    assert result.status == EvidenceStatus.FAILED
    assert result.failure_reason == "Elasticsearch search timed out"


def test_investigation_request_rejects_server_file_paths():
    with pytest.raises(ValidationError):
        InvestigationRequest(question="Investigate this", uploaded_file="/etc/passwd")


def test_upload_filename_cannot_escape_upload_directory(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(api_main.settings, "upload_dir", str(upload_dir))
    upload = UploadFile(filename="../../escape.txt", file=BytesIO(b"test upload"))

    target, original_name = api_main._store_upload(upload)

    assert target.parent == upload_dir.resolve()
    assert target.read_bytes() == b"test upload"
    assert original_name == "escape.txt"
    assert not (tmp_path / "escape.txt").exists()


def test_planner_can_select_search_then_metadata_then_finish(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    search = SearchClient("http://localhost:9200", "doc_chunks")
    document, _ = DocumentRepository(db).register(
        filename="incident.txt",
        file_path=str(tmp_path / "incident.txt"),
        content_hash="planner-test-hash",
        source=DocSource.CORPUS,
    )
    search.index_document(
        document.id,
        "incident.txt",
        "The outage was caused by a database failure.",
        "corpus",
        page=1,
    )
    llm = FakeLLM(
        planned_actions=[
            {
                "required_information": ["incident cause"],
                "action": "search_documents",
                "arguments": {"query": "What caused the outage?", "top_k": 3},
            },
            {
                "required_information": ["incident cause"],
                "action": "fetch_document_metadata",
                "arguments": {"document_id": document.id},
            },
            {"required_information": ["incident cause"], "action": "finish", "arguments": {}},
        ]
    )
    agent = InvestigationAgent(db=db, search_client=search, llm=llm)

    result = agent.investigate("What caused the outage?", top_k=3)

    assert result.status == EvidenceStatus.READY
    assert llm.planner_calls == 3
    assert any("selected fetch_document_metadata" in step for step in result.trace)


def test_planner_can_process_user_upload_as_direct_evidence(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    search = SearchClient("http://localhost:9200", "doc_chunks")
    upload_path = tmp_path / "inspection.txt"
    upload_path.write_text("Asset EQ-77 safety test result: PASSED.", encoding="utf-8")
    llm = FakeLLM(
        planned_actions=[
            {"action": "process_upload", "arguments": {}},
            {"action": "finish", "arguments": {}},
        ]
    )
    agent = InvestigationAgent(db, search, llm)

    result = agent.investigate(
        "What safety test result is recorded for asset EQ-77?",
        uploaded_file=str(upload_path),
        uploaded_filename="inspection.txt",
    )

    assert result.status == EvidenceStatus.READY
    assert any(chunk.source == "upload" and "PASSED" in chunk.text for chunk in result.evidence)
    assert any("selected process_upload" in step for step in result.trace)


def test_planner_chains_search_upload_research_and_finish(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    search = SearchClient("http://localhost:9200", "doc_chunks")
    search.index_document(
        "incident",
        "incident.txt",
        "Incident INC-2041 recovery completed at 06:40, exceeding the four-hour threshold.",
        "corpus",
        page=2,
    )
    upload_path = tmp_path / "confirmation.txt"
    upload_path.write_text("Customer confirmation for INC-2041 was signed on 2026-10-25.", encoding="utf-8")
    llm = FakeLLM(
        planned_actions=[
            {"action": "search_documents", "arguments": {"query": "INC-2041 recovery threshold", "top_k": 3}},
            {"action": "process_upload", "arguments": {}},
            {"action": "search_documents", "arguments": {"query": "INC-2041 customer confirmation signed", "top_k": 3}},
            {"action": "finish", "arguments": {}},
        ]
    )
    llm.settings.agent_max_steps = 6
    agent = InvestigationAgent(db, search, llm)

    result = agent.investigate(
        "Did the outage qualify for a credit review, and was customer confirmation signed?",
        uploaded_file=str(upload_path),
        uploaded_filename="confirmation.txt",
        top_k=3,
    )

    selected_actions = [entry for entry in result.trace if "selected " in entry]
    assert result.status == EvidenceStatus.READY
    assert len(selected_actions) == 4
    assert any(chunk.filename == "incident.txt" for chunk in result.evidence)
    assert any(chunk.filename == "confirmation.txt" for chunk in result.evidence)


def test_planner_rejects_unallowlisted_action_and_arbitrary_paths():
    with pytest.raises(ToolExecutionError, match="disallowed action"):
        InvestigationAgent._validate_action(
            {"action": "execute_shell", "arguments": {}},
            document_ids=set(),
            upload_available=False,
            top_k=5,
        )


def test_planner_invalid_action_returns_failed(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    search = SearchClient("http://localhost:9200", "doc_chunks")
    invalid_action = {"action": "execute_shell", "arguments": {}}
    llm = FakeLLM(planned_actions=[invalid_action, invalid_action])
    result = InvestigationAgent(db, search, llm).investigate("Investigate the incident")

    assert result.status == EvidenceStatus.FAILED
    assert "disallowed action" in result.failure_reason


def test_planner_stops_repeated_actions(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    search = SearchClient("http://localhost:9200", "doc_chunks")
    repeated_search = {
        "action": "search_documents",
        "arguments": {"query": "incident", "top_k": 2},
    }
    llm = FakeLLM(planned_actions=[repeated_search, repeated_search, repeated_search])
    result = InvestigationAgent(db, search, llm).investigate("Investigate the incident", top_k=2)

    assert result.status == EvidenceStatus.INSUFFICIENT
    assert any("repeated the same action" in warning for warning in result.warnings)


def test_planner_repeated_search_uses_evidence_already_retrieved(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    search = SearchClient("http://localhost:9200", "doc_chunks")
    search.index_document(
        "python-lists",
        "python_notes.txt",
        "A Python list is an ordered sequence enclosed in square brackets.",
        "corpus",
        page=1,
    )
    repeated_search = {
        "action": "search_documents",
        "arguments": {"query": "What is a list in Python?", "top_k": 3},
    }
    llm = FakeLLM(planned_actions=[repeated_search, repeated_search, repeated_search])
    agent = InvestigationAgent(db, search, llm)

    result = agent.investigate("What is a list in Python?", top_k=3)

    assert result.status == EvidenceStatus.READY
    assert any("Proceeding with already gathered evidence" in entry for entry in result.trace)
    assert any(chunk.filename == "python_notes.txt" for chunk in result.evidence)


def test_planner_stops_at_configured_step_limit(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    search = SearchClient("http://localhost:9200", "doc_chunks")
    search.index_document("doc-1", "incident.txt", "Incident details are recorded.", "corpus", page=1)
    llm = FakeLLM(
        planned_actions=[
            {"action": "search_documents", "arguments": {"query": "incident details", "top_k": 2}},
            {"action": "search_documents", "arguments": {"query": "recorded incident", "top_k": 2}},
        ]
    )
    llm.settings.agent_max_steps = 2
    result = InvestigationAgent(db, search, llm).investigate("Investigate the incident", top_k=2)

    assert result.status == EvidenceStatus.INSUFFICIENT
    assert any("step limit" in warning for warning in result.warnings)


def test_remote_search_creates_vector_mapping_and_never_falls_back_to_memory(monkeypatch):
    class FakeIndices:
        def __init__(self):
            self.created = None

        def exists(self, index):
            return False

        def create(self, index, mappings):
            self.created = (index, mappings)

    class FakeElasticsearch:
        def __init__(self, url):
            self.indices = FakeIndices()
            self.indexed = []

        def index(self, **kwargs):
            self.indexed.append(kwargs)

    monkeypatch.setattr(search_module, "Elasticsearch", FakeElasticsearch)
    client = SearchClient("http://localhost:9200", "test-index", use_remote=True)

    client.index_document("doc-1", "sample.txt", "Sample text", "corpus", embedding=[0.1, 0.2])

    assert client._es.indices.created[0] == "test-index"
    assert client._es.indices.created[1]["properties"]["embedding"]["dims"] == 2
    assert len(client._es.indexed) == 1
    assert client._memory == []


def test_remote_indexing_failure_is_reported_not_hidden(monkeypatch):
    class FailedIndices:
        def exists(self, index):
            raise ConnectionError("Elasticsearch is unavailable")

    class FailedElasticsearch:
        def __init__(self, url):
            self.indices = FailedIndices()

    monkeypatch.setattr(search_module, "Elasticsearch", FailedElasticsearch)
    client = SearchClient("http://localhost:9200", "test-index", use_remote=True)

    with pytest.raises(ConnectionError, match="Elasticsearch is unavailable"):
        client.index_document("doc-1", "sample.txt", "Sample text", "corpus", embedding=[0.1, 0.2])
    assert client._memory == []


def test_planner_rejects_upload_path_argument():
    with pytest.raises(ToolExecutionError, match="unsupported arguments"):
        InvestigationAgent._validate_action(
            {"action": "process_upload", "arguments": {"path": "/etc/passwd"}},
            document_ids=set(),
            upload_available=True,
            top_k=5,
        )


def test_low_confidence_ocr_is_saved_for_review_not_indexed(tmp_path, monkeypatch):
    service = IngestionService(db_path=str(tmp_path / "test.db"))
    service.ocr.min_confidence = 70.0
    file_path = tmp_path / "scan.png"
    file_path.write_bytes(b"mock scanned image")
    document, created = service.register_upload(str(file_path), original_name=file_path.name)
    indexed_chunks = []
    monkeypatch.setattr(service.ocr, "extract_pages", lambda _: [("uncertain scanned text", True, 42.0)])
    monkeypatch.setattr(service.search_client, "index_document", lambda **kwargs: indexed_chunks.append(kwargs))

    result = service.process_document(document.id)

    assert created is True
    assert result["status"] == "needs_review"
    assert result["confidence"] == 42.0
    assert indexed_chunks == []
    saved_page = service.repo.get_pages(document.id)[0]
    assert saved_page.method.value == "ocr"
    assert saved_page.ocr_confidence == 42.0
    assert saved_page.needs_review is True


def test_ocr_confidence_uses_valid_tesseract_word_scores(monkeypatch):
    from app.ocr import processor

    monkeypatch.setattr(processor.pytesseract, "image_to_string", lambda *args, **kwargs: "sample text")
    monkeypatch.setattr(
        processor.pytesseract,
        "image_to_data",
        lambda *args, **kwargs: {"conf": ["-1", "80", "bad", "60"]},
    )

    text, confidence = OCRProcessor()._extract_ocr(object())

    assert text == "sample text"
    assert confidence == 70.0


def test_mixed_pdf_preserves_page_numbers_and_ocr_method(tmp_path, monkeypatch):
    import pymupdf

    pdf_path = tmp_path / "mixed.pdf"
    pdf = pymupdf.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "Native page one")
    pdf.new_page()
    third = pdf.new_page()
    third.insert_text((72, 72), "Native page three")
    pdf.save(pdf_path)
    pdf.close()

    service = IngestionService(db_path=str(tmp_path / "test.db"))
    document, _ = service.register_upload(str(pdf_path), original_name=pdf_path.name)
    indexed_pages = []
    monkeypatch.setattr(service.ocr, "_extract_ocr", lambda _: ("Scanned page two", 90.0))
    monkeypatch.setattr(service.search_client, "index_document", lambda **kwargs: indexed_pages.append(kwargs["page"]))

    result = service.process_document(document.id)
    pages = service.repo.get_pages(document.id)

    assert result["status"] == "indexed"
    assert [(page.page_number, page.method.value) for page in pages] == [
        (1, "native"),
        (2, "ocr"),
        (3, "native"),
    ]
    assert indexed_pages == [1, 2, 3]
