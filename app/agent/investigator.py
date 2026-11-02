from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.db.database import Database
from app.db.repository import DocumentRepository
from app.evidence.models import EvidenceChunk, EvidenceStatus, InvestigationResult
from app.ingestion.service import IngestionService
from app.llm.client import LLMClient
from app.retrieval.reranker import Reranker
from app.retrieval.search import SearchClient, SearchHit
from app.tools.investigation_tools import InvestigationToolRegistry


class ToolExecutionError(Exception):
    pass


class PlannerValidationError(ToolExecutionError):
    pass


class InvestigationAgent:
    """Simplified fixed-pipeline investigation agent.

    This version is intentionally more conservative and more model-friendly than the
    original planner-loop design. It performs a bounded series of retrieval steps and
    then asks the model for a single grounded final answer.
    """

    def __init__(self, db: Database, search_client: SearchClient, llm: LLMClient | None = None):
        self.db = db
        self.repo = DocumentRepository(db)
        self.search = search_client
        self.llm = llm or LLMClient()
        self.embedding = search_client.embedding_service
        self.reranker = Reranker(self.embedding)
        self.ingestion = IngestionService(
            db_path=str(db.path),
            es_url=search_client.url,
            es_index=search_client.index,
            search_client=search_client,
        )
        self.tools = InvestigationToolRegistry(db, search_client, self.ingestion)

    def investigate(
        self,
        question: str,
        uploaded_file: str | None = None,
        top_k: int = 5,
        uploaded_filename: str | None = None,
    ) -> InvestigationResult:
        trace: list[str] = ["Started fixed-pipeline investigation."]
        upload_warnings: list[str] = []
        evidence: list[EvidenceChunk] = []

        try:
            if uploaded_file:
                upload_res = self.tools.process_new_upload(
                    uploaded_file,
                    original_name=uploaded_filename or Path(uploaded_file).name,
                    index_document=True,
                )
                if not upload_res.ok:
                    raise ToolExecutionError(upload_res.error or "Upload processing failed")

                payload = upload_res.payload or {}
                doc_id = payload.get("document_id")
                trace.append(f"Uploaded document id={doc_id} duplicate={payload.get('duplicate')}")
                if doc_id:
                    doc = self.repo.get(doc_id)
                    pages = self.repo.get_pages(doc_id)
                    review_count = sum(page.needs_review for page in pages)
                    if review_count:
                        upload_warnings.append(f"Excluded {review_count} low-confidence OCR page(s) from evidence.")
                    for page in pages:
                        if page.needs_review or not page.text.strip():
                            continue
                        evidence.append(
                            EvidenceChunk(
                                doc_id=doc_id,
                                filename=doc.filename if doc else Path(uploaded_file).name,
                                source=doc.source.value if doc else "upload",
                                page=page.page_number,
                                text=page.text[:1200],
                                score=0.85,
                                metadata={"method": page.method.value, "needs_review": page.needs_review},
                            )
                        )

            queries = self._generate_queries(question, trace)
            if question.strip() and question.strip() not in queries:
                queries.append(question.strip())

            seen_keys: set[tuple[Any, ...]] = {(e.doc_id, e.page, e.text) for e in evidence}
            for query in queries[:3]:
                res = self.tools.search_documents(query, top_k)
                if not res.ok:
                    trace.append(f"Search failed for {query!r}: {res.error}")
                    continue
                hits = [SearchHit(**hit) for hit in (res.payload or {}).get("hits", [])]
                ranked = self.reranker.rerank(question, hits)
                for hit in ranked:
                    chunk_text = (hit.text or "")[:1200]
                    key = (hit.doc_id, hit.page, chunk_text)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    evidence.append(
                        EvidenceChunk(
                            doc_id=hit.doc_id,
                            filename=hit.filename,
                            source=hit.source,
                            page=hit.page,
                            text=chunk_text,
                            score=float(hit.score),
                            metadata=hit.metadata or {},
                        )
                    )

            evidence = evidence[-20:]
            trace.append(f"Collected {len(evidence)} evidence chunk(s) from {len(queries)} query plan(s).")

            if not evidence:
                return InvestigationResult(
                    question=question,
                    status=EvidenceStatus.INSUFFICIENT,
                    answer=None,
                    confidence=0.0,
                    evidence=[],
                    missing_information=["No relevant documents were found for the question."],
                    warnings=["No relevant retrieval results were found.", *upload_warnings],
                    sources=[],
                    trace=trace,
                )

            return self._finalize(question, evidence, upload_warnings, trace)

        except ToolExecutionError as exc:
            trace.append(f"Tool error: {exc}")
            return self._failed(question, evidence, str(exc), trace)
        except Exception as exc:  # pragma: no cover
            trace.append(f"Unexpected error: {exc}")
            return self._failed(question, evidence, str(exc), trace)

    def _generate_queries(self, question: str, trace: list[str]) -> list[str]:
        prompt = (
            "Write 1 to 3 short search queries that would find documents answering the question. "
            "Return JSON only with a key named 'queries'. No prose.\n"
            f"Question: {question}"
        )
        try:
            raw = self.llm.generate(prompt)
            queries = raw.get("queries") if isinstance(raw, dict) else None
            if isinstance(queries, list):
                cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
                if cleaned:
                    trace.append(f"LLM queries: {cleaned}")
                    return cleaned[:3]
        except Exception as exc:  # pragma: no cover
            trace.append(f"Query expansion failed ({exc}); falling back to raw question.")
        return [question.strip()] if question.strip() else []

    def _finalize(
        self,
        question: str,
        evidence: list[EvidenceChunk],
        upload_warnings: list[str],
        trace: list[str],
    ) -> InvestigationResult:
        sources = list(dict.fromkeys(item.filename for item in evidence))
        has_heuristic_conflict = self._detect_contradictions(question, evidence)

        prompt = (
            "Answer the question using ONLY the evidence below. "
            "Follow this procedure: (1) identify the decisive fact needed to answer the question, "
            "(2) prefer primary evidence such as logs with timestamps, contracts, or signed decisions over "
            "secondary summaries or review notes, (3) if a secondary report differs from primary evidence, "
            "note the difference in reasoning but do not mark it as a contradiction unless the primary evidence itself conflicts. "
            "Return JSON only with keys: answer, confidence, has_contradiction, contradiction_reasoning, "
            "needs_more_evidence, missing_information, reasoning. "
            "The answer value must be a detailed natural-language string of 2 to 4 sentences, never true, false, "
            "yes, or no by itself. Start with the direct conclusion, then explain the relevant policy or contract "
            "rule and the supporting facts from the evidence. Mention any unresolved point, such as whether an "
            "approved credit was actually issued. Keep the answer concise but informative. "
            "If the evidence does not fully support an answer, set answer to null and needs_more_evidence to true.\n\n"
            f"Question: {question}\n\nEvidence:\n"
            + "\n\n---\n\n".join(
                f"File: {e.filename}\nPage: {e.page}\nText: {e.text[:800]}" for e in evidence
            )
        )
        payload = self.llm.generate(prompt)

        llm_has_conflict = self._norm_bool(payload.get("has_contradiction", False))
        primary_conflict_resolved = (
            not has_heuristic_conflict
            and llm_has_conflict
            and self._primary_source_resolves_conflict(question, evidence)
        )
        if primary_conflict_resolved:
            llm_has_conflict = False

        if has_heuristic_conflict or llm_has_conflict:
            reason = payload.get("contradiction_reasoning") or "Retrieved sources conflict."
            trace.append("Contradiction detected.")
            return InvestigationResult(
                question=question,
                status=EvidenceStatus.CONTRADICTORY,
                answer=None,
                confidence=0.35,
                evidence=evidence,
                missing_information=[],
                warnings=[str(reason), *upload_warnings],
                sources=sources,
                trace=trace,
            )

        answer = payload.get("answer")
        confidence = self._norm_conf(payload.get("confidence", 0.0))
        needs_more = self._norm_bool(payload.get("needs_more_evidence", False))
        missing = self._norm_list(payload.get("missing_information"))
        reasoning = payload.get("reasoning") or "Evidence judged insufficient."

        if not answer or needs_more or confidence < 0.55:
            trace.append("LLM judged evidence insufficient.")
            return InvestigationResult(
                question=question,
                status=EvidenceStatus.INSUFFICIENT,
                answer=None,
                confidence=confidence,
                evidence=evidence,
                missing_information=missing or ["Insufficient evidence in retrieved documents."],
                warnings=[reasoning, *upload_warnings],
                sources=sources,
                trace=trace,
            )

        answer_text = self._format_answer_text(question, answer, reasoning)

        trace.append("Answer produced.")
        return InvestigationResult(
            question=question,
            status=EvidenceStatus.READY,
            answer=answer_text,
            confidence=confidence,
            evidence=evidence,
            missing_information=[],
            warnings=upload_warnings,
            sources=sources,
            trace=trace,
        )

    @staticmethod
    def _failed(question: str, evidence: list[EvidenceChunk], reason: str, trace: list[str]) -> InvestigationResult:
        return InvestigationResult(
            question=question,
            status=EvidenceStatus.FAILED,
            answer=None,
            confidence=0.0,
            evidence=evidence,
            missing_information=[],
            warnings=[f"Investigation failed: {reason}"],
            sources=[item.filename for item in evidence],
            trace=trace,
            failure_reason=reason,
        )

    @staticmethod
    def _norm_conf(value: Any) -> float:
        if isinstance(value, str):
            lowered = value.strip().lower()
            mapping = {"high": 0.85, "medium": 0.65, "low": 0.25}
            if lowered in mapping:
                return mapping[lowered]
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _norm_bool(value: Any) -> bool:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0", ""}:
                return False
        return bool(value)

    @staticmethod
    def _norm_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item.strip()]
        return []

    def _detect_contradictions(self, question: str, evidence: list[EvidenceChunk]) -> bool:
        if len(evidence) < 2:
            return False

        normalized_question = self._clean_text(question)
        asks_eligibility = bool(re.search(
            r"\b(qualify|qualifies|qualified|eligible|eligibility|entitled|credit review|"
            r"should we (file|request|claim)|service credit)\b",
            normalized_question,
        ))
        asks_duration = bool(re.search(
            r"\b(duration|lasted|how long|how many hours|recovery time)\b",
            normalized_question,
        ))

        if asks_eligibility:
            return self._eligibility_contradiction(evidence)

        if asks_duration:
            return self._duration_contradiction(evidence)

        # Default safeguard for explicit yes/no contradictions not tied to a
        # threshold or duration question.
        texts = [self._clean_text(chunk.text) for chunk in evidence]
        for idx, text in enumerate(texts):
            if re.search(r"(?:does\s+not\s+qualify|not\s+qualify|not\s+eligible|ineligible|doesn\'t\s+qualify)", text):
                for other_idx, other_text in enumerate(texts):
                    if other_idx == idx:
                        continue
                    if re.search(r"(?:qualif(?:y|ies|ied)|eligible|credit\s+review)", other_text):
                        return True
        return False

    def _eligibility_contradiction(self, evidence: list[EvidenceChunk]) -> bool:
        texts = [self._clean_text(chunk.text) for chunk in evidence]

        support_terms = [
            "qualifies for a service credit review",
            "qualify for a service credit review",
            "eligible for a service credit",
            "service credit eligibility",
            "credit review",
            "exceeds four hours",
            "more than four hours",
            "threshold was crossed",
            "approve a credit review",
            "crossed the sla",
        ]
        deny_terms = [
            "does not qualify",
            "not qualify",
            "not eligible",
            "ineligible",
            "doesn't qualify",
        ]
        saw_support = any(any(term in text for term in support_terms) for text in texts)
        saw_deny = any(any(term in text for term in deny_terms) for text in texts)
        if saw_support and not saw_deny:
            return False

        threshold_hours = None
        for text in texts:
            if "credit" in text.lower() or "service credit" in text.lower():
                match = re.search(r"(?:within|exceed[s]?|more than|over|after)\s+(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", text, flags=re.I)
                if match:
                    threshold_hours = float(match.group(1))
                    break

        if threshold_hours is None:
            return False

        above: set[str] = set()
        below: set[str] = set()
        for chunk in evidence:
            text = self._clean_text(chunk.text)
            if re.search(r"(?:more than|over|exceeded|crossed|above)\s+(?:the\s+)?(?:four|4)\s*(?:hours?|hrs?)|exceed[s]?\s+(?:the\s+)?(?:four|4)", text):
                above.add(chunk.doc_id)
            if re.search(r"(?:under|below|not exceed|did not exceed|less than|within)\s+(?:the\s+)?(?:four|4)\s*(?:hours?|hrs?)", text):
                below.add(chunk.doc_id)
            matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)", text, flags=re.I)
            for value in matches:
                hrs = float(value)
                if hrs > threshold_hours:
                    above.add(chunk.doc_id)
                elif hrs < threshold_hours:
                    below.add(chunk.doc_id)

        if above and below and above != below:
            primary_logs = [chunk for chunk in evidence if "log" in (chunk.filename or "").lower() and re.search(r"\d{2}:\d{2}", chunk.text)]
            if primary_logs:
                return False
            return True

        return False

    def _duration_contradiction(self, evidence: list[EvidenceChunk]) -> bool:
        claims: dict[str, set[tuple[str, float]]] = {}
        for chunk in evidence:
            ids = {m.upper() for m in re.findall(r"\bINC-\d+\b", chunk.text, flags=re.I)}
            if not ids:
                continue
            matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)", chunk.text, flags=re.I)
            for value in matches:
                hrs = float(value)
                for inc_id in ids:
                    claims.setdefault(inc_id, set()).add((chunk.doc_id, hrs))

        for rows in claims.values():
            values = {value for _, value in rows}
            doc_ids = {doc_id for doc_id, _ in rows}
            if len(doc_ids) > 1 and len(values) > 1 and max(values) - min(values) > 0.5:
                return True
        return False

    def _primary_source_resolves_conflict(self, question: str, evidence: list[EvidenceChunk]) -> bool:
        normalized_question = self._clean_text(question)
        if not any(keyword in normalized_question for keyword in [
            "qualify",
            "eligible",
            "credit",
            "threshold",
            "service credit",
        ]):
            return False

        primary_log = any(
            "log" in (chunk.filename or "").lower() and re.search(r"\d{2}:\d{2}", chunk.text)
            for chunk in evidence
        )
        if not primary_log:
            return False

        support_terms = ["more than four hours", "exceeds four hours", "threshold was crossed", "credit review", "approve a credit review"]
        denied_terms = ["does not qualify", "not qualify", "not eligible", "ineligible"]
        text_blob = " ".join(self._clean_text(chunk.text) for chunk in evidence)
        return any(term in text_blob for term in support_terms) and not any(term in text_blob for term in denied_terms)

    @staticmethod
    def _format_answer_text(question: str, answer: Any, reasoning: Any = None) -> str:
        if isinstance(answer, bool):
            fallback = str(reasoning).strip() if reasoning else ""
            if fallback and len(fallback.split()) >= 12:
                return fallback
            return (
                "The evidence supports the conclusion, but the model did not provide a detailed explanation."
                if answer
                else "The evidence does not support the requested conclusion."
            )

        if answer is None:
            return "No answer was supported by the available evidence."

        if isinstance(answer, (int, float)):
            return str(answer)

        normalized = str(answer).strip()
        if normalized.lower() in {"true", "false"}:
            fallback = str(reasoning).strip() if reasoning else ""
            if fallback and len(fallback.split()) >= 12:
                return fallback
            return (
                "The evidence supports the conclusion, but the model did not provide a detailed explanation."
                if normalized.lower() == "true"
                else "The evidence does not support the requested conclusion."
            )
        return normalized

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip().lower()
