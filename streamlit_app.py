import streamlit as st
import requests

API_URL = "http://localhost:8001/investigate"
UPLOAD_API_URL = "http://localhost:8001/upload"
REQUEST_TIMEOUT_SECONDS = 200

st.set_page_config(page_title="Investigation Agent", page_icon="", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    .eyebrow { color: #64748b; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
    .page-title { color: #172033; font-size: 2.35rem; font-weight: 750; margin: 0.15rem 0 0.35rem; }
    .page-subtitle { color: #64748b; font-size: 1rem; margin-bottom: 1.8rem; }
    .panel { background: #ffffff; border: 1px solid #dbe3ec; border-radius: 10px; padding: 1.25rem 1.35rem; box-shadow: 0 8px 24px rgba(27, 44, 72, 0.06); }
    .panel-label { color: #64748b; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; }
    .answer-copy { color: #172033; font-size: 1.12rem; line-height: 1.65; }
    .evidence-title { color: #172033; font-size: 1.02rem; font-weight: 700; margin-bottom: 0.2rem; }
    .evidence-meta { color: #64748b; font-size: 0.82rem; }
    .status-ready { color: #087443; }
    .status-contradictory { color: #b45309; }
    .status-insufficient, .status-failed { color: #b42318; }
    div[data-testid="stMetric"] { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Evidence workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Investigation Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Ask a question and review the reasoning, source documents, and confidence behind the result.</div>',
    unsafe_allow_html=True,
)

if "uploaded_document" not in st.session_state:
    st.session_state.uploaded_document = None

upload_clicked = False
run_question = False
uploaded_file = None
question = ""
top_k = 5
mode = "Question only"

with st.container(border=True):
    st.markdown("#### Investigation request")
    mode = st.radio("Workflow", ["Question only", "Upload a document"], horizontal=True)
    top_k = st.slider("Evidence sources", 1, 10, 5)

    if mode == "Question only":
        question = st.text_area(
            "Question",
            "Did the outage qualify for a service credit review?",
            height=100,
        )
        run_question = st.button("Run investigation", type="primary", use_container_width=True)
    else:
        st.markdown("#### Step 1: Upload and index document")
        uploaded_file = st.file_uploader(
            "Document",
            type=["txt", "pdf", "png", "jpg", "jpeg"],
            help="Choose a document, then upload it before asking questions.",
        )
        upload_clicked = st.button("Upload document", type="primary", use_container_width=True)

        uploaded_document = st.session_state.uploaded_document
        if uploaded_document:
            st.success(
                f"Uploaded: {uploaded_document.get('filename', 'document')}"
                f" · Status: {uploaded_document.get('status', 'processed')}"
            )
            st.markdown("#### Step 2: Ask a question")
            question = st.text_area(
                "Ask a question about the uploaded document",
                "What does this document say about the incident?",
                height=100,
                key="uploaded_question",
            )
            run_question = st.button("Ask question", type="primary", use_container_width=True)
            if st.button("Remove uploaded document"):
                st.session_state.uploaded_document = None
                st.rerun()

if upload_clicked:
    if uploaded_file is None:
        st.warning("Select a document before continuing.")
        st.stop()

    with st.spinner("Uploading and indexing document..."):
        try:
            resp = requests.post(
                UPLOAD_API_URL,
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            st.session_state.uploaded_document = resp.json()
        except requests.RequestException as exc:
            st.error(f"The investigation service could not be reached: {exc}")
            st.stop()
    st.rerun()

if run_question:
    if mode == "Upload a document" and not st.session_state.uploaded_document:
        st.warning("Upload and index a document before asking a question.")
        st.stop()

    payload = {"question": question, "top_k": top_k}
    with st.spinner("Reviewing evidence..."):
        try:
            resp = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            st.error(f"The investigation service could not be reached: {exc}")
            st.stop()

    status = str(data.get("status", "unknown"))
    status_class = f"status-{status}" if status in {"ready", "contradictory", "insufficient", "failed"} else ""
    confidence = float(data.get("confidence") or 0)
    confidence_display = f"{confidence:.0%}"

    st.markdown("### Investigation result")
    summary = st.columns([1.25, 1, 2.75])
    with summary[0]:
        st.markdown('<div class="panel-label">Status</div>', unsafe_allow_html=True)
        st.markdown(f'<h3 class="{status_class}">{status.title()}</h3>', unsafe_allow_html=True)
    with summary[1]:
        st.metric("Confidence", confidence_display)
    with summary[2]:
        sources = data.get("sources") or []
        st.metric("Sources reviewed", len(sources))

    st.markdown('<div class="panel-label">Answer</div>', unsafe_allow_html=True)
    answer = data.get("answer")
    if answer:
        st.markdown(f'<div class="panel answer-copy">{answer}</div>', unsafe_allow_html=True)
    else:
        st.info("No supported answer was returned for this question.")

    evidence = data.get("evidence") or []
    st.markdown(f"### Evidence ({len(evidence)})")
    if evidence:
        for index, item in enumerate(evidence, start=1):
            page = item.get("page")
            page_label = f"Page {page}" if page is not None else "Page not specified"
            score = float(item.get("score") or 0)
            with st.container(border=True):
                heading = st.columns([4, 1])
                with heading[0]:
                    st.markdown(f'<div class="evidence-title">{index}. {item.get("filename", "Unknown source")}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="evidence-meta">{page_label} · {item.get("source", "Unknown source type")}</div>',
                        unsafe_allow_html=True,
                    )
                with heading[1]:
                    st.metric("Match score", f"{score:.2f}")
                st.write(item.get("snippet", "No excerpt available."))
    else:
        st.info("No evidence was returned.")

    warnings = data.get("warnings") or []
    if warnings:
        with st.expander("Warnings", expanded=True):
            for warning in warnings:
                st.warning(warning)

    missing = data.get("missing_information") or []
    if missing:
        with st.expander("Missing information", expanded=True):
            for item in missing:
                st.write(f"- {item}")

    trace = data.get("trace") or []
    if trace:
        with st.expander("Investigation trace"):
            for item in trace:
                st.write(f"- {item}")