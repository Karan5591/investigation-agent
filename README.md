# Investigation Agent

A FastAPI-based evidence investigation system that answers questions using a local OpenAI-compatible LLM, Elasticsearch, SQLite metadata, and optional OCR/upload processing.

This project is designed to be easy for a team to clone and run locally on a standard developer machine.

## What this project does

- Accepts a natural-language question
- Retrieves relevant evidence from a local document corpus
- Optionally indexes uploaded documents
- Runs a bounded investigation flow
- Returns evidence-backed answers with trace information
- Detects insufficient evidence and contradiction cases
- Supports local LLMs through the OpenAI-compatible API
- Provides a Streamlit frontend for easy manual testing

## Project structure

- `app/` — application code
  - `app/api/` — FastAPI API and schemas
  - `app/agent/` — investigation orchestration logic
  - `app/db/` — SQLite metadata and repository logic
  - `app/evidence/` — evidence models and result types
  - `app/ingestion/` — upload processing and indexing
  - `app/llm/` — OpenAI-compatible LLM client
  - `app/ocr/` — OCR extraction
  - `app/retrieval/` — embeddings, rerank, search
  - `app/tools/` — allowed investigation tools
- `data/` — sample documents, uploaded files, and app DB
- `docs/` — design/docs
- `scripts/` — helper scripts
- `tests/` — unit/integration tests
- `docker-compose.yml` — Elasticsearch service definition
- `.env.example` — environment file template
- `requirements.txt` — Python dependencies
- `streamlit_app.py` — Streamlit frontend for manual testing

## Prerequisites

Before running this project, make sure the following are installed:

- Python 3.11 or 3.12
- pip
- Docker + Docker Compose
- Git
- RapidOCR is installed automatically with the Python dependencies for scanned documents or images

RapidOCR runs locally through `rapidocr_onnxruntime`; no separate Tesseract executable is required. Text-based PDFs and text files do not require OCR.
- Ollama for the recommended local LLM setup, or another OpenAI-compatible provider

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd investigation-agent
```

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- FastAPI and Uvicorn
- OpenAI client
- Elasticsearch client
- sentence-transformers and torch
- PDF + OCR dependencies
- Streamlit
- pytest

## 4. Configure environment variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then review the values in `.env` and update them as needed:

- `OPENAI_API_KEY`: leave blank for a local Ollama model, or set your provider API key for a remote OpenAI-compatible endpoint
- `OPENAI_MODEL`: set to your model name, for example `llama3.2:3b` for local use or your hosted model name for remote use
- `OPENAI_BASE_URL`: use `http://localhost:11434/v1` for local Ollama, or your provider's API base URL for remote use
- `ES_URL`: usually `http://localhost:9200`
- `SQLITE_PATH`: usually `data/app.db`

Recommended local setup using Ollama:

```env
OPENAI_API_KEY=
OPENAI_MODEL=llama3.2:3b
OPENAI_BASE_URL=http://localhost:11434/v1
ES_URL=http://localhost:9200
ES_INDEX=document_chunks
ES_ENABLED=true
SQLITE_PATH=data/app.db
```

To use a local model with Ollama:

1. Install Ollama on the machine.

On Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On Windows, download and install Ollama from:

- https://ollama.com/download/windows

Restart the terminal after installation, then verify that Ollama is available:

```bash
ollama --version
```

2. Pull the model:

```bash
ollama pull llama3.2:3b
```

3. Start the Ollama server:

```bash
ollama serve
```

4. Set your `.env` values to:

```env
OPENAI_API_KEY=
OPENAI_MODEL=llama3.2:3b
OPENAI_BASE_URL=http://localhost:11434/v1
ES_URL=http://localhost:9200
ES_INDEX=document_chunks
ES_ENABLED=true
SQLITE_PATH=data/app.db
```

This allows the application to talk to the local model through the OpenAI-compatible API without requiring an OpenAI account.

If you are using another OpenAI-compatible endpoint (for example Airawat or another remote provider), set:

```env
OPENAI_API_KEY=<your-key>
OPENAI_MODEL=<your-model>
OPENAI_BASE_URL=<your-provider-base-url>
```

Do not commit `.env` to Git. It should remain local to each machine.

## 5. Start Elasticsearch

From the project root:

```bash
docker compose up -d
```

This starts the Elasticsearch container defined in `docker-compose.yml`.

Check that Elasticsearch is running:

```bash
curl http://localhost:9200
```

You should get a JSON response from Elasticsearch.

## 6. Start the local model and backend API

Start Ollama in one terminal:

```bash
ollama serve
```

Start the backend in another terminal. Use `--host 0.0.0.0` so the Streamlit process can reach the API:

From the project root:

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8001
```

The app should start and listen on:

- http://localhost:8001

Health check:

```bash
curl http://localhost:8001/health
```

Expected response:

```json
{"status":"ok"}
```

## 7. Start Streamlit for manual testing

Streamlit is the primary way to test this project manually. Start it from the project root in another terminal:

```bash
streamlit run streamlit_app.py
```

Open the frontend in a browser:

- http://localhost:8501

The Streamlit page provides three workflows:

### Question only

Use the default **Question only** workflow to ask a question against the seeded sample corpus. Set the number of evidence sources, then click **Run investigation**.

The result shows:

- investigation status
- confidence score
- number of sources reviewed
- detailed answer
- evidence cards with filename, page, source type, excerpt, and match score
- warnings, missing information, and investigation trace when available

### Upload a document: two-step workflow

Choose **Upload a document**. The frontend then guides you through two explicit steps:

1. **Step 1: Upload and index document** — select a supported document and click **Upload document**.
2. **Step 2: Ask a question** — after the upload status confirms success, enter a question and click **Ask question**.

This sends the document to the ingestion pipeline. The document is processed and indexed so it can be used in later investigations.

Supported file types in the frontend include text files, PDFs, PNG images, and JPG/JPEG images.

The Streamlit upload limit is configured to 20 MB in `.streamlit/config.toml`. Files larger than 20 MB are rejected by Streamlit before they are sent to the backend.

The Streamlit frontend waits up to 200 seconds for an investigation or upload request to finish. This is a response timeout, not a file-size limit.

The document remains available in the current Streamlit session while you ask questions. Click **Remove uploaded document** to clear it and start with another file.

## 8. Sample questions for Streamlit

Try these questions in the **Question only** workflow:

- `Did the outage qualify for a service credit review?`
- `What duration did the reports give for incident INC-2041?`
- `What is the company office address in Tokyo?`

The first should produce an evidence-backed service-credit result, the second exercises contradictory evidence, and the third should demonstrate insufficient evidence handling.

## 9. Optional API and Swagger testing

The backend also exposes Swagger UI at:

- http://localhost:8001/docs

Use Swagger when you need to test the API endpoints directly. The available endpoints are:

- `GET /health`
- `POST /investigate`
- `POST /upload`
- `POST /investigate-upload`

### Optional cURL: health check

```bash
curl http://localhost:8001/health
```

### Optional cURL: investigation

```bash
curl -X POST http://localhost:8001/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Did the outage qualify for a service credit review?",
    "top_k": 5
  }'
```

### Optional cURL: upload

```bash
curl -X POST http://localhost:8001/upload \
  -F "file=@/path/to/document.pdf"
```

### Optional cURL: upload and investigate

```bash
curl -X POST http://localhost:8001/investigate-upload \
  -F "question=What does the uploaded note say about the outage?" \
  -F "file=@data/sample_docs/uploaded_note.txt"
```

The Streamlit 20 MB limit applies to the frontend. Direct API uploads through cURL or Swagger may still require separate server or reverse-proxy limits.

## 10. Seed the sample corpus

On startup, the application automatically seeds a demo corpus from `data/sample_docs` into the database and Elasticsearch if the index is empty or stale.

This means that a fresh clone can run the demo without manual seeding.

## 11. OCR support

This project uses RapidOCR for scanned pages or image-based files. RapidOCR is installed by `pip install -r requirements.txt`; no separate Tesseract installation is required.

For PDFs, PyMuPDF first extracts native text page by page. RapidOCR is used only for pages where no usable text is found. This avoids running OCR unnecessarily on text-based pages.

The first OCR request may take longer while the local RapidOCR model initializes.

## 12. Resetting local state

If you want to clear stale local app data while testing, delete the SQLite DB and Elasticsearch index:

```bash
rm -f data/app.db
curl -X DELETE http://localhost:9200/document_chunks
```

Then restart the backend and the app will re-seed the sample corpus.

## 13. Troubleshooting

### Docker Elasticsearch is not starting

Check the container status:

```bash
docker ps -a
```

Check logs:

```bash
docker logs investigation-es
```

### LLM connection error

Check that Ollama is running:

```bash
ollama serve
```

Or verify the provider URL in `.env` matches your actual endpoint.

### Port already in use

If port 8001 is already occupied, change the port in the uvicorn command:

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8002
```

### Missing dependencies

Reinstall dependencies:

```bash
pip install -r requirements.txt
```

## 14. Testing the project

Run unit tests from the project root:

```bash
pytest tests/unit -q
```

For a focused check of the investigation workflow:

```bash
pytest tests/unit/test_investigation_workflow.py -q
```

## 15. Summary

For a new team member, the shortest path to running this project is:

1. Clone repo
2. Create `.venv`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env`
5. Run `docker compose up -d`
6. Start Ollama or configure a remote OpenAI-compatible endpoint
7. Start the API with `uvicorn app.api.main:app --host 0.0.0.0 --port 8001`
8. Start Streamlit with `streamlit run streamlit_app.py`
9. Open `http://localhost:8501` and test the three workflows

This is enough for a team to run the application on a fresh machine.
