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
- `streamlit_app.py` — optional frontend UI

## Prerequisites

Before running this project, make sure the following are installed:

- Python 3.11 or 3.12
- pip
- Docker + Docker Compose
- Git
- Optional: Tesseract OCR for scanned documents
- Optional: Ollama for a local LLM

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
- `OPENAI_MODEL`: set to your model name, for example `qwen2.5:7b` for local use or your hosted model name for remote use
- `OPENAI_BASE_URL`: use `http://localhost:11434/v1` for local Ollama, or your provider's API base URL for remote use
- `ES_URL`: usually `http://localhost:9200`
- `SQLITE_PATH`: usually `data/app.db`

Recommended local setup (Ollama):

```env
OPENAI_API_KEY=
OPENAI_MODEL=qwen2.5:7b
OPENAI_BASE_URL=http://localhost:11434/v1
ES_URL=http://localhost:9200
ES_INDEX=document_chunks
ES_ENABLED=true
SQLITE_PATH=data/app.db
```

To use a local model with Ollama:

1. Install Ollama.
2. Pull the model:

```bash
ollama pull qwen2.5:7b
```

3. Start the Ollama server:

```bash
ollama serve
```

4. Set your `.env` values to:

```env
OPENAI_API_KEY=
OPENAI_MODEL=qwen2.5:7b
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

## 6. Start a local LLM (Ollama recommended)

If you want to run with a local model, install Ollama and pull a model:

```bash
ollama pull qwen2.5:7b
ollama serve
```

The app uses the OpenAI-compatible chat API, so this works without changing application code.

If you prefer a remote OpenAI-compatible API, you can skip Ollama and set the `.env` values to that endpoint instead.

## 7. Start the backend API

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

## 8. Seed the sample corpus

On startup, the application automatically seeds a demo corpus from `data/sample_docs` into the database and Elasticsearch if the index is empty or stale.

This means that a fresh clone can run the demo without manual seeding.

## 9. Test the API

### Example 1: service credit review question

```bash
curl -X POST http://localhost:8001/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Did the outage qualify for a service credit review?",
    "top_k": 5
  }'
```

### Example 2: unsupported question

```bash
curl -X POST http://localhost:8001/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the company office address in Tokyo?",
    "top_k": 5
  }'
```

Expected result: the app should abstain with `status: insufficient` if the evidence does not support the answer.

### Example 3: contradiction question

```bash
curl -X POST http://localhost:8001/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What duration did the reports give for incident INC-2041?",
    "top_k": 10
  }'
```

Expected result: `status: contradictory` if the different reports disagree on the decisive fact.

## 10. Optional: run the Streamlit frontend

The project includes a lightweight Streamlit app for simple manual testing.

Start it from the project root:

```bash
streamlit run streamlit_app.py
```

Then open:

- http://localhost:8501

## 11. Optional: upload a document

Upload a file to the API:

```bash
curl -X POST http://localhost:8001/upload \
  -F "file=@/path/to/file.pdf"
```

Or send a question with an uploaded document:

```bash
curl -X POST http://localhost:8001/investigate-upload \
  -F "question=What does the uploaded note say about the outage?" \
  -F "file=@data/sample_docs/uploaded_note.txt"
```

## 12. OCR support

This project supports OCR for scanned pages or image-based files.

Install Tesseract:

- Ubuntu/Debian:
  ```bash
  sudo apt-get update
  sudo apt-get install -y tesseract-ocr
  ```

- macOS:
  ```bash
  brew install tesseract
  ```

- Windows: install Tesseract and ensure it is on the PATH.

## 13. Resetting local state

If you want to clear stale local app data while testing, delete the SQLite DB and Elasticsearch index:

```bash
rm -f data/app.db
curl -X DELETE http://localhost:9200/document_chunks
```

Then restart the backend and the app will re-seed the sample corpus.

## 14. Troubleshooting

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

## 15. Testing the project

Run unit tests from the project root:

```bash
pytest tests/unit -q
```

For a focused check of the investigation workflow:

```bash
pytest tests/unit/test_investigation_workflow.py -q
```

## 16. Summary

For a new team member, the shortest path to running this project is:

1. Clone repo
2. Create `.venv`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env`
5. Run `docker compose up -d`
6. Start Ollama or configure a remote OpenAI-compatible endpoint
7. Start API with `uvicorn app.api.main:app --host 0.0.0.0 --port 8001`
8. Hit `/health` and `/investigate`

This is enough for a team to run the application on a fresh machine.
