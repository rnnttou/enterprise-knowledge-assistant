# Enterprise Knowledge Assistant (RAG)

> Ask questions about your company's PDF documents and get answers **with cited sources**.

![status](https://img.shields.io/badge/status-complete-brightgreen)
![python](https://img.shields.io/badge/python-3.11-blue)

Built by **[Rania Bentatou](https://github.com/rnnttou)**.

---

## What it does

1. **Upload PDFs** (policies, manuals, reports…)
2. **Split** them into small chunks of text
3. **Embed** each chunk with Google Gemini (free tier), turning text into vectors
4. **Store** the vectors in ChromaDB
5. **Answer questions**: find the most relevant chunks, then let Gemini write an answer that cites which document and page each fact came from

```
PDF ──► chunks ──► embeddings ──► ChromaDB
                                      │
question ──► hybrid search (BM25 + vectors) ──► top chunks ──► Gemini ──► answer + sources
```

## Tech stack

| Layer            | Tool                          |
|------------------|-------------------------------|
| LLM + embeddings | Google Gemini API (free tier) |
| Vector store     | ChromaDB                      |
| Keyword search   | BM25 (`rank-bm25`)            |
| API              | FastAPI                       |
| UI               | Streamlit                     |
| Packaging        | Docker / docker compose       |
| Tests            | pytest                        |

## Roadmap

- [x] **Step 1**: project skeleton and configuration
- [x] **Step 2**: load PDFs and split them into chunks
- [x] **Step 3**: embeddings with Gemini, stored in ChromaDB
- [x] **Step 4**: question answering with cited sources
- [x] **Step 5**: FastAPI backend
- [x] **Step 6**: Streamlit UI
- [x] **Step 7**: hybrid retrieval (BM25 + vector search)
- [x] **Step 8**: conversation memory
- [x] **Step 9**: Docker packaging
- [x] **Step 10**: evaluation on 30–50 test questions (retrieval hit rate + latency)

## Getting started

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/rnnttou/enterprise-knowledge-assistant.git
cd enterprise-knowledge-assistant
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your Gemini API key

Get a free key at <https://aistudio.google.com/apikey>, then:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Open `.env` and paste your key after `GEMINI_API_KEY=`.

### 3. Check the setup

```bash
python -m app.config
```

### 4. Index your PDFs

Put PDFs in `data/`, then embed them into ChromaDB (stored in `chroma_db/`):

```bash
python -m app.vector_store
```

Re-running is cheap: unchanged files are skipped, changed files have their chunks replaced, and deleted files are removed from the index. The free Gemini tier is rate-limited, so the first run pauses and retries when it hits the limit.

PDFs in a sub-folder are only visible to that group: `data/hr/salaries.pdf` can be read by `hr`, while files directly in `data/` are public.

Try a search:

```bash
python -m app.vector_store --query "How many vacation days do employees get?"
python -m app.vector_store --query "salary bands" --groups hr
```

### 5. Ask a question

```bash
python -m app.qa "How many vacation days do employees get?"
python -m app.qa "salary bands" --groups hr
```

The chat model only sees the retrieved excerpts (never the full corpus), answers from them alone, and cites which excerpt each fact came from. Answers print alongside the source document and page for each excerpt used.

### 6. Run the API

```bash
uvicorn app.api:app --reload
```

Then open <http://127.0.0.1:8000/docs> for interactive API docs, or call it directly:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do employees get?", "groups": ["hr"], "k": 5}'
curl -X POST http://127.0.0.1:8000/index
```

`/ask` runs the same retrieval + citation pipeline as `python -m app.qa` and returns
`{"answer": ..., "citations": [{"marker", "source", "page", "snippet"}], "refused": ..., "conversation_id": ...}`.
`refused` is `true` when the documents don't contain the answer. `/index` re-runs the indexer, so new PDFs dropped into `data/` can be picked up without a server restart.

Pass the `conversation_id` from a response back into your next request to ask a
follow-up ("What about for part-time staff?") that's understood in context of
the earlier question and answer (Step 8). Omit it to start a new conversation.
History is kept in memory per `conversation_id`, capped at the last 6 turns,
and is lost on restart.

### 7. Run the Streamlit UI

With the API running (step 6) in one terminal, start the UI in another:

```bash
streamlit run app/streamlit_app.py
```

Then open <http://localhost:8501>. It's a thin client over the API: pick your
groups and chunk count in the sidebar, ask questions in the chat box, and
expand "Sources" under an answer to see the cited pages and snippets. Point it
at a different backend with `API_URL=http://host:port streamlit run app/streamlit_app.py`.

### 8. Run the tests

```bash
pytest
```

The tests use a fake embedder, so they run offline and don't use your API quota.

### 9. Run everything with Docker

With `.env` set up (step 2) and PDFs in `data/` (step 4):

```bash
docker compose up --build
```

This builds one image and runs it as two containers: `api` (FastAPI on
<http://localhost:8088>, mapped from the container's port 8000 - see the
comment in `docker-compose.yml` if you hit a port-binding error on
Windows/WSL) and `ui` (Streamlit on <http://localhost:8501>, pointed at `api`
over the compose network). `data/` and `chroma_db/` are mounted as volumes, so
PDFs and the index persist across restarts. Index your documents with
`curl -X POST http://localhost:8088/index` or the UI's "Re-index documents"
button.

### 10. Evaluate retrieval quality

```bash
python -m app.evaluate
```

Runs `tests/eval_questions.json` (38 hand-written questions over the sample
PDFs in `data/`) through hybrid_search and reports hit rate (did the right
document show up in the top-k?) and latency, without spending any Gemini
quota. Add `--full` to run the complete answer pipeline instead and also see
the refusal rate; see [Evaluation](#evaluation) for a sample run.

## Project structure

```
enterprise-knowledge-assistant/
├── app/                # application code
│   ├── config.py       # settings loaded from .env
│   ├── chunking.py     # PDF loading and splitting (Step 2)
│   ├── embeddings.py   # Gemini embeddings (Step 3)
│   ├── vector_store.py # ChromaDB indexing and search (Step 3)
│   ├── qa.py           # question answering with cited sources (Step 4)
│   ├── api.py          # FastAPI backend (Step 5)
│   ├── streamlit_app.py # Streamlit UI (Step 6)
│   └── evaluate.py     # retrieval hit rate and latency (Step 10)
├── chroma_db/          # the vector index (created on first run, not committed)
├── data/               # put your PDFs here (not committed)
├── tests/              # unit tests and evaluation questions
├── Dockerfile           # image shared by the api and ui containers (Step 9)
├── docker-compose.yml   # runs api + ui together (Step 9)
├── .env.example        # template for secrets
├── requirements.txt
└── README.md
```

## Evaluation

`python -m app.evaluate` against the sample PDFs in `data/` (employee
handbook, NIST Cybersecurity Framework 2.0, two ANRT telecom market reports)
over the 38 questions in `tests/eval_questions.json`:

```
38 questions, hit rate 100.0%, avg latency 1817ms, p95 5055ms
```

Hit rate is whether the expected source document appears in hybrid_search's
top-5 results; latency is per-question retrieval time (embedding the query +
vector search + BM25 + fusion), no chat-model call included. Re-run it after
changing chunking, retrieval, or `k` to see the effect; numbers will vary with
your own documents and questions.

## License

MIT
