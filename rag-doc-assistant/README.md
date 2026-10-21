# AI Document Assistant (RAG)

Ask questions about your own PDF/TXT documents. Built with Python, OpenAI, FAISS and FastAPI.

## How it works
**Indexing:** document -> extract text -> chunk (800 chars, 150 overlap) -> embed (text-embedding-3-small) -> store in FAISS
**Querying:** question -> embed -> FAISS top-4 cosine search -> prompt with retrieved context -> gpt-4o-mini answers with sources

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env       # Mac/Linux: cp .env.example .env  (then add your OpenAI key)
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000 (web UI) or http://127.0.0.1:8000/docs (API docs).

## API
| Method | Path | Description |
|---|---|---|
| POST | /upload | Upload PDF/TXT and index it |
| POST | /ask | `{"question": "..."}` -> answer + sources |
| GET | /documents | List indexed documents |
| GET | /health | Status and chunk count |

## Tests and evaluation
```bash
pytest
python -m scripts.evaluate
```

## Docker
```bash
docker build -t rag-doc-assistant .
docker run -p 8000:8000 --env-file .env rag-doc-assistant
```

## Future improvements
Hybrid BM25 + vector search, re-ranking, OCR for scanned PDFs, streaming responses, chat memory, managed vector DB.
