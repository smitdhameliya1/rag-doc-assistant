import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.ingest import process_file
from app.rag import ingest_chunks, answer_question, store

app = FastAPI(title="AI Document Assistant (RAG)")
UPLOAD_DIR = "data/docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class Question(BaseModel):
    question: str


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": store.index.ntotal}


@app.get("/documents")
def documents():
    return store.documents()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF and TXT files are supported")
    path = os.path.join(UPLOAD_DIR, os.path.basename(file.filename))
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks = process_file(path, os.path.basename(file.filename))
    if not chunks:
        raise HTTPException(422, "No readable text found (scanned PDF?)")
    ingest_chunks(chunks)
    return {"file": file.filename, "chunks_added": len(chunks)}


@app.post("/ask")
def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(400, "Question is empty")
    return answer_question(q.question)
