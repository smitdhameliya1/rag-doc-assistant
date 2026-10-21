from pypdf import PdfReader
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text(path: str) -> list[dict]:
    """Return a list of {page, text}."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return [{"page": i + 1, "text": p.extract_text() or ""}
                for i, p in enumerate(reader.pages)]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [{"page": 1, "text": f.read()}]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = " ".join(text.split())  # normalize whitespace
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def process_file(path: str, filename: str) -> list[dict]:
    """Return chunks with metadata, ready to embed."""
    out = []
    for page in extract_text(path):
        for chunk in chunk_text(page["text"]):
            out.append({"text": chunk, "source": filename, "page": page["page"]})
    return out
