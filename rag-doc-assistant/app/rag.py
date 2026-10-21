from openai import OpenAI
from app.config import OPENAI_API_KEY, CHAT_MODEL, TOP_K
from app.embeddings import embed_texts
from app.vector_store import VectorStore

client = OpenAI(api_key=OPENAI_API_KEY)
store = VectorStore()

SYSTEM_PROMPT = (
    "You are a document assistant. Answer ONLY using the provided context. "
    "If the answer is not in the context, say \"I couldn't find that in the documents.\" "
    "Be concise and mention the source file and page when relevant."
)


def ingest_chunks(chunks: list[dict]):
    vectors = embed_texts([c["text"] for c in chunks])
    store.add(vectors, chunks)


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    q_vec = embed_texts([question])[0]
    return store.search(q_vec, k)


def answer_question(question: str) -> dict:
    hits = retrieve(question)
    if not hits:
        return {"answer": "No documents indexed yet.", "sources": []}

    context = "\n\n".join(
        f"[Source: {h['source']}, page {h['page']}]\n{h['text']}" for h in hits
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [{"file": h["source"], "page": h["page"],
                     "score": round(h["score"], 3)} for h in hits],
    }
