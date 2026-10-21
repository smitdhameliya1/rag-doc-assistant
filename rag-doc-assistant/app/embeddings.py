from openai import OpenAI
from app.config import OPENAI_API_KEY, EMBED_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), 100):  # batch of 100
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 100])
        vectors.extend(d.embedding for d in resp.data)
    return vectors
