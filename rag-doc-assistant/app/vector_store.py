import os
import json
import faiss
import numpy as np
from app.config import EMBED_DIM, INDEX_DIR


class VectorStore:
    def __init__(self):
        os.makedirs(INDEX_DIR, exist_ok=True)
        self.index_path = os.path.join(INDEX_DIR, "faiss.index")
        self.meta_path = os.path.join(INDEX_DIR, "meta.json")
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            # Inner product on normalized vectors == cosine similarity
            self.index = faiss.IndexFlatIP(EMBED_DIM)
            self.meta = []

    def add(self, vectors, metadatas):
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self.meta.extend(metadatas)
        self._save()

    def search(self, query_vec, k):
        if self.index.ntotal == 0:
            return []
        q = np.array([query_vec], dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, k)
        return [{**self.meta[i], "score": float(s)}
                for s, i in zip(scores[0], ids[0]) if i != -1]

    def documents(self):
        counts = {}
        for m in self.meta:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return [{"file": f, "chunks": c} for f, c in counts.items()]

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f)
