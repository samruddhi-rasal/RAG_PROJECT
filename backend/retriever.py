import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

with open(os.path.join(FAISS_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

passages = metadata["passages"]

# 🔥 Better model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load index
index = faiss.read_index(os.path.join(FAISS_DIR, "rag.index"))

def retrieve(query, top_k=5):
    query_embedding = model.encode([query], normalize_embeddings=True).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "passage": passages[idx],
            "score": float(distances[0][i])
        })

    return results