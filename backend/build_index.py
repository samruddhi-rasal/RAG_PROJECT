import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "passages.json")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

os.makedirs(FAISS_DIR, exist_ok=True)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    passages = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔥 Normalize embeddings for cosine similarity
embeddings = model.encode(passages, normalize_embeddings=True)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

# 🔥 Inner product index for cosine similarity
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, os.path.join(FAISS_DIR, "rag.index"))

with open(os.path.join(FAISS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump({"passages": passages}, f)

print("Index rebuilt successfully.")