from retriever import retrieve
from sentence_transformers import SentenceTransformer
import numpy as np
import re

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def is_valid_sentence(sentence):
    # Ignore very short titles
    if len(sentence.split()) < 6:
        return False
    
    # Ignore map/title-like sentences
    if "map of" in sentence.lower():
        return False

    return True

def generate_answer(query):
    try:
        docs = retrieve(query, top_k=5)

        if not docs:
            return {
                "answer": "No relevant information found.",
                "confidence": 0.0,
                "evidence": []
            }

        query_embedding = sentence_model.encode(
            query, normalize_embeddings=True
        )

        best_sentence = ""
        best_score = -1
        best_passage = ""

        for doc in docs:
            passage = doc["passage"]
            sentences = re.split(r'(?<=[.!?]) +', passage)

            valid_sentences = [
                s for s in sentences if is_valid_sentence(s)
            ]

            if not valid_sentences:
                continue

            sentence_embeddings = sentence_model.encode(
                valid_sentences,
                normalize_embeddings=True
            )

            scores = np.dot(sentence_embeddings, query_embedding)

            max_index = np.argmax(scores)
            max_score = float(scores[max_index])

            if max_score > best_score:
                best_score = max_score
                best_sentence = valid_sentences[max_index]
                best_passage = passage

        return {
            "answer": best_sentence.strip(),
            "confidence": round(best_score, 3),
            "evidence": [best_passage]
        }

    except Exception as e:
        print("RAG ERROR:", e)
        return {
            "answer": "Backend error",
            "confidence": 0.0,
            "evidence": []
        }