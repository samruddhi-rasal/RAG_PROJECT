import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

from retriever import retrieve
from rag_pipeline import generate_answer

# =====================================================
# DOWNLOAD NLTK RESOURCES (RUNS ONLY FIRST TIME)
# =====================================================

nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

# =====================================================
# CONFIG
# =====================================================

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

TOP_K = 5

TEST_DATA = [
    {
        "question": "Where is Uruguay located?",
        "ground_truth": "Uruguay is a country located in the southeastern part of South America.",
        "ground_passage_keyword": "southeastern part of South America"
    },
    {
        "question": "What is the capital of Uruguay?",
        "ground_truth": "Montevideo is the capital of Uruguay.",
        "ground_passage_keyword": "Montevideo"
    }
]

# =====================================================
# NORMALIZATION
# =====================================================

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

# =====================================================
# RETRIEVAL METRICS
# =====================================================

def evaluate_retrieval():
    recall_scores = []
    mrr_scores = []

    for item in TEST_DATA:
        question = item["question"]
        keyword = item["ground_passage_keyword"]

        docs = retrieve(question, top_k=TOP_K)

        found = False
        rank = 0

        for i, doc in enumerate(docs):
            if keyword.lower() in doc["passage"].lower():
                found = True
                rank = i + 1
                break

        recall_scores.append(1 if found else 0)

        if found:
            mrr_scores.append(1 / rank)
        else:
            mrr_scores.append(0)

    return np.mean(recall_scores), np.mean(mrr_scores)

# =====================================================
# ANSWER METRICS
# =====================================================

def exact_match(pred, truth):
    return 1 if normalize(pred) == normalize(truth) else 0


def f1_score(pred, truth):
    pred_tokens = normalize(pred)
    truth_tokens = normalize(truth)

    common = set(pred_tokens) & set(truth_tokens)

    if len(common) == 0:
        return 0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


def semantic_similarity(pred, truth):
    embeddings = EMBED_MODEL.encode([pred, truth], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))

# =====================================================
# ANSWER QUALITY EVALUATION
# =====================================================

def evaluate_answer_quality():
    exact_scores = []
    f1_scores = []
    semantic_scores = []

    for item in TEST_DATA:
        question = item["question"]
        truth = item["ground_truth"]

        prediction = generate_answer(question)["answer"]

        em = exact_match(prediction, truth)
        f1 = f1_score(prediction, truth)
        sem = semantic_similarity(prediction, truth)

        exact_scores.append(em)
        f1_scores.append(f1)
        semantic_scores.append(sem)

        print("\nQuestion:", question)
        print("Prediction:", prediction)
        print("Ground Truth:", truth)
        print("Exact Match:", em)
        print("F1 Score:", round(f1, 3))
        print("Semantic Similarity:", round(sem, 3))
        print("----------------------------------")

    return (
        np.mean(exact_scores),
        np.mean(f1_scores),
        np.mean(semantic_scores)
    )

# =====================================================
# RUN FULL EVALUATION
# =====================================================

def run_full_evaluation():
    print("\n===== RAG EVALUATION REPORT =====\n")

    recall, mrr = evaluate_retrieval()
    em, f1, sem = evaluate_answer_quality()

    print("\n===== FINAL METRICS =====\n")

    print("Retrieval Metrics")
    print("-----------------")
    print(f"Recall@{TOP_K}: {recall:.3f}")
    print(f"MRR: {mrr:.3f}\n")

    print("Answer Quality Metrics")
    print("----------------------")
    print(f"F1 Score: {f1:.3f}")
    print(f"Semantic Similarity: {sem:.3f}")

    print("\n=================================\n")


if __name__ == "__main__":
    run_full_evaluation()