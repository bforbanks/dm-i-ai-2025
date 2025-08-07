import os
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
TOPIC_DIR = Path("data/topics")
STATEMENT_DIR = Path("data/train/statements")
ANSWER_DIR = Path("data/train/answers")
TOPIC_MAP = json.loads(Path("data/topics.json").read_text())

# Models
MINILM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
# Placeholder for ColBERT-style model if you decide to test it later
# COLBERT_MODEL = SentenceTransformer("colbert-ir/colbertv2.0")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_statements():
    records = []
    for path in sorted(STATEMENT_DIR.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((ANSWER_DIR / f"statement_{sid}.json").read_text())
        records.append((statement, ans["statement_topic"]))
    return records

def load_topic_docs() -> Dict[int, str]:
    topic_docs = {}
    for topic_path in TOPIC_DIR.iterdir():
        topic_id = TOPIC_MAP[topic_path.name]
        contents = []
        for md_file in topic_path.glob("*.md"):
            text = md_file.read_text(encoding="utf-8").strip()
            contents.append(text)
        topic_docs[topic_id] = "\n".join(contents)
    return topic_docs

# ---------------------------------------------------------------------------
# Dense embedding search
# ---------------------------------------------------------------------------
def dense_search(query: str, topic_docs: Dict[int, str], model) -> List[Tuple[int, float]]:
    corpus = list(topic_docs.values())
    topic_ids = list(topic_docs.keys())

    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    top_k_indices = np.argsort(-scores.cpu().numpy())
    return [(topic_ids[i], scores[i].item()) for i in top_k_indices[:10]]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_dense(model_name: str, model):
    print(f"\nEvaluating dense model: {model_name}")
    topic_docs = load_topic_docs()
    statements = load_statements()

    top_k_hits = [0] * 10

    for query, true_topic in tqdm(statements):
        results = dense_search(query, topic_docs, model)
        for k in range(10):
            if any(r[0] == true_topic for r in results[:k+1]):
                top_k_hits[k] += 1

    for k in range(10):
        acc = top_k_hits[k] / len(statements)
        print(f"Top-{k+1}: {acc:.3f} ({top_k_hits[k]}/{len(statements)})")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Evaluate MiniLM
    evaluate_dense("MiniLM (all-MiniLM-L6-v2)", MINILM_MODEL)

    # Placeholder: if you want to try ColBERT later, uncomment this
    # evaluate_dense("ColBERT", COLBERT_MODEL)
