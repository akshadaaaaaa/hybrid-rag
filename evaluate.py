"""
evaluate.py — Retrieval evaluation for HybridRAG
Compares semantic vs keyword vs hybrid search using:
  - Precision@K
  - Mean Reciprocal Rank (MRR)
  - Normalised Discounted Cumulative Gain (NDCG@K)

Usage:
    python evaluate.py

Expects:
    data/eval_queries.json  — ground truth queries + relevant chunks
    data/complaints.txt     — your document corpus (one chunk per line)
"""

import json
import math
import numpy as np
from hybrid_search import HybridSearch


# ── helpers ──────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of top-K retrieved results that are relevant."""
    retrieved_k = retrieved[:k]
    hits = sum(1 for chunk in retrieved_k if any(rel in chunk or chunk in rel for rel in relevant))
    return hits / k


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1/rank of the first relevant result. 0 if none found."""
    for rank, chunk in enumerate(retrieved, start=1):
        if any(rel in chunk or chunk in rel for rel in relevant):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Normalised DCG@K. Rewards relevant results appearing earlier."""
    def dcg(hits):
        return sum(
            hit / math.log2(i + 2)          # i+2 because enumerate starts at 0
            for i, hit in enumerate(hits[:k])
        )

    hits = [
        1 if any(rel in chunk or chunk in rel for rel in relevant) else 0
        for chunk in retrieved[:k]
    ]
    ideal_hits = sorted(hits, reverse=True)   # best possible ordering

    actual_dcg = dcg(hits)
    ideal_dcg  = dcg(ideal_hits)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ── evaluation runner ─────────────────────────────────────────────────────────

def evaluate(corpus_path: str, eval_path: str, k: int = 3) -> dict:
    """
    Run evaluation across all search methods.

    Args:
        corpus_path : path to .txt corpus (one chunk per line)
        eval_path   : path to eval_queries.json
        k           : number of results to evaluate at

    Returns:
        dict of metrics per search method
    """
    # Load corpus
    with open(corpus_path, "r") as f:
        chunks = [line.strip() for line in f if line.strip()]

    # Load ground truth
    with open(eval_path, "r") as f:
        eval_queries = json.load(f)

    # Index
    searcher = HybridSearch()
    searcher.add_documents(chunks)

    methods = ["semantic", "keyword", "hybrid"]
    results = {m: {"precision": [], "mrr": [], "ndcg": []} for m in methods}

    for item in eval_queries:
        query    = item["query"]
        relevant = item["relevant_chunks"]

        for method in methods:
            if method == "semantic":
                raw = searcher.semantic_search(query, k=k)
            elif method == "keyword":
                raw = searcher.keyword_search(query, k=k)
            else:
                raw = searcher.hybrid_search(query, k=k)

            retrieved = [chunk for chunk, _ in raw]

            results[method]["precision"].append(precision_at_k(retrieved, relevant, k))
            results[method]["mrr"].append(reciprocal_rank(retrieved, relevant))
            results[method]["ndcg"].append(ndcg_at_k(retrieved, relevant, k))

    # Aggregate
    summary = {}
    for method in methods:
        summary[method] = {
            f"Precision@{k}": round(np.mean(results[method]["precision"]), 4),
            "MRR":            round(np.mean(results[method]["mrr"]), 4),
            f"NDCG@{k}":      round(np.mean(results[method]["ndcg"]), 4),
        }

    return summary


# ── pretty print ──────────────────────────────────────────────────────────────

def print_results(summary: dict, k: int = 3) -> None:
    print("\n" + "=" * 52)
    print(f"  RETRIEVAL EVALUATION RESULTS  (K={k})")
    print("=" * 52)
    print(f"{'Method':<12} {'Precision@K':>12} {'MRR':>8} {'NDCG@K':>10}")
    print("-" * 52)
    for method, metrics in summary.items():
        p  = metrics[f"Precision@{k}"]
        m  = metrics["MRR"]
        n  = metrics[f"NDCG@{k}"]
        print(f"{method:<12} {p:>12.4f} {m:>8.4f} {n:>10.4f}")
    print("=" * 52)

    # Which method wins each metric?
    print("\n  Winner per metric:")
    for metric_key in [f"Precision@{k}", "MRR", f"NDCG@{k}"]:
        winner = max(summary, key=lambda m: summary[m][metric_key])
        print(f"  {metric_key:<14} → {winner}")
    print()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CORPUS = "data/complaints.txt"     # adjust filename if different
    EVALS  = "data/eval_queries.json"
    K      = 3

    summary = evaluate(CORPUS, EVALS, k=K)
    print_results(summary, k=K)