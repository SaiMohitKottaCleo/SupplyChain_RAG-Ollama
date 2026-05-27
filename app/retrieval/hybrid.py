from typing import List, Dict, Any
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Index
from app.config import TOP_K_DENSE, TOP_K_SPARSE, TOP_K_FINAL, SIMILARITY_THRESHOLD
from loguru import logger


class HybridRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval via
    Reciprocal Rank Fusion (RRF).

    Why RRF over learned fusion?
      - No training data needed
      - Parameter-free (k=60 is a robust universal constant)
      - Robust to score scale differences (BM25 scores are unbounded;
        cosine scores are 0-1 — RRF uses only rank, ignoring raw values)
      - Empirically matches or beats learned fusion on most IR benchmarks

    One HybridRetriever per collection. Instantiated lazily by ContextAssembler.
    """

    RRF_K = 60

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.dense  = DenseRetriever(collection_name)
        self.sparse = BM25Index(collection_name)

    def _rrf_score(self, rank: int) -> float:
        """RRF contribution for a document at zero-indexed rank position."""
        return 1.0 / (self.RRF_K + rank + 1)

    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        n: int = TOP_K_FINAL,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-n chunks using hybrid RRF fusion.

        Algorithm:
          1. Dense retrieval  → top-5 by cosine similarity
          2. BM25 retrieval   → top-5 by keyword score
          3. For every unique chunk, sum its RRF contributions
             (1/(60+rank+1) for each list it appears in)
          4. Sort by total RRF score
          5. Filter out chunks below SIMILARITY_THRESHOLD (dense score)
          6. Return top-n

        Deduplication key: first 120 chars of text.
        Using full text would be slower; 120 chars identifies the chunk
        uniquely in practice.
        """
        dense_results  = self.dense.retrieve(query_embedding, TOP_K_DENSE)
        sparse_results = self.sparse.search(query, TOP_K_SPARSE)

        # Map from dedup_key → {result dict, accumulated rrf score, scores}
        rrf_map: Dict[str, Dict] = {}

        for rank, result in enumerate(dense_results):
            key = result["text"][:120]
            if key not in rrf_map:
                rrf_map[key] = {"result": result, "rrf": 0.0, "dense_score": 0.0, "bm25_score": 0.0}
            rrf_map[key]["rrf"]         += self._rrf_score(rank)
            rrf_map[key]["dense_score"]  = result.get("score", 0.0)

        for rank, result in enumerate(sparse_results):
            key = result["text"][:120]
            if key not in rrf_map:
                rrf_map[key] = {"result": result, "rrf": 0.0, "dense_score": 0.0, "bm25_score": 0.0}
            rrf_map[key]["rrf"]        += self._rrf_score(rank)
            rrf_map[key]["bm25_score"]  = result.get("bm25_score", 0.0)

        # Sort by RRF score descending
        sorted_items = sorted(rrf_map.values(), key=lambda x: x["rrf"], reverse=True)

        # Build final result list — attach fusion scores, apply threshold
        final = []
        for item in sorted_items[:n * 2]:  # over-fetch, then filter
            result              = item["result"].copy()
            result["rrf_score"]   = round(item["rrf"], 6)
            result["dense_score"] = round(item["dense_score"], 4)
            result["bm25_score"]  = round(item["bm25_score"], 4)

            # Include if: dense score meets threshold, OR BM25 strongly matched.
            # BM25_RESCUE_THRESHOLD = 0.5: only rescue via BM25 if it scored
            # meaningfully — filters out stopword-only false positives.
            # (In a tiny test corpus, stopwords get inflated IDF scores;
            # in production this threshold keeps only true keyword hits.)
            BM25_RESCUE_THRESHOLD = 0.5
            dense_ok = result["dense_score"] >= SIMILARITY_THRESHOLD
            bm25_ok  = item["bm25_score"] >= BM25_RESCUE_THRESHOLD

            if dense_ok or bm25_ok:
                final.append(result)

        logger.debug(f"HybridRetriever '{self.collection_name}': "
                     f"dense={len(dense_results)} sparse={len(sparse_results)} "
                     f"fused={len(final)}")
        return final[:n]
