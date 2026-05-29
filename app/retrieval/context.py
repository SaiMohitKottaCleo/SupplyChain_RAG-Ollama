from typing import List, Dict, Any, Tuple
from app.retrieval.classifier import QueryClassifier
from app.retrieval.hybrid import HybridRetriever
from app.ingestion.embedder import Embedder
from app.config import TOP_K_FINAL, SIMILARITY_THRESHOLD
from loguru import logger


class ContextAssembler:
    """
    Orchestrates the full retrieval pipeline for a single query.

    Responsibilities:
      1. Classify query → relevant collections
      2. Embed query (once, reused across all collection searches)
      3. Hybrid-retrieve from each relevant collection
      4. Deduplicate results across collections
      5. Sort globally by RRF score
      6. Return assembled context + confidence signal

    Retrievers are lazily instantiated and cached — creating a
    HybridRetriever loads the BM25 index from disk, so we only
    pay that cost once per collection per process lifetime.
    """

    def __init__(self):
        self.classifier = QueryClassifier()
        self.embedder   = Embedder()
        self._retrievers: Dict[str, HybridRetriever] = {}

    def _get_retriever(self, collection_name: str) -> HybridRetriever:
        """Return cached retriever, creating it on first access."""
        if collection_name not in self._retrievers:
            self._retrievers[collection_name] = HybridRetriever(collection_name)
        return self._retrievers[collection_name]

    def assemble(
        self,
        query: str,
    ) -> Tuple[List[Dict[str, Any]], float, List[str]]:
        """
        Full retrieval pipeline. Returns:
          chunks              : list of relevant chunk dicts (text + metadata + scores)
          max_confidence      : highest dense similarity score found (0-1)
                                Used by generator to decide whether to answer or abstain
          collections_searched: names of collections that were queried
        """
        # Step 1: Classify → top 3 collections maximum
        collection_rankings = self.classifier.classify(query)
        collections_to_search = [col for col, _ in collection_rankings[:3]]

        logger.debug(f"Query classified to: {collections_to_search}")

        # Step 2: Embed query once
        query_embedding = self.embedder.embed_one(query)

        # Step 3: Retrieve from each collection
        all_chunks: List[Dict[str, Any]] = []
        max_confidence = 0.0

        for collection_name in collections_to_search:
            retriever = self._get_retriever(collection_name)
            results   = retriever.retrieve(query, query_embedding, n=TOP_K_FINAL)

            for chunk in results:
                chunk["collection"] = collection_name
                all_chunks.append(chunk)
                max_confidence = max(max_confidence, chunk.get("dense_score", 0.0))

        # Step 4: Deduplicate by text content
        # Same chunk can appear in multiple collections if data overlaps.
        # Dedup key: first 150 chars — unique enough, faster than full text.
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for chunk in all_chunks:
            key = chunk["text"][:150]
            if key not in seen:
                seen.add(key)
                deduped.append(chunk)

        # Step 5: Sort globally by RRF score (best first)
        deduped.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

        # Return up to TOP_K_FINAL * 2 chunks (generator will use top TOP_K_FINAL)
        final_chunks = deduped[: TOP_K_FINAL * 2]

        logger.debug(
            f"ContextAssembler: {len(all_chunks)} raw → "
            f"{len(deduped)} deduped → {len(final_chunks)} returned | "
            f"max_confidence={max_confidence:.3f}"
        )

        return final_chunks, max_confidence, collections_to_search
