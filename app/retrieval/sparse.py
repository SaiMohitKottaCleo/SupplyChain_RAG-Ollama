import pickle
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from app.config import CHROMA_DIR
from loguru import logger


class BM25Index:
    """
    BM25 sparse retrieval index — one per ChromaDB collection.

    Persisted to disk at: chroma_db/bm25_{collection_name}.pkl
    Stored alongside the ChromaDB files so they stay in sync.

    Why BM25 alongside ChromaDB?
      ChromaDB stores vectors and text, but has no keyword search capability.
      BM25 fills that gap — exact token matching for EDI codes, acronyms,
      segment identifiers, and technical terminology the embedding model
      struggles with.

    Lifecycle:
      1. BM25Index("edi_standards")  — loads existing index from disk if present
      2. .add(chunks)                — rebuilds index with new chunks, saves
      3. .search(query, n)           — returns top-n keyword matches
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.index_path      = CHROMA_DIR / f"bm25_{collection_name}.pkl"
        self.bm25: BM25Okapi = None
        self.corpus: List[Dict[str, Any]] = []  # full text + metadata of every chunk
        self._load()

    # ── Tokenizer ──────────────────────────────────────────────────────────

    # Minimal stopword set — common English function words with near-zero
    # semantic value. Filtering them prevents inflated BM25 scores from
    # word overlap in tiny corpora. Kept small deliberately: EDI abbreviations
    # and codes must NOT be filtered (e.g. "as2", "of" in "bill of lading").
    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "it", "its", "this", "that", "and", "or", "but", "in", "on",
        "at", "to", "for", "of", "with", "by", "from", "what", "how",
        "does", "do", "did", "have", "has", "can", "not", "no", "any",
    }

    def _tokenize(self, text: str) -> List[str]:
        """
        Whitespace + lowercase tokenizer with stopword removal.

        Preserves exact token strings for EDI codes ("isa06", "ak5*r").
        Removes high-frequency English function words that carry no
        discriminative signal and inflate BM25 scores in small corpora.
        Stemming is intentionally skipped: "rejected" and "reject" are
        different enough in EDI context to treat separately.
        """
        return [
            token for token in text.lower().split()
            if token not in self.STOPWORDS and len(token) > 1
        ]

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self):
        """Load index from disk if it exists. Silent no-op if not found."""
        if not self.index_path.exists():
            return
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self.corpus = data["corpus"]
            tokenized   = [self._tokenize(d["text"]) for d in self.corpus]
            self.bm25   = BM25Okapi(tokenized)
            logger.info(f"BM25 index loaded: '{self.collection_name}' "
                        f"({len(self.corpus)} docs)")
        except Exception as e:
            logger.warning(f"Could not load BM25 index for '{self.collection_name}': {e}")
            self.corpus = []
            self.bm25   = None

    def _save(self):
        """Persist the corpus to disk. BM25 object is rebuilt from corpus on load."""
        with open(self.index_path, "wb") as f:
            pickle.dump({"corpus": self.corpus}, f)

    # ── Write path ─────────────────────────────────────────────────────────

    def add(self, chunks: List[Dict[str, Any]]):
        """
        Add new chunks and rebuild the BM25 index.

        BM25 requires rebuilding because IDF scores depend on the full corpus.
        Adding one new document changes every term's IDF value.
        Rebuilding is O(n × avg_doc_length) — fast for our corpus sizes.
        """
        self.corpus.extend(chunks)
        tokenized  = [self._tokenize(d["text"]) for d in self.corpus]
        self.bm25  = BM25Okapi(tokenized)
        self._save()
        logger.info(f"BM25 index updated: '{self.collection_name}' "
                    f"({len(self.corpus)} docs total)")

    # ── Read path ──────────────────────────────────────────────────────────

    def search(self, query: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Return top-n chunks by BM25 score.

        Scores are raw BM25 values (not normalized to 0-1).
        Chunks with score=0 are excluded — they matched zero query terms.
        Results include "bm25_score" key for use in RRF fusion.
        """
        if not self.bm25 or not self.corpus:
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Pair each score with its corpus index, sort descending
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        results = []
        for idx, score in ranked:
            if score > 0:  # only include actual matches
                results.append({
                    **self.corpus[idx],
                    "bm25_score": float(score),
                })
        return results
