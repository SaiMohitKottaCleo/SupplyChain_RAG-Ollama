from sentence_transformers import SentenceTransformer
from typing import List
from app.config import EMBEDDING_MODEL
from loguru import logger


class Embedder:
    """
    Wraps the sentence-transformers embedding model.

    Model: all-MiniLM-L6-v2
      - 22M parameters, 6 transformer layers
      - Output: 384-dimensional unit vectors (L2-normalized)
      - Speed: ~5ms per sentence on CPU, ~1ms on GPU
      - Trained via contrastive learning on 1B+ sentence pairs

    Singleton: the model is loaded once per process and reused.
    Loading costs ~400ms and ~90MB RAM — paying that per query is unacceptable.
    """

    _instance = None   # class-level reference — shared across all callers

    def __new__(cls):
        """
        __new__ runs before __init__ every time you write Embedder().
        We intercept it: if an instance already exists, return that one.
        If not, create it, load the model, store it, and return it.
        This guarantees at most one model load per process lifetime.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL} ...")
            cls._instance.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Embedding model ready. Output dim: {cls._instance.model.get_sentence_embedding_dimension()}")
        return cls._instance

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts. Returns one 384-dim vector per text.

        batch_size=32: texts are processed 32 at a time through the model.
        Batching amortizes the fixed overhead of GPU/CPU kernel launches.
        Larger batches = faster throughput, but more peak memory.

        normalize_embeddings=True: forces all output vectors to unit length.
        This enables cosine similarity via dot product — critical for ChromaDB.

        show_progress_bar: only shown for large ingestion jobs (>100 texts).
        Queries are single texts — no progress bar clutter.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # sentence-transformers returns numpy arrays; ChromaDB wants Python lists
        return embeddings.tolist()

    def embed_one(self, text: str) -> List[float]:
        """Convenience wrapper — embed a single query string."""
        return self.embed([text])[0]
