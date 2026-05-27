import chromadb
from typing import List, Dict, Any
from app.config import CHROMA_DIR, COLLECTIONS
from loguru import logger


class ChromaClient:
    """
    Manages all ChromaDB collections — one per query category.

    ChromaDB 1.x notes (vs the 0.5.x spec):
      - Settings import removed; PersistentClient takes only path in 1.x
      - Telemetry disabled via ANONYMIZED_TELEMETRY env var instead
      - Collection metadata key "hnsw:space" sets the distance metric
      - list_collections() returns Collection objects (same as before)
      - add/query API is identical to 0.5.x

    All methods are classmethods — no instance needed.
    One shared _client per process (singleton at the class level).
    """

    _client = None

    # ── Client management ──────────────────────────────────────────────────

    @classmethod
    def get_client(cls) -> chromadb.PersistentClient:
        """
        Return the shared PersistentClient, creating it once if needed.
        PersistentClient writes all data to CHROMA_DIR on disk.
        On restart, it reads existing collections back into memory.
        """
        if cls._client is None:
            cls._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            logger.info(f"ChromaDB client ready. Storage: {CHROMA_DIR}")
        return cls._client

    @classmethod
    def get_or_create_collection(cls, name: str) -> chromadb.Collection:
        """
        Get an existing collection or create it if it doesn't exist.

        hnsw:space=cosine: with normalized embeddings, cosine distance
        is the correct metric. Returns distance = 1 - cosine_similarity,
        so our similarity score = 1 - distance (clean 0-1 range).
        """
        client = cls.get_client()
        return client.get_or_create_collection(
            name=name,
            metadata={
                "description": COLLECTIONS.get(name, ""),
                "hnsw:space":  "cosine",
            },
        )

    @classmethod
    def list_collections(cls) -> List[str]:
        """Return names of all existing collections."""
        client = cls.get_client()
        return [c.name for c in client.list_collections()]

    # ── Write path ─────────────────────────────────────────────────────────

    @classmethod
    def add_chunks(
        cls,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store chunks + their embeddings in a collection.

        IDs must be globally unique within a collection. We use
        "{collection}_{offset+i}" where offset = current count.
        This means re-running ingestion appends rather than overwrites.

        ChromaDB requires all metadata values to be strings, ints, or floats.
        We coerce everything to str to avoid silent type errors.
        """
        if not chunks:
            return 0

        collection = cls.get_or_create_collection(collection_name)
        offset     = collection.count()

        ids       = [f"{collection_name}_{offset + i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {k: str(v) for k, v in c.get("metadata", {}).items()}
            for c in chunks
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Stored {len(chunks)} chunks in '{collection_name}' "
                    f"(total: {offset + len(chunks)})")
        return len(chunks)

    # ── Read path ──────────────────────────────────────────────────────────

    @classmethod
    def query(
        cls,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find the n most similar chunks to query_embedding.

        Returns list of dicts, each with:
          text     : the chunk's raw text
          metadata : source, page, chunk_index, etc.
          distance : raw cosine distance (0=identical, 1=orthogonal)
          score    : cosine similarity = 1 - distance (higher = more relevant)

        Sorted by score descending (ChromaDB sorts by distance ascending).
        """
        collection = cls.get_or_create_collection(collection_name)

        if collection.count() == 0:
            return []

        # n_results cannot exceed collection size
        n = min(n_results, collection.count())

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            output.append({
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "score":    round(1.0 - distance, 4),  # cosine similarity
            })

        # Already sorted by distance asc = score desc
        return output

    # ── Stats ──────────────────────────────────────────────────────────────

    @classmethod
    def get_collection_stats(cls) -> Dict[str, int]:
        """Return chunk count per collection. Used by the UI sidebar."""
        stats = {}
        for name in COLLECTIONS.keys():
            try:
                stats[name] = cls.get_or_create_collection(name).count()
            except Exception:
                stats[name] = 0
        return stats
