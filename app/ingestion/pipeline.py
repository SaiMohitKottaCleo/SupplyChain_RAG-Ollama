from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger

from app.ingestion.loaders import DocumentLoader
from app.ingestion.chunker import SemanticChunker
from app.ingestion.embedder import Embedder
from app.database.chroma import ChromaClient
from app.retrieval.sparse import BM25Index
from app.config import COLLECTIONS


class IngestionPipeline:
    """
    Orchestrates the full ingestion pipeline:
        Load → Chunk → Embed → Store (ChromaDB + BM25)

    Both stores are always updated together to keep dense and sparse
    retrieval indices in sync. A mismatch would cause hybrid retrieval
    to return inconsistent rankings.

    Singleton-style shared instances for Embedder and ChromaClient
    (they are themselves singletons — this just holds references).
    """

    def __init__(self):
        self.loader   = DocumentLoader()
        self.chunker  = SemanticChunker()
        self.embedder = Embedder()
        self.chroma   = ChromaClient()

    def ingest(
        self,
        source: str,
        collection_name: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a single source into the named collection.

        Args:
            source:           file path, URL, or raw text string
            collection_name:  one of the 8 COLLECTIONS
            extra_metadata:   optional dict merged into every chunk's metadata

        Returns:
            {
              "status":          "ok" | "error"
              "source":          original source string
              "collection":      collection_name
              "docs_loaded":     number of raw documents loaded
              "chunks_created":  number of semantic chunks produced
              "chunks_stored":   number of chunks actually stored (deduped)
              "error":           error message if status == "error"
            }
        """
        if collection_name not in COLLECTIONS:
            return {
                "status":  "error",
                "source":  source,
                "collection": collection_name,
                "error":   f"Unknown collection '{collection_name}'. "
                           f"Valid: {COLLECTIONS}",
            }

        logger.info(f"Ingesting '{source}' → collection '{collection_name}'")

        # ── Step 1: Load ──────────────────────────────────────────────────
        try:
            docs = self.loader.load(source)
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return {"status": "error", "source": source,
                    "collection": collection_name, "error": str(e)}

        if not docs:
            return {"status": "error", "source": source,
                    "collection": collection_name,
                    "error": "No content extracted from source"}

        # Merge extra_metadata into every doc
        if extra_metadata:
            for doc in docs:
                doc["metadata"].update(extra_metadata)

        logger.info(f"  Loaded {len(docs)} document section(s)")

        # ── Step 2: Chunk ─────────────────────────────────────────────────
        try:
            chunks = self.chunker.chunk_many(docs)
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            return {"status": "error", "source": source,
                    "collection": collection_name, "error": str(e)}

        if not chunks:
            return {"status": "error", "source": source,
                    "collection": collection_name,
                    "error": "No chunks produced after chunking"}

        logger.info(f"  Chunked into {len(chunks)} semantic chunks")

        # ── Step 3: Embed + store in ChromaDB ────────────────────────────
        texts = [c["text"] for c in chunks]
        try:
            embeddings = self.embedder.embed(texts)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return {"status": "error", "source": source,
                    "collection": collection_name, "error": str(e)}

        try:
            stored = self.chroma.add_chunks(collection_name, chunks, embeddings)
        except Exception as e:
            logger.error(f"ChromaDB storage failed: {e}")
            return {"status": "error", "source": source,
                    "collection": collection_name, "error": str(e)}

        logger.info(f"  Stored {stored} chunks in ChromaDB '{collection_name}'")

        # ── Step 4: Store in BM25 ─────────────────────────────────────────
        try:
            bm25 = BM25Index(collection_name)
            bm25.add(chunks)
        except Exception as e:
            logger.error(f"BM25 storage failed: {e}")
            return {"status": "error", "source": source,
                    "collection": collection_name, "error": str(e)}

        logger.info(f"  BM25 index updated for '{collection_name}'")

        return {
            "status":         "ok",
            "source":         source,
            "collection":     collection_name,
            "docs_loaded":    len(docs),
            "chunks_created": len(chunks),
            "chunks_stored":  stored,
        }

    def ingest_many(
        self,
        sources: List[Dict[str, str]],
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Ingest multiple sources. Each item: {"source": ..., "collection": ...}.

        progress_callback(i, total, result) — optional; used by Streamlit
        progress bar and by the CLI ingest script.
        """
        results = []
        total   = len(sources)
        for i, item in enumerate(sources):
            result = self.ingest(
                source=item["source"],
                collection_name=item["collection"],
                extra_metadata=item.get("metadata"),
            )
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, total, result)
        return results
