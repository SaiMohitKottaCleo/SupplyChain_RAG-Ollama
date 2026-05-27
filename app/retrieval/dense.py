from typing import List, Dict, Any
from app.database.chroma import ChromaClient
from app.config import TOP_K_DENSE


class DenseRetriever:
    """
    Thin adapter over ChromaClient for dense (vector) retrieval.

    Keeps the retrieval interface uniform with BM25Index:
      both expose .retrieve() / .search() returning List[Dict].

    The HybridRetriever calls both with the same interface,
    making it easy to swap either backend without changing fusion logic.
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def retrieve(
        self,
        query_embedding: List[float],
        n: int = TOP_K_DENSE,
    ) -> List[Dict[str, Any]]:
        """
        Return top-n chunks by cosine similarity.
        Results include 'score' (cosine similarity, 0-1).
        """
        return ChromaClient.query(
            self.collection_name,
            query_embedding,
            n_results=n,
        )
