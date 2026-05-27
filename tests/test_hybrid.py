import sys
sys.path.insert(0, '.')

from app.ingestion.embedder import Embedder
from app.retrieval.hybrid import HybridRetriever

e   = Embedder()
ret = HybridRetriever("edi_standards")

# ── Test 1: Exact code query — BM25 strength ─────────────────────────────
print("=== Test 1: Exact code (BM25 wins, dense confirms) ===")
q1      = "AK5*R rejected acknowledgment"
q1_vec  = e.embed_one(q1)
results = ret.retrieve(q1, q1_vec, n=3)
for r in results:
    print(f"  rrf={r['rrf_score']:.5f}  dense={r['dense_score']:.3f}  "
          f"bm25={r['bm25_score']:.3f}  chunk={r['metadata']['chunk_index']}")
print()

# ── Test 2: Semantic query — dense strength, BM25 confusion ──────────────
print("=== Test 2: Semantic query (dense wins, BM25 corrected) ===")
q2      = "What is the outermost envelope wrapping an X12 interchange?"
q2_vec  = e.embed_one(q2)
results2 = ret.retrieve(q2, q2_vec, n=3)
for r in results2:
    print(f"  rrf={r['rrf_score']:.5f}  dense={r['dense_score']:.3f}  "
          f"bm25={r['bm25_score']:.3f}  chunk={r['metadata']['chunk_index']}")
print("  (Chunk 0 = ISA segment — should rank highest or near-highest)\n")

# ── Test 3: Unrelated query — threshold filters it out ───────────────────
print("=== Test 3: Off-topic query (should return 0 results) ===")
q3      = "What is the recipe for French onion soup?"
q3_vec  = e.embed_one(q3)
results3 = ret.retrieve(q3, q3_vec, n=3)
print(f"  Results: {len(results3)}  (expected: 0 — all below threshold or no BM25 hits)")

print("\nhybrid.py OK")
