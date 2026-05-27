import sys
sys.path.insert(0, '.')

from app.retrieval.sparse import BM25Index

# Same 4 chunks as the ChromaDB test
chunks = [
    {"text": "The ISA segment is the interchange control header in X12 EDI. "
             "It contains sender ID, receiver ID, and the interchange control number.",
     "metadata": {"source": "test_doc.txt", "chunk_index": "0"}},
    {"text": "The GS segment opens a functional group. It identifies the type "
             "of transaction sets within the group and assigns a control number.",
     "metadata": {"source": "test_doc.txt", "chunk_index": "1"}},
    {"text": "A 997 functional acknowledgment confirms receipt of a transaction set. "
             "AK5*A means accepted. AK5*R means rejected.",
     "metadata": {"source": "test_doc.txt", "chunk_index": "2"}},
    {"text": "The X12 850 purchase order transaction set is sent by a buyer to a "
             "seller to place an order for goods or services.",
     "metadata": {"source": "test_doc.txt", "chunk_index": "3"}},
]

idx = BM25Index("edi_standards")
idx.add(chunks)

# ── Test 1: exact code match — BM25 strength ─────────────────────────────
print("=== Test 1: Exact EDI code match ===")
results = idx.search("AK5*R rejected", n=3)
for r in results:
    print(f"  bm25={r['bm25_score']:.4f}  chunk={r['metadata']['chunk_index']}  "
          f"{r['text'][:70]}...")

top = results[0]["metadata"]["chunk_index"]
assert top == "2", f"Expected chunk 2, got {top}"
print(f"  PASS: top result = chunk 2 (997/AK5 chunk)\n")

# ── Test 2: natural language — semantic query (BM25's weakness) ──────────
print("=== Test 2: Natural language (BM25 weakness) ===")
results2 = idx.search("What is the outermost envelope of an interchange?", n=3)
for r in results2:
    print(f"  bm25={r['bm25_score']:.4f}  chunk={r['metadata']['chunk_index']}  "
          f"{r['text'][:70]}...")
print("  (Note: 'outermost envelope' = ISA, but BM25 can only match exact words)")
print()

# ── Test 3: term that appears nowhere ────────────────────────────────────
print("=== Test 3: Zero-match query ===")
results3 = idx.search("French onion soup")
print(f"  Results: {len(results3)}  (expected: 0 — no matching terms)")
assert len(results3) == 0

print("\nsparse.py OK")
