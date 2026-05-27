import sys
sys.path.insert(0, '.')

from app.ingestion.embedder import Embedder
from app.database.chroma import ChromaClient

e = Embedder()

# ── Step 1: store 4 EDI chunks ──────────────────────────────────────────
chunks = [
    {
        "text": "The ISA segment is the interchange control header in X12 EDI. "
                "It contains sender ID, receiver ID, and the interchange control number.",
        "metadata": {"source": "test_doc.txt", "chunk_index": "0"},
    },
    {
        "text": "The GS segment opens a functional group. It identifies the type "
                "of transaction sets within the group and assigns a control number.",
        "metadata": {"source": "test_doc.txt", "chunk_index": "1"},
    },
    {
        "text": "A 997 functional acknowledgment confirms receipt of a transaction set. "
                "AK5*A means accepted. AK5*R means rejected.",
        "metadata": {"source": "test_doc.txt", "chunk_index": "2"},
    },
    {
        "text": "The X12 850 purchase order transaction set is sent by a buyer to a "
                "seller to place an order for goods or services.",
        "metadata": {"source": "test_doc.txt", "chunk_index": "3"},
    },
]

embeddings = e.embed([c["text"] for c in chunks])
added = ChromaClient.add_chunks("edi_standards", chunks, embeddings)
print(f"Stored {added} chunks into 'edi_standards'")

# ── Step 2: query with a natural language question ───────────────────────
query   = "What does AK5*R mean in an EDI acknowledgment?"
q_vec   = e.embed_one(query)
results = ChromaClient.query("edi_standards", q_vec, n_results=3)

print(f"\nQuery : {query}")
print(f"Results (best match first):")
for i, r in enumerate(results):
    print(f"  [{i+1}] score={r['score']:.4f}  chunk_index={r['metadata']['chunk_index']}")
    print(f"       {r['text'][:90]}...")

# ── Step 3: verify the top result IS the 997/AK5 chunk ──────────────────
top_chunk_index = results[0]["metadata"]["chunk_index"]
assert top_chunk_index == "2", f"Expected chunk 2, got {top_chunk_index}"
print(f"\nAssertion passed: top result = chunk 2 (the 997/AK5 chunk)")

# ── Step 4: collection stats ─────────────────────────────────────────────
stats = ChromaClient.get_collection_stats()
print(f"\nCollection stats:")
for name, count in stats.items():
    marker = "<-- has data" if count > 0 else ""
    print(f"  {name}: {count} chunks  {marker}")

print("\nchroma.py OK")
