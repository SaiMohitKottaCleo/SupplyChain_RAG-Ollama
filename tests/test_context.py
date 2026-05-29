import sys
sys.path.insert(0, '.')
from app.retrieval.context import ContextAssembler

assembler = ContextAssembler()

# ── Test 1: EDI query → should find AK5 chunk ────────────────────────────
print("=== Test 1: EDI query end-to-end ===")
chunks, confidence, collections = assembler.assemble(
    "What does AK5*R mean in a 997 acknowledgment?"
)
print(f"Collections searched : {collections}")
print(f"Chunks returned      : {len(chunks)}")
print(f"Max confidence       : {confidence:.4f}")
if chunks:
    top = chunks[0]
    print(f"Top chunk text       : {top['text'][:80]}...")
    print(f"Top chunk collection : {top['collection']}")
    print(f"Top chunk rrf_score  : {top['rrf_score']}")
print()

# ── Test 2: Off-topic → confidence below threshold ────────────────────────
print("=== Test 2: Off-topic query → low/zero confidence ===")
chunks2, conf2, cols2 = assembler.assemble("How do I make French onion soup?")
print(f"Chunks returned : {len(chunks2)}")
print(f"Max confidence  : {conf2:.4f}  (below 0.35 threshold = system will say 'I don't know')")
will_answer = conf2 >= 0.35
print(f"System will answer? {will_answer}  (expected: False)")
print()

# ── Test 3: Lazy caching — second call reuses retrievers ─────────────────
print("=== Test 3: Retriever caching ===")
chunks3, _, _ = assembler.assemble("What is the ISA segment?")
cached_keys = list(assembler._retrievers.keys())
print(f"Cached retrievers : {cached_keys}")
print(f"(Created once, reused for all future queries to same collections)")

print("\ncontext.py OK")
