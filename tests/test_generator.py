import sys
sys.path.insert(0, '.')
from app.generation.generator import AnswerGenerator

gen = AnswerGenerator()

print("=== LLM availability check ===")
print(f"Ollama online: {gen.llm.is_available()}\n")

# ── Test 1: Question the system CAN answer (from our 4 test chunks) ──────
print("=== Test 1: Answerable EDI question ===")
result = gen.answer("What does AK5*R mean in a 997 acknowledgment?")
print(f"Confident     : {result['confident']}")
print(f"Confidence    : {result['confidence']:.3f}")
print(f"Chunks used   : {result['chunks_used']}")
print(f"Sources       : {result['sources']}")
print(f"Collections   : {result['collections_searched']}")
print()
print("--- Answer ---")
print(result['answer'])
print()

# ── Test 2: Question the system CANNOT answer (off-topic) ────────────────
print("=== Test 2: Unanswerable off-topic question ===")
result2 = gen.answer("What is the capital of France?")
print(f"Confident  : {result2['confident']}  (expected: False)")
print(f"Confidence : {result2['confidence']:.3f}  (expected: < 0.35)")
print()
print("--- Response ---")
print(result2['answer'][:200])
print()
print("generator.py OK")
