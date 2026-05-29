from typing import Dict, Any
from app.generation.llm import OllamaLLM
from app.generation.prompts import SYSTEM_PROMPT, build_user_message, NO_CONTEXT_RESPONSE
from app.retrieval.context import ContextAssembler
from app.config import SIMILARITY_THRESHOLD, TOP_K_FINAL
from loguru import logger


class AnswerGenerator:
    """
    Runs the full RAG pipeline: retrieve → assemble → generate → cite.

    answer(query) returns a dict with everything the UI needs:
      answer              : the generated text (or NO_CONTEXT_RESPONSE)
      sources             : list of source document names cited
      confidence          : max retrieval similarity score (0-1)
      confident           : bool — did we exceed SIMILARITY_THRESHOLD?
      collections_searched: which ChromaDB collections were queried
      chunks_used         : how many context chunks fed to the LLM
      chunks              : the raw chunk dicts (for UI debug / transparency panel)
    """

    def __init__(self):
        self.llm       = OllamaLLM()
        self.assembler = ContextAssembler()

    def answer(self, query: str) -> Dict[str, Any]:
        """
        Full RAG pipeline for one query.

        Step 1 — Retrieve: ContextAssembler classifies, embeds, hybrid-retrieves,
                 deduplicates, and returns the best context chunks.

        Step 2 — Gate: if max similarity < SIMILARITY_THRESHOLD, the retrieved
                 context is too weak to ground an answer. Return the honest
                 "I don't know" response WITHOUT calling the LLM at all.
                 This saves ~2-5 seconds of inference time and prevents the
                 LLM from trying to answer from noise chunks.

        Step 3 — Generate: build the prompt (system + user message with injected
                 context), call Ollama, get the answer.

        Step 4 — Extract sources: deduplicate source names from all used chunks.
        """
        logger.info(f"Query: {query[:80]}")

        # ── Step 1: Retrieve ──────────────────────────────────────────────
        chunks, confidence, collections_searched = self.assembler.assemble(query)

        # ── Step 2: Confidence gate ───────────────────────────────────────
        if confidence < SIMILARITY_THRESHOLD or not chunks:
            logger.info(f"Below threshold ({confidence:.3f} < {SIMILARITY_THRESHOLD}) — abstaining")
            return {
                "answer":               NO_CONTEXT_RESPONSE,
                "sources":              [],
                "confidence":           confidence,
                "confident":            False,
                "collections_searched": collections_searched,
                "chunks_used":          0,
                "chunks":               [],
            }

        # Use only the top TOP_K_FINAL chunks for the LLM context window
        top_chunks = chunks[:TOP_K_FINAL]

        # ── Step 3: Generate ──────────────────────────────────────────────
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(query, top_chunks)},
        ]

        logger.info(f"Generating answer from {len(top_chunks)} chunks "
                    f"(confidence={confidence:.3f})")
        answer_text = self.llm.generate(messages)

        # ── Step 4: Extract sources ───────────────────────────────────────
        sources = []
        for chunk in top_chunks:
            source = chunk.get("metadata", {}).get("source", "Unknown")
            if source not in sources:
                sources.append(source)

        return {
            "answer":               answer_text,
            "sources":              sources,
            "confidence":           confidence,
            "confident":            True,
            "collections_searched": collections_searched,
            "chunks_used":          len(top_chunks),
            "chunks":               top_chunks,
        }
