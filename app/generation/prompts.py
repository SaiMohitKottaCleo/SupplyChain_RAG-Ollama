SYSTEM_PROMPT = """You are the Cleo Supply Chain Intelligence assistant — an expert in EDI \
standards, supply chain operations, logistics, compliance regulations, and Cleo's \
integration platform.

STRICT RULES — FOLLOW EXACTLY:
1. Answer ONLY using the provided context chunks. Do not use outside knowledge.
2. If the context does not contain enough information to answer, respond with exactly:
   "I don't have reliable information on this topic in my knowledge base."
3. Cite every factual claim: [Source: <source_name>, Chunk <chunk_index>]
4. Do not speculate, infer beyond the context, or fill gaps with general knowledge.
5. For Cleo-specific questions (what Cleo supports, allows, or provides), only answer
   from chunks labelled collection: cleo_company. Do not guess Cleo's capabilities.
6. Be precise and professional. Users are supply chain engineers and operations teams.

RESPONSE FORMAT:
- Lead with a direct answer
- Support with specific details from the context
- Cite every source inline
- End with a "Sources:" section listing all documents used"""


def build_user_message(query: str, context_chunks: list) -> str:
    """
    Inject retrieved context chunks into the user message.

    Each chunk is labelled with its source, collection, and index —
    so the model can produce accurate citations and the user can
    verify exactly which document the answer came from.
    """
    if not context_chunks:
        context_str = "No relevant context found in the knowledge base."
    else:
        parts = []
        for i, chunk in enumerate(context_chunks):
            meta        = chunk.get("metadata", {})
            source      = meta.get("source", "Unknown")
            chunk_idx   = meta.get("chunk_index", str(i))
            collection  = chunk.get("collection", "general")
            score       = chunk.get("dense_score", 0)
            parts.append(
                f"[Chunk {i+1} | Source: {source} | "
                f"Collection: {collection} | "
                f"Chunk index: {chunk_idx} | "
                f"Relevance: {score:.2f}]\n"
                f"{chunk['text']}"
            )
        context_str = "\n\n---\n\n".join(parts)

    return (
        f"CONTEXT FROM KNOWLEDGE BASE:\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"USER QUESTION: {query}\n\n"
        f"Answer using only the context above. Cite every claim."
    )


# Returned when confidence is below SIMILARITY_THRESHOLD
# or when no chunks were retrieved.
NO_CONTEXT_RESPONSE = (
    "I don't have reliable information on this topic in my current knowledge base.\n\n"
    "To get an accurate answer, you can:\n"
    "1. **Upload a relevant document** — use the upload feature to add PDFs, DOCX, or TXT files\n"
    "2. **Rephrase your question** — try more specific EDI terms or supply chain terminology\n"
    "3. **Check what's covered** — see the Knowledge Base section in the sidebar\n\n"
    "I will not guess or generate an answer without grounding it in verified documentation."
)
