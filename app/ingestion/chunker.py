import re
from typing import List, Dict, Any
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


class SemanticChunker:
    """
    Splits documents into retrieval-sized chunks that respect semantic boundaries.

    Strategy (in priority order):
      1. Split at paragraph breaks (double newlines) — strongest boundary
      2. Split at numbered list items (1. 2. 3.) — each item is a unit
      3. Split at EDI segment definitions (ISA: GS: ST: etc.) — domain-specific
      4. Split at Markdown headers (# ## ###) — structural boundary
      5. Fall back to token-window splitting if no boundaries found

    Overlap: the last ~64 tokens of each chunk are prepended to the next.
    This preserves cross-boundary context without duplicating full chunks.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    # ── Semantic boundary detection ────────────────────────────────────────

    def _split_into_semantic_units(self, text: str) -> List[str]:
        """
        Split text at semantic boundaries using regex.
        Returns a list of non-empty text units.
        """
        patterns = [
            r'\n\n+',              # paragraph break (strongest signal)
            r'\n(?=\d+[\.\)])',    # numbered list: "1." or "1)"
            r'\n(?=[A-Z]{2,3}:)', # EDI segment defs: "ISA:" "GS:" "ST:"
            r'\n(?=#{1,4} )',      # Markdown headers: "# Section"
        ]
        combined_pattern = '|'.join(patterns)
        units = re.split(combined_pattern, text)
        return [u.strip() for u in units if u.strip()]

    # ── Token counting (approximation) ────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """
        Approximate token count: 4 characters ≈ 1 token.

        Why approximate? Running a real tokenizer (like tiktoken or HuggingFace
        tokenizers) on every chunk is slow. The 4-char heuristic is accurate to
        within ~10% for English prose. For EDI data (many short codes), real
        token count may be slightly higher, but 512 tokens of wiggle room
        absorbs this error.
        """
        return max(1, len(text) // 4)

    # ── Core chunking logic ────────────────────────────────────────────────

    def chunk(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document dict into multiple chunk dicts.

        Algorithm:
          - Split document text into semantic units
          - Accumulate units into a window until chunk_size is reached
          - When the window is full: emit chunk, keep overlap tail, continue
          - Emit final chunk at end of document
        """
        text     = doc["text"]
        metadata = doc.get("metadata", {})
        units    = self._split_into_semantic_units(text)

        chunks        = []
        current_units: List[str] = []
        current_tokens = 0
        chunk_index    = 0

        for unit in units:
            unit_tokens = self._count_tokens(unit)

            # If adding this unit would overflow, emit current window as a chunk
            if current_tokens + unit_tokens > self.chunk_size and current_units:
                chunk_text = "\n\n".join(current_units)

                # Only emit if chunk meets minimum size (skip stub chunks)
                if self._count_tokens(chunk_text) >= MIN_CHUNK_SIZE // 4:
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_index":  chunk_index,
                            "chunk_tokens": current_tokens,
                        }
                    })
                    chunk_index += 1

                # Build overlap tail: walk backward through current_units,
                # collecting units until we've accumulated ~overlap tokens.
                overlap_units:  List[str] = []
                overlap_tokens = 0
                for prev_unit in reversed(current_units):
                    prev_tokens = self._count_tokens(prev_unit)
                    if overlap_tokens + prev_tokens <= self.overlap:
                        overlap_units.insert(0, prev_unit)
                        overlap_tokens += prev_tokens
                    else:
                        break

                current_units  = overlap_units
                current_tokens = overlap_tokens

            current_units.append(unit)
            current_tokens += unit_tokens

        # Emit the final remaining chunk
        if current_units:
            chunk_text = "\n\n".join(current_units)
            if self._count_tokens(chunk_text) >= MIN_CHUNK_SIZE // 4:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index":  chunk_index,
                        "chunk_tokens": current_tokens,
                    }
                })

        return chunks

    def chunk_many(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk a list of documents. Returns flat list of all chunks."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))
        return all_chunks
