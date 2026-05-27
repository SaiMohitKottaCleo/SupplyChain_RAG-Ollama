from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent          # cleo-rag/
DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
SYNTHETIC_QA_DIR = DATA_DIR / "synthetic_qa"
CHROMA_DIR     = BASE_DIR / "chroma_db"

# Create directories if they don't exist
for _d in [RAW_DIR, PROCESSED_DIR, SYNTHETIC_QA_DIR, CHROMA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Embedding model ───────────────────────────────────────────────────────
# Runs on CPU. ~90MB download on first use.
# 384-dimensional output vectors, trained for semantic similarity.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ── Ollama LLM ────────────────────────────────────────────────────────────
# Q4_K_M quantization — fits in 4GB VRAM with room for KV cache.
# phi3:mini = 3.8B params, ~2.2GB on disk
OLLAMA_MODEL    = "phi3:mini"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── ChromaDB collections ──────────────────────────────────────────────────
# One collection per query category. Each has its own HNSW index and
# BM25 index. The query classifier routes queries to the right collection(s).
COLLECTIONS = {
    "edi_standards":         "EDI standards, X12 transaction sets, segments, loops, acknowledgments",
    "supply_chain_concepts": "Supply chain terminology, inventory, logistics, demand planning",
    "integration_onboarding":"Trading partner onboarding, connection setup, testing, compliance",
    "troubleshooting":       "Error resolution, 997 rejections, error codes, debugging",
    "logistics_shipping":    "Freight, carriers, shipping labels, GS1, SCAC codes, transportation",
    "compliance_regulations":"FSMA, DSCSA, retailer EDI mandates, HIPAA, regulatory requirements",
    "cleo_company":          "Cleo products, capabilities, policies, what Cleo supports",
    "uploaded_documents":    "User-uploaded documents — runtime ingestion",
}

# ── Chunking parameters ───────────────────────────────────────────────────
# 512 tokens ≈ 2048 characters — the sweet spot for retrieval quality.
# Too small: loses context. Too large: dilutes the embedding signal.
CHUNK_SIZE     = 512   # approximate tokens per chunk
CHUNK_OVERLAP  = 64    # tokens of overlap between consecutive chunks
MIN_CHUNK_SIZE = 100   # discard chunks shorter than this (noise)

# ── Retrieval parameters ──────────────────────────────────────────────────
TOP_K_DENSE    = 5     # candidates from vector (semantic) search
TOP_K_SPARSE   = 5     # candidates from BM25 (keyword) search
TOP_K_FINAL    = 4     # after RRF fusion, feed this many chunks to the LLM

# If the best retrieval score is below this, we say "I don't know"
# instead of hallucinating a low-confidence answer.
SIMILARITY_THRESHOLD = 0.35

# ── Generation parameters ─────────────────────────────────────────────────
MAX_TOKENS   = 1024    # max tokens in LLM response
TEMPERATURE  = 0.1     # near-zero → deterministic, consistent answers
                       # high temperature → creative but inconsistent
