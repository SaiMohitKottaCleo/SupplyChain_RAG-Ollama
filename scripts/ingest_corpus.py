"""
Bulk ingestion script: scans data/raw/ and ingests every document.

Collection routing is determined by the subfolder name:
    data/raw/edi_standards/         → edi_standards
    data/raw/supply_chain_concepts/ → supply_chain_concepts
    data/raw/troubleshooting/       → troubleshooting
    ... (one folder per collection name)

Any file directly in data/raw/ (no subfolder) goes to → uploaded_documents

Supported file types: .pdf, .docx, .txt, .md

Usage:
    python scripts/ingest_corpus.py
    python scripts/ingest_corpus.py --dry-run     (list files without ingesting)
    python scripts/ingest_corpus.py --collection edi_standards  (one collection only)
"""
import sys
import argparse
from pathlib import Path

# Ensure project root is on sys.path when run from scripts/ or project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.pipeline import IngestionPipeline
from app.config import COLLECTIONS, DATA_DIR
from loguru import logger


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def discover_files(data_raw: Path, target_collection: str = None):
    """
    Walk data/raw/ and build a list of (file_path, collection_name) pairs.

    Routing logic:
      - data/raw/<collection_name>/file.pdf  →  collection_name
      - data/raw/file.pdf                    →  uploaded_documents
    """
    pairs = []

    for path in sorted(data_raw.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        # Determine collection from parent folder name
        parent = path.parent.name
        if parent in COLLECTIONS:
            collection = parent
        else:
            collection = "uploaded_documents"

        if target_collection and collection != target_collection:
            continue

        pairs.append((path, collection))

    return pairs


def run(dry_run: bool = False, target_collection: str = None):
    data_raw = DATA_DIR / "raw"

    if not data_raw.exists():
        logger.error(f"data/raw/ not found: {data_raw}")
        sys.exit(1)

    pairs = discover_files(data_raw, target_collection)

    if not pairs:
        logger.warning("No documents found in data/raw/")
        print("\n  Put your documents here:")
        for col in COLLECTIONS:
            print(f"    data/raw/{col}/your_file.pdf")
        return

    print(f"\n{'DRY RUN — ' if dry_run else ''}Found {len(pairs)} document(s):\n")
    for path, col in pairs:
        rel = path.relative_to(data_raw)
        print(f"  [{col:30s}]  {rel}")

    if dry_run:
        print("\n(Dry run — nothing ingested. Remove --dry-run to ingest.)")
        return

    print()
    pipeline  = IngestionPipeline()
    ok_count  = 0
    err_count = 0

    def on_progress(i, total, result):
        nonlocal ok_count, err_count
        status = result["status"]
        src    = Path(result["source"]).name if result["source"] else "?"
        col    = result["collection"]
        if status == "ok":
            ok_count += 1
            chunks = result.get("chunks_stored", 0)
            print(f"  [{i:3d}/{total}] OK      {src} → {col} ({chunks} chunks)")
        else:
            err_count += 1
            print(f"  [{i:3d}/{total}] ERROR   {src} → {col}: {result.get('error', '?')}")

    sources = [
        {"source": str(path), "collection": col}
        for path, col in pairs
    ]

    pipeline.ingest_many(sources, progress_callback=on_progress)

    print(f"\n{'─'*60}")
    print(f"  Done: {ok_count} ingested, {err_count} failed")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into CSCI RAG")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without ingesting")
    parser.add_argument("--collection", type=str, default=None,
                        help="Only ingest files for this collection")
    args = parser.parse_args()
    run(dry_run=args.dry_run, target_collection=args.collection)
