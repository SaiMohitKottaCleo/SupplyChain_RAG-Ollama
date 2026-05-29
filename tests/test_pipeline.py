import sys, tempfile, os
sys.path.insert(0, '.')
from app.ingestion.pipeline import IngestionPipeline

pipe = IngestionPipeline()

# ── Test 1: ingest a raw text string ──────────────────────────────────────────
print("=== Test 1: ingest raw text ===")
result = pipe.ingest(
    source=(
        "EDI 856 Ship Notice / Manifest\n\n"
        "The 856 transaction set is used to describe the contents and configuration "
        "of a shipment in various levels of detail. It can be used to list the "
        "contents of a shipment of goods as well as details at the line, order, "
        "carton, and pallet levels.\n\n"
        "Key segments:\n"
        "BSN: Beginning Segment for Ship Notice\n"
        "DTM: Date/Time Reference\n"
        "HL: Hierarchical Level\n"
        "TD1: Carrier Details (Quantity and Weight)\n"
        "TD5: Carrier Details (Routing Sequence)\n"
    ),
    collection_name="edi_standards",
    extra_metadata={"source": "edi_856_test", "source_type": "text"},
)
print(f"Status         : {result['status']}")
print(f"Docs loaded    : {result.get('docs_loaded')}")
print(f"Chunks created : {result.get('chunks_created')}")
print(f"Chunks stored  : {result.get('chunks_stored')}")

# ── Test 2: ingest a temp .txt file ──────────────────────────────────────────
print("\n=== Test 2: ingest .txt file ===")
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                  delete=False, encoding='utf-8') as f:
    f.write(
        "Supply Chain Visibility\n\n"
        "Real-time visibility in supply chains refers to the ability to track "
        "goods, shipments, and inventory levels at every stage of the supply "
        "chain process. This includes in-transit visibility, warehouse management, "
        "and last-mile delivery tracking.\n\n"
        "Benefits include reduced carrying costs, improved customer satisfaction, "
        "and faster response to disruptions."
    )
    tmp_path = f.name

result2 = pipe.ingest(
    source=tmp_path,
    collection_name="supply_chain_concepts",
    extra_metadata={"source": "visibility_test", "source_type": "txt"},
)
os.unlink(tmp_path)
print(f"Status         : {result2['status']}")
print(f"Chunks stored  : {result2.get('chunks_stored')}")

# ── Test 3: invalid collection name ──────────────────────────────────────────
print("\n=== Test 3: invalid collection ===")
result3 = pipe.ingest(source="hello", collection_name="nonexistent_collection")
print(f"Status : {result3['status']}  (expected: error)")
print(f"Error  : {result3.get('error')[:60]}")

print("\npipeline.py OK")
